"""Ownership boundaries for the existing homogeneous read_file policy."""

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import Field, ValidationError

import core.brain_batch_policy as policy
import core.brain_provider as provider_module
import core.protocol.controller as controller_module
import tools.registry as registry_module
from core.models import ReadFileRequest
from core.protocol.controller import CortexController
from core.protocol.enums import BrainOutcomeKind as Kind, ControllerDecisionType as Decision
from test_brain_outcomes import SequenceModel, brain_input, native_action
from test_brain_read_file_batch import initial_state, input_for, native_batch, proposal
from tools.file_ops import get_file_tools
from tools.registry import ToolRegistry, get_tool_argument_schema, get_tool_definition


def change_read_definition(monkeypatch, **changes):
    monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", tuple(
        replace(definition, **changes) if definition.name == "read_file" else definition
        for definition in registry_module.TOOL_DEFINITIONS
    ))


def test_eligible_read_file_normalizes_defaults_and_coercion_from_executable():
    registry = ToolRegistry.from_tools(get_file_tools(str(Path.cwd())))
    schema = get_tool_argument_schema("read_file", registry=registry)
    assert schema is get_tool_definition("read_file").args_schema
    arguments = {"path": "a", "offset": "2"}
    assert policy.normalized_batch_arguments("read_file", arguments, registry=registry) == (
        schema.model_validate(arguments).model_dump(mode="json")
    )


def test_read_only_metadata_does_not_grant_batch_eligibility():
    assert get_tool_definition("git_status").mutating is False
    with pytest.raises(ValueError, match="tool_not_batch_eligible"):
        policy.validate_read_only_batch([("git_status", {}), ("git_status", {})], {"git_status"})


def test_registry_mutating_metadata_vetoes_an_eligible_tool(monkeypatch):
    change_read_definition(monkeypatch, mutating=True)
    with pytest.raises(ValueError, match="tool_not_batch_eligible"):
        policy.normalized_batch_arguments("read_file", {"path": "a"})


def test_changed_authoritative_schema_is_shared_by_executable_policy_and_provider(monkeypatch):
    class RestrictedRead(ReadFileRequest):
        limit: int = Field(default=7, gt=0, le=7)

    change_read_definition(monkeypatch, args_schema=RestrictedRead)
    tools = get_file_tools(str(Path.cwd()))
    registry = ToolRegistry.from_tools(tools)
    assert registry.by_name["read_file"].args_schema is RestrictedRead
    assert policy.normalized_batch_arguments("read_file", {"path": "a"})["limit"] == 7
    with pytest.raises(ValidationError):
        policy.normalized_batch_arguments("read_file", {"path": "a", "limit": 8})
    model = SequenceModel(native_batch(args=[{"path": "a"}, {"path": "b", "limit": 8}]))
    output = provider_module.LangChainBrainProvider(brain_llm=model, executable_tools=tools).generate(
        brain_input(), (),
    )
    assert output.kind == Kind.INVALID_OUTPUT
    assert len(model.calls) == 1


@pytest.mark.parametrize("missing", ["definition", "declared_schema", "executable", "bound_schema"])
def test_missing_executable_or_schema_fails_closed(monkeypatch, missing):
    registry = None
    if missing == "definition":
        monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", tuple(
            definition for definition in registry_module.TOOL_DEFINITIONS
            if definition.name != "read_file"
        ))
    elif missing == "declared_schema":
        change_read_definition(monkeypatch, args_schema=None)
    elif missing == "executable":
        registry = ToolRegistry.from_tools([])
    else:
        registry = ToolRegistry.from_tools([SimpleNamespace(name="read_file", args_schema=None)])
    with pytest.raises(ValueError, match="tool_not_batch_eligible|schema_unavailable"):
        policy.normalized_batch_arguments("read_file", {"path": "a"}, registry=registry)


def test_bound_schema_lookup_and_batch_normalization_use_the_registered_executable():
    class BoundRead(ReadFileRequest):
        limit: int = Field(default=3, gt=0, le=3)

    registry = ToolRegistry.from_tools([SimpleNamespace(name="read_file", args_schema=BoundRead)])
    assert get_tool_argument_schema("read_file", registry=registry) is BoundRead
    assert policy.normalized_batch_arguments("read_file", {"path": "a"}, registry=registry)["limit"] == 3
    with pytest.raises(ValidationError):
        policy.normalized_batch_arguments("read_file", {"path": "a", "limit": 4}, registry=registry)


def test_provider_candidates_do_not_depend_on_batch_eligibility(monkeypatch):
    action = proposal()
    tools = get_file_tools(str(Path.cwd()))
    authorized = {tool.name: tool for tool in tools}
    monkeypatch.setattr(policy, "_BATCH_ELIGIBLE_TOOLS", frozenset())
    assert provider_module._is_valid_native_batch(native_batch(), authorized)
    assert not provider_module._is_permitted_native_batch(native_batch(), authorized)
    # Static eligibility changes are consumed by both provider and Controller.
    decision = CortexController(24).decide(input_for(initial_state(), action))
    assert decision.decision_type == Decision.TERMINATE
    assert decision.pending_tool_request is None
    model = SequenceModel(native_batch(), native_action("read_file", {"path": "chosen"}))
    output = provider_module.LangChainBrainProvider(brain_llm=model, executable_tools=tools).generate(
        brain_input(), (),
    )
    assert output.tool_request.arguments == {"path": "chosen"}
    assert len(model.calls) == 2


def test_provider_has_no_tool_name_specific_batch_candidate_branch():
    assert '"read_file"' not in inspect.getsource(provider_module._is_valid_native_batch)


def test_provider_batch_without_a_bound_argument_schema_cannot_execute_as_batch():
    model = SequenceModel(native_batch(), native_action("brain_step_failed", {"message": "Cannot proceed"}))
    tool = SimpleNamespace(name="read_file", args_schema=None)
    assert not provider_module._is_permitted_native_batch(native_batch(), {"read_file": tool})
    provider = provider_module.LangChainBrainProvider(
        brain_llm=model, executable_tools=[tool],
    )
    output = provider.generate(brain_input(), ())
    assert output.kind == Kind.STEP_FAILED
    assert output.tool_requests is None
    assert len(model.calls) == 2


def test_controller_capability_ceiling_is_independent_of_batch_policy(monkeypatch):
    action = proposal()
    state = initial_state()
    plan = state.protocol_visible.active_plan.model_copy(update={"available_tools": ("list_files",)})
    state = state.model_copy(update={"protocol_visible": state.protocol_visible.model_copy(
        update={"active_plan": plan},
    )})
    monkeypatch.setattr(controller_module, "validate_read_only_batch", lambda *args: ())
    decision = CortexController(24).decide(input_for(state, action))
    assert decision.decision_type == Decision.TERMINATE
    assert decision.pending_tool_request is None


def test_controller_scope_validation_is_independent_of_batch_policy(monkeypatch):
    action = proposal().model_copy(update={"step_id": "another-step"})
    monkeypatch.setattr(controller_module, "validate_read_only_batch", lambda *args: ())
    with pytest.raises(ValueError, match="step does not match active step"):
        CortexController(24).decide(input_for(initial_state(), action))


def test_configured_async_submission_mode_still_requires_lifecycle_preparation():
    # read_file is statically batch eligible, but a configured async execution
    # mode must not bypass Controller's singleton submission preparation.
    action = proposal()
    controller = CortexController(24, async_submission_tool_names=("read_file",))
    decision = controller.decide(input_for(initial_state(), action))
    assert decision.decision_type == Decision.TERMINATE
    assert decision.reason == "invalid_tool_batch:async_tool_not_batch_eligible"
    assert decision.pending_tool_request is None
