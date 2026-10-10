"""Mixed metadata-enabled groups retain ordinary authorization and execution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import tools.registry as registry_module
from core.brain_batch_policy import is_oversized_read_only_batch, validate_read_only_batch
from core.brain_normalization import normalize_brain_output
from core.brain_provider import LangChainBrainProvider, _is_permitted_native_batch
from core.protocol.completion_identity import evidence_identity
from core.protocol.controller import CortexController, apply_controller_decision_to_state
from core.protocol.enums import BrainOutcomeKind as Kind, ControllerDecisionType as Decision
from core.protocol.models import BrainOutcome, ExecutionState
from core.runtime.tool_result_integration import SerializedToolRuntimePort, integrate_tool_result
from test_brain_metadata_batch import context_for, executables, injected_proposal, state_for
from test_brain_outcomes import SequenceModel
from test_brain_read_file_batch import input_for, native_batch
from tools.registry import ToolDefinition


def mixed_group(names=("list_files", "read_file")):
    return native_batch(len(names), names=names, args=[
        {"pattern": f"*.ext{i}"} if name == "find_files" else {"path": f"Snake/path-{i}"}
        for i, name in enumerate(names)
    ])


def assert_boundaries(raw, tools, schema_for, *, accepted, context=None):
    context = context or context_for()
    authorized = {tool.name: tool for tool in tools if tool.name in context.active_plan.available_tools}
    candidates = [(call["name"], call["args"]) for call in raw.tool_calls]
    if accepted:
        validate_read_only_batch(candidates, authorized, argument_schema_for=schema_for)
    else:
        with pytest.raises(ValueError):
            validate_read_only_batch(candidates, authorized, argument_schema_for=schema_for)
    assert _is_permitted_native_batch(raw, authorized, argument_schema_for=schema_for) is accepted
    model = SequenceModel(raw, raw)
    output = LangChainBrainProvider(brain_llm=model, executable_tools=tools).generate(context, ())
    normalized = normalize_brain_output(raw, context, set(authorized), argument_schema_for=schema_for)
    assert output.kind == normalized.kind == (Kind.TOOL_REQUESTED if accepted else Kind.INVALID_OUTPUT)
    decision = CortexController(24, argument_schema_for=schema_for).decide(
        input_for(state_for(context), output if accepted else injected_proposal(raw)),
    )
    if accepted:
        assert len(model.calls) == 1
        assert output.tool_requests == normalized.tool_requests
        assert [(request.tool_name, request.arguments) for request in output.tool_requests] == candidates
        assert len({request.request_id for request in output.tool_requests}) == len(candidates)
        assert decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME
        assert decision.pending_tool_request == output.tool_requests[0]
        assert decision.tool_request_continuation.remaining == output.tool_requests[1:]
    else:
        assert len(model.calls) <= 2
        assert output.tool_request is output.tool_requests is None
        assert decision.decision_type == Decision.TERMINATE
        assert decision.pending_tool_request is decision.tool_request_continuation is None
    return output, model


@pytest.mark.parametrize("names", [
    ("list_files", "read_file"), ("find_files", "read_file"), ("list_files", "find_files"),
    ("read_file", "find_files", "list_files", "read_file"),
])
def test_mixed_groups_accepted_at_all_boundaries(executables, names):
    assert_boundaries(mixed_group(names), *executables, accepted=True)


def test_snake_example_preserves_exact_arguments_and_order(executables):
    raw = native_batch(4, names=["list_files", "list_files", "read_file", "read_file"], args=[
        {"path": "Snake/ui/menu"}, {"path": "Snake/ui/scoreboard"},
        {"path": "Snake/app/SnakeGame.ui"}, {"path": "Snake/app/SnakeGame_scene.cpp"},
    ])
    snapshot = raw.model_dump()
    assert_boundaries(raw, *executables, accepted=True)
    assert raw.model_dump() == snapshot


@pytest.mark.parametrize("invalid", [
    "unauthorized_first", "unauthorized_last", "singleton_only", "mutating", "unregistered",
    "schema_missing", "invalid_args", "malformed", "duplicate",
    "brain_step_completed", "brain_step_failed", "brain_replan_requested",
])
def test_one_invalid_member_rejects_entire_mixed_group(executables, monkeypatch, invalid):
    tools, schema_for = executables
    context = context_for()
    raw = mixed_group()
    if invalid.startswith("unauthorized"):
        context = context_for(("read_file",) if invalid.endswith("first") else ("list_files",))
    elif invalid == "singleton_only":
        raw.tool_calls[1].update(name="search_text", args={"query": "Snake"})
    elif invalid == "mutating":
        monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", tuple(
            replace(definition, max_batch_calls=24) if definition.name == "write_file" else definition
            for definition in registry_module.TOOL_DEFINITIONS
        ))
        raw.tool_calls[1].update(name="write_file", args={"path": "a", "content": "x"})
    elif invalid == "unregistered":
        monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", tuple(
            definition for definition in registry_module.TOOL_DEFINITIONS if definition.name != "read_file"
        ))
    elif invalid == "schema_missing":
        original = schema_for
        schema_for = lambda name: None if name == "read_file" else original(name)
        tools = [SimpleNamespace(name=tool.name, args_schema=None) if tool.name == "read_file" else tool
                 for tool in tools]
    elif invalid == "invalid_args":
        raw.tool_calls[1]["args"] = {"path": "a", "limit": 0}
    elif invalid == "malformed":
        raw.tool_calls[1]["args"] = []
    elif invalid == "duplicate":
        raw = native_batch(3, names=["read_file", "list_files", "read_file"], args=[
            {"path": "a"}, {"path": "Snake/ui"}, {"limit": "10000", "offset": 0, "path": "a"},
        ])
    else:
        raw.tool_calls[1].update(name=invalid, args={"message": "Done"})
    assert_boundaries(raw, tools, schema_for, accepted=False, context=context)


def test_same_effective_arguments_for_different_tools_are_distinct(executables, monkeypatch):
    class Query(BaseModel):
        query: str
        limit: int = 10

    names = ("lookup_a", "lookup_b")
    monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", (
        *registry_module.TOOL_DEFINITIONS,
        *(ToolDefinition(name, "general", max_batch_calls=24) for name in names),
    ))
    tools = [SimpleNamespace(name=name, args_schema=Query) for name in names]
    raw = native_batch(names=names, args=[{"query": "a"}, {"limit": "10", "query": "a"}])
    assert_boundaries(raw, tools, lambda name: Query, accepted=True, context=context_for(names))


@pytest.mark.parametrize("count", [24, 25])
@pytest.mark.parametrize("larger_metadata", [False, True])
def test_overall_limit_cannot_be_multiplied_by_mixing_tools(executables, monkeypatch, count, larger_metadata):
    if larger_metadata:
        monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", tuple(
            replace(definition, max_batch_calls=30) if definition.name in {"list_files", "read_file"} else definition
            for definition in registry_module.TOOL_DEFINITIONS
        ))
    raw = mixed_group(tuple("list_files" if i % 2 else "read_file" for i in range(count)))
    assert_boundaries(raw, *executables, accepted=count == 24)


@pytest.mark.parametrize("read_count,list_count", [(2, 22), (3, 1)])
@pytest.mark.parametrize("reverse", [False, True])
def test_per_tool_limit_counts_only_that_tools_members(executables, monkeypatch, read_count, list_count, reverse):
    monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", tuple(
        replace(definition, max_batch_calls=2) if definition.name == "read_file" else definition
        for definition in registry_module.TOOL_DEFINITIONS
    ))
    names = ("read_file",) * read_count + ("list_files",) * list_count
    raw = mixed_group(names[::-1] if reverse else names)
    assert_boundaries(raw, *executables, accepted=read_count == 2)


@pytest.mark.parametrize("limit", [2, 24])
def test_mixed_size_only_correction_retains_limits_selection_and_one_retry(executables, monkeypatch, limit):
    tools, schema_for = executables
    monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", tuple(
        replace(definition, max_batch_calls=limit) if definition.name == "read_file" else definition
        for definition in registry_module.TOOL_DEFINITIONS
    ))
    names = ("list_files",) * (24 - limit) + ("read_file",) * (limit + 1)
    # Keep at least one different tool when exercising the overall 24-call cap.
    if limit == 24:
        names = ("list_files",) + ("read_file",) * 24
    raw = mixed_group(names)
    candidates = [(call["name"], call["args"]) for call in raw.tool_calls]
    assert is_oversized_read_only_batch(candidates, context_for().active_plan.available_tools,
                                       argument_schema_for=schema_for)
    corrected = raw.model_copy(update={"tool_calls": raw.tool_calls[:-2] + raw.tool_calls[-1:]})
    model = SequenceModel(raw, corrected)
    output = LangChainBrainProvider(brain_llm=model, executable_tools=tools).generate(context_for(), ())
    assert output.kind == Kind.TOOL_REQUESTED and len(model.calls) == 2
    instruction = model.calls[1][-1].content
    assert "at most 24" in instruction
    assert f"Per-tool call limits: list_files: 24, read_file: {limit}." in instruction
    assert "same tool" not in instruction and "homogeneous" not in instruction.lower()
    assert [(request.tool_name, request.arguments) for request in output.tool_requests] == [
        (call["name"], call["args"]) for call in corrected.tool_calls
    ]


def test_configured_async_member_remains_excluded_by_controller(executables):
    tools, schema_for = executables
    output, _ = assert_boundaries(mixed_group(), tools, schema_for, accepted=True)
    decision = CortexController(24, argument_schema_for=schema_for,
                                async_submission_tool_names=("read_file",)).decide(
        input_for(state_for(context_for()), output),
    )
    assert decision.decision_type == Decision.TERMINATE
    assert decision.reason == "invalid_tool_batch:async_tool_not_batch_eligible"
    assert decision.pending_tool_request is decision.tool_request_continuation is None


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("first_fails", [False, True])
def test_mixed_execution_checkpoint_resume_preserves_order_failures_and_provenance(
    executables, tmp_path, captured, first_fails,
):
    tools, schema_for = executables
    (tmp_path / "Snake/ui").mkdir(parents=True)
    (tmp_path / "Snake/app").mkdir()
    (tmp_path / "Snake/app/SnakeGame.ui").write_text("UI", encoding="utf-8")
    (tmp_path / "Snake/app/SnakeGame_scene.cpp").write_text("Scene", encoding="utf-8")
    raw = native_batch(4, names=["list_files", "read_file", "find_files", "read_file"], args=[
        {"path": "missing" if first_fails else "Snake/ui"}, {"path": "Snake/app/SnakeGame.ui"},
        {"path": "Snake", "pattern": "*.cpp"}, {"path": "Snake/app/SnakeGame_scene.cpp"},
    ])
    context = context_for()
    output, _ = assert_boundaries(raw, tools, schema_for, accepted=True)
    state = state_for(context)
    controller = CortexController(24, argument_schema_for=schema_for)
    observed = []
    runtime = SerializedToolRuntimePort(tools)

    def execute(state, decision):
        request = decision.pending_tool_request
        assert request == output.tool_requests[len(observed)]
        assert state.protocol_visible.pending_tool_request == request
        assert len(state.working.tool_execution_history) == len(observed)
        assert request not in decision.tool_request_continuation.remaining
        observed.append(request)
        result = runtime.execute(request)
        assert result.success is (not first_fails or len(observed) != 1)
        return integrate_tool_result(state, decision, result)

    first = controller.decide(input_for(state, output))
    state = execute(apply_controller_decision_to_state(state, first), first)
    second = controller.decide(input_for(state))
    state = apply_controller_decision_to_state(state, second)
    if captured:
        state = execute(state, second)
    state = ExecutionState.model_validate_json(state.model_dump_json())
    controller = CortexController(24, argument_schema_for=schema_for)
    while len(observed) < len(output.tool_requests):
        decision = controller.decide(input_for(state))
        assert decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME
        state = execute(apply_controller_decision_to_state(state, decision), decision)
    dispatch = controller.decide(input_for(state))
    assert dispatch.decision_type == Decision.DISPATCH_BRAIN
    state = apply_controller_decision_to_state(state, dispatch)
    assert state.protocol_visible.pending_tool_request is state.protocol_visible.tool_request_continuation is None
    records = state.working.tool_execution_history
    assert observed == list(output.tool_requests)
    assert [record.result.request_id for record in records] == [request.request_id for request in observed]
    assert [(record.tool_name, record.arguments) for record in records] == [
        (call["name"], call["args"]) for call in raw.tool_calls
    ]
    assert all((record.execution_id, record.plan_id, record.plan_revision, record.step_id) ==
               (context.identity.execution_id, "p1", 1, "s1") for record in records)
    completion = controller.decide(input_for(state, BrainOutcome(
        outcome=Kind.STEP_COMPLETED, step_id="s1", message="Inspected Snake files",
    )))
    assert completion.completion_evidence.tool_request_ids == tuple(request.request_id for request in observed)
    assert completion.completion_evidence.evidence_id == evidence_identity(records)
