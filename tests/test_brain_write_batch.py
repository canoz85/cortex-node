"""Explicit mutating opt-in shares batch authority and ordinary tool execution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import tools.registry as registry_module
from core.brain_batch_policy import batch_call_limit, validate_tool_batch
from core.brain_provider import LangChainBrainProvider
from core.protocol.controller import CortexController, apply_controller_decision_to_state
from core.protocol.enums import BrainOutcomeKind as Kind, ControllerDecisionType as Decision
from core.runtime.tool_result_integration import SerializedToolRuntimePort, integrate_tool_result
from test_brain_metadata_batch import context_for, executables, state_for
from test_brain_mixed_batch import assert_boundaries
from test_brain_outcomes import SequenceModel
from test_brain_read_file_batch import input_for, native_batch
from test_duplicate_successful_call import scoped_record
from tools.registry import get_tool_definition


def writes(count=2):
    return native_batch(count, names=["write_file"] * count, args=[
        {"path": f"output-{index}.txt", "content": f"value-{index}"} for index in range(count)
    ])


@pytest.mark.parametrize("count", [2, 24])
def test_explicit_write_opt_in_is_accepted_at_all_boundaries(executables, count):
    tools, schema_for = executables
    definition = get_tool_definition("write_file")
    assert definition.mutating is True and definition.max_batch_calls == 24
    assert batch_call_limit("write_file", argument_schema_for=schema_for) == 24
    raw = writes(count)
    normalized = validate_tool_batch(
        [(call["name"], call["args"]) for call in raw.tool_calls], {"write_file"},
        argument_schema_for=schema_for,
    )
    assert len(normalized) == count and all(args["overwrite"] is True for args in normalized)
    output, model = assert_boundaries(raw, tools, schema_for, accepted=True)
    assert any("- write_file: up to 24" in message.content for message in model.calls[0])
    assert len(output.tool_requests) == count


@pytest.mark.parametrize("limit,count", [(24, 25), (2, 3)])
def test_write_batch_limits_remain_enforced_at_all_boundaries(executables, monkeypatch, limit, count):
    monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", tuple(
        replace(definition, max_batch_calls=limit) if definition.name == "write_file" else definition
        for definition in registry_module.TOOL_DEFINITIONS
    ))
    assert_boundaries(writes(count), *executables, accepted=False)


def test_opted_in_writes_can_mix_with_read_only_tools(executables):
    raw = writes(3)
    raw.tool_calls[1].update(name="list_files", args={"path": "."})
    assert_boundaries(raw, *executables, accepted=True)


@pytest.mark.parametrize("equivalent", [False, True])
def test_duplicate_and_default_equivalent_writes_are_rejected(executables, equivalent):
    raw = writes()
    raw.tool_calls[1]["args"] = dict(raw.tool_calls[0]["args"])
    if equivalent:
        raw.tool_calls[1]["args"]["overwrite"] = True
    tools, schema_for = executables
    with pytest.raises(ValueError, match="duplicate_effective_batch_invocation"):
        validate_tool_batch([(call["name"], call["args"]) for call in raw.tool_calls],
                            {"write_file"}, argument_schema_for=schema_for)
    assert_boundaries(raw, tools, schema_for, accepted=False)


@pytest.mark.parametrize("invalid", ["unauthorized", "schema", "args", "no_opt_in"])
def test_mutating_opt_in_does_not_bypass_other_batch_checks(executables, monkeypatch, invalid):
    tools, schema_for = executables
    raw = writes()
    context = context_for()
    if invalid == "unauthorized":
        context = context_for(("read_file",))
    elif invalid == "schema":
        original = schema_for
        schema_for = lambda name: None if name == "write_file" else original(name)
        tools = [SimpleNamespace(name=tool.name, args_schema=None) if tool.name == "write_file" else tool
                 for tool in tools]
    elif invalid == "args":
        raw.tool_calls[-1]["args"] = {"path": "missing-content"}
    else:
        monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", tuple(
            replace(definition, max_batch_calls=1) if definition.name == "write_file" else definition
            for definition in registry_module.TOOL_DEFINITIONS
        ))
    assert_boundaries(raw, tools, schema_for, accepted=False, context=context)


def test_write_batch_executes_sequentially_and_keeps_workspace_enforcement(executables, tmp_path):
    tools, schema_for = executables
    raw = writes()
    raw.tool_calls[0]["args"]["path"] = "../outside-write-batch.txt"
    output, _ = assert_boundaries(raw, tools, schema_for, accepted=True)
    context = context_for()
    state = state_for(context)
    controller = CortexController(24, argument_schema_for=schema_for)
    runtime = SerializedToolRuntimePort(tools)
    for index, request in enumerate(output.tool_requests):
        decision = controller.decide(input_for(state, output if index == 0 else None))
        assert decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME
        assert decision.pending_tool_request == request
        state = apply_controller_decision_to_state(state, decision)
        result = runtime.execute(request)
        assert result.success is (index == 1)
        if index == 0:
            assert result.error_code == "FILE_WRITE_FAILED"
        state = integrate_tool_result(state, decision, result)
    assert not (tmp_path.parent / "outside-write-batch.txt").exists()
    assert (tmp_path / "output-1.txt").read_text(encoding="utf-8") == "value-1"
    assert len(state.working.tool_execution_history) == 2


def test_successful_write_replay_rejects_whole_batch(executables):
    tools, schema_for = executables
    raw = writes()
    context = context_for()
    record = scoped_record(context, "write_file", raw.tool_calls[0]["args"])
    context = context.model_copy(update={"tool_execution_history": (record,)})
    output = LangChainBrainProvider(brain_llm=SequenceModel(raw, raw, raw), executable_tools=tools).generate(context, ())
    assert output.kind == Kind.INVALID_OUTPUT and output.error_code == "duplicate_successful_call"
    assert output.tool_requests is None
    accepted, _ = assert_boundaries(raw, tools, schema_for, accepted=True)
    state = state_for(context)
    state = state.model_copy(update={"working": state.working.model_copy(update={"tool_execution_history": (record,)})})
    decision = CortexController(24, argument_schema_for=schema_for).decide(input_for(state, accepted))
    assert decision.decision_type != Decision.DISPATCH_TOOL_RUNTIME
    assert decision.pending_tool_request is None


def test_opted_in_write_still_cannot_bypass_async_lifecycle(executables):
    tools, schema_for = executables
    output, _ = assert_boundaries(writes(), tools, schema_for, accepted=True)
    decision = CortexController(24, argument_schema_for=schema_for,
                                async_submission_tool_names=("write_file",)).decide(
        input_for(state_for(context_for()), output),
    )
    assert decision.reason == "invalid_tool_batch:async_tool_not_batch_eligible"
    assert decision.pending_tool_request is None
