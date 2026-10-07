"""Bounded native-call cardinality correction at the Brain provider boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from core.brain import BrainService, _build_execution_messages
from core.brain_provider import LangChainBrainProvider
from core.protocol.enums import BrainOutcomeKind as Kind
from test_brain_outcomes import (
    SequenceModel, brain_input, controller_input, evidence_context, native_action,
)
from test_brain_ollama_boundary import boundary, git_brain_input, native


CORRECTION = (
    "The previous response contained multiple native calls and was not accepted. "
    "Choose exactly one next action from the useful candidates for this turn. "
    "Return exactly one native call using a currently bound tool. "
    "Do not batch or parallelize calls. "
    "Leave content empty."
)


class ReadArguments(BaseModel):
    path: str


def batch():
    # Valid candidates, but duplicate effective invocations are not executable.
    return AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {"path": "a.py"}, "id": "rejected-a"},
        {"name": "read_file", "args": {"path": "a.py", "offset": 0}, "id": "rejected-b"},
    ])


def setup_provider(*replies):
    model = SequenceModel(*replies)
    model.bind_tools = Mock(wraps=model.bind_tools)
    tool = SimpleNamespace(name="read_file", args_schema=ReadArguments, invoke=Mock())
    provider = LangChainBrainProvider(brain_llm=model, executable_tools=[tool])
    return provider, model, tool


def test_batch_correction_selects_one_action_without_executing_or_recording_candidates(
    monkeypatch, caplog,
):
    import core.brain_normalization as normalization
    import core.brain_provider as provider_module

    create_request = Mock(wraps=normalization._tool_request)
    monkeypatch.setattr(normalization, "_tool_request", create_request)
    exchanges = Mock()
    monkeypatch.setattr(provider_module, "log_llm_exchange", exchanges)
    invocations = Mock()
    monkeypatch.setattr(provider_module, "begin_provider_invocation", invocations)
    usage = Mock()
    monkeypatch.setattr(provider_module, "add_response_usage", usage)
    rejected = batch()
    provider, model, tool = setup_provider(
        rejected, native_action("read_file", {"path": "b.py"}),
    )
    context = evidence_context()
    snapshot = context.model_dump(mode="json")
    evidence_before = _build_execution_messages(system_prompt="active", brain_input=context)

    with caplog.at_level("DEBUG", logger="core.brain_provider"):
        outcome = BrainService(provider=provider, agent_system_prompt="active").run(context)

    assert outcome.kind == Kind.TOOL_REQUESTED
    assert outcome.tool_request.arguments == {"path": "b.py"}
    assert len(model.calls) == 2
    model.bind_tools.assert_called_once()
    assert model.calls[1][:-2] == model.calls[0]
    assert model.calls[1][-2] is rejected
    assert model.calls[1][-1].content == CORRECTION
    tool.invoke.assert_not_called()
    create_request.assert_called_once()
    assert create_request.call_args.args[:2] == ("read_file", {"path": "b.py"})
    assert invocations.call_count == usage.call_count == exchanges.call_count == 2
    assert context.model_dump(mode="json") == snapshot
    assert _build_execution_messages(system_prompt="active", brain_input=context) == evidence_before
    visible = controller_input(context, outcome).model_copy(update={
        "tool_execution_history": context.tool_execution_history,
    })
    assert visible.tool_execution_history == context.tool_execution_history
    assert len(visible.tool_execution_history) == 1
    assert visible.tool_execution_history[0].result.request_id == "req1"
    assert visible.brain_result.tool_request == outcome.tool_request
    assert visible.brain_result.completion_evidence is None
    logs = [r.message for r in caplog.records if "Brain native-call compliance" in r.message]
    assert len(logs) == 2
    assert "call_count=2" in logs[0] and "['read_file', 'read_file']" in logs[0]
    assert "correction_triggered=True" in logs[0]
    assert "call_count=1" in logs[1] and "correction_exhausted=False" in logs[1]


def test_batch_correction_can_return_a_lifecycle_call():
    provider, model, tool = setup_provider(
        batch(), native_action("brain_replan_requested", {
            "reason": "The current strategy cannot proceed.", "constraints": ["Use available evidence"],
        }),
    )
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.REPLAN_REQUESTED
    assert outcome.replan_request.constraints == ("Use available evidence",)
    assert outcome.tool_request is None
    assert len(model.calls) == 2
    tool.invoke.assert_not_called()


@pytest.mark.parametrize("corrected,error", [
    (batch(), "exactly_one_tool_call_required"),
    (AIMessage(content=""), "native_tool_call_required"),
    (native_action("unknown", {}), "unknown_tool"),
    (AIMessage(content="", invalid_tool_calls=[{
        "name": "read_file", "args": "{bad", "id": "bad", "error": "invalid JSON",
    }]), "invalid_native_tool_call"),
])
def test_invalid_correction_is_exhausted_without_a_third_invocation(corrected, error, caplog):
    provider, model, tool = setup_provider(batch(), corrected)
    with caplog.at_level("DEBUG", logger="core.brain_provider"):
        outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.error_code == error
    assert outcome.tool_request is None
    assert outcome.completion_evidence is None
    assert len(model.calls) == 2
    tool.invoke.assert_not_called()
    assert any("attempt=2" in r.message and "correction_exhausted=True" in r.message
               for r in caplog.records)


@pytest.mark.parametrize("bad_call", [
    {"name": "unknown", "args": {}, "id": "bad"},
    {"name": "write_file", "args": {"path": "a", "content": "x"}, "id": "bad"},
    {"name": "read_file", "args": "not an object", "id": "bad"},
    {"name": "read_file", "args": {}, "id": "bad"},
    {"name": "read_file", "args": {"path": []}, "id": "bad"},
    {"name": "read_file", "args": {"path": "a.py"}, "id": "bad", "extra": True},
    {"name": "brain_step_completed", "args": {
        "message": "Done", "exact_collection_source_record_index": 0,
    }, "id": "bad"},
    {"name": "brain_step_failed", "args": {"message": ""}, "id": "bad"},
    {"name": "brain_replan_requested", "args": {"reason": "Changed", "constraints": "bad"}, "id": "bad"},
])
def test_invalid_batch_candidates_do_not_trigger_cardinality_correction(bad_call, caplog):
    raw = SimpleNamespace(content="", tool_calls=[batch().tool_calls[0], bad_call])
    provider, model, tool = setup_provider(raw)
    with caplog.at_level("DEBUG", logger="core.brain_provider"):
        outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.error_code == "exactly_one_tool_call_required"
    assert len(model.calls) == 1
    tool.invoke.assert_not_called()
    assert any("correction_triggered=False" in r.message for r in caplog.records)


def test_batch_with_invalid_native_calls_is_not_cardinality_only():
    raw = AIMessage(content="Also do this", tool_calls=batch().tool_calls, invalid_tool_calls=[{
        "name": "read_file", "args": "{bad", "id": "bad", "error": "invalid JSON",
    }])
    provider, model, _ = setup_provider(raw)
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert len(model.calls) == 1


def test_batch_correction_provider_exception_is_not_retried():
    provider, model, _ = setup_provider(batch(), RuntimeError("offline"))
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.PROVIDER_FAILURE
    assert outcome.error_code == "RuntimeError"
    assert len(model.calls) == 2


def test_rejected_batch_survives_ollama_serialization_with_identical_bound_tools():
    rejected = {"content": "", "tool_calls": [
        {"function": {"name": "git_status", "arguments": {}}},
        {"function": {"name": "git_diff", "arguments": {}}},
    ]}
    provider, requests = boundary([rejected, native("git_diff", {})])
    outcome = BrainService(provider=provider, agent_system_prompt="active").run(git_brain_input())
    assert outcome.kind == Kind.TOOL_REQUESTED
    assert outcome.tool_request.tool_name == "git_diff"
    assert len(requests) == 2
    assert requests[0]["tools"] == requests[1]["tools"]
    assert requests[0]["messages"] == requests[1]["messages"][:-2]
    assert requests[1]["messages"][-2]["role"] == "assistant"
    assert [c["function"]["name"] for c in requests[1]["messages"][-2]["tool_calls"]] == [
        "git_status", "git_diff",
    ]
    assert requests[1]["messages"][-1]["content"] == CORRECTION
    assert not any(m["role"] == "tool" for m in requests[1]["messages"])
