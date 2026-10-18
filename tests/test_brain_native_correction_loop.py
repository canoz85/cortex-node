"""One shared bounded budget revalidates every native-action correction."""

import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

import core.brain_normalization as normalization
from core.brain_provider import LangChainBrainProvider
from core.logging.live_status import LiveStatus
from core.protocol.bridge import build_brain_input
from core.protocol.enums import BrainOutcomeKind as Kind
from test_brain_argument_correction import context_for, setup_probe
from test_brain_ollama_boundary import boundary, git_brain_input, native
from test_brain_outcomes import SequenceModel, native_action
from test_brain_read_file_batch import native_batch
from test_duplicate_successful_call import execute_once, input_for, setup


PSEUDO = AIMessage(content='brain_step_completed{"message":"Used prior evidence."}')
DONE = native_action("brain_step_completed", {"message": "Used prior evidence."})
VALID = native_action("schema_probe", {"path": "repaired"})
INVALID = native_action("schema_probe", {"path": 12})
BATCH = native_batch(names=["schema_probe", "schema_probe"], args=[{"path": "a"}, {"path": "b"}])


@pytest.mark.parametrize("replies,kind,error", [
    ([VALID], Kind.TOOL_REQUESTED, None),
    ([PSEUDO, VALID], Kind.TOOL_REQUESTED, None),
    ([INVALID, PSEUDO, VALID], Kind.TOOL_REQUESTED, None),
    ([BATCH, PSEUDO, VALID], Kind.TOOL_REQUESTED, None),
    ([INVALID, BATCH, VALID], Kind.TOOL_REQUESTED, None),
    ([PSEUDO, PSEUDO, PSEUDO], Kind.INVALID_OUTPUT, "unexpected_response_content"),
    ([INVALID, INVALID, INVALID], Kind.INVALID_OUTPUT, "invalid_native_tool_arguments"),
    ([BATCH, BATCH, BATCH], Kind.INVALID_OUTPUT, "exactly_one_tool_call_required"),
])
def test_each_response_reuses_checks_and_only_accepted_calls_create_requests(replies, kind, error, monkeypatch, caplog):
    provider, model, tool = setup_probe(*replies, VALID)  # A fourth reply must remain unused on exhaustion.
    context = context_for(tool.name)
    before = context.model_dump_json()
    create_request = Mock(wraps=normalization._tool_request)
    monkeypatch.setattr(normalization, "_tool_request", create_request)
    invoke = model.invoke

    def inspect(messages):
        create_request.assert_not_called()
        tool.invoke.assert_not_called()
        assert context.model_dump_json() == before
        return invoke(messages)

    model.invoke = inspect
    with caplog.at_level("DEBUG", logger="core.brain_provider"):
        outcome = provider.generate(context, ())
    assert outcome.kind == kind and outcome.error_code == error
    assert len(model.calls) == len(replies) <= 3
    assert create_request.call_count == (1 if kind == Kind.TOOL_REQUESTED else 0)
    assert context.model_dump_json() == before and context.tool_execution_history == ()
    tool.invoke.assert_not_called()
    logs = [record.message for record in caplog.records if "Brain native-call compliance" in record.message]
    assert len(logs) == len(replies)
    for attempt, line in enumerate(logs, start=1):
        assert f"attempt={attempt}" in line
    assert f"accepted={kind != Kind.INVALID_OUTPUT}" in logs[-1]
    assert f"correction_exhausted={kind == Kind.INVALID_OUTPUT}" in logs[-1]
    for index in range(1, len(model.calls)):
        assert model.calls[index][:-2] == model.calls[index - 1]
        assert not any(message.type == "tool" for message in model.calls[index])


@pytest.mark.parametrize("corrections,kind,reasons", [
    ([DONE], Kind.STEP_COMPLETED, ["duplicate_successful_call", None]),
    ([PSEUDO, DONE], Kind.STEP_COMPLETED, ["duplicate_successful_call", "missing_native_call", None]),
    (["duplicate", DONE], Kind.STEP_COMPLETED, ["duplicate_successful_call", "duplicate_successful_call", None]),
    (["duplicate", "duplicate"], Kind.INVALID_OUTPUT, ["duplicate_successful_call"] * 3),
])
def test_mutating_replay_never_executes_twice_while_corrections_are_revalidated(setup, corrections, kind, reasons,
                                                                           monkeypatch, caplog):
    case = setup
    arguments = {"path": "result.txt", "content": "Saved exactly once."}
    state, result = execute_once(case, "write_file", arguments)
    assert result.success and len(case.calls) == 1
    context = build_brain_input({"execution_state": state})
    before = state.model_dump_json()
    path = case.root / "result.txt"
    modified = path.stat().st_mtime_ns
    duplicate = native_action("write_file", arguments)
    replies = [duplicate, *(duplicate if item == "duplicate" else item for item in corrections)]
    model = SequenceModel(*replies, DONE)
    create_request = Mock(wraps=normalization._tool_request)
    monkeypatch.setattr(normalization, "_tool_request", create_request)
    with caplog.at_level("DEBUG", logger="core.brain_provider"):
        outcome = LangChainBrainProvider(brain_llm=model, executable_tools=case.tools).generate(context, ())
    assert outcome.kind == kind and len(model.calls) == len(replies) <= 3
    assert outcome.tool_request is outcome.tool_requests is None
    if kind == Kind.INVALID_OUTPUT:
        assert outcome.error_code == "duplicate_successful_call" and outcome.completion_evidence is None
    create_request.assert_not_called()
    assert state.model_dump_json() == before and len(state.working.tool_execution_history) == 1
    assert len(case.calls) == 1 and path.read_text() == arguments["content"]
    assert path.stat().st_mtime_ns == modified
    logs = [record.message for record in caplog.records if "Brain native-call compliance" in record.message]
    assert len(logs) == len(reasons)
    for attempt, (line, reason) in enumerate(zip(logs, reasons), start=1):
        assert f"attempt={attempt}" in line and f"correction_reason={reason}" in line
    continuation = case.driver.turn(state, input_for(state, outcome), dispatch_worker=False)
    assert continuation.decision.pending_tool_request is None and len(case.calls) == 1
    assert continuation.execution_state.working.tool_execution_history == state.working.tool_execution_history


def test_all_three_exchanges_are_raw_logged_and_counted_before_canonicalization(tmp_path, monkeypatch):
    path = tmp_path / "corrections.jsonl"
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", str(path))
    final = VALID.model_copy(update={"content": "Incidental explanation beside a real native call."})
    replies = [INVALID, PSEUDO, final]
    replies = [reply.model_copy(update={"usage_metadata": {
        "input_tokens": index * 10, "output_tokens": index, "total_tokens": index * 11,
    }}) for index, reply in enumerate(replies, start=1)]
    provider, model, _tool = setup_probe(*replies)
    status = LiveStatus(enabled=False)
    status.start("planner")
    try:
        outcome = provider.generate(context_for("schema_probe"), ())
        assert outcome.kind == Kind.TOOL_REQUESTED and len(model.calls) == 3
        assert status.provider_invocations_by_worker["brain"] == 3
        assert status.usage_by_worker["brain"] == {
            "input_tokens": 60, "output_tokens": 6, "total_tokens": 66, "calls": 3,
        }
        assert outcome.usage.prompt_tokens == 30 and outcome.usage.completion_tokens == 3
    finally:
        status.stop()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    for index, record in enumerate(records):
        assert record["response"]["content"] == replies[index].content
        assert record["response"]["tool_calls"] == replies[index].tool_calls
        assert len(record["messages"]) == len(model.calls[0]) + 2 * index
        assert record["worker"] == "brain" and record["operation"] == "step"
    assert records[1]["response"]["tool_calls"] == []
    assert "invalid_native_tool_arguments" in records[1]["messages"][-1]["content"]
    assert "no native call" in records[2]["messages"][-1]["content"]
    assert final.content  # Canonicalization did not mutate the original response.


def test_second_correction_survives_real_native_binding_without_schema_changes():
    provider, requests = boundary([
        {"content": "No native action."},
        {"content": 'brain_step_completed{"message":"Done"}'},
        native("brain_step_completed", {"message": "Done"}),
    ])
    outcome = provider.generate(git_brain_input(), ())
    assert outcome.kind == Kind.STEP_COMPLETED and len(requests) == 3
    assert requests[0]["tools"] == requests[1]["tools"] == requests[2]["tools"]
    assert requests[2]["messages"][:-2] == requests[1]["messages"]
    assert requests[2]["messages"][-2]["content"].startswith("brain_step_completed{")
    assert not any(message["role"] == "tool" for message in requests[2]["messages"])
