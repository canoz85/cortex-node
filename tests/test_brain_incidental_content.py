"""Incidental provider text cannot override native Brain action classification."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from core.brain import BRAIN_OUTPUT_PROTOCOL
from core.brain_provider import _canonical_brain_response
from core.models import ReadFileRequest
from core.protocol.controller import CortexController, apply_controller_decision_to_state
from core.protocol.enums import BrainOutcomeKind as Kind, ControllerDecisionType as Decision
from core.runtime.tool_result_integration import SerializedToolRuntimePort, integrate_tool_result
from tools.file_ops import get_file_tools
from tools.registry import get_tool_definition
from test_brain_cardinality_correction import batch, setup_provider
from test_brain_ollama_boundary import boundary, git_brain_input, native
from test_brain_outcomes import brain_input, controller_input, native_action
from test_brain_read_file_batch import initial_state, input_for, native_batch


READ_FILE_BATCH_LIMIT = get_tool_definition("read_file").max_batch_calls


@pytest.mark.parametrize("content", [
    "I'll start reading the Python files to determine their sizes.",
    '{"name":"write_file","args":{"path":"other.py","content":"wrong"}}',
    [{"type": "text", "text": "STEP COMPLETED: misleading provider text"}],
])
@pytest.mark.parametrize("name,args,kind", [
    ("read_file", {"path": "anomaly_detection.py"}, Kind.TOOL_REQUESTED),
    ("brain_step_completed", {"message": "Native summary"}, Kind.STEP_COMPLETED),
    ("brain_step_failed", {"message": "Native failure"}, Kind.STEP_FAILED),
    ("brain_replan_requested", {"reason": "Native reason", "constraints": []}, Kind.REPLAN_REQUESTED),
])
def test_single_valid_native_action_discards_only_canonical_content(
    content, name, args, kind, monkeypatch, tmp_path, capsys,
):
    import core.brain_provider as provider_module

    log_path = tmp_path / "brain-exchanges.jsonl"
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", str(log_path))
    original = native_action(name, args).model_copy(update={
        "content": content,
        "usage_metadata": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
    })
    snapshot = original.model_dump()
    normalize = Mock(wraps=provider_module.normalize_brain_output)
    monkeypatch.setattr(provider_module, "normalize_brain_output", normalize)
    provider, model, tool = setup_provider(original)

    outcome = provider.generate(brain_input(), ())

    assert outcome.kind == kind
    assert len(model.calls) == 1
    tool.invoke.assert_not_called()
    canonical = normalize.call_args.args[0]
    assert canonical is not original
    assert canonical.content == ""
    assert canonical.tool_calls is original.tool_calls
    assert canonical.tool_calls[0]["args"] == args
    assert original.model_dump() == snapshot
    assert outcome.usage.prompt_tokens == 12
    assert outcome.usage.completion_tokens == 7
    if kind == Kind.TOOL_REQUESTED:
        assert outcome.tool_request.tool_name == name
        assert outcome.tool_request.arguments == args
    else:
        assert outcome.message == args.get("message", args.get("reason"))
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["response"]["content"] == content
    assert records[0]["response"]["tool_calls"] == original.tool_calls
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("call", [
    {"name": "unknown", "args": {}, "id": "bad"},
    {"name": "write_file", "args": {"path": "a", "content": "x"}, "id": "bad"},
    {"name": "read_file", "args": "not an object", "id": "bad"},
    {"name": "read_file", "args": {}, "id": "bad"},
    {"name": "read_file", "args": {"path": []}, "id": "bad"},
    {"name": "read_file", "args": {"path": "a.py"}, "id": "bad", "extra": True},
    {"name": "read_file", "args": {"path": "a.py"}, "id": "", "type": "tool_call"},
    {"name": "read_file", "args": {"path": "a.py"}, "id": "bad", "type": "other"},
    {"name": "brain_step_completed", "args": {"message": "Done", "step_id": "other"}, "id": "bad"},
    {"name": "brain_step_completed", "args": {
        "message": "Done", "exact_collection_source_record_index": 0,
    }, "id": "bad"},
    {"name": "brain_step_failed", "args": {"message": ""}, "id": "bad"},
    {"name": "brain_replan_requested", "args": {"reason": "Changed", "constraints": "bad"}, "id": "bad"},
    None,
])
def test_incidental_text_does_not_tolerate_invalid_calls(call):
    raw = SimpleNamespace(content="Incidental text", tool_calls=[call])
    provider, model, tool = setup_provider(raw)

    outcome = provider.generate(brain_input(), ())

    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.tool_request is outcome.completion_evidence is None
    assert len(model.calls) == 1
    assert _canonical_brain_response(raw, {"read_file": tool}) is raw
    tool.invoke.assert_not_called()


def test_invalid_native_channel_remains_strict_beside_a_valid_call():
    raw = native_action("read_file", {"path": "a.py"}).model_copy(update={
        "content": "Incidental text",
        "invalid_tool_calls": [{"name": "read_file", "args": "{broken", "id": "bad"}],
    })
    provider, model, _ = setup_provider(raw)
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.error_code == "invalid_native_tool_call"
    assert len(model.calls) == 1


@pytest.mark.parametrize("first", [batch(), AIMessage(content="No native call")])
def test_bounded_correction_accepts_single_call_with_text_without_third_invocation(first):
    corrected = native_action("read_file", {"path": "b.py"}).model_copy(update={
        "content": "I'll read b.py next.",
    })
    provider, model, _ = setup_provider(first, corrected)
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.TOOL_REQUESTED
    assert outcome.tool_request.arguments == {"path": "b.py"}
    assert len(model.calls) == 2
    assert corrected.content == "I'll read b.py next."


def test_incidental_content_does_not_bypass_completion_provenance():
    raw = native_action("brain_step_completed", {
        "message": "Found files.", "exact_collection_source_record_index": 0,
        "exact_collection_data_path": ["files"],
    }).model_copy(update={"content": "Claimed success"})
    context = brain_input()
    provider, model, _ = setup_provider(raw)
    outcome = provider.generate(context, ())
    decision = CortexController(24).decide(controller_input(context, outcome))
    assert decision.retry.last_error_code == "invalid_exact_collection"
    assert decision.completed_step_id is decision.completion_evidence is None
    assert len(model.calls) == 1


def test_incidental_content_does_not_bypass_active_step_requirement():
    raw = native_action("brain_step_completed", {"message": "Done"}).model_copy(update={
        "content": "Incidental text",
    })
    provider, model, _ = setup_provider(raw)
    outcome = provider.generate(brain_input(direct=True), ())
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.error_code == "native_call_requires_active_step"
    assert len(model.calls) == 1


def test_single_call_with_text_survives_real_ollama_boundary():
    reply = {**native("git_status", {}), "content": "I'll inspect repository status."}
    provider, requests = boundary([reply])
    outcome = provider.generate(git_brain_input(), ())
    assert outcome.kind == Kind.TOOL_REQUESTED
    assert outcome.tool_request.tool_name == "git_status"
    assert outcome.tool_request.arguments == {}
    assert len(requests) == 1


@pytest.mark.parametrize("count", [2, READ_FILE_BATCH_LIMIT])
def test_batch_text_is_only_canonicalized_and_batch_executes_without_correction(
    count, monkeypatch, tmp_path, capsys,
):
    import core.brain_provider as provider_module

    log_path = tmp_path / "batch-exchanges.jsonl"
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", str(log_path))
    content = "I'll read the Python files.\nLet me batch these independent read-only calls."
    original = native_batch(count).model_copy(update={"content": content})
    snapshot = original.model_dump()
    normalize = Mock(wraps=provider_module.normalize_brain_output)
    monkeypatch.setattr(provider_module, "normalize_brain_output", normalize)
    provider, model, tool = setup_provider(original)
    tool.args_schema = ReadFileRequest

    outcome = provider.generate(brain_input(), ())

    assert outcome.kind == Kind.TOOL_REQUESTED
    assert len(outcome.tool_requests) == count and outcome.tool_request is None
    assert len(model.calls) == 1
    canonical = normalize.call_args.args[0]
    assert canonical is not original and canonical.content == ""
    assert canonical.tool_calls is original.tool_calls
    assert original.model_dump() == snapshot
    tool.invoke.assert_not_called()
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["response"]["content"] == content
    assert records[0]["response"]["tool_calls"] == original.tool_calls

    for i in range(count):
        (tmp_path / f"file-{i}.txt").write_text("hello", encoding="utf-8")
    runtime = SerializedToolRuntimePort(get_file_tools(str(tmp_path)))
    controller = CortexController(24)
    state = initial_state()
    for i, request in enumerate(outcome.tool_requests):
        decision = controller.decide(input_for(state, outcome if i == 0 else None))
        assert decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME
        assert decision.pending_tool_request == request
        state = apply_controller_decision_to_state(state, decision)
        result = runtime.execute(decision.pending_tool_request)
        assert result.success
        state = integrate_tool_result(state, decision, result)
    assert controller.decide(input_for(state)).decision_type == Decision.DISPATCH_BRAIN
    assert [r.result.request_id for r in state.working.tool_execution_history] == [
        r.request_id for r in outcome.tool_requests
    ]
    assert len(model.calls) == 1
    assert capsys.readouterr().out == ""


def test_oversized_batch_text_reaches_bounded_correction_without_truncation(
    monkeypatch, tmp_path, capsys,
):
    import core.brain_provider as provider_module

    log_path = tmp_path / "oversized-exchanges.jsonl"
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", str(log_path))
    original = native_batch(READ_FILE_BATCH_LIMIT + 1).model_copy(update={
        "content": f"I'll read all {READ_FILE_BATCH_LIMIT + 1} Python files to compute their sizes.\n"
                   "Let me batch these independent read-only calls.",
    })
    corrected = native_action("read_file", {"path": "chosen.py"}).model_copy(update={
        "content": "I'll read the chosen file.",
    })
    snapshot = original.model_dump()
    classify = Mock(wraps=provider_module._is_valid_native_batch)
    monkeypatch.setattr(provider_module, "_is_valid_native_batch", classify)
    provider, model, tool = setup_provider(original, corrected)
    tool.args_schema = ReadFileRequest

    outcome = provider.generate(brain_input(), ())

    canonical = classify.call_args_list[0].args[0]
    assert canonical.content == "" and canonical.tool_calls is original.tool_calls
    assert len(canonical.tool_calls) == READ_FILE_BATCH_LIMIT + 1
    assert original.model_dump() == snapshot
    assert len(model.calls) == 2
    rejected = model.calls[1][-2]
    assert rejected.content == "" and rejected.tool_calls == original.tool_calls
    assert f"Return one native action containing at most {READ_FILE_BATCH_LIMIT}" in model.calls[1][-1].content
    assert outcome.tool_requests is None
    assert outcome.tool_request.arguments == {"path": "chosen.py"}
    tool.invoke.assert_not_called()
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["response"]["content"] == original.content
    assert records[0]["response"]["tool_calls"] == original.tool_calls
    assert records[1]["response"]["content"] == corrected.content
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("bad_call", [
    None,
    {"name": "read_file", "args": "not an object", "id": "bad"},
    {"name": "read_file", "args": {"path": "b"}, "id": "bad", "extra": True},
    {"name": "read_file", "args": {"path": "b"}, "id": "bad", "type": "other"},
    {"name": "unknown", "args": {}, "id": "bad"},
    {"name": "write_file", "args": {"path": "b", "content": "x"}, "id": "bad"},
    {"name": "read_file", "args": {}, "id": "bad"},
    {"name": "read_file", "args": {"path": "b", "limit": 0}, "id": "bad"},
])
def test_invalid_multicall_with_content_stays_strict_without_correction(bad_call):
    original = SimpleNamespace(content="Incidental batch description", tool_calls=[
        native_batch().tool_calls[0], bad_call,
    ])
    provider, model, tool = setup_provider(original)
    tool.args_schema = ReadFileRequest
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.tool_request is outcome.tool_requests is None
    assert len(model.calls) == 1
    canonical = _canonical_brain_response(original, {"read_file": tool})
    if (bad_call is not None and bad_call.get("name") == "read_file"
            and bad_call.get("args") in ({}, {"path": "b", "limit": 0})):
        # Parseable argument objects lose incidental text, then fail argument
        # validation. Canonicalization must not confer execution eligibility.
        assert canonical.content == "" and canonical.tool_calls is original.tool_calls
    else:
        assert canonical is original
    tool.invoke.assert_not_called()


def test_registered_but_unauthorized_multicall_with_content_is_strict():
    original = native_batch().model_copy(update={"content": "I'll read these files."})
    provider, model, tool = setup_provider(original)
    context = brain_input()
    context = context.model_copy(update={"active_plan": context.active_plan.model_copy(
        update={"available_tools": ("list_files",)})})
    outcome = provider.generate(context, ())
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.tool_requests is outcome.tool_request is None
    assert len(model.calls) == 1
    assert original.content == "I'll read these files."
    tool.invoke.assert_not_called()


def test_static_native_action_contract_does_not_advertise_read_only_batching():
    assert "homogeneous batch" not in BRAIN_OUTPUT_PROTOCOL
    assert "read-only tool" not in BRAIN_OUTPUT_PROTOCOL
