"""Strict textual compatibility stays inside the provider's native validation path."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict

import core.brain_provider as module
from core.protocol.enums import BrainOutcomeKind as Kind
from test_brain_argument_correction import context_for
from test_brain_ollama_boundary import boundary, git_brain_input
from test_brain_outcomes import SequenceModel, brain_input, native_action
from test_duplicate_successful_call import scoped_record
from tools.file_ops import get_file_tools


class ListArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str


def setup(content):
    raw = AIMessage(content=content)
    model = SequenceModel(raw, raw, raw)
    tool = SimpleNamespace(name="list_files", args_schema=ListArguments, invoke=Mock())
    provider = module.LangChainBrainProvider(brain_llm=model, executable_tools=[tool])
    return provider, model, tool, raw


@pytest.mark.parametrize("content,kind", [
    ('brain_step_completed{"message":"done"}', Kind.STEP_COMPLETED),
    ('list_files{"path":"some/path"}', Kind.TOOL_REQUESTED),
    ('brain_step_failed{"message":"failed"}', Kind.STEP_FAILED),
    ('brain_replan_requested{"reason":"changed","constraints":[]}', Kind.REPLAN_REQUESTED),
])
def test_exact_bound_action_recovers_through_normalization(content, kind, monkeypatch, caplog, capsys, tmp_path):
    log_path = tmp_path / "exchanges.jsonl"
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", str(log_path))
    provider, model, tool, raw = setup(content)
    snapshot = raw.model_dump()
    normalize = Mock(wraps=module.normalize_brain_output)
    monkeypatch.setattr(module, "normalize_brain_output", normalize)
    with caplog.at_level("DEBUG", logger="core.brain_provider"):
        outcome = provider.generate(brain_input(), ())
    assert outcome.kind == kind and len(model.calls) == 1
    canonical = normalize.call_args.args[0]
    assert canonical.content == "" and len(canonical.tool_calls) == 1
    assert raw.model_dump() == snapshot
    expected = module.normalize_brain_output(canonical, brain_input(), {tool.name})
    assert outcome.model_copy(update={"usage": expected.usage}) == expected
    assert "recovered_from_text=True" in caplog.text
    record, = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert record["response"]["content"] == content and record["response"]["tool_calls"] == []
    tool.invoke.assert_not_called()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("content", [
    'list_files{"path":}', 'unknown_tool{"path":"a"}',
    'Please list_files{"path":"a"}', 'list_files{"path":"a"} please',
    '```json\nlist_files{"path":"a"}\n```',
    'list_files{"path":"a"}list_files{"path":"b"}',
    'list_files{"path":"a"}\nbrain_step_completed{"message":"done"}',
    'list_files[]', 'list_files{"path":NaN}',
    'list_files{"path":"a","path":"b"}',
    [{"type": "text", "text": 'list_files{"path":"a"}'}],
])
def test_non_exact_text_is_not_executable(content, caplog):
    provider, model, tool, raw = setup(content)
    with caplog.at_level("DEBUG", logger="core.brain_provider"):
        outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT and outcome.tool_request is None
    assert module._canonical_brain_response(raw, {tool.name: tool}) is raw
    assert "recovered_from_text=True" not in caplog.text
    tool.invoke.assert_not_called()


def test_registered_but_unbound_tool_is_not_recovered():
    provider, model, tool, raw = setup('list_files{"path":"a"}')
    outcome = provider.generate(context_for("read_file"), ())
    assert outcome.kind == Kind.INVALID_OUTPUT and outcome.tool_request is None
    assert len(model.calls) == 3


@pytest.mark.parametrize("arguments", [{}, {"path": 12}, {"path": "a", "extra": True}])
def test_recovered_schema_errors_use_native_argument_validation(arguments, monkeypatch):
    provider, model, tool, raw = setup("list_files" + json.dumps(arguments))
    feedback = Mock(wraps=module._native_argument_feedback)
    monkeypatch.setattr(module, "_native_argument_feedback", feedback)
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT and outcome.error_code == "invalid_native_tool_arguments"
    assert outcome.tool_request is None and len(model.calls) == 3
    assert feedback.call_args.args[0].tool_calls[0]["args"] == arguments
    tool.invoke.assert_not_called()


@pytest.mark.parametrize("arguments", [{}, {"message": "done", "extra": True}])
def test_recovered_lifecycle_keeps_normal_validation(arguments):
    provider, model, tool, raw = setup("brain_step_completed" + json.dumps(arguments))
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT and outcome.completion_evidence is None
    assert len(model.calls) == 1


def test_native_call_bypasses_recovery(monkeypatch):
    provider, model, tool, raw = setup("unused")
    native = native_action("list_files", {"path": "native"}).model_copy(update={
        "content": 'brain_step_completed{"message":"text must not win"}',
    })
    provider.brain_llm = SequenceModel(native)
    recovery = Mock(side_effect=AssertionError("native calls must bypass recovery"))
    monkeypatch.setattr(module, "_recover_textual_call", recovery)
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.TOOL_REQUESTED and outcome.tool_request.arguments == {"path": "native"}
    recovery.assert_not_called()


def test_invalid_native_channel_blocks_textual_recovery():
    provider, model, tool, raw = setup('brain_step_completed{"message":"done"}')
    raw.invalid_tool_calls = [{"name": "list_files", "args": "{", "id": "bad", "error": "invalid"}]
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT and outcome.error_code == "invalid_native_tool_call"
    assert len(model.calls) == 1


def test_recovered_tool_keeps_duplicate_protection():
    provider, model, tool, raw = setup('list_files{"path":"a"}')
    context = brain_input()
    record = scoped_record(context, tool.name, {"path": "a"})
    context = context.model_copy(update={"tool_execution_history": (record,)})
    outcome = provider.generate(context, ())
    assert outcome.error_code == "duplicate_successful_call" and outcome.tool_request is None
    assert len(model.calls) == 3
    tool.invoke.assert_not_called()


def test_recovery_does_not_bypass_active_step_requirement():
    provider, model, tool, raw = setup('brain_step_completed{"message":"done"}')
    outcome = provider.generate(brain_input(direct=True), ())
    assert outcome.kind == Kind.INVALID_OUTPUT and outcome.error_code == "native_call_requires_active_step"


def test_textual_lifecycle_recovers_across_real_ollama_boundary():
    provider, requests = boundary([{"content": 'brain_step_completed{"message":"done"}'}])
    outcome = provider.generate(git_brain_input(), ())
    assert outcome.kind == Kind.STEP_COMPLETED and outcome.message == "done"
    assert len(requests) == 1


def test_real_bound_list_files_schema_accepts_recovered_arguments():
    model = SequenceModel(AIMessage(content='list_files{"path":"some/path"}'))
    provider = module.LangChainBrainProvider(brain_llm=model, executable_tools=get_file_tools("."))
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.TOOL_REQUESTED
    assert outcome.tool_request.tool_name == "list_files"
    assert outcome.tool_request.arguments == {"path": "some/path"}
    assert len(model.calls) == 1


def test_domain_normalizer_remains_native_only():
    raw = AIMessage(content='brain_step_completed{"message":"done"}')
    outcome = module.normalize_brain_output(raw, brain_input(), {"list_files"})
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.error_code == "unexpected_response_content"
