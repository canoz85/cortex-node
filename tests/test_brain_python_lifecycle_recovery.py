from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

import core.brain_provider as module
from core.protocol.enums import BrainOutcomeKind as Kind
from test_brain_outcomes import SequenceModel, brain_input, native_action


def provider_for(content):
    raw = AIMessage(content=content)
    model = SequenceModel(raw, raw, raw)
    tool = SimpleNamespace(name="read_file", invoke=Mock())
    return module.LangChainBrainProvider(brain_llm=model, executable_tools=[tool]), model, tool, raw


@pytest.mark.parametrize("content,kind,expected_message", [
    ('brain_step_completed(message="done")', Kind.STEP_COMPLETED, "done"),
    ('brain_step_completed(message="Bitti: İstanbul, %100! \\"Harika\\" sonuç.")',
     Kind.STEP_COMPLETED, 'Bitti: İstanbul, %100! "Harika" sonuç.'),
    ('brain_step_failed(message="Cannot continue.")', Kind.STEP_FAILED, "Cannot continue."),
    ('brain_replan_requested(reason="Yol değişti.", constraints=["Yeni yolu kullan", "Veriyi koru"])',
     Kind.REPLAN_REQUESTED, "Yol değişti."),
])
def test_python_style_lifecycle_call_recovers_through_native_path(content, kind, expected_message, caplog):
    provider, model, tool, raw = provider_for(content)
    with caplog.at_level("DEBUG", logger="core.brain_provider"):
        outcome = provider.generate(brain_input(), ())
    assert outcome.kind == kind
    assert outcome.message == expected_message
    assert len(model.calls) == 1
    assert "recovered_from_text=True" in caplog.text
    assert module._canonical_brain_response(raw, {tool.name: tool}) is not raw
    tool.invoke.assert_not_called()


@pytest.mark.parametrize("content", [
    'brain_step_unknown(message="done")',
    'brain_step_completed(message="done", extra="no")',
    'brain_step_completed()',
    'I finished: brain_step_completed(message="done")',
    'brain_step_completed(message="done") extra prose',
    'brain_step_completed(message="first") brain_step_failed(message="second")',
    'brain_step_completed(message="unterminated)',
    'brain_step_completed(message="done"',
    'brain_step_completed(message=str("done"))',
    'brain_step_completed(message="done".__class__)',
    'brain_replan_requested(reason="change")',
])
def test_non_exact_or_non_schema_python_lifecycle_text_is_not_recovered(content, caplog):
    provider, model, tool, raw = provider_for(content)
    with caplog.at_level("DEBUG", logger="core.brain_provider"):
        outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.completion_evidence is None
    assert "recovered_from_text=True" not in caplog.text
    assert module._canonical_brain_response(raw, {tool.name: tool}) is raw
    tool.invoke.assert_not_called()


def test_native_lifecycle_call_takes_precedence_over_textual_recovery(monkeypatch):
    provider, _, tool, _ = provider_for("unused")
    native = native_action("brain_step_completed", {"message": "native wins"})
    provider.brain_llm = SequenceModel(native)
    recovery = Mock(side_effect=AssertionError("text recovery must not run"))
    monkeypatch.setattr(module, "_recover_textual_call", recovery)
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.STEP_COMPLETED
    assert outcome.message == "native wins"
    recovery.assert_not_called()
    tool.invoke.assert_not_called()


def test_invalid_native_calls_block_python_lifecycle_recovery():
    provider, model, tool, raw = provider_for('brain_step_completed(message="done")')
    raw.invalid_tool_calls = [{"name": "sql_query", "args": "{", "id": "bad", "error": "invalid"}]
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.error_code == "invalid_native_tool_call"
    assert len(model.calls) == 1
    tool.invoke.assert_not_called()


def test_textual_json_lifecycle_recovery_is_unchanged():
    provider, model, tool, _ = provider_for('brain_step_completed{"message":"JSON completion"}')
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.STEP_COMPLETED
    assert outcome.message == "JSON completion"
    assert len(model.calls) == 1
    tool.invoke.assert_not_called()


def test_missing_action_correction_includes_lifecycle_actions_and_empty_content():
    invalid = AIMessage(content="not an action")
    provider, _, tool, _ = provider_for(invalid.content)
    native = native_action("brain_step_completed", {"message": "recovered natively"})
    provider.brain_llm = SequenceModel(invalid, native)
    outcome = provider.generate(brain_input(), ())
    correction = provider.brain_llm.calls[1][-1].content
    assert outcome.kind == Kind.STEP_COMPLETED
    assert "exactly one native action" in correction
    assert "authorized executable tool call" in correction
    assert all(name in correction for name in (
        "brain_step_completed", "brain_step_failed", "brain_replan_requested",
    ))
    assert "assistant content empty" in correction
    tool.invoke.assert_not_called()
