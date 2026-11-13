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


@pytest.mark.parametrize("content,kind,expected_message", [
    ("brain_step_completed(message='The query successfully retrieved...\n- Annem: 1\n- Can: 2')",
     Kind.STEP_COMPLETED, "The query successfully retrieved...\n- Annem: 1\n- Can: 2"),
    ("brain_step_failed(message='Could not continue\\nwithout evidence')",
     Kind.STEP_FAILED, "Could not continue\nwithout evidence"),
    ("brain_replan_requested(reason='Use the updated plan', constraints=['Keep scope', 'Use new evidence'])",
     Kind.REPLAN_REQUESTED, "Use the updated plan"),
])
def test_single_quoted_python_lifecycle_call_recovers_through_native_path(content, kind, expected_message, caplog):
    provider, model, tool, raw = provider_for(content)
    with caplog.at_level("DEBUG", logger="core.brain_provider"):
        outcome = provider.generate(brain_input(), ())
    assert outcome.kind == kind
    assert outcome.message == expected_message
    assert len(model.calls) == 1
    assert "recovered_from_text=True" in caplog.text
    assert module._canonical_brain_response(raw, {tool.name: tool}) is not raw
    tool.invoke.assert_not_called()


@pytest.mark.parametrize("content,kind,expected_message", [
    ('brain_step_completed{message:<|"|>done<|"|>}', Kind.STEP_COMPLETED, "done"),
    ('brain_step_completed{message:<|"|>NexFolioM kayıtlarına göre,\n\n- AHE: 1 hesap\n- YKB: 1 hesap<|"|>}',
     Kind.STEP_COMPLETED, "NexFolioM kayıtlarına göre,\n\n- AHE: 1 hesap\n- YKB: 1 hesap"),
    ('brain_step_failed{message:<|"|>Veri alınamadı.<|"|>}', Kind.STEP_FAILED, "Veri alınamadı."),
    ('brain_replan_requested{reason:<|"|>Yeni veri bulundu.<|"|>, constraints:[<|"|>Mevcut kapsamı koru<|"|>]}',
     Kind.REPLAN_REQUESTED, "Yeni veri bulundu."),
])
def test_gemma_style_lifecycle_call_recovers_through_native_path(content, kind, expected_message, caplog):
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
    'brain_step_completed{message:<|"|>unclosed}',
    'brain_step_completed{message:|"|>missing opener<|"|>}',
    'brain_step_completed{message:<|"|>missing closer}',
    'brain_step_completed{message:<|"|>done<|"|>, extra:<|"|>no<|"|>}',
    'prefix brain_step_completed{message:<|"|>done<|"|>}',
    'brain_step_completed{message:<|"|>done<|"|>} trailing text',
    'brain_step_completed{message:<|"|>one<|"|>}brain_step_failed{message:<|"|>two<|"|>}',
    'brain_step_unknown{message:<|"|>done<|"|>}',
    'brain_step_completed{message:call("done")}',
    'brain_replan_requested{reason:<|"|>change<|"|>}',
])
def test_invalid_gemma_lifecycle_text_is_not_recovered(content, caplog):
    provider, _, tool, raw = provider_for(content)
    with caplog.at_level("DEBUG", logger="core.brain_provider"):
        outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert "recovered_from_text=True" not in caplog.text
    assert module._canonical_brain_response(raw, {tool.name: tool}) is raw
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
    "brain_step_completed(message='unclosed)",
    "brain_step_completed(message='first' + 'second')",
    "brain_step_completed(message=f'interpolated {value}')",
    "brain_step_completed(message=build('value'))",
    "brain_step_completed(message='done') trailing prose",
    "brain_step_completed(message='one') brain_step_failed(message='two')",
    "read_file(path='not recoverable')",
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
    native = native_action("brain_step_completed", {"message": "native wins"}).model_copy(update={
        "content": 'brain_step_completed{message:<|"|>text must not win<|"|>}',
    })
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


@pytest.mark.parametrize("name,content,kind,expected", [
    ("brain_step_completed", "brain_step_completed:  Done  ", Kind.STEP_COMPLETED, "Done"),
    ("brain_step_failed", "brain_step_failed: Cannot proceed", Kind.STEP_FAILED, "Cannot proceed"),
    ("brain_replan_requested", "brain_replan_requested: Change strategy", Kind.REPLAN_REQUESTED, "Change strategy"),
])
def test_colon_style_lifecycle_recovery(content, name, kind, expected):
    provider, model, tool, _ = provider_for(content)
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == kind
    assert outcome.message == expected
    assert len(model.calls) == 1
    assert outcome.replan_request.constraints == () if name == "brain_replan_requested" else True
    tool.invoke.assert_not_called()


@pytest.mark.parametrize("content", [
    "brain_step_unknown: Done",
    "brain_step_completed:",
    "prefix brain_step_completed: Done",
    "brain_step_completed: Done\nbrain_step_failed: Nope",
])
def test_invalid_colon_lifecycle_text_is_rejected(content):
    provider, _, tool, raw = provider_for(content)
    outcome = provider.generate(brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert module._canonical_brain_response(raw, {tool.name: tool}) is raw
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


def test_identifiable_action_repair_prompt_preserves_previous_action():
    invalid = AIMessage(content="brain_step_completed: ")
    provider, _, _, _ = provider_for(invalid.content)
    provider.brain_llm = SequenceModel(invalid, native_action("brain_step_completed", {"message": "Done"}))
    outcome = provider.generate(brain_input(), ())
    correction = provider.brain_llm.calls[1][-1].content
    assert outcome.kind == Kind.STEP_COMPLETED
    assert "Repair only the previous response into native-call format" in correction
    assert "Preserve the same intended action and arguments" in correction
    assert "Do not reconsider the task" in correction
    assert "choose another action" in correction
    assert "Do not write the action as assistant text" in correction
    assert "Do not output textual function-call syntax such as action(...)" in correction
    assert "Invoke the already-bound intended action natively" in correction
    assert "returned as a native tool call" in correction
    assert "Leave assistant content empty" in correction
    assert "Return exactly one native action" not in correction
    assert "authorized executable tool call" not in correction


def test_unidentifiable_prose_keeps_existing_fallback_instruction():
    invalid = AIMessage(content="I think the next step may be useful.")
    provider, _, _, _ = provider_for(invalid.content)
    provider.brain_llm = SequenceModel(invalid, native_action("brain_step_completed", {"message": "Done"}))
    provider.generate(brain_input(), ())
    correction = provider.brain_llm.calls[1][-1].content
    assert "Return exactly one native action" in correction
    assert "preserve the same intended action" not in correction
