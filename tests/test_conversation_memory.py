"""Slice 1: typed, authority-aware conversation memory without runtime wiring."""

import pytest
from pydantic import ValidationError

from core.memory import (
    ConversationContinuity, ConversationMemory, FactCategory, FactClaim,
    MemoryFact, MemoryLimits, MemorySource, MemoryUpdate, OpenQuestion,
    QuestionStatus, SourceKind, TurnStatus, merge_memory,
)


def human(turn=1):
    return MemorySource(kind=SourceKind.HUMAN, turn_index=turn, turn_id=f"turn-{turn}")


def inferred(turn=1):
    return MemorySource(kind=SourceKind.INFERRED, turn_index=turn, turn_id=f"turn-{turn}")


def accepted(turn=1):
    return MemorySource(kind=SourceKind.ACCEPTED_RESULT, turn_index=turn,
                        execution_id=f"exec-{turn}", plan_id="plan", plan_revision=1,
                        step_id="read", accepted_result_ref=f"result-{turn}")


def tool(turn=1):
    return MemorySource(kind=SourceKind.TOOL, turn_index=turn,
                        execution_id=f"exec-{turn}", tool_request_id=f"request-{turn}",
                        tool_result_ref=f"result-{turn}", tool_success=True)


def fact(text, source, *, category=FactCategory.USER_PREFERENCE, key="editor",
         claim=FactClaim.OBSERVATION):
    return MemoryFact(category=category, scope_key=key, text=text,
                      source=source, claim=claim)


def project(text, source, *, key="repository", claim=FactClaim.OBSERVATION):
    return fact(text, source, category=FactCategory.PROJECT_REPOSITORY,
                key=key, claim=claim)


def question(status=QuestionStatus.OPEN, resolved_turn=None):
    return OpenQuestion(question_id="q1", text="Which file?", source_turn_index=1,
                        source_turn_id="turn-1", status=status,
                        resolved_turn_index=resolved_turn)


def test_empty_round_trip_and_malformed_serialization():
    empty = ConversationMemory()
    assert empty.facts == () and empty.questions == ()
    assert ConversationMemory.model_validate_json(empty.model_dump_json()) == empty
    populated = merge_memory(empty, MemoryUpdate(
        facts=(fact("Use Vim", human()), project("Python repository", accepted())),
        questions=(question(),),
    ))
    assert ConversationMemory.model_validate_json(populated.model_dump_json()) == populated
    for bad in ('{"schema_version":2}', '{"schema_version":1,"facts":"oops"}',
                '{"schema_version":1,"unknown":4}', '{bad'):
        with pytest.raises(ValidationError):
            ConversationMemory.model_validate_json(bad)


def test_explicit_user_and_accepted_project_facts_can_be_added():
    result = merge_memory(ConversationMemory(), MemoryUpdate(facts=(
        fact("Prefer short answers", human()),
        project("Three Python files were read", accepted(),
                claim=FactClaim.EXECUTION_OUTCOME),
    )))
    assert {item.source.kind for item in result.facts} == {
        SourceKind.HUMAN, SourceKind.ACCEPTED_RESULT,
    }


def test_assistant_prose_is_not_project_authority():
    with pytest.raises(ValidationError):
        MemorySource(kind="assistant", turn_index=1, turn_id="turn-1")
    with pytest.raises(ValidationError):
        project("Repo uses Rust", human())
    candidate = project("Possibly Rust", inferred())
    assert candidate.source.kind == SourceKind.INFERRED
    assert candidate.claim == FactClaim.OBSERVATION


def test_newer_human_correction_supersedes_and_inference_cannot_overwrite():
    original = merge_memory(ConversationMemory(), MemoryUpdate(
        facts=(fact("Use Vim", human(1)),)))
    correction = merge_memory(original, MemoryUpdate(
        facts=(fact("Use VS Code", human(2)),)))
    assert correction.facts[0].text == "Use VS Code"
    weakened = merge_memory(correction, MemoryUpdate(
        facts=(fact("Use Emacs", inferred(3)),)))
    assert weakened == correction
    assert original.facts[0].text == "Use Vim"


def test_accepted_semantics_outrank_raw_tool_and_inference():
    strong = merge_memory(ConversationMemory(), MemoryUpdate(
        facts=(project("Task completed", accepted(1), claim=FactClaim.EXECUTION_OUTCOME),)))
    changed = merge_memory(strong, MemoryUpdate(facts=(
        project("Task failed", tool(2)), project("Task uncertain", inferred(3)),
    )))
    assert changed == strong
    reverse = merge_memory(ConversationMemory(), MemoryUpdate(facts=(
        project("Task uncertain", inferred(1)), project("Tool ran", tool(2)),
        project("Task completed", accepted(3), claim=FactClaim.EXECUTION_OUTCOME),
    )))
    assert reverse.facts[0].text == "Task completed"


def test_tool_success_cannot_establish_task_completion():
    with pytest.raises(ValidationError):
        project("Task completed", tool(), claim=FactClaim.EXECUTION_OUTCOME)
    with pytest.raises(ValidationError):
        MemorySource(kind=SourceKind.TOOL, turn_index=1, execution_id="e",
                     tool_request_id="r", tool_result_ref="x", tool_success=False)


def test_duplicate_updates_are_idempotent_and_inputs_immutable():
    original = ConversationMemory()
    update = MemoryUpdate(facts=(fact("Use Vim", human()),))
    once = merge_memory(original, update)
    assert merge_memory(once, update) == once
    assert original == ConversationMemory()
    with pytest.raises(ValidationError):
        once.facts = ()


@pytest.mark.parametrize("source", [
    {"kind": "human", "turn_index": 1},
    {"kind": "inferred", "turn_index": 1, "turn_id": "t", "execution_id": "e"},
    {"kind": "accepted_result", "turn_index": 1, "execution_id": "e"},
    {"kind": "tool", "turn_index": 1, "execution_id": "e",
     "tool_request_id": "r", "tool_result_ref": "x"},
])
def test_invalid_provenance_rejected(source):
    with pytest.raises(ValidationError):
        MemorySource.model_validate(source)


def test_continuity_is_not_controller_state_and_expires():
    continuity = ConversationContinuity(topic="Count source files", status=TurnStatus.COMPLETED,
        outcome_note="Answered", source_turn_index=1, source_turn_id="turn-1",
        source_execution_id="exec-1", expires_after_turn=2)
    assert not hasattr(continuity, "active_step")
    memory = merge_memory(ConversationMemory(), MemoryUpdate(continuity=continuity))
    assert memory.continuity == continuity
    later = merge_memory(memory, MemoryUpdate(facts=(fact("Use Vim", human(3)),)))
    assert later.continuity == ConversationContinuity()


def test_question_resolves_and_stays_resolved_on_duplicate_update():
    memory = merge_memory(ConversationMemory(), MemoryUpdate(questions=(question(),)))
    resolution = MemoryUpdate(resolve_question_ids=("q1",), resolution_turn_index=2)
    resolved = merge_memory(memory, resolution)
    assert resolved.questions[0].status == QuestionStatus.RESOLVED
    assert resolved.questions[0].resolved_turn_index == 2
    assert merge_memory(resolved, resolution) == resolved
    assert merge_memory(resolved, MemoryUpdate(questions=(question(),))) == resolved


def test_question_origin_must_be_human():
    with pytest.raises(ValidationError):
        OpenQuestion(question_id="q", text="Invented?", source_kind=SourceKind.INFERRED,
                     source_turn_index=1, source_turn_id="turn-1")


def test_budget_removes_whole_weaker_records_and_resolved_questions_first():
    update = MemoryUpdate(facts=(
        fact("Explicit preference", human(1), key="preferred"),
        fact("Tentative preference", inferred(2), key="tentative"),
        project("Accepted fact", accepted(1), key="accepted"),
        project("Tentative fact", inferred(2), key="tentative"),
    ), questions=(question(), OpenQuestion(
        question_id="q2", text="Old question", source_turn_index=1,
        source_turn_id="turn-1", status=QuestionStatus.RESOLVED,
        resolved_turn_index=2,
    )))
    limited = merge_memory(ConversationMemory(), update, limits=MemoryLimits(
        max_user_facts=1, max_project_facts=1, max_questions=1,
        max_total_chars=3000,
    ))
    assert {item.text for item in limited.facts} == {
        "Explicit preference", "Accepted fact",
    }
    assert [item.question_id for item in limited.questions] == ["q1"]
    tighter = merge_memory(limited, MemoryUpdate(), limits=MemoryLimits(max_total_chars=500))
    assert ConversationMemory.model_validate_json(tighter.model_dump_json()) == tighter
    assert all(item.text in {"Explicit preference", "Accepted fact"} for item in tighter.facts)


def test_per_record_limit_rejects_proposal_without_changing_previous_memory():
    original = merge_memory(ConversationMemory(), MemoryUpdate(
        facts=(fact("Use Vim", human()),)))
    with pytest.raises(ValueError, match="per-record"):
        merge_memory(original, MemoryUpdate(facts=(fact("Long record", human(2)),)),
                     limits=MemoryLimits(max_record_chars=8))
    assert original.facts[0].text == "Use Vim"


def test_total_budget_keeps_stronger_fact_and_existing_continuity_is_bounded():
    memory = merge_memory(ConversationMemory(), MemoryUpdate(
        facts=(fact("Explicit", human(), key="strong"),
               fact("Tentative", inferred(2), key="weak")),
        continuity=ConversationContinuity(topic="A long prior topic", status=TurnStatus.COMPLETED,
            source_turn_index=2, source_turn_id="turn-2"),
    ))
    bounded = merge_memory(memory, MemoryUpdate(), limits=MemoryLimits(
        max_record_chars=12, max_total_chars=600,
    ))
    assert bounded.continuity == ConversationContinuity()
    assert [item.text for item in bounded.facts] == ["Explicit"]
