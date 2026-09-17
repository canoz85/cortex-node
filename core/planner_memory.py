"""Deterministic, bounded projection of derived memory for Planner context."""

from __future__ import annotations

from dataclasses import dataclass

from core.memory import ConversationMemory, QuestionStatus, SourceKind
from core.protocol.models import (
    PlannerMemoryContext, PlannerMemoryContinuity, PlannerMemoryFact,
    PlannerMemoryQuestion,
)


@dataclass(frozen=True)
class PlannerMemoryLimits:
    max_user_facts: int = 6
    max_project_facts: int = 6
    max_open_questions: int = 3
    max_record_chars: int = 240
    max_continuity_chars: int = 400
    max_total_chars: int = 3_000

    def __post_init__(self) -> None:
        if min(self.max_user_facts, self.max_project_facts, self.max_open_questions) < 0:
            raise ValueError("Planner memory item limits cannot be negative")
        if min(self.max_record_chars, self.max_continuity_chars) < 1:
            raise ValueError("Planner memory text limits must be positive")
        if self.max_total_chars < len(PlannerMemoryContext().model_dump_json()):
            raise ValueError("Planner memory total budget is below the empty envelope")


_AUTHORITY = {
    SourceKind.HUMAN: "explicit_user",
    SourceKind.ACCEPTED_RESULT: "controller_accepted",
    SourceKind.TOOL: "tool_supported",
    SourceKind.INFERRED: "inferred",
}
_RANK = {
    SourceKind.HUMAN: 3,
    SourceKind.ACCEPTED_RESULT: 3,
    SourceKind.TOOL: 2,
    SourceKind.INFERRED: 0,
}


def project_planner_memory(
    memory: ConversationMemory,
    *,
    limits: PlannerMemoryLimits | None = None,
    current_turn_index: int | None = None,
) -> PlannerMemoryContext:
    """Select whole records by authority then recency; never mutate memory."""
    limits = limits or PlannerMemoryLimits()
    user_facts: list[PlannerMemoryFact] = []
    project_facts: list[PlannerMemoryFact] = []
    questions: list[PlannerMemoryQuestion] = []
    continuity: PlannerMemoryContinuity | None = None

    def snapshot() -> PlannerMemoryContext:
        return PlannerMemoryContext(
            user_facts=tuple(user_facts), project_facts=tuple(project_facts),
            continuity=continuity, open_questions=tuple(questions),
        )

    source = memory.continuity
    continuity_texts = (source.topic, source.outcome_note, source.pending_follow_up)
    continuity_is_current = (
        current_turn_index is None
        or source.expires_after_turn is None
        or current_turn_index <= source.expires_after_turn
    )
    if (
        continuity_is_current
        and any(continuity_texts)
        and all(len(text) <= limits.max_record_chars for text in continuity_texts)
        and sum(map(len, continuity_texts)) <= limits.max_continuity_chars
    ):
        candidate = PlannerMemoryContinuity(
            topic=source.topic, terminal_status=source.status.value,
            outcome_note=source.outcome_note, pending_follow_up=source.pending_follow_up,
        )
        continuity = candidate
        if len(snapshot().model_dump_json()) > limits.max_total_chars:
            continuity = None

    open_questions = sorted(
        (q for q in memory.questions if q.status == QuestionStatus.OPEN),
        key=lambda q: (-q.source_turn_index, q.question_id),
    )
    for question in open_questions:
        if len(questions) >= limits.max_open_questions:
            break
        if len(question.text) > limits.max_record_chars:
            continue
        candidate = PlannerMemoryQuestion(
            text=question.text, source_turn_index=question.source_turn_index,
        )
        questions.append(candidate)
        if len(snapshot().model_dump_json()) > limits.max_total_chars:
            questions.pop()

    facts = sorted(
        memory.facts,
        key=lambda f: (-_RANK[f.source.kind], -f.source.turn_index,
                       f.category.value, f.scope_key, f.text),
    )
    for fact in facts:
        target = user_facts if fact.category.is_user else project_facts
        maximum = limits.max_user_facts if fact.category.is_user else limits.max_project_facts
        if len(target) >= maximum or len(fact.text) > limits.max_record_chars:
            continue
        target.append(PlannerMemoryFact(
            category=fact.category.value, text=fact.text,
            authority=_AUTHORITY[fact.source.kind],
            source_turn_index=fact.source.turn_index,
        ))
        if len(snapshot().model_dump_json()) > limits.max_total_chars:
            target.pop()

    return snapshot()
