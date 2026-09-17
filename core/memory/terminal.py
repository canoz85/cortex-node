"""Application-turn evidence and conservative deterministic memory extraction."""

from __future__ import annotations

from hashlib import sha256

from pydantic import Field, model_validator

from .models import (
    ConversationContinuity, MemoryModel, MemoryUpdate, OpenQuestion,
    TurnStatus,
)


class AcceptedCompletion(MemoryModel):
    """Narrow copy of a Controller-bound accepted step conclusion."""

    step_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    plan_revision: int = Field(ge=1)


class CompletedTurnEvidence(MemoryModel):
    """One terminal application turn; never protocol or execution state."""

    turn_id: str = Field(min_length=1)
    turn_index: int = Field(ge=1)
    user_request: str = Field(min_length=1)
    execution_id: str = Field(min_length=1)
    status: TurnStatus
    direct_response: bool = False
    accepted_completions: tuple[AcceptedCompletion, ...] = ()
    terminal_reason: str = ""
    accepted_answer: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_terminal(self) -> "CompletedTurnEvidence":
        if self.status == TurnStatus.UNKNOWN:
            raise ValueError("completed turn evidence requires terminal status")
        if self.direct_response and self.accepted_completions:
            raise ValueError("direct response cannot carry accepted plan completions")
        return self


def _short(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[:limit - 1].rstrip() + "…"


def _question_id(request: str) -> str:
    normalized = " ".join(request.strip().casefold().split())
    return sha256(normalized.encode("utf-8")).hexdigest()[:16]


def extract_memory_update(evidence: CompletedTurnEvidence) -> MemoryUpdate:
    """Propose continuity and explicit question state, never durable prose facts."""
    accepted = evidence.accepted_completions
    if evidence.status == TurnStatus.COMPLETED:
        if accepted:
            outcome = _short("; ".join(item.summary for item in accepted))
        elif evidence.direct_response and evidence.accepted_answer:
            outcome = _short(evidence.accepted_answer)
        else:
            outcome = "Completed."
    elif evidence.status == TurnStatus.FAILED:
        outcome = _short(evidence.terminal_reason) or "Execution failed."
    else:
        outcome = _short(evidence.terminal_reason) or "Execution cancelled."

    explicit_question = evidence.user_request.strip().endswith("?")
    answered = evidence.status == TurnStatus.COMPLETED and evidence.accepted_answer is not None
    question_id = _question_id(evidence.user_request) if explicit_question else None
    unresolved = explicit_question and not answered
    continuity = ConversationContinuity(
        topic=_short(evidence.user_request),
        status=evidence.status,
        outcome_note=outcome,
        pending_follow_up=_short(evidence.user_request) if unresolved else "",
        source_turn_index=evidence.turn_index,
        source_turn_id=evidence.turn_id,
        source_execution_id=evidence.execution_id,
        expires_after_turn=evidence.turn_index + 3,
    )
    question = OpenQuestion(
        question_id=question_id,
        text=_short(evidence.user_request),
        source_turn_index=evidence.turn_index,
        source_turn_id=evidence.turn_id,
    ) if unresolved else None
    return MemoryUpdate(
        continuity=continuity,
        questions=(question,) if question is not None else (),
        resolve_question_ids=(question_id,) if question_id and answered else (),
        resolution_turn_index=evidence.turn_index if question_id and answered else None,
    )
