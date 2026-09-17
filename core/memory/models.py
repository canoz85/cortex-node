"""Immutable, versioned application memory. These values are not protocol state."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemoryModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class SourceKind(StrEnum):
    HUMAN = "human"
    ACCEPTED_RESULT = "accepted_result"
    TOOL = "tool"
    INFERRED = "inferred"


class FactCategory(StrEnum):
    USER_PROFILE = "user_profile"
    USER_PREFERENCE = "user_preference"
    USER_CONSTRAINT = "user_constraint"
    USER_GOAL = "user_goal"
    PROJECT_REPOSITORY = "project_repository"
    PROJECT_ARCHITECTURE = "project_architecture"
    PROJECT_ENVIRONMENT = "project_environment"
    PROJECT_TOOLING = "project_tooling"
    PROJECT_WORKFLOW = "project_workflow"
    PROJECT_DOMAIN = "project_domain"

    @property
    def is_user(self) -> bool:
        return self.value.startswith("user_")


class FactClaim(StrEnum):
    OBSERVATION = "observation"
    EXECUTION_OUTCOME = "execution_outcome"


class TurnStatus(StrEnum):
    UNKNOWN = "unknown"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QuestionStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


class MemorySource(MemoryModel):
    kind: SourceKind
    turn_index: int = Field(ge=0)
    turn_id: str | None = Field(default=None, min_length=1)
    execution_id: str | None = Field(default=None, min_length=1)
    plan_id: str | None = Field(default=None, min_length=1)
    plan_revision: int | None = Field(default=None, ge=1)
    step_id: str | None = Field(default=None, min_length=1)
    accepted_result_ref: str | None = Field(default=None, min_length=1)
    tool_request_id: str | None = Field(default=None, min_length=1)
    tool_result_ref: str | None = Field(default=None, min_length=1)
    tool_success: bool | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "MemorySource":
        execution_fields = (self.execution_id, self.plan_id, self.plan_revision, self.step_id,
                            self.accepted_result_ref, self.tool_request_id, self.tool_result_ref,
                            self.tool_success)
        if self.kind in (SourceKind.HUMAN, SourceKind.INFERRED):
            if self.turn_id is None or any(value is not None for value in execution_fields):
                raise ValueError("human/inferred source requires turn_id and no execution references")
        elif self.kind == SourceKind.ACCEPTED_RESULT:
            if not all((self.execution_id, self.plan_id, self.plan_revision,
                        self.step_id, self.accepted_result_ref)):
                raise ValueError("accepted result requires execution, plan, step, and result references")
            if any(value is not None for value in
                   (self.tool_request_id, self.tool_result_ref, self.tool_success)):
                raise ValueError("accepted result cannot carry raw tool references")
        elif self.kind == SourceKind.TOOL:
            if not all((self.execution_id, self.tool_request_id, self.tool_result_ref)):
                raise ValueError("tool source requires execution, request, and result references")
            if self.tool_success is not True or self.accepted_result_ref is not None:
                raise ValueError("only successful tool evidence is eligible; it is not an accepted result")
            if (self.plan_id is None) != (self.plan_revision is None):
                raise ValueError("tool plan ID and revision must be supplied together")
        return self


class MemoryFact(MemoryModel):
    category: FactCategory
    scope_key: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=4096)
    source: MemorySource
    claim: FactClaim = FactClaim.OBSERVATION

    @model_validator(mode="after")
    def validate_claim(self) -> "MemoryFact":
        if self.category.is_user and self.source.kind not in (SourceKind.HUMAN, SourceKind.INFERRED):
            raise ValueError("user memory must originate from a human or labeled inference")
        if not self.category.is_user and self.source.kind == SourceKind.HUMAN:
            raise ValueError("an unverified user project claim is not a project fact")
        if self.claim == FactClaim.EXECUTION_OUTCOME and (
            self.category.is_user or self.source.kind != SourceKind.ACCEPTED_RESULT
        ):
            raise ValueError("only an accepted semantic result can establish execution outcome")
        return self


class ConversationContinuity(MemoryModel):
    topic: str = Field(default="", max_length=4096)
    status: TurnStatus = TurnStatus.UNKNOWN
    outcome_note: str = Field(default="", max_length=4096)
    pending_follow_up: str = Field(default="", max_length=4096)
    source_turn_index: int | None = Field(default=None, ge=0)
    source_turn_id: str | None = Field(default=None, min_length=1)
    source_execution_id: str | None = Field(default=None, min_length=1)
    expires_after_turn: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_continuity(self) -> "ConversationContinuity":
        populated = bool(self.topic or self.outcome_note or self.pending_follow_up or
                         self.status != TurnStatus.UNKNOWN)
        if populated and (self.source_turn_index is None or self.source_turn_id is None):
            raise ValueError("continuity requires a source turn")
        if (self.source_turn_index is None) != (self.source_turn_id is None):
            raise ValueError("continuity source turn is incomplete")
        if self.expires_after_turn is not None and (
            self.source_turn_index is None or self.expires_after_turn < self.source_turn_index
        ):
            raise ValueError("continuity expiry must follow its source turn")
        return self


class OpenQuestion(MemoryModel):
    question_id: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=4096)
    source_kind: Literal[SourceKind.HUMAN] = SourceKind.HUMAN
    source_turn_index: int = Field(ge=0)
    source_turn_id: str = Field(min_length=1)
    status: QuestionStatus = QuestionStatus.OPEN
    resolved_turn_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_resolution(self) -> "OpenQuestion":
        if (self.status == QuestionStatus.RESOLVED) != (self.resolved_turn_index is not None):
            raise ValueError("resolved question requires resolution turn; open question cannot have one")
        if self.resolved_turn_index is not None and self.resolved_turn_index < self.source_turn_index:
            raise ValueError("question cannot resolve before it was asked")
        return self


class ConversationMemory(MemoryModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    facts: tuple[MemoryFact, ...] = ()
    continuity: ConversationContinuity = Field(default_factory=ConversationContinuity)
    questions: tuple[OpenQuestion, ...] = ()

    @model_validator(mode="after")
    def unique_records(self) -> "ConversationMemory":
        identities = [(fact.category, fact.scope_key) for fact in self.facts]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate fact scope")
        question_ids = [question.question_id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("duplicate question ID")
        return self


class MemoryUpdate(MemoryModel):
    """Untrusted proposal. Only merge_memory may turn it into derived memory."""

    facts: tuple[MemoryFact, ...] = ()
    continuity: ConversationContinuity | None = None
    questions: tuple[OpenQuestion, ...] = ()
    resolve_question_ids: tuple[str, ...] = ()
    resolution_turn_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_resolution(self) -> "MemoryUpdate":
        if bool(self.resolve_question_ids) != (self.resolution_turn_index is not None):
            raise ValueError("question resolutions require a resolution turn")
        if any(not question_id for question_id in self.resolve_question_ids):
            raise ValueError("question ID cannot be empty")
        return self


class MemoryLimits(MemoryModel):
    max_user_facts: int = Field(default=24, ge=0)
    max_project_facts: int = Field(default=24, ge=0)
    max_questions: int = Field(default=12, ge=0)
    max_record_chars: int = Field(default=500, ge=1, le=4096)
    max_total_chars: int = Field(default=16000, ge=300)
