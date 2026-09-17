"""Curated application memory proposals and deterministic provenance binding."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.memory import (
    ConversationMemory, FactCategory, FactClaim, MemoryFact, MemorySource,
    MemoryUpdate, SourceKind,
    TurnStatus,
)
from core.memory.models import MemoryModel
from core.memory.terminal import AcceptedCompletion, CompletedTurnEvidence


class ExistingFactHint(MemoryModel):
    category: FactCategory
    scope_key: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=200)
    source_kind: SourceKind


class MemoryUpdateRequest(MemoryModel):
    """Bounded trusted evidence; no graph history or mutable session."""

    turn_id: str = Field(min_length=1)
    turn_index: int = Field(ge=1)
    execution_id: str = Field(min_length=1)
    user_request: str = Field(min_length=1, max_length=4_000)
    terminal_status: TurnStatus
    accepted_answer: str | None = Field(default=None, max_length=600)
    accepted_completions: tuple[AcceptedCompletion, ...] = Field(default=(), max_length=4)
    existing_facts: tuple[ExistingFactHint, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def validate_input_budget(self) -> "MemoryUpdateRequest":
        if any(len(item.summary) > 1_200 for item in self.accepted_completions):
            raise ValueError("accepted summary exceeds memory updater input limit")
        if sum(len(item.text) + len(item.scope_key) for item in self.existing_facts) > 2_000:
            raise ValueError("existing fact hints exceed memory updater input limit")
        return self


class LLMProposalModel(BaseModel):
    """JSON-facing proposal values; accepted memory remains strict and immutable."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProposedMemoryFact(LLMProposalModel):
    category: FactCategory
    scope_key: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=500)
    source_handle: str = Field(min_length=1, max_length=160)
    evidence_quote: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_record_text(self) -> "ProposedMemoryFact":
        if not self.scope_key.strip() or self.scope_key != self.scope_key.strip():
            raise ValueError("memory scope key must be nonblank and trimmed")
        if not self.text.strip() or self.text != self.text.strip():
            raise ValueError("memory fact text must be nonblank and trimmed")
        return self


class MemoryProposal(LLMProposalModel):
    facts: tuple[ProposedMemoryFact, ...] = Field(default=(), max_length=8)


class MemoryProposalProvider(Protocol):
    def generate(self, request: MemoryUpdateRequest) -> MemoryProposal: ...


class MemoryUpdater(Protocol):
    def propose(self, request: MemoryUpdateRequest) -> MemoryUpdate: ...


_RANK = {
    SourceKind.HUMAN: 3, SourceKind.ACCEPTED_RESULT: 3,
    SourceKind.TOOL: 2, SourceKind.INFERRED: 0,
}


def build_memory_update_request(
    evidence: CompletedTurnEvidence,
    existing: ConversationMemory,
) -> MemoryUpdateRequest:
    """Select a small correction context and accepted sources for one turn."""
    if len(evidence.user_request) > 4_000:
        raise ValueError("current user request exceeds memory updater input limit")
    facts = sorted(
        existing.facts,
        key=lambda f: (-_RANK[f.source.kind], -f.source.turn_index,
                       f.category.value, f.scope_key),
    )
    hints: list[ExistingFactHint] = []
    hint_chars = 0
    for fact in facts:
        if len(hints) >= 12 or len(fact.text) > 200:
            continue
        size = len(fact.text) + len(fact.scope_key)
        if hint_chars + size > 2_000:
            continue
        hints.append(ExistingFactHint(
            category=fact.category, scope_key=fact.scope_key,
            text=fact.text, source_kind=fact.source.kind,
        ))
        hint_chars += size
    accepted = tuple(
        item for item in evidence.accepted_completions
        if len(item.summary) <= 1_200
    )[:4]
    answer = evidence.accepted_answer
    return MemoryUpdateRequest(
        turn_id=evidence.turn_id, turn_index=evidence.turn_index,
        execution_id=evidence.execution_id, user_request=evidence.user_request,
        terminal_status=evidence.status,
        accepted_answer=answer if answer is not None and len(answer) <= 600 else None,
        accepted_completions=accepted, existing_facts=tuple(hints),
    )


class LLMMemoryUpdater:
    """Accept only source-bound proposals; merge and persistence belong elsewhere."""

    def __init__(self, provider: MemoryProposalProvider):
        self.provider = provider

    def propose(self, request: MemoryUpdateRequest) -> MemoryUpdate:
        proposal = self.provider.generate(request)
        if not isinstance(proposal, MemoryProposal):
            raise ValueError("memory provider returned no typed proposal")
        sources = {
            f"accepted:{item.evidence_id}": item
            for item in request.accepted_completions
        }
        if len(sources) != len(request.accepted_completions):
            raise ValueError("duplicate accepted completion handles")
        facts: list[MemoryFact] = []
        seen: set[tuple[FactCategory, str]] = set()
        for proposed in proposal.facts:
            identity = (proposed.category, proposed.scope_key)
            if identity in seen:
                raise ValueError("duplicate fact scope in memory proposal")
            seen.add(identity)
            quote = proposed.evidence_quote.strip()
            if quote != proposed.evidence_quote or len(quote) < 4 or not any(c.isalnum() for c in quote):
                raise ValueError("memory evidence quote must be exact")
            if proposed.category.is_user:
                if proposed.text.casefold() == quote.casefold():
                    raise ValueError("user fact text must be a normalized durable fact")
                if (
                    proposed.source_handle != "human_current"
                    or quote not in request.user_request
                    or quote.endswith("?")
                ):
                    raise ValueError("user fact has unsupported human evidence")
                source = MemorySource(
                    kind=SourceKind.HUMAN, turn_index=request.turn_index,
                    turn_id=request.turn_id,
                )
            else:
                if proposed.text != quote:
                    raise ValueError("project fact text must equal its accepted evidence span")
                accepted = sources.get(proposed.source_handle)
                if accepted is None or quote not in accepted.summary:
                    raise ValueError("project fact has unsupported accepted evidence")
                source = MemorySource(
                    kind=SourceKind.ACCEPTED_RESULT,
                    turn_index=request.turn_index,
                    turn_id=request.turn_id,
                    execution_id=request.execution_id,
                    plan_id=accepted.plan_id,
                    plan_revision=accepted.plan_revision,
                    step_id=accepted.step_id,
                    accepted_result_ref=accepted.evidence_id,
                )
            facts.append(MemoryFact(
                category=proposed.category, scope_key=proposed.scope_key,
                text=proposed.text, source=source,
                claim=FactClaim.OBSERVATION,
            ))
        return MemoryUpdate(facts=tuple(facts))
