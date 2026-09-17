"""Pure, source-aware merge and structural budget for derived memory."""

from __future__ import annotations

from .models import (
    ConversationContinuity, ConversationMemory, FactCategory, MemoryFact,
    MemoryLimits, MemoryUpdate, OpenQuestion, QuestionStatus, SourceKind,
)


def _rank(fact: MemoryFact) -> int:
    if fact.category.is_user:
        return 2 if fact.source.kind == SourceKind.HUMAN else 0
    return {SourceKind.ACCEPTED_RESULT: 3, SourceKind.TOOL: 2,
            SourceKind.INFERRED: 0}[fact.source.kind]


def _preferred(left: MemoryFact, right: MemoryFact) -> MemoryFact:
    def key(fact: MemoryFact) -> tuple:
        return (_rank(fact), fact.source.turn_index, fact.source.turn_id or "",
                fact.model_dump_json())
    return max((left, right), key=key)


def _fact_prune_key(fact: MemoryFact) -> tuple:
    return (_rank(fact), fact.source.turn_index, fact.category.value, fact.scope_key)


def _question_prune_key(question: OpenQuestion) -> tuple:
    return (question.status == QuestionStatus.OPEN, question.source_turn_index,
            question.question_id)


def _check_text(update: MemoryUpdate, limits: MemoryLimits) -> None:
    records = [*(fact.text for fact in update.facts),
               *(question.text for question in update.questions)]
    if update.continuity is not None:
        records.extend((update.continuity.topic, update.continuity.outcome_note,
                        update.continuity.pending_follow_up))
    if any(len(text) > limits.max_record_chars for text in records):
        raise ValueError("memory update exceeds per-record text limit")


def _bounded(memory: ConversationMemory, limits: MemoryLimits) -> ConversationMemory:
    facts = list(memory.facts)
    questions = list(memory.questions)
    continuity = memory.continuity
    if any(len(text) > limits.max_record_chars for text in (
        continuity.topic, continuity.outcome_note, continuity.pending_follow_up
    )):
        continuity = ConversationContinuity()

    def count(is_user: bool) -> int:
        return sum(fact.category.is_user == is_user for fact in facts)

    for is_user, maximum in ((True, limits.max_user_facts),
                             (False, limits.max_project_facts)):
        while count(is_user) > maximum:
            victim = min((fact for fact in facts if fact.category.is_user == is_user),
                         key=_fact_prune_key)
            facts.remove(victim)
    while len(questions) > limits.max_questions:
        questions.remove(min(questions, key=_question_prune_key))

    def build() -> ConversationMemory:
        return ConversationMemory(
            facts=tuple(sorted(facts, key=lambda f: (f.category.value, f.scope_key))),
            continuity=continuity,
            questions=tuple(sorted(questions, key=lambda q: q.question_id)),
        )

    candidate = build()
    while len(candidate.model_dump_json()) > limits.max_total_chars:
        if questions and any(q.status == QuestionStatus.RESOLVED for q in questions):
            questions.remove(min((q for q in questions if q.status == QuestionStatus.RESOLVED),
                                 key=_question_prune_key))
        elif facts and (not questions or _rank(min(facts, key=_fact_prune_key)) < 2):
            facts.remove(min(facts, key=_fact_prune_key))
        elif questions:
            questions.remove(min(questions, key=_question_prune_key))
        elif facts:
            facts.remove(min(facts, key=_fact_prune_key))
        elif continuity != ConversationContinuity():
            continuity = ConversationContinuity()
        else:
            raise ValueError("memory budget is smaller than the empty memory envelope")
        candidate = build()
    return candidate


def merge_memory(
    existing: ConversationMemory,
    update: MemoryUpdate,
    *,
    limits: MemoryLimits | None = None,
) -> ConversationMemory:
    """Validate a proposal, resolve scoped conflicts, and return bounded new memory.

    Invalid typed provenance or overlong proposed text raises ValueError. Unknown
    question IDs are ignored. Neither input is mutated.
    """
    limits = limits or MemoryLimits()
    _check_text(update, limits)
    facts: dict[tuple[FactCategory, str], MemoryFact] = {
        (fact.category, fact.scope_key): fact for fact in existing.facts
        if len(fact.text) <= limits.max_record_chars
    }
    for fact in update.facts:
        identity = (fact.category, fact.scope_key)
        facts[identity] = _preferred(facts[identity], fact) if identity in facts else fact

    questions: dict[str, OpenQuestion] = {
        question.question_id: question for question in existing.questions
        if len(question.text) <= limits.max_record_chars
    }
    for question in update.questions:
        old = questions.get(question.question_id)
        if old is None:
            questions[question.question_id] = question
        elif old.status == QuestionStatus.OPEN and question.status == QuestionStatus.RESOLVED:
            questions[question.question_id] = question if question.source_turn_index >= old.source_turn_index else old
        elif old.status == QuestionStatus.RESOLVED and question.status == QuestionStatus.OPEN:
            if question.source_turn_index > (old.resolved_turn_index or 0):
                questions[question.question_id] = question
        elif (question.source_turn_index, question.model_dump_json()) > (
            old.source_turn_index, old.model_dump_json()
        ):
            questions[question.question_id] = question
    for question_id in update.resolve_question_ids:
        old = questions.get(question_id)
        if old is not None and update.resolution_turn_index >= old.source_turn_index:
            resolved_turn = max(update.resolution_turn_index, old.resolved_turn_index or 0)
            questions[question_id] = old.model_copy(update={
                "status": QuestionStatus.RESOLVED,
                "resolved_turn_index": resolved_turn,
            })

    continuity = existing.continuity
    proposed = update.continuity
    if proposed is not None and (
        proposed.source_turn_index if proposed.source_turn_index is not None else -1,
        proposed.model_dump_json(),
    ) > (
        continuity.source_turn_index if continuity.source_turn_index is not None else -1,
        continuity.model_dump_json(),
    ):
        continuity = proposed
    if continuity.expires_after_turn is not None:
        observed_turns = [0, *(fact.source.turn_index for fact in facts.values()),
                          *(question.source_turn_index for question in questions.values())]
        if proposed is not None and proposed.source_turn_index is not None:
            observed_turns.append(proposed.source_turn_index)
        latest_turn = max(observed_turns)
        if latest_turn > continuity.expires_after_turn:
            continuity = ConversationContinuity()

    return _bounded(ConversationMemory(
        facts=tuple(facts.values()), continuity=continuity,
        questions=tuple(questions.values()),
    ), limits)
