"""Whole-turn compaction after confirmed application memory maintenance."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage

from core.application_session import RecentConversationLimits, bounded_recent_conversation
from core.memory import OpenQuestion, QuestionStatus


@dataclass(frozen=True)
class CompactionLimits:
    target_turns: int = 12
    minimum_verbatim_turns: int = 8

    def __post_init__(self) -> None:
        if not 1 <= self.minimum_verbatim_turns <= self.target_turns <= RecentConversationLimits().max_turns:
            raise ValueError("compaction turn limits must fit the hard recent-history ceiling")


def compact_recent_conversation(
    messages: tuple[BaseMessage, ...] | list[BaseMessage],
    *,
    completed_turn_count: int,
    maintenance_start_turn: int | None,
    maintenance_end_turn: int | None,
    questions: tuple[OpenQuestion, ...] = (),
    limits: CompactionLimits = CompactionLimits(),
) -> tuple[BaseMessage, ...]:
    """Remove whole oldest eligible turns while retaining verbatim recent context.

    Old sessions without a verified maintenance streak keep the Slice 2 hard
    ceiling. A missing/failed maintenance turn prevents normal compaction until
    that gap has aged out of the hard window.
    """
    bounded = bounded_recent_conversation(messages)
    turns: list[list[BaseMessage]] = []
    for message in bounded:
        if isinstance(message, HumanMessage):
            turns.append([message])
        elif turns:
            turns[-1].append(message)
    if (
        len(turns) <= limits.target_turns
        or maintenance_start_turn is None
        or maintenance_end_turn is None
    ):
        return bounded
    first_turn_index = completed_turn_count - len(turns) + 1
    if first_turn_index < maintenance_start_turn:
        return bounded

    open_texts = {
        " ".join(question.text.casefold().split())
        for question in questions if question.status == QuestionStatus.OPEN
    }
    keep = [True] * len(turns)
    remaining = len(turns)
    for offset, turn in enumerate(turns[:-limits.minimum_verbatim_turns]):
        if remaining <= limits.target_turns:
            break
        turn_index = first_turn_index + offset
        if turn_index > maintenance_end_turn:
            continue
        user_text = " ".join(str(turn[0].content).casefold().split())
        if user_text in open_texts:
            continue
        keep[offset] = False
        remaining -= 1
    return tuple(message for kept, turn in zip(keep, turns) if kept for message in turn)
