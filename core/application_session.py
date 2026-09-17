"""Bounded, trusted conversation state owned by the application session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from core.graph_messages import (
    ACCEPTED_FINALIZER_PROVENANCE,
    CONVERSATION_PROVENANCE_KEY,
)
from core.memory import ConversationMemory


@dataclass(frozen=True)
class RecentConversationLimits:
    max_turns: int = 16
    max_messages: int = 32
    max_message_chars: int = 8_000
    max_total_chars: int = 64_000

    def __post_init__(self) -> None:
        if min(self.max_turns, self.max_messages, self.max_message_chars, self.max_total_chars) < 1:
            raise ValueError("recent conversation limits must be positive")


@dataclass(frozen=True)
class ApplicationSession:
    conversation_memory: ConversationMemory = field(default_factory=ConversationMemory)
    recent_conversation: tuple[BaseMessage, ...] = ()
    # Compatibility input for the old graph state; never interpreted as typed memory.
    legacy_rolling_summary: str = ""
    completed_turn_count: int = 0
    maintenance_start_turn: int | None = None
    maintenance_end_turn: int | None = None
    last_enrichment_status: Literal[
        "not_run", "disabled", "accepted", "no_op", "failed_safe",
    ] = "not_run"

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_memory, ConversationMemory):
            raise TypeError("conversation_memory must be typed ConversationMemory")
        if not isinstance(self.legacy_rolling_summary, str):
            raise TypeError("legacy_rolling_summary must be a string")
        if not isinstance(self.completed_turn_count, int) or self.completed_turn_count < 0:
            raise ValueError("completed_turn_count must be a nonnegative integer")
        if (self.maintenance_start_turn is None) != (self.maintenance_end_turn is None):
            raise ValueError("maintenance streak must have both endpoints")
        if self.maintenance_start_turn is not None and not (
            1 <= self.maintenance_start_turn <= self.maintenance_end_turn <= self.completed_turn_count
        ):
            raise ValueError("maintenance streak must belong to completed turns")
        if self.last_enrichment_status not in {
            "not_run", "disabled", "accepted", "no_op", "failed_safe",
        }:
            raise ValueError("invalid memory enrichment status")
        object.__setattr__(
            self, "recent_conversation", bounded_recent_conversation(self.recent_conversation),
        )


def bounded_recent_conversation(
    messages: list[BaseMessage] | tuple[BaseMessage, ...],
    limits: RecentConversationLimits = RecentConversationLimits(),
) -> tuple[BaseMessage, ...]:
    """Keep whole recent turns, stripping untrusted message metadata and chatter.

    An oversized message discards its whole turn. No retained message is clipped.
    """
    turns: list[list[BaseMessage]] = []
    for message in messages:
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            turns.append([HumanMessage(content=message.content)])
        elif (
            isinstance(message, AIMessage)
            and isinstance(message.content, str)
            and message.additional_kwargs.get(CONVERSATION_PROVENANCE_KEY)
            == ACCEPTED_FINALIZER_PROVENANCE
            and not message.tool_calls
            and not message.invalid_tool_calls
            and turns
            and len(turns[-1]) == 1
        ):
            turns[-1].append(AIMessage(
                content=message.content,
                additional_kwargs={
                    CONVERSATION_PROVENANCE_KEY: ACCEPTED_FINALIZER_PROVENANCE,
                },
            ))

    eligible = [
        turn for turn in turns
        if all(len(message.content) <= limits.max_message_chars for message in turn)
    ]
    retained: list[list[BaseMessage]] = []
    message_count = 0
    total_chars = 0
    for turn in reversed(eligible):
        turn_chars = sum(len(message.content) for message in turn)
        if (
            len(retained) >= limits.max_turns
            or message_count + len(turn) > limits.max_messages
            or total_chars + turn_chars > limits.max_total_chars
        ):
            break
        retained.append(turn)
        message_count += len(turn)
        total_chars += turn_chars
    return tuple(message for turn in reversed(retained) for message in turn)
