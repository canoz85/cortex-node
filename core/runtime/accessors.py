"""Runtime read accessors for protocol model migration.

This module is the read boundary between the legacy runtime dictionary and the
new immutable protocol contracts. It is intentionally stateless, deterministic,
and translation-free.

The bridge module owns translation. This module only reads already-attached
protocol models so future controller migration can centralize protocol access.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.protocol.models import (
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    ProtocolVisibleState,
    RetryMetadata,
    WorkingState,
)


StateSource = Mapping[str, Any] | ExecutionState | None


def _execution_state_from_source(state: StateSource) -> ExecutionState | None:
    """Return an attached ExecutionState without reconstructing it."""
    if isinstance(state, ExecutionState):
        return state
    if isinstance(state, Mapping):
        candidate = state.get("execution_state")
        return candidate if isinstance(candidate, ExecutionState) else None
    return None


def _protocol_state_from_source(state: StateSource) -> ProtocolVisibleState | None:
    """Return attached protocol-visible state when already available."""
    execution_state = _execution_state_from_source(state)
    if execution_state is None:
        return None
    return execution_state.protocol_visible


def _working_state_from_source(state: StateSource) -> WorkingState | None:
    """Return attached working state when already available."""
    execution_state = _execution_state_from_source(state)
    if execution_state is None:
        return None
    return execution_state.working


def get_execution_state(state: StateSource) -> ExecutionState | None:
    """Read the attached ExecutionState without translating legacy runtime data."""
    return _execution_state_from_source(state)


def get_protocol_state(state: StateSource) -> ProtocolVisibleState | None:
    """Read the attached Protocol-visible State from the runtime migration boundary."""
    return _protocol_state_from_source(state)


def get_working_state(state: StateSource) -> WorkingState | None:
    """Read the attached Working State from the runtime migration boundary."""
    return _working_state_from_source(state)


def get_execution_identity(state: StateSource) -> ExecutionIdentity | None:
    """Read the execution identity from attached protocol state, if present."""
    protocol_state = _protocol_state_from_source(state)
    if protocol_state is None:
        return None
    return protocol_state.identity


def get_execution_cursor(state: StateSource) -> ExecutionCursor | None:
    """Read the execution cursor from attached protocol state, if present."""
    protocol_state = _protocol_state_from_source(state)
    if protocol_state is None:
        return None
    return protocol_state.cursor


def get_active_plan(state: StateSource) -> ExecutionPlan | None:
    """Read the active plan from attached protocol state, if present."""
    protocol_state = _protocol_state_from_source(state)
    if protocol_state is None:
        return None
    return protocol_state.active_plan


def get_active_step(state: StateSource) -> ExecutionStep | None:
    """Read the active step from attached protocol state, if present."""
    protocol_state = _protocol_state_from_source(state)
    if protocol_state is None:
        return None
    return protocol_state.active_step


def get_retry_metadata(state: StateSource) -> RetryMetadata | None:
    """Read retry metadata from attached protocol state, if present."""
    protocol_state = _protocol_state_from_source(state)
    if protocol_state is None:
        return None
    return protocol_state.retry


__all__ = [
    "get_execution_state",
    "get_protocol_state",
    "get_working_state",
    "get_execution_identity",
    "get_execution_cursor",
    "get_active_plan",
    "get_active_step",
    "get_retry_metadata",
]
