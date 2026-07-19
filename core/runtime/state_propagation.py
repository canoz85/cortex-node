"""Transport-only propagation helpers for graph state updates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def propagate_execution_state(previous_state: Mapping[str, Any] | None, node_update: Any) -> Any:
    """Preserve an attached execution_state on partial node updates.

    If a node explicitly returns execution_state, that value is left untouched.
    Otherwise, the existing object reference from the previous state is copied
    onto the returned update without reconstruction or mutation.
    """
    if not isinstance(node_update, Mapping):
        return node_update

    if "execution_state" in node_update:
        return node_update

    if not isinstance(previous_state, Mapping) or "execution_state" not in previous_state:
        return node_update

    return {**node_update, "execution_state": previous_state.get("execution_state")}


__all__ = ["propagate_execution_state"]