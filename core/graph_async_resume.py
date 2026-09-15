"""LangGraph persistence/presentation adapter for portable async continuation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.protocol.enums import ControllerDecisionType
from core.protocol.models import ControllerDecision


class LangGraphAsyncResumeAdapter:
    """Persist portable async turns without using graph nodes as resume tokens."""

    def __init__(self, compiled_graph: Any, controller_adapter: Any) -> None:
        self._compiled_graph = compiled_graph
        self._controller_adapter = controller_adapter

    def load(self, config: dict[str, Any]) -> Mapping[str, Any]:
        snapshot = self._compiled_graph.get_state(config)
        state = getattr(snapshot, "values", None)
        if not isinstance(state, Mapping):
            raise RuntimeError("Checkpoint does not contain graph state.")
        return state

    def resume(
        self,
        *,
        config: dict[str, Any],
        state: Mapping[str, Any],
        update: Mapping[str, Any],
    ):
        current = dict(state)
        persisted: dict[str, Any] = {}
        emitted: list[dict[str, dict[str, Any]]] = []

        def apply(values: Mapping[str, Any]) -> None:
            for key, value in values.items():
                if key == "messages" and isinstance(value, list):
                    current[key] = [*current.get(key, []), *value]
                    persisted[key] = [*persisted.get(key, []), *value]
                else:
                    current[key] = value
                    persisted[key] = value

        apply(update)

        for _ in range(100):
            turn_update = self._controller_adapter(current)
            if not isinstance(turn_update, dict):
                raise TypeError("Controller adapter returned no state update.")
            apply(turn_update)
            emitted.append({"async_runtime": turn_update})

            decision = turn_update.get("controller_decision")
            if not isinstance(decision, ControllerDecision):
                raise RuntimeError("Controller continuation returned no decision.")
            if (
                decision.decision_type == ControllerDecisionType.AWAIT_ASYNC_JOB
                or decision.terminal
            ):
                break
        else:
            raise RuntimeError("Async portable continuation exceeded turn limit.")

        self._compiled_graph.update_state(config, persisted)
        return iter(emitted)


__all__ = ["LangGraphAsyncResumeAdapter"]
