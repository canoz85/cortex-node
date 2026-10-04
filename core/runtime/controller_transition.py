"""Framework-neutral execution transitions authorized by the Controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.protocol.controller import apply_controller_decision_to_state as _apply_controller_decision_to_state
from core.protocol.models import ControllerDecision, ControllerInput, ExecutionState


class ControllerPort(Protocol):
    """Minimal Controller capability required for one execution transition."""

    def decide(self, controller_input: ControllerInput) -> ControllerDecision: ...


@dataclass(frozen=True, slots=True)
class ControllerTransition:
    """The decision and immutable state produced by one Controller turn."""

    decision: ControllerDecision
    execution_state: ExecutionState


class ControllerCoordinator:
    """Invoke Controller once and apply its decision once."""

    def __init__(self, controller: ControllerPort):
        self._controller = controller

    def transition(
        self,
        execution_state: ExecutionState,
        controller_input: ControllerInput,
    ) -> ControllerTransition:
        if not isinstance(execution_state, ExecutionState):
            raise TypeError("execution_state must be an ExecutionState")
        if not isinstance(controller_input, ControllerInput):
            raise TypeError("controller_input must be a ControllerInput")

        protocol = execution_state.protocol_visible
        if controller_input.identity != protocol.identity:
            raise ValueError("ControllerInput identity does not match ExecutionState")
        if controller_input.cursor != protocol.cursor:
            raise ValueError("ControllerInput cursor does not match ExecutionState")

        decision = self._controller.decide(controller_input)
        updated_state = _apply_controller_decision_to_state(execution_state, decision)
        return ControllerTransition(
            decision=decision,
            execution_state=updated_state,
        )

__all__ = [
    "ControllerCoordinator",
    "ControllerPort",
    "ControllerTransition",
]
