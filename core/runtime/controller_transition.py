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
        if controller_input.tool_request_continuation != protocol.tool_request_continuation:
            raise ValueError("ControllerInput tool continuation does not match ExecutionState")
        proposal = controller_input.brain_result
        if protocol.tool_request_continuation is not None or (
            proposal is not None and proposal.tool_requests is not None
        ):
            if (controller_input.active_plan != protocol.active_plan
                    or controller_input.active_step != protocol.active_step
                    or controller_input.pending_tool_request != protocol.pending_tool_request):
                raise ValueError("ControllerInput batch scope does not match ExecutionState")

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
