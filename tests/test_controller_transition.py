import ast
from pathlib import Path

import pytest

from core.protocol.enums import ControllerDecisionType, ExecutionPhase, WorkerRole
from core.protocol.models import (
    ControllerDecision,
    ControllerInput,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionState,
    ProtocolVisibleState,
)
from core.runtime.controller_transition import (
    ControllerCoordinator,
    apply_controller_decision_to_state,
)


def _state(*, iteration: int = 4) -> ExecutionState:
    return ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=ExecutionIdentity(
                execution_id="portable-transition",
                protocol_version="1.0",
            ),
            cursor=ExecutionCursor(
                phase=ExecutionPhase.EXECUTING,
                current_worker=WorkerRole.CONTROLLER,
                controller_iteration=iteration,
            ),
        )
    )


def _input(state: ExecutionState) -> ControllerInput:
    protocol = state.protocol_visible
    return ControllerInput(
        identity=protocol.identity,
        cursor=protocol.cursor,
        context=ExecutionContext(user_request="Continue the execution"),
    )


class _RecordingController:
    def __init__(self, decision: ControllerDecision):
        self.decision = decision
        self.inputs: list[ControllerInput] = []

    def decide(self, controller_input: ControllerInput) -> ControllerDecision:
        self.inputs.append(controller_input)
        return self.decision


def test_portable_transition_module_has_no_graph_framework_imports():
    module_path = Path(__file__).parents[1] / "core" / "runtime" / "controller_transition.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    assert not any(name.startswith(("langgraph", "langchain")) for name in imports)
    assert "core.state" not in imports


def test_coordinator_returns_one_controller_decision_and_immutable_updated_state():
    state = _state(iteration=6)
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_BRAIN,
        reason="Continue reasoning.",
        next_worker=WorkerRole.BRAIN,
        cursor=state.protocol_visible.cursor.model_copy(
            update={
                "current_worker": WorkerRole.BRAIN,
                "controller_iteration": 7,
            }
        ),
    )
    controller = _RecordingController(decision)

    transition = ControllerCoordinator(controller).transition(state, _input(state))

    assert controller.inputs == [_input(state)]
    assert transition.decision is decision
    assert transition.execution_state is not state
    assert state.protocol_visible.cursor.controller_iteration == 6
    assert state.protocol_visible.cursor.current_worker == WorkerRole.CONTROLLER
    assert transition.execution_state.protocol_visible.cursor.controller_iteration == 7
    assert transition.execution_state.protocol_visible.cursor.current_worker == WorkerRole.BRAIN


def test_reapplying_same_immutable_decision_does_not_advance_cursor_again():
    state = _state(iteration=2)
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_BRAIN,
        reason="Continue reasoning.",
        next_worker=WorkerRole.BRAIN,
        cursor=state.protocol_visible.cursor.model_copy(
            update={
                "current_worker": WorkerRole.BRAIN,
                "controller_iteration": 3,
            }
        ),
    )

    once = apply_controller_decision_to_state(state, decision)
    twice = apply_controller_decision_to_state(once, decision)

    assert once == twice
    assert twice.protocol_visible.cursor.controller_iteration == 3


def test_coordinator_accepts_no_graph_state_or_unrelated_graph_fields():
    state = _state()
    controller = _RecordingController(
        ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_BRAIN,
            reason="Continue reasoning.",
            next_worker=WorkerRole.BRAIN,
            cursor=state.protocol_visible.cursor,
        )
    )

    with pytest.raises(TypeError, match="ExecutionState"):
        ControllerCoordinator(controller).transition(  # type: ignore[arg-type]
            {"execution_state": state, "messages": [], "steps": 99},
            _input(state),
        )


def test_coordinator_rejects_input_for_a_different_checkpoint_cursor():
    state = _state(iteration=4)
    stale_input = _input(state).model_copy(
        update={
            "cursor": state.protocol_visible.cursor.model_copy(
                update={"controller_iteration": 3}
            )
        }
    )
    controller = _RecordingController(
        ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_BRAIN,
            reason="Would be stale.",
            next_worker=WorkerRole.BRAIN,
            cursor=state.protocol_visible.cursor,
        )
    )

    with pytest.raises(ValueError, match="cursor"):
        ControllerCoordinator(controller).transition(state, stale_input)
    assert controller.inputs == []
