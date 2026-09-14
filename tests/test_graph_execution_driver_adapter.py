from core.graph_controller import create_controller_node
from core.protocol.enums import ExecutionPhase
from core.protocol.models import (
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionState,
    ProtocolVisibleState,
)
from core.runtime.execution_driver import ExecutionDriver


def test_live_graph_controller_invokes_execution_driver(monkeypatch):
    calls = []
    original = ExecutionDriver.transition

    def observe(self, execution_state, controller_input):
        calls.append((execution_state, controller_input))
        return original(self, execution_state, controller_input)

    monkeypatch.setattr(ExecutionDriver, "transition", observe)
    state = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=ExecutionIdentity(execution_id="live-driver", protocol_version="1"),
        cursor=ExecutionCursor(phase=ExecutionPhase.INITIALIZING),
    ))

    update = create_controller_node()({
        "execution_state": state,
        "user_request": "Inspect the workspace",
    })

    assert len(calls) == 1
    assert calls[0][0] is state
    assert update["controller_decision"].planning_request is not None
