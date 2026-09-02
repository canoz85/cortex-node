import pytest
from langgraph.graph import END

from core.graph_routing import route_after_controller
from core.graph_state_machine import map_controller_decision
from core.protocol.enums import (
    BrainOutcome,
    ControllerDecisionType,
    EventType,
    ExecutionPhase,
    ExecutionStatus,
    PlannerOutcome,
    StepStatus,
    WorkerRole,
)
from core.protocol.models import ControllerDecision, ExecutionCursor


def _decision(decision_type: ControllerDecisionType) -> ControllerDecision:
    if decision_type == ControllerDecisionType.DISPATCH_SUMMARY:
        return ControllerDecision(
            decision_type=decision_type,
            execution_status=ExecutionStatus.COMPLETED,
            cursor=ExecutionCursor(
                phase=ExecutionPhase.COMPLETED,
                current_worker=WorkerRole.SUMMARY,
            ),
            terminal=True,
        )
    if decision_type == ControllerDecisionType.CANCEL:
        return ControllerDecision(
            decision_type=decision_type,
            execution_status=ExecutionStatus.CANCELLED,
            cursor=ExecutionCursor(
                phase=ExecutionPhase.CANCELLED,
                current_worker=WorkerRole.CONTROLLER,
            ),
            terminal=True,
        )
    if decision_type == ControllerDecisionType.TERMINATE:
        return ControllerDecision(
            decision_type=decision_type,
            execution_status=ExecutionStatus.FAILED,
            cursor=ExecutionCursor(
                phase=ExecutionPhase.FAILED,
                current_worker=WorkerRole.CONTROLLER,
            ),
            terminal=True,
        )
    return ControllerDecision(decision_type=decision_type)


@pytest.mark.parametrize(
    ("decision_type", "expected_route"),
    [
        (ControllerDecisionType.DISPATCH_PLANNER, "planner"),
        (ControllerDecisionType.DISPATCH_BRAIN, "brain"),
        (ControllerDecisionType.DISPATCH_TOOL_RUNTIME, "tools"),
        (ControllerDecisionType.DISPATCH_SUMMARY, END),
        (ControllerDecisionType.AWAIT_ASYNC_JOB, END),
        (ControllerDecisionType.PAUSE, END),
        (ControllerDecisionType.CANCEL, END),
        (ControllerDecisionType.TERMINATE, END),
    ],
)
def test_existing_topology_maps_controller_decisions_only(
    decision_type,
    expected_route,
):
    assert map_controller_decision(_decision(decision_type)) == expected_route


def test_route_after_controller_reads_stored_decision():
    decision = _decision(ControllerDecisionType.DISPATCH_BRAIN)

    assert route_after_controller({"controller_decision": decision}) == "brain"


def test_route_after_controller_fails_closed_without_decision():
    assert route_after_controller({}) == END


def test_protocol_enums_contain_no_graph_only_node_names():
    graph_only_names = {
        "tools",
        "capture_tool_output",
        "summarize_memory",
        "__end__",
    }
    protocol_values = {
        member.value
        for enum_type in (
            BrainOutcome,
            ControllerDecisionType,
            EventType,
            ExecutionPhase,
            ExecutionStatus,
            PlannerOutcome,
            StepStatus,
            WorkerRole,
        )
        for member in enum_type
    }

    assert protocol_values.isdisjoint(graph_only_names)
