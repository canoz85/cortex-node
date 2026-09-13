from core.graph_controller import create_controller_node
from core.graph_state_machine import apply_controller_decision_to_state
from core.protocol.bridge import build_controller_input
from core.protocol.controller import CortexController
from core.protocol.enums import (
    BrainOutcome,
    ControllerDecisionType,
    ExecutionPhase,
    ExecutionStatus,
    PlannerOutcome,
    StepStatus,
    WorkerRole,
)
from core.protocol.models import (
    BrainResult,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    PlannerResult,
    ProtocolVisibleState,
    ToolRequest,
    ToolResult,
)


IDENTITY = ExecutionIdentity(execution_id="iteration-test", protocol_version="1")
CONTEXT = ExecutionContext(user_request="Inspect a file")


def _initial_state() -> ExecutionState:
    return ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=IDENTITY,
        cursor=ExecutionCursor(phase=ExecutionPhase.INITIALIZING),
    ))


def _apply(state: ExecutionState, decision) -> ExecutionState:
    return apply_controller_decision_to_state(state, decision)


def _planned_state(controller: CortexController) -> ExecutionState:
    initial = _initial_state()
    planning = controller.decide(build_controller_input({
        "execution_state": initial,
        "context": CONTEXT,
    }))
    planning_state = _apply(initial, planning)
    request = planning.planning_request
    plan = ExecutionPlan(
        plan_id="plan-1",
        steps=(
            ExecutionStep(step_id="step-1", title="Read the file"),
            ExecutionStep(step_id="step-2", title="Inspect the contents"),
        ),
    )
    accepted = controller.decide(build_controller_input({
        "execution_state": planning_state,
        "context": CONTEXT,
        "planner_result": PlannerResult(
            outcome=PlannerOutcome.EXECUTION_PLAN,
            request_id=request.request_id,
            proposed_plan=plan,
        ),
    }))
    assert accepted.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert accepted.cursor.controller_iteration == 1
    return _apply(planning_state, accepted)


def test_brain_dispatches_advance_one_protocol_iteration_and_tool_round_trip_cannot_bypass_limit():
    controller = CortexController(max_reasoning_steps=2)
    state = _planned_state(controller)
    assert state.protocol_visible.cursor.controller_iteration == 1

    request = ToolRequest(
        request_id="read-1",
        tool_name="read_file",
        arguments={"path": "README.md"},
    )
    tool_dispatch = controller.decide(build_controller_input({
        "execution_state": state,
        "context": CONTEXT,
        "brain_result": BrainResult(
            outcome=BrainOutcome.TOOL_REQUEST,
            step_id="step-1",
            tool_request=request,
        ),
    }))
    assert tool_dispatch.decision_type == ControllerDecisionType.DISPATCH_TOOL_RUNTIME
    state = _apply(state, tool_dispatch)
    assert state.protocol_visible.cursor.controller_iteration == 1

    brain_dispatch = controller.decide(build_controller_input({
        "execution_state": state,
        "context": CONTEXT,
        "tool_result": ToolResult(
            request_id=request.request_id,
            success=True,
            message="read",
        ),
    }))
    assert brain_dispatch.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert brain_dispatch.cursor.controller_iteration == 2
    assert brain_dispatch.next_step_id == "step-2"
    state = _apply(state, brain_dispatch)

    blocked = controller.decide(build_controller_input({
        "execution_state": state,
        "context": CONTEXT,
        "brain_result": BrainResult(outcome=BrainOutcome.CONTINUE),
    }))
    assert blocked.decision_type == ControllerDecisionType.TERMINATE
    assert blocked.reason == "max_steps"
    assert blocked.execution_status == ExecutionStatus.FAILED


def test_checkpoint_round_trip_preserves_iteration_and_controller_decision_is_idempotent():
    controller = CortexController(max_reasoning_steps=3)
    state = _planned_state(controller)
    decision = controller.decide(build_controller_input({
        "execution_state": state,
        "context": CONTEXT,
        "brain_result": BrainResult(outcome=BrainOutcome.CONTINUE),
    }))
    assert decision.cursor.controller_iteration == 2

    once = _apply(state, decision)
    twice = _apply(once, decision)
    assert twice.protocol_visible.cursor.controller_iteration == 2

    restored = ExecutionState.model_validate_json(twice.model_dump_json())
    assert restored.protocol_visible.cursor.controller_iteration == 2
    resumed = controller.decide(build_controller_input({
        "execution_state": restored,
        "context": CONTEXT,
        "brain_result": BrainResult(outcome=BrainOutcome.CONTINUE),
    }))
    assert resumed.cursor.controller_iteration == 3


def test_attached_execution_state_enforces_limit_without_legacy_steps():
    step = ExecutionStep(
        step_id="step-1",
        title="Read the file",
        status=StepStatus.ACTIVE,
        attempt=1,
    )
    state = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=IDENTITY,
        cursor=ExecutionCursor(
            phase=ExecutionPhase.EXECUTING,
            step_id=step.step_id,
            step_attempt=1,
            current_worker=WorkerRole.BRAIN,
            controller_iteration=2,
        ),
        active_plan=ExecutionPlan(plan_id="plan-1", steps=(step,)),
        active_step=step,
    ))

    update = create_controller_node(
        controller=CortexController(max_reasoning_steps=2),
    )({
        "execution_state": state,
        "context": CONTEXT,
        "brain_result": BrainResult(outcome=BrainOutcome.CONTINUE),
    })

    assert update["controller_decision"].decision_type == ControllerDecisionType.TERMINATE
    assert update["controller_decision"].reason == "max_steps"
    assert update["execution_state"].protocol_visible.cursor.controller_iteration == 2
