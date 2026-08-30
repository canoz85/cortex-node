from core.graph_state_machine import apply_controller_decision_to_state
from core.protocol.enums import ControllerDecisionType, ExecutionPhase, StepStatus, WorkerRole
from core.protocol.models import (
    ControllerDecision,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    ProtocolVisibleState,
    ToolRequest,
)


def _build_state(
    *,
    cursor: ExecutionCursor,
    active_plan: ExecutionPlan | None = None,
    active_step: ExecutionStep | None = None,
    pending_tool_request: ToolRequest | None = None,
    completed_step_ids: tuple[str, ...] = tuple(),
) -> ExecutionState:
    return ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=ExecutionIdentity(execution_id="run-1", protocol_version="1.0"),
            cursor=cursor,
            active_plan=active_plan,
            active_step=active_step,
            pending_tool_request=pending_tool_request,
            completed_step_ids=completed_step_ids,
        )
    )


def test_apply_syncs_cursor_when_plan_is_accepted_and_dispatched_to_brain():
    step_1 = ExecutionStep(
        step_id="step-1",
        title="Step 1",
        description="first",
        status=StepStatus.PENDING,
        attempt=1,
    )
    plan = ExecutionPlan(
        plan_id="plan-1",
        revision=2,
        objective="Do work",
        steps=(step_1,),
    )
    state = _build_state(
        cursor=ExecutionCursor(
            phase=ExecutionPhase.INITIALIZING,
            step_id=None,
            current_worker=None,
        )
    )

    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_BRAIN,
        reason="Plan accepted.",
        next_worker=WorkerRole.BRAIN,
        accepted_plan=plan,
        next_step_id="step-1",
    )

    updated = apply_controller_decision_to_state(state, decision)
    cursor = updated.protocol_visible.cursor

    assert cursor.phase == ExecutionPhase.EXECUTING
    assert cursor.current_worker == WorkerRole.BRAIN
    assert cursor.step_id == "step-1"
    assert cursor.plan_revision == 2
    assert cursor.step_attempt == 1
    assert updated.protocol_visible.active_step is not None
    assert updated.protocol_visible.active_step.step_id == "step-1"


def test_apply_syncs_cursor_when_dispatching_tool_runtime():
    step_1 = ExecutionStep(
        step_id="step-1",
        title="Step 1",
        description="first",
        status=StepStatus.ACTIVE,
        attempt=2,
    )
    plan = ExecutionPlan(
        plan_id="plan-1",
        revision=3,
        objective="Do work",
        steps=(step_1,),
    )
    state = _build_state(
        cursor=ExecutionCursor(
            phase=ExecutionPhase.EXECUTING,
            step_id="step-1",
            current_worker=WorkerRole.BRAIN,
            plan_revision=3,
            step_attempt=2,
        ),
        active_plan=plan,
        active_step=step_1,
    )

    request = ToolRequest(
        request_id="req-1",
        tool_name="read_file",
        arguments={"path": "README.md"},
        requested_by=WorkerRole.BRAIN,
    )

    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
        reason="tool_request",
        next_worker=WorkerRole.TOOL_RUNTIME,
        pending_tool_request=request,
    )

    updated = apply_controller_decision_to_state(state, decision)
    cursor = updated.protocol_visible.cursor

    assert cursor.phase == ExecutionPhase.EXECUTING
    assert cursor.current_worker == WorkerRole.TOOL_RUNTIME
    assert cursor.step_id == "step-1"
    assert cursor.plan_revision == 3
    assert cursor.step_attempt == 2
    assert updated.protocol_visible.pending_tool_request == request


def test_apply_marks_completed_and_activates_next_step():
    step_1 = ExecutionStep(
        step_id="step-1",
        title="Step 1",
        description="first",
        status=StepStatus.ACTIVE,
        attempt=1,
    )
    step_2 = ExecutionStep(
        step_id="step-2",
        title="Step 2",
        description="second",
        status=StepStatus.PENDING,
        attempt=0,
    )
    plan = ExecutionPlan(
        plan_id="plan-1",
        revision=4,
        objective="Do work",
        steps=(step_1, step_2),
    )
    state = _build_state(
        cursor=ExecutionCursor(
            phase=ExecutionPhase.EXECUTING,
            step_id="step-1",
            current_worker=WorkerRole.BRAIN,
            plan_revision=4,
            step_attempt=1,
        ),
        active_plan=plan,
        active_step=step_1,
    )

    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_BRAIN,
        reason="Advance to next step.",
        next_worker=WorkerRole.BRAIN,
        completed_step_id="step-1",
        next_step_id="step-2",
    )

    updated = apply_controller_decision_to_state(state, decision)
    cursor = updated.protocol_visible.cursor

    assert updated.protocol_visible.completed_step_ids == ("step-1",)
    assert updated.protocol_visible.active_step is not None
    assert updated.protocol_visible.active_step.step_id == "step-2"
    assert cursor.step_id == "step-2"
    assert cursor.step_attempt == 0


def test_apply_clears_active_step_and_tool_request_on_termination():
    step_1 = ExecutionStep(
        step_id="step-1",
        title="Step 1",
        description="first",
        status=StepStatus.ACTIVE,
        attempt=2,
    )
    plan = ExecutionPlan(
        plan_id="plan-1",
        revision=5,
        objective="Do work",
        steps=(step_1,),
    )
    request = ToolRequest(
        request_id="req-2",
        tool_name="run_python",
        arguments={"code": "print('x')"},
        requested_by=WorkerRole.BRAIN,
    )
    state = _build_state(
        cursor=ExecutionCursor(
            phase=ExecutionPhase.EXECUTING,
            step_id="step-1",
            current_worker=WorkerRole.TOOL_RUNTIME,
            plan_revision=5,
            step_attempt=2,
        ),
        active_plan=plan,
        active_step=step_1,
        pending_tool_request=request,
        completed_step_ids=("step-0",),
    )

    decision = ControllerDecision(
        decision_type=ControllerDecisionType.TERMINATE,
        reason="stop",
        terminal=True,
    )

    updated = apply_controller_decision_to_state(state, decision)
    cursor = updated.protocol_visible.cursor

    assert cursor.phase == ExecutionPhase.TERMINATING
    assert cursor.step_id is None
    assert updated.protocol_visible.active_step is None
    assert updated.protocol_visible.pending_tool_request is None
    assert updated.protocol_visible.completed_step_ids == ("step-0",)


def test_apply_preserves_active_step_while_awaiting_async_job():
    step_1 = ExecutionStep(
        step_id="step-1",
        title="Generate image",
        status=StepStatus.ACTIVE,
        attempt=1,
    )
    plan = ExecutionPlan(
        plan_id="plan-1",
        revision=1,
        objective="Generate image",
        steps=(step_1,),
    )
    state = _build_state(
        cursor=ExecutionCursor(
            phase=ExecutionPhase.EXECUTING,
            step_id="step-1",
            current_worker=WorkerRole.TOOL_RUNTIME,
        ),
        active_plan=plan,
        active_step=step_1,
        pending_tool_request=ToolRequest(
            request_id="req-1",
            tool_name="run_comfy_workflow",
        ),
    )

    updated = apply_controller_decision_to_state(
        state,
        ControllerDecision(
            decision_type=ControllerDecisionType.AWAIT_ASYNC_JOB,
            reason="Awaiting async job prompt-1.",
            next_worker=WorkerRole.CONTROLLER,
            cursor=ExecutionCursor(
                phase=ExecutionPhase.WAITING,
                step_id="step-1",
                current_worker=WorkerRole.CONTROLLER,
            ),
            async_job_id="prompt-1",
            requires_checkpoint=True,
            clear_pending_tool_request=True,
        ),
    )

    assert updated.protocol_visible.cursor.phase == ExecutionPhase.WAITING
    assert updated.protocol_visible.cursor.current_worker == WorkerRole.CONTROLLER
    assert updated.protocol_visible.active_step == step_1
    assert updated.protocol_visible.pending_tool_request is None
