import pytest

from core.protocol.bridge import build_controller_input
from core.protocol.controller import CortexController
from core.graph_state_machine import apply_controller_decision_to_state
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
    ControllerDecision,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    PlannerResult,
    ProtocolVisibleState,
    ReplanRequest,
    RetryMetadata,
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
        cursor=ExecutionCursor(
            phase=ExecutionPhase.EXECUTING,
            step_id="step-1",
            current_worker=WorkerRole.BRAIN,
        ),
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
        execution_status=ExecutionStatus.FAILED,
        cursor=ExecutionCursor(
            phase=ExecutionPhase.FAILED,
            current_worker=WorkerRole.CONTROLLER,
        ),
        clear_active_step=True,
        clear_pending_tool_request=True,
        terminal=True,
    )

    updated = apply_controller_decision_to_state(state, decision)
    cursor = updated.protocol_visible.cursor

    assert cursor.phase == ExecutionPhase.FAILED
    assert updated.protocol_visible.status == ExecutionStatus.FAILED
    assert cursor.step_id is None
    assert updated.protocol_visible.active_step is None
    assert updated.protocol_visible.pending_tool_request is None
    assert updated.protocol_visible.completed_step_ids == ("step-0",)


def test_apply_copies_controller_owned_retry_metadata_without_inference():
    step = ExecutionStep(
        step_id="step-1",
        title="Step 1",
        status=StepStatus.ACTIVE,
        attempt=2,
    )
    plan = ExecutionPlan(plan_id="plan-1", steps=(step,))
    state = _build_state(
        cursor=ExecutionCursor(
            phase=ExecutionPhase.EXECUTING,
            step_id="step-1",
            step_attempt=1,
        ),
        active_plan=plan,
        active_step=step.model_copy(update={"attempt": 1}),
    )
    retry = RetryMetadata(
        step_id="step-1",
        retry_count=1,
        max_retries=2,
        last_error_message="temporary failure",
    )
    decision = ControllerDecision(
        accepted_plan=plan,
        decision_type=ControllerDecisionType.DISPATCH_BRAIN,
        reason="retry_step",
        next_worker=WorkerRole.BRAIN,
        cursor=ExecutionCursor(
            phase=ExecutionPhase.EXECUTING,
            step_id="step-1",
            step_attempt=2,
            current_worker=WorkerRole.BRAIN,
        ),
        failed_step_id="step-1",
        failure_reason="temporary failure",
        next_step_id="step-1",
        retry=retry,
    )

    updated = apply_controller_decision_to_state(state, decision)

    assert updated.protocol_visible.status == ExecutionStatus.NON_TERMINAL
    assert updated.protocol_visible.retry == retry
    assert updated.protocol_visible.cursor.step_attempt == 2
    assert updated.protocol_visible.active_step == step


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


def _retryable_state():
    step = ExecutionStep(
        step_id="step-1", title="First", status=StepStatus.ACTIVE, attempt=1
    )
    state = _build_state(
        cursor=ExecutionCursor(
            phase=ExecutionPhase.EXECUTING,
            step_id=step.step_id,
            step_attempt=step.attempt,
            current_worker=WorkerRole.BRAIN,
        ),
        active_plan=ExecutionPlan(
            plan_id="plan-1",
            steps=(step, ExecutionStep(step_id="step-2", title="Second")),
        ),
        active_step=step,
    )
    return state.model_copy(
        update={
            "protocol_visible": state.protocol_visible.model_copy(
                update={"retry": RetryMetadata(max_retries=1)}
            )
        }
    )


def _apply_worker_result(state, **worker_result):
    decision = CortexController(max_reasoning_steps=10).decide(
        build_controller_input({"execution_state": state, **worker_result})
    )
    return apply_controller_decision_to_state(state, decision)


@pytest.mark.parametrize("retry_next_step", [False, True])
def test_retried_step_can_complete_and_next_step_can_proceed(retry_next_step):
    initial = _retryable_state()
    retried = _apply_worker_result(
        initial,
        brain_result=BrainResult(outcome=BrainOutcome.STEP_FAILED, message="temporary"),
    )
    assert retried.protocol_visible.retry.step_id == "step-1"
    assert retried.protocol_visible.retry.retry_count == 1
    assert retried.protocol_visible.active_step.attempt == 2

    continued = _apply_worker_result(
        retried, brain_result=BrainResult(outcome=BrainOutcome.CONTINUE)
    )
    assert continued.protocol_visible.retry == retried.protocol_visible.retry

    advanced = _apply_worker_result(
        continued, brain_result=BrainResult(outcome=BrainOutcome.STEP_COMPLETED)
    )
    assert advanced.protocol_visible.active_step.step_id == "step-2"
    assert advanced.protocol_visible.cursor.step_attempt == 0
    assert advanced.protocol_visible.retry == RetryMetadata(max_retries=1)

    if retry_next_step:
        advanced = _apply_worker_result(
            advanced,
            brain_result=BrainResult(outcome=BrainOutcome.STEP_FAILED, message="retry second"),
        )
        assert advanced.protocol_visible.status == ExecutionStatus.NON_TERMINAL
        assert advanced.protocol_visible.retry.step_id == "step-2"
        assert advanced.protocol_visible.retry.retry_count == 1

    finished = _apply_worker_result(
        advanced, brain_result=BrainResult(outcome=BrainOutcome.STEP_COMPLETED)
    )
    assert finished.protocol_visible.completed_step_ids == ("step-1", "step-2")
    assert finished.protocol_visible.active_step is None
    assert finished.protocol_visible.retry == RetryMetadata(max_retries=1)

    terminal = _apply_worker_result(
        finished,
        brain_result=BrainResult(outcome=BrainOutcome.FINAL_ANSWER, final_answer="Done"),
    )
    assert terminal.protocol_visible.status == ExecutionStatus.COMPLETED
    assert terminal.protocol_visible.cursor.phase == ExecutionPhase.COMPLETED
    assert initial.protocol_visible.retry.retry_count == 0
    assert retried.protocol_visible.retry.retry_count == 1


@pytest.mark.parametrize("replacement_step_id", ["step-1", "replacement-1"])
def test_failed_step_replan_and_replacement_plan_can_proceed(replacement_step_id):
    retried = _apply_worker_result(
        _retryable_state(),
        brain_result=BrainResult(outcome=BrainOutcome.STEP_FAILED, message="temporary"),
    )
    replanning = _apply_worker_result(
        retried,
        brain_result=BrainResult(
            outcome=BrainOutcome.REPLAN_REQUEST,
            replan_request=ReplanRequest(reason="replace plan", failed_step_id="step-1"),
        ),
    )
    assert replanning.protocol_visible.active_step is None
    assert replanning.protocol_visible.active_plan.steps[0].status == StepStatus.FAILED
    assert replanning.protocol_visible.retry == RetryMetadata(max_retries=1)

    accepted = _apply_worker_result(
        replanning,
        planner_result=PlannerResult(
            outcome=PlannerOutcome.EXECUTION_PLAN,
            proposed_plan=ExecutionPlan(
                plan_id="plan-2",
                revision=2,
                steps=(ExecutionStep(step_id=replacement_step_id, title="Replacement"),),
            ),
        ),
    )
    assert accepted.protocol_visible.active_step.step_id == replacement_step_id
    assert accepted.protocol_visible.retry == RetryMetadata(max_retries=1)

    new_retry = _apply_worker_result(
        accepted,
        brain_result=BrainResult(outcome=BrainOutcome.STEP_FAILED, message="new failure"),
    )
    assert new_retry.protocol_visible.status == ExecutionStatus.NON_TERMINAL
    assert new_retry.protocol_visible.retry.step_id == replacement_step_id
    assert new_retry.protocol_visible.retry.retry_count == 1
    assert new_retry.protocol_visible.active_step.attempt == 1

    finished = _apply_worker_result(
        new_retry, brain_result=BrainResult(outcome=BrainOutcome.STEP_COMPLETED)
    )
    assert finished.protocol_visible.active_step is None
    assert finished.protocol_visible.active_plan.steps[0].status == StepStatus.COMPLETED
    assert finished.protocol_visible.retry == RetryMetadata(max_retries=1)
