import pytest

from core.protocol.controller import CortexController
from core.protocol.bridge import build_brain_input
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
    ControllerInput,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    PlannerResult,
    ProtocolVisibleState,
    ReplanRequest,
    RetryMetadata,
)


IDENTITY = ExecutionIdentity(execution_id="stage-1", protocol_version="1.0")
CONTEXT = ExecutionContext(user_request="complete the task", role=WorkerRole.CONTROLLER)


def _controller() -> CortexController:
    return CortexController(max_reasoning_steps=10)


def _active_execution(
    *,
    step_id: str = "step-1",
    cursor_step_id: str | None = "step-1",
    plan_step_id: str = "step-1",
    attempt: int = 1,
    retry_count: int = 0,
    max_retries: int = 0,
) -> dict:
    active_step = ExecutionStep(
        step_id=step_id,
        title="Do work",
        status=StepStatus.ACTIVE,
        attempt=attempt,
    )
    plan_step = active_step.model_copy(update={"step_id": plan_step_id})
    return {
        "cursor": ExecutionCursor(
            phase=ExecutionPhase.EXECUTING,
            step_id=cursor_step_id,
            step_attempt=attempt,
            current_worker=WorkerRole.BRAIN,
        ),
        "active_plan": ExecutionPlan(
            plan_id="plan-1",
            objective="Complete the task",
            steps=(plan_step,),
        ),
        "active_step": active_step,
        "retry": RetryMetadata(
            step_id=step_id,
            retry_count=retry_count,
            max_retries=max_retries,
        ),
    }


def _input(**updates) -> ControllerInput:
    payload = {
        "identity": IDENTITY,
        "cursor": ExecutionCursor(phase=ExecutionPhase.INITIALIZING),
        "context": CONTEXT,
    }
    payload.update(updates)
    return ControllerInput(**payload)


def test_direct_response_context_is_explicit_and_final_answer_completes():
    direct = _controller().decide(
        _input(
            cursor=ExecutionCursor(phase=ExecutionPhase.PLANNING),
            planner_result=PlannerResult(outcome=PlannerOutcome.DIRECT_RESPONSE),
        )
    )

    assert direct.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert direct.direct_response is True
    assert direct.execution_status == ExecutionStatus.NON_TERMINAL

    completed = _controller().decide(
        _input(
            cursor=direct.cursor,
            brain_result=BrainResult(
                outcome=BrainOutcome.FINAL_ANSWER,
                final_answer="Done.",
            ),
        )
    )

    assert completed.decision_type == ControllerDecisionType.DISPATCH_SUMMARY
    assert completed.execution_status == ExecutionStatus.COMPLETED
    assert completed.cursor is not None
    assert completed.cursor.phase == ExecutionPhase.COMPLETED
    assert completed.terminal is True

def test_final_answer_after_completed_plan_does_not_require_active_step():
    execution = _active_execution()

    completed = _controller().decide(
        _input(
            **{
                **execution,
                "active_step": None,
                "cursor": execution["cursor"].model_copy(
                    update={
                        "step_id": None,
                        "step_attempt": None,
                    }
                ),
                "brain_result": BrainResult(
                    outcome=BrainOutcome.FINAL_ANSWER,
                    final_answer="Done.",
                ),
            }
        )
    )

    assert completed.decision_type == ControllerDecisionType.DISPATCH_SUMMARY
    assert completed.execution_status == ExecutionStatus.COMPLETED
    assert completed.completed_step_id is None
    assert completed.cursor is not None
    assert completed.cursor.phase == ExecutionPhase.COMPLETED
    assert completed.terminal is True

def test_direct_response_marker_crosses_bridge_without_graph_state():
    direct = _controller().decide(
        _input(
            cursor=ExecutionCursor(phase=ExecutionPhase.PLANNING),
            planner_result=PlannerResult(outcome=PlannerOutcome.DIRECT_RESPONSE),
        )
    )
    execution_state = ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=IDENTITY,
            cursor=direct.cursor,
        )
    )

    brain_input = build_brain_input(
        {
            "execution_state": execution_state,
            "controller_decision": direct,
            "messages": (),
        }
    )

    assert brain_input.direct_response is True


def test_cancelled_and_failed_termination_carry_matching_status_and_cursor():
    cancelled = _controller().decide(_input(cancel_requested=True))

    assert cancelled.execution_status == ExecutionStatus.CANCELLED
    assert cancelled.cursor is not None
    assert cancelled.cursor.phase == ExecutionPhase.CANCELLED
    assert cancelled.terminal is True

    failed = _controller().decide(
        _input(
            cursor=ExecutionCursor(
                phase=ExecutionPhase.EXECUTING,
                controller_iteration=10,
            )
        )
    )

    assert failed.execution_status == ExecutionStatus.FAILED
    assert failed.cursor is not None
    assert failed.cursor.phase == ExecutionPhase.FAILED
    assert failed.terminal is True


def test_planner_failure_terminates_with_failed_status():
    decision = _controller().decide(
        _input(
            cursor=ExecutionCursor(phase=ExecutionPhase.PLANNING),
            planner_result=PlannerResult(
                outcome=PlannerOutcome.FAILED,
                message="planner unavailable",
            ),
        )
    )

    assert decision.decision_type == ControllerDecisionType.TERMINATE
    assert decision.execution_status == ExecutionStatus.FAILED
    assert decision.failure_reason == "planner unavailable"
    assert decision.cursor is not None
    assert decision.cursor.phase == ExecutionPhase.FAILED


def test_step_failed_retries_same_step_when_budget_remains():
    execution = _active_execution(retry_count=0, max_retries=2, attempt=1)
    decision = _controller().decide(
        _input(
            **execution,
            brain_result=BrainResult(
                outcome=BrainOutcome.STEP_FAILED,
                message="temporary failure",
                proposed_step_status=StepStatus.FAILED,
            ),
        )
    )

    assert decision.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert decision.execution_status == ExecutionStatus.NON_TERMINAL
    assert decision.failed_step_id == "step-1"
    assert decision.failure_reason == "temporary failure"
    assert decision.retry is not None
    assert decision.retry.retry_count == 1
    assert decision.retry.max_retries == 2
    assert decision.accepted_plan is not None
    assert decision.accepted_plan.steps[0].status == StepStatus.ACTIVE
    assert decision.accepted_plan.steps[0].attempt == 2
    assert decision.cursor is not None
    assert decision.cursor.step_id == "step-1"
    assert decision.cursor.step_attempt == 2


def test_step_failed_marks_step_failed_and_terminates_when_retries_exhausted():
    execution = _active_execution(retry_count=1, max_retries=1, attempt=2)
    decision = _controller().decide(
        _input(
            **execution,
            brain_result=BrainResult(
                outcome=BrainOutcome.STEP_FAILED,
                message="permanent failure",
                proposed_step_status=StepStatus.FAILED,
            ),
        )
    )

    assert decision.decision_type == ControllerDecisionType.TERMINATE
    assert decision.execution_status == ExecutionStatus.FAILED
    assert decision.cursor is not None
    assert decision.cursor.phase == ExecutionPhase.FAILED
    assert decision.failed_step_id == "step-1"
    assert decision.failure_reason == "permanent failure"
    assert decision.accepted_plan is not None
    assert decision.accepted_plan.steps[0].status == StepStatus.FAILED
    assert decision.retry is not None
    assert decision.retry.retry_count == 1
    assert decision.clear_active_step is True
    assert decision.terminal is True


def test_explicit_replan_marks_current_step_failed_and_dispatches_planner():
    execution = _active_execution(retry_count=1, max_retries=2, attempt=2)
    decision = _controller().decide(
        _input(
            **execution,
            brain_result=BrainResult(
                outcome=BrainOutcome.REPLAN_REQUEST,
                message="current plan cannot continue",
                replan_request=ReplanRequest(
                    reason="dependency changed",
                    failed_step_id="step-1",
                ),
            ),
        )
    )

    assert decision.decision_type == ControllerDecisionType.DISPATCH_PLANNER
    assert decision.execution_status == ExecutionStatus.NON_TERMINAL
    assert decision.requires_replan is True
    assert decision.failed_step_id == "step-1"
    assert decision.accepted_plan is not None
    assert decision.accepted_plan.steps[0].status == StepStatus.FAILED
    assert decision.clear_active_step is True
    assert decision.cursor is not None
    assert decision.cursor.phase == ExecutionPhase.REPLANNING
    assert decision.retry == RetryMetadata(max_retries=2)


@pytest.mark.parametrize("has_next_step", [False, True])
def test_step_completion_clears_retry_history_but_preserves_budget(has_next_step):
    execution = _active_execution(retry_count=1, max_retries=2, attempt=2)
    execution["retry"] = execution["retry"].model_copy(
        update={"last_error_code": "temporary", "last_error_message": "retry me"}
    )
    if has_next_step:
        execution["active_plan"] = execution["active_plan"].model_copy(
            update={
                "steps": (
                    execution["active_step"],
                    ExecutionStep(step_id="step-2", title="Next step"),
                )
            }
        )

    decision = _controller().decide(
        _input(
            **execution,
            brain_result=BrainResult(outcome=BrainOutcome.STEP_COMPLETED),
        )
    )

    assert decision.retry == RetryMetadata(max_retries=2)
    assert decision.next_step_id == ("step-2" if has_next_step else None)
    assert decision.clear_active_step is (not has_next_step)
    assert decision.execution_status == ExecutionStatus.NON_TERMINAL
    assert execution["retry"].retry_count == 1
    assert execution["retry"].last_error_code == "temporary"


@pytest.mark.parametrize("replacement_step_id", ["step-1", "replacement-1"])
def test_accepting_replacement_plan_clears_old_retry_history(replacement_step_id):
    execution = _active_execution(retry_count=2, max_retries=2, attempt=3)
    execution["retry"] = execution["retry"].model_copy(
        update={"last_error_code": "old-plan", "last_error_message": "replan me"}
    )
    replacement = ExecutionPlan(
        plan_id="plan-2",
        revision=2,
        steps=(ExecutionStep(step_id=replacement_step_id, title="Replacement"),),
    )

    decision = _controller().decide(
        _input(
            **execution,
            planner_result=PlannerResult(
                outcome=PlannerOutcome.EXECUTION_PLAN,
                proposed_plan=replacement,
            ),
        )
    )

    assert decision.accepted_plan == replacement
    assert decision.retry == RetryMetadata(max_retries=2)
    assert decision.next_step_id == replacement_step_id
    assert decision.cursor.step_attempt == 0
    assert decision.execution_status == ExecutionStatus.NON_TERMINAL


@pytest.mark.parametrize(
    ("execution_updates", "error_match"),
    [
        ({"active_plan": None}, "active plan"),
        ({"active_step": None}, "active step"),
        ({"active_plan": None, "active_step": None, "cursor_step_id": None}, "active plan"),
        ({"cursor_step_id": None}, "cursor step"),
        ({"cursor_step_id": "stale-step"}, "cursor step"),
        ({"plan_step_id": "other-step"}, "active plan"),
        ({"retry_step_id": "stale-step"}, "retry metadata"),
    ],
)
@pytest.mark.parametrize("outcome", [BrainOutcome.STEP_FAILED, BrainOutcome.STEP_COMPLETED])
def test_step_results_reject_missing_or_stale_active_step_identifiers(
    execution_updates,
    error_match,
    outcome,
):
    active_kwargs = {}
    if "cursor_step_id" in execution_updates:
        active_kwargs["cursor_step_id"] = execution_updates["cursor_step_id"]
    if "plan_step_id" in execution_updates:
        active_kwargs["plan_step_id"] = execution_updates["plan_step_id"]

    execution = _active_execution(max_retries=1, **active_kwargs)
    if "active_plan" in execution_updates:
        execution["active_plan"] = execution_updates["active_plan"]
    if "active_step" in execution_updates:
        execution["active_step"] = execution_updates["active_step"]
    if "retry_step_id" in execution_updates:
        execution["retry"] = execution["retry"].model_copy(
            update={"step_id": execution_updates["retry_step_id"]}
        )

    with pytest.raises(ValueError, match=error_match):
        _controller().decide(
            _input(
                **execution,
                brain_result=BrainResult(
                    outcome=outcome,
                    message="failure",
                ),
            )
        )


def test_replan_request_rejects_stale_failed_step_identifier():
    execution = _active_execution(max_retries=1)

    with pytest.raises(ValueError, match="replan request"):
        _controller().decide(
            _input(
                **execution,
                brain_result=BrainResult(
                    outcome=BrainOutcome.REPLAN_REQUEST,
                    replan_request=ReplanRequest(
                        reason="dependency changed",
                        failed_step_id="stale-step",
                    ),
                ),
            )
        )


def test_controller_decision_rejects_terminal_status_disagreement():
    with pytest.raises(ValueError, match="terminal"):
        ControllerDecision(
            decision_type=ControllerDecisionType.TERMINATE,
            reason="invalid",
            execution_status=ExecutionStatus.NON_TERMINAL,
            terminal=True,
        )
