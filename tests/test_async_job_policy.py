from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from core.protocol.controller import CortexController
from core.protocol.enums import (
    AsyncJobStatus,
    BrainOutcome,
    CancellationSource,
    ControllerDecisionType,
    ExecutionPhase,
    WorkerRole,
)
from core.protocol.models import (
    AsyncJobPolicy,
    BrainResult,
    ControllerInput,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionStep,
    ToolExecutionRecord,
    ToolRequest,
    ToolResult,
)


STARTED_AT = datetime(2026, 8, 30, 10, 0, tzinfo=timezone.utc)


def _record(
    request_id: str,
    status: AsyncJobStatus,
    *,
    job_id: str = "prompt-1",
    observed_at: datetime = STARTED_AT,
    success: bool | None = None,
    tool_name: str = "get_comfy_history",
    error_code: str | None = None,
    data: dict | None = None,
) -> ToolExecutionRecord:
    if success is None:
        success = status not in {AsyncJobStatus.FAILED, AsyncJobStatus.CANCELLED}
    return ToolExecutionRecord(
        step_id="step-1",
        tool_name=tool_name,
        result=ToolResult(
            request_id=request_id,
            signature=f'{tool_name}:{{"prompt_id":"{job_id}"}}',
            success=success,
            message=f"Observed {status.value}.",
            error_code=error_code,
            data=data,
            is_async_job=True,
            async_job_id=job_id,
            async_job_status=status,
            async_terminal=status in {
                AsyncJobStatus.COMPLETED,
                AsyncJobStatus.FAILED,
                AsyncJobStatus.CANCELLED,
            },
            async_observed_at_utc=observed_at,
        ),
    )


def _input(
    *,
    history: tuple[ToolExecutionRecord, ...] = (),
    tool_result: ToolResult | None = None,
    pending_tool_request: ToolRequest | None = None,
    brain_result: BrainResult | None = None,
    policy: AsyncJobPolicy | None = None,
    cancel_requested: bool = False,
) -> ControllerInput:
    return ControllerInput(
        identity=ExecutionIdentity(execution_id="run-1", protocol_version="1.0"),
        cursor=ExecutionCursor(
            phase=ExecutionPhase.EXECUTING,
            step_id="step-1",
            current_worker=WorkerRole.CONTROLLER,
        ),
        context=ExecutionContext(
            user_request="Generate an image",
            role=WorkerRole.CONTROLLER,
        ),
        active_step=ExecutionStep(step_id="step-1", title="Generate image"),
        pending_tool_request=pending_tool_request,
        brain_result=brain_result,
        tool_result=tool_result,
        async_policy=policy or AsyncJobPolicy(),
        cancel_requested=cancel_requested,
        tool_execution_history=history,
    )


def _controller(now: datetime) -> CortexController:
    return CortexController(max_reasoning_steps=10, now_utc=lambda: now)


def test_async_job_policy_defaults_and_cross_field_validation():
    policy = AsyncJobPolicy()

    assert policy.visibility_grace_seconds == 15
    assert policy.poll_interval_seconds == 5
    assert policy.max_poll_interval_seconds == 30
    assert policy.execution_timeout_seconds == 1800
    assert policy.max_poll_failures == 3
    assert policy.max_submission_attempts == 1

    with pytest.raises(ValidationError):
        AsyncJobPolicy(poll_interval_seconds=10, max_poll_interval_seconds=5)

    with pytest.raises(ValidationError):
        AsyncJobPolicy(visibility_grace_seconds=20, execution_timeout_seconds=10)


def test_missing_history_after_visibility_grace_remains_nonterminal():
    submitted = _record(
        "submit-1",
        AsyncJobStatus.SUBMITTED,
        tool_name="run_comfy_workflow",
    )
    unknown = _record(
        "poll-1",
        AsyncJobStatus.UNKNOWN,
        observed_at=STARTED_AT + timedelta(seconds=16),
    )
    controller_input = _input(
        history=(submitted, unknown),
        tool_result=unknown.result,
    )

    decision = _controller(STARTED_AT + timedelta(seconds=16)).decide(
        controller_input
    )

    assert decision.decision_type == ControllerDecisionType.AWAIT_ASYNC_JOB
    assert "visibility grace" in decision.reason
    assert decision.terminal is False


def test_poll_transport_failure_uses_separate_budget_then_pauses():
    submitted = _record(
        "submit-1",
        AsyncJobStatus.SUBMITTED,
        tool_name="run_comfy_workflow",
    )
    failures = tuple(
        _record(
            f"poll-{index}",
            AsyncJobStatus.UNKNOWN,
            observed_at=STARTED_AT + timedelta(seconds=index),
            success=False,
            error_code="COMFY_CONNECTION_FAILED",
        )
        for index in range(1, 4)
    )

    retry_decision = _controller(STARTED_AT + timedelta(seconds=2)).decide(
        _input(
            history=(submitted, *failures[:2]),
            tool_result=failures[1].result,
        )
    )
    exhausted_decision = _controller(STARTED_AT + timedelta(seconds=3)).decide(
        _input(
            history=(submitted, *failures),
            tool_result=failures[-1].result,
        )
    )

    assert retry_decision.decision_type == ControllerDecisionType.AWAIT_ASYNC_JOB
    assert exhausted_decision.decision_type == ControllerDecisionType.PAUSE
    assert exhausted_decision.reason == "async_poll_failure_budget_exhausted:3"
    assert exhausted_decision.reconciliation_required is True


def test_ambiguous_submission_does_not_consume_poll_failure_budget():
    request = ToolRequest(
        request_id="submit-1",
        tool_name="run_comfy_workflow",
        arguments={"workflow_json": {}, "prompt_id": "prompt-1"},
    )
    ambiguous_submission = _record(
        "submit-1",
        AsyncJobStatus.UNKNOWN,
        success=False,
        tool_name="run_comfy_workflow",
        error_code="COMFY_CONNECTION_FAILED",
        data={"submission_outcome": "ambiguous"},
    )
    policy = AsyncJobPolicy(max_poll_failures=1)

    decision = _controller(STARTED_AT).decide(
        _input(
            history=(ambiguous_submission,),
            tool_result=ambiguous_submission.result,
            pending_tool_request=request,
            policy=policy,
        )
    )

    assert decision.decision_type == ControllerDecisionType.AWAIT_ASYNC_JOB
    assert decision.resume_after_utc == STARTED_AT + timedelta(seconds=5)


def test_deadline_triggers_exactly_one_final_provider_reconciliation_boundary():
    policy = AsyncJobPolicy(
        visibility_grace_seconds=2,
        poll_interval_seconds=2,
        max_poll_interval_seconds=10,
        execution_timeout_seconds=30,
    )
    submitted = _record(
        "submit-1",
        AsyncJobStatus.SUBMITTED,
        tool_name="run_comfy_workflow",
    )
    before_deadline = _record(
        "poll-before-deadline",
        AsyncJobStatus.RUNNING,
        observed_at=STARTED_AT + timedelta(seconds=29),
    )
    after_deadline = _record(
        "poll-after-deadline",
        AsyncJobStatus.RUNNING,
        observed_at=STARTED_AT + timedelta(seconds=31),
    )

    reconcile_decision = _controller(STARTED_AT + timedelta(seconds=31)).decide(
        _input(
            history=(submitted, before_deadline),
            tool_result=before_deadline.result,
            policy=policy,
        )
    )
    timeout_decision = _controller(STARTED_AT + timedelta(seconds=31)).decide(
        _input(
            history=(submitted, before_deadline, after_deadline),
            tool_result=after_deadline.result,
            policy=policy,
        )
    )

    assert reconcile_decision.decision_type == ControllerDecisionType.AWAIT_ASYNC_JOB
    assert reconcile_decision.resume_after_utc == STARTED_AT + timedelta(seconds=31)
    assert "Final reconciliation" in reconcile_decision.reason
    assert timeout_decision.decision_type == ControllerDecisionType.PAUSE
    assert timeout_decision.reason == "async_execution_timeout_after_reconciliation"
    assert timeout_decision.reconciliation_required is False


def test_two_nonterminal_jobs_for_same_step_are_paused_for_reconciliation():
    first = _record("submit-1", AsyncJobStatus.SUBMITTED, job_id="prompt-1")
    second = _record("submit-2", AsyncJobStatus.SUBMITTED, job_id="prompt-2")

    decision = _controller(STARTED_AT).decide(
        _input(history=(first, second), tool_result=second.result)
    )

    assert decision.decision_type == ControllerDecisionType.PAUSE
    assert decision.reason == "multiple_active_async_jobs_for_step:prompt-1,prompt-2"
    assert decision.reconciliation_required is True


def test_second_submission_is_blocked_after_terminal_first_attempt():
    submitted = _record(
        "submit-1",
        AsyncJobStatus.SUBMITTED,
        tool_name="run_comfy_workflow",
    )
    failed = _record("poll-failed", AsyncJobStatus.FAILED)
    new_request = ToolRequest(
        request_id="submit-2",
        tool_name="run_comfy_workflow",
        arguments={"workflow_json": {}},
    )

    decision = _controller(STARTED_AT).decide(
        _input(
            history=(submitted, failed),
            brain_result=BrainResult(
                outcome=BrainOutcome.TOOL_REQUEST,
                tool_request=new_request,
            ),
        )
    )

    assert decision.decision_type == ControllerDecisionType.PAUSE
    assert decision.reason == "async_submission_attempt_limit_reached:1"
    assert decision.reconciliation_required is True


def test_controller_preallocates_provider_id_before_first_submission():
    request = ToolRequest(
        request_id="submit-1",
        tool_name="run_comfy_workflow",
        arguments={"workflow_json": {}},
    )

    decision = _controller(STARTED_AT).decide(
        _input(
            brain_result=BrainResult(
                outcome=BrainOutcome.TOOL_REQUEST,
                tool_request=request,
            )
        )
    )

    assert decision.decision_type == ControllerDecisionType.DISPATCH_TOOL_RUNTIME
    assert decision.pending_tool_request is not None
    assert decision.pending_tool_request.arguments["client_id"] == "run-1"
    assert UUID(decision.pending_tool_request.arguments["prompt_id"]).version == 5


@pytest.mark.parametrize(
    ("max_attempts", "expected_decision"),
    [
        (1, ControllerDecisionType.PAUSE),
        (2, ControllerDecisionType.DISPATCH_BRAIN),
    ],
)
def test_confirmed_absent_submission_obeys_retry_limit(
    max_attempts,
    expected_decision,
):
    submitted = _record(
        "submit-1",
        AsyncJobStatus.SUBMITTED,
        tool_name="run_comfy_workflow",
    )
    absent = _record(
        "poll-absent",
        AsyncJobStatus.UNKNOWN,
        observed_at=STARTED_AT + timedelta(seconds=16),
        data={"provider_visibility": "absent"},
    )
    policy = AsyncJobPolicy(max_submission_attempts=max_attempts)

    decision = _controller(STARTED_AT + timedelta(seconds=16)).decide(
        _input(
            history=(submitted, absent),
            tool_result=absent.result,
            policy=policy,
        )
    )

    assert decision.decision_type == expected_decision
    if max_attempts == 1:
        assert decision.reconciliation_required is False
    else:
        assert "retry policy permits" in decision.reason


def test_ambiguous_submission_failure_never_immediately_reposts():
    request = ToolRequest(
        request_id="submit-1",
        tool_name="run_comfy_workflow",
        arguments={"workflow_json": {}},
    )
    ambiguous_result = ToolResult(
        request_id="submit-1",
        success=False,
        message="POST timed out.",
        error_code="COMFY_CONNECTION_FAILED",
    )

    decision = _controller(STARTED_AT).decide(
        _input(
            pending_tool_request=request,
            tool_result=ambiguous_result,
        )
    )

    assert decision.decision_type == ControllerDecisionType.PAUSE
    assert decision.reconciliation_required is True
    assert decision.clear_pending_tool_request is True


def test_definitive_invalid_submission_can_return_to_brain_for_correction():
    request = ToolRequest(
        request_id="submit-1",
        tool_name="run_comfy_workflow",
        arguments={"workflow_json": {}},
    )
    invalid_result = ToolResult(
        request_id="submit-1",
        success=False,
        message="Provider rejected invalid workflow.",
        error_code="COMFY_INVALID_PAYLOAD",
    )

    decision = _controller(STARTED_AT).decide(
        _input(
            pending_tool_request=request,
            tool_result=invalid_result,
        )
    )

    assert decision.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert decision.reason == "Tool failed."

    invalid_record = ToolExecutionRecord(
        step_id="step-1",
        tool_name="run_comfy_workflow",
        result=invalid_result,
    )
    corrected_request = ToolRequest(
        request_id="submit-2",
        tool_name="run_comfy_workflow",
        arguments={"workflow_json": {"1": {}}},
    )
    corrected_decision = _controller(STARTED_AT).decide(
        _input(
            history=(invalid_record,),
            brain_result=BrainResult(
                outcome=BrainOutcome.TOOL_REQUEST,
                tool_request=corrected_request,
            ),
        )
    )

    assert (
        corrected_decision.decision_type
        == ControllerDecisionType.DISPATCH_TOOL_RUNTIME
    )


def test_local_cancel_signal_preempts_worker_results():
    running = _record("poll-1", AsyncJobStatus.RUNNING)

    decision = _controller(STARTED_AT).decide(
        _input(
            history=(running,),
            tool_result=running.result,
            cancel_requested=True,
        )
    )

    assert decision.decision_type == ControllerDecisionType.CANCEL
    assert decision.reason == "local_async_cancellation_requested"
    assert decision.cancellation_source == CancellationSource.LOCAL
    assert decision.terminal is True
