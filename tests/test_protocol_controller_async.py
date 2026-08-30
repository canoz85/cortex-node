import pytest
from datetime import datetime, timedelta, timezone
from langgraph.graph import END

from core.graph_state_machine import map_controller_decision
from core.protocol.controller import CortexController
from core.protocol.enums import (
    AsyncJobStatus,
    CancellationSource,
    ControllerDecisionType,
    ExecutionPhase,
    WorkerRole,
)
from core.protocol.models import (
    ControllerInput,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionStep,
    ToolExecutionRecord,
    ToolRequest,
    ToolResult,
)


OBSERVED_AT = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)


def _controller(now_utc=OBSERVED_AT) -> CortexController:
    return CortexController(
        max_reasoning_steps=10,
        now_utc=lambda: now_utc,
    )


def _controller_input(status: AsyncJobStatus) -> ControllerInput:
    request = ToolRequest(request_id="req-1", tool_name="run_comfy_workflow")
    return ControllerInput(
        identity=ExecutionIdentity(execution_id="run-1", protocol_version="1.0"),
        cursor=ExecutionCursor(
            phase=ExecutionPhase.EXECUTING,
            step_id="step-1",
            current_worker=WorkerRole.TOOL_RUNTIME,
        ),
        context=ExecutionContext(user_request="Generate an image", role=WorkerRole.CONTROLLER),
        active_step=ExecutionStep(step_id="step-1", title="Generate image"),
        pending_tool_request=request,
        tool_result=ToolResult(
            request_id="req-1",
            success=status not in {AsyncJobStatus.FAILED, AsyncJobStatus.CANCELLED},
            message=f"ComfyUI job is {status.value}.",
            is_async_job=True,
            async_job_id="prompt-1",
            async_job_status=status,
            async_terminal=status in {
                AsyncJobStatus.COMPLETED,
                AsyncJobStatus.FAILED,
                AsyncJobStatus.CANCELLED,
            },
            async_observed_at_utc=OBSERVED_AT,
        ),
    )


@pytest.mark.parametrize(
    "status",
    [AsyncJobStatus.SUBMITTED, AsyncJobStatus.RUNNING, AsyncJobStatus.UNKNOWN],
)
def test_controller_awaits_nonterminal_async_job(status):
    decision = _controller().decide(_controller_input(status))

    assert decision.decision_type == ControllerDecisionType.AWAIT_ASYNC_JOB
    assert decision.async_job_id == "prompt-1"
    assert decision.requires_checkpoint is True
    assert decision.clear_pending_tool_request is True
    assert decision.cursor is not None
    assert decision.cursor.phase == ExecutionPhase.WAITING
    assert decision.cursor.current_worker == WorkerRole.CONTROLLER
    assert decision.resume_after_utc == OBSERVED_AT + timedelta(seconds=5)
    assert decision.execution_deadline_utc == OBSERVED_AT + timedelta(minutes=30)
    assert map_controller_decision(decision) == END


def test_controller_resumes_brain_only_after_async_job_completion():
    decision = _controller().decide(
        _controller_input(AsyncJobStatus.COMPLETED)
    )

    assert decision.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert decision.clear_pending_tool_request is True
    assert decision.reason == "Async job completed."


def test_controller_keeps_existing_failure_policy_for_terminal_async_failure():
    decision = _controller().decide(_controller_input(AsyncJobStatus.FAILED))

    assert decision.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert decision.reason == "Tool failed."


def test_controller_treats_provider_cancellation_as_terminal_cancellation():
    decision = _controller().decide(_controller_input(AsyncJobStatus.CANCELLED))

    assert decision.decision_type == ControllerDecisionType.CANCEL
    assert decision.reason == "remote_async_job_cancelled"
    assert decision.cancellation_source == CancellationSource.PROVIDER
    assert decision.terminal is True


def test_controller_rejects_async_tool_result_request_id_mismatch():
    controller_input = _controller_input(AsyncJobStatus.SUBMITTED).model_copy(
        update={
            "pending_tool_request": ToolRequest(
                request_id="expected-request",
                tool_name="run_comfy_workflow",
            )
        }
    )

    decision = _controller().decide(controller_input)

    assert decision.decision_type == ControllerDecisionType.TERMINATE
    assert decision.reason == "tool_result_request_id_mismatch"


def test_controller_sync_success_still_dispatches_brain():
    controller_input = _controller_input(AsyncJobStatus.SUBMITTED).model_copy(
        update={
            "tool_result": ToolResult(
                request_id="req-1",
                success=True,
                message="Synchronous tool completed.",
            )
        }
    )

    decision = _controller().decide(controller_input)

    assert decision.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert decision.reason == "Tool completed."


def test_controller_uses_terminal_history_over_stale_nonterminal_observation():
    def record(request_id: str, status: AsyncJobStatus) -> ToolExecutionRecord:
        result = ToolResult(
            request_id=request_id,
            success=status != AsyncJobStatus.FAILED,
            message=f"Observed {status.value}.",
            is_async_job=True,
            async_job_id="prompt-1",
            async_job_status=status,
            async_terminal=status in {
                AsyncJobStatus.COMPLETED,
                AsyncJobStatus.FAILED,
                AsyncJobStatus.CANCELLED,
            },
        )
        return ToolExecutionRecord(
            step_id="step-1",
            tool_name="get_comfy_history",
            result=result,
        )

    completed = record("req-completed", AsyncJobStatus.COMPLETED)
    stale_running = record("req-stale", AsyncJobStatus.RUNNING)
    controller_input = _controller_input(AsyncJobStatus.RUNNING).model_copy(
        update={
            "pending_tool_request": None,
            "tool_result": stale_running.result,
            "tool_execution_history": (completed, stale_running),
        }
    )

    decision = _controller().decide(controller_input)

    assert decision.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert decision.reason == "Async job completed."


def test_nonterminal_poll_failure_awaits_before_generic_failure_policy():
    failed_poll_records = tuple(
        ToolExecutionRecord(
            step_id="step-1",
            tool_name="get_comfy_history",
            result=ToolResult(
                request_id=f"poll-{index}",
                signature='get_comfy_history:{"prompt_id": "prompt-1"}',
                success=False,
                message="Polling connection failed.",
                is_async_job=True,
                async_job_id="prompt-1",
                async_job_status=AsyncJobStatus.UNKNOWN,
                async_terminal=False,
            ),
        )
        for index in range(1, 3)
    )
    controller_input = _controller_input(AsyncJobStatus.UNKNOWN).model_copy(
        update={
            "pending_tool_request": None,
            "tool_result": failed_poll_records[-1].result,
            "tool_execution_history": failed_poll_records,
        }
    )

    decision = _controller().decide(controller_input)

    assert decision.decision_type == ControllerDecisionType.AWAIT_ASYNC_JOB
    assert decision.async_job_id == "prompt-1"


def test_terminal_async_failure_uses_existing_replan_threshold():
    failed_records = tuple(
        ToolExecutionRecord(
            step_id="step-1",
            tool_name="get_comfy_history",
            result=ToolResult(
                request_id=f"poll-{index}",
                signature='get_comfy_history:{"prompt_id": "prompt-1"}',
                success=False,
                message="Provider reported failure.",
                is_async_job=True,
                async_job_id="prompt-1",
                async_job_status=AsyncJobStatus.FAILED,
                async_terminal=True,
            ),
        )
        for index in range(1, 4)
    )
    controller_input = _controller_input(AsyncJobStatus.FAILED).model_copy(
        update={
            "pending_tool_request": None,
            "tool_result": failed_records[-1].result,
            "tool_execution_history": failed_records,
        }
    )

    decision = _controller().decide(controller_input)

    assert decision.decision_type == ControllerDecisionType.DISPATCH_PLANNER
    assert "Repeated tool failure threshold (3)" in decision.reason


def test_controller_increases_poll_delay_from_recorded_observations():
    submitted = ToolExecutionRecord(
        step_id="step-1",
        tool_name="run_comfy_workflow",
        result=ToolResult(
            request_id="submit-1",
            success=True,
            message="Submitted.",
            is_async_job=True,
            async_job_id="prompt-1",
            async_job_status=AsyncJobStatus.SUBMITTED,
            async_terminal=False,
            async_observed_at_utc=OBSERVED_AT,
        ),
    )
    running = ToolExecutionRecord(
        step_id="step-1",
        tool_name="get_comfy_history",
        result=ToolResult(
            request_id="poll-1",
            success=True,
            message="Running.",
            is_async_job=True,
            async_job_id="prompt-1",
            async_job_status=AsyncJobStatus.RUNNING,
            async_terminal=False,
            async_observed_at_utc=OBSERVED_AT + timedelta(seconds=2),
        ),
    )
    controller_input = _controller_input(AsyncJobStatus.RUNNING).model_copy(
        update={
            "pending_tool_request": None,
            "tool_result": running.result,
            "tool_execution_history": (submitted, running),
        }
    )

    decision = _controller().decide(controller_input)

    assert decision.decision_type == ControllerDecisionType.AWAIT_ASYNC_JOB
    assert decision.resume_after_utc == OBSERVED_AT + timedelta(seconds=12)
