from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from core.completion import CompletionService
from core.protocol.controller import CortexController
from core.protocol.enums import (
    AsyncJobStatus,
    BrainOutcome,
    ControllerDecisionType,
    ExecutionPhase,
    StepStatus,
    WorkerRole,
)
from core.protocol.models import (
    BrainResult,
    ControllerInput,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    ProtocolVisibleState,
    ToolExecutionRecord,
    ToolRequest,
    ToolResult,
    WorkingState,
)
from core.runtime.async_wake import AsyncExecutionWake
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver
from core.runtime.portable_orchestration import PortableExecutionRuntime
from core.runtime.tool_result_integration import integrate_tool_result
from core.runtime.tool_result_integration import SerializedToolRuntimePort
from core.tool_output import parse_tool_result, unwrap_tool_output
import tools.comfy_ops as comfy_ops
from tools.comfy_ops import get_comfy_tools


NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


class UnusedPort:
    def run(self, _value):
        raise AssertionError("unexpected worker dispatch")

    def execute(self, _value):
        raise AssertionError("unexpected tool dispatch")


class RecordingBrain:
    def __init__(self):
        self.inputs = []

    def run(self, value):
        self.inputs.append(value)
        return BrainResult(outcome=BrainOutcome.CONTINUE)


class PollPort:
    def __init__(self, status):
        self.status = status
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        return ToolResult(
            request_id=request.request_id,
            success=True,
            message=self.status.value,
            is_async_job=True,
            async_job_id="job-1",
            async_job_status=self.status,
            async_terminal=self.status == AsyncJobStatus.COMPLETED,
            async_observed_at_utc=NOW,
        )


def _runtime(brain):
    unused = UnusedPort()
    return PortableExecutionRuntime(
        driver=ExecutionDriver(
            coordinator=ControllerCoordinator(
                CortexController(max_reasoning_steps=10, now_utc=lambda: NOW)
            ),
            planner=unused,
            brain=brain,
            tool_runtime=unused,
            finalizer=unused,
        ),
        completion_service=CompletionService(),
    )


def _submitted_state_and_input():
    identity = ExecutionIdentity(execution_id="portable-async", protocol_version="1")
    step = ExecutionStep(
        step_id="step-1",
        title="Async work",
        status=StepStatus.ACTIVE,
        attempt=1,
    )
    plan = ExecutionPlan(plan_id="plan-1", revision=1, steps=(step,))
    submission = ToolResult(
        request_id="submit-1",
        success=True,
        message="submitted",
        is_async_job=True,
        async_job_id="job-1",
        async_job_status=AsyncJobStatus.SUBMITTED,
        async_terminal=False,
        async_observed_at_utc=NOW,
    )
    record = ToolExecutionRecord(
        execution_id=identity.execution_id,
        plan_id=plan.plan_id,
        plan_revision=plan.revision,
        step_id=step.step_id,
        tool_name="submit_async",
        result=submission,
    )
    cursor = ExecutionCursor(
        phase=ExecutionPhase.EXECUTING,
        current_worker=WorkerRole.TOOL_RUNTIME,
        step_id=step.step_id,
        plan_revision=plan.revision,
        step_attempt=step.attempt,
    )
    state = ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=identity,
            cursor=cursor,
            active_plan=plan,
            active_step=step,
            pending_tool_request=ToolRequest(
                request_id="submit-1",
                tool_name="submit_async",
            ),
        ),
        working=WorkingState(
            last_tool_result=submission,
            tool_execution_history=(record,),
        ),
    )
    controller_input = ControllerInput(
        identity=identity,
        cursor=cursor,
        context=ExecutionContext(user_request="run async work"),
        active_plan=plan,
        active_step=step,
        pending_tool_request=state.protocol_visible.pending_tool_request,
        tool_result=submission,
        tool_execution_history=(record,),
    )
    return state, controller_input


@pytest.mark.parametrize(
    ("status", "expected_decision", "brain_calls"),
    [
        (AsyncJobStatus.RUNNING, ControllerDecisionType.AWAIT_ASYNC_JOB, 0),
        (AsyncJobStatus.COMPLETED, ControllerDecisionType.DISPATCH_BRAIN, 1),
    ],
)
def test_portable_async_lifecycle_without_langgraph_nodes(
    status,
    expected_decision,
    brain_calls,
):
    brain = RecordingBrain()
    runtime = _runtime(brain)
    state, controller_input = _submitted_state_and_input()

    waiting = runtime.turn(state, controller_input, dispatch_worker=False)
    assert waiting.driver_turn.decision.decision_type == ControllerDecisionType.AWAIT_ASYNC_JOB

    wake = AsyncExecutionWake(
        execution_id="portable-async",
        async_job_id="job-1",
    )
    authorized = runtime.authorize_async_wake(
        waiting.driver_turn.execution_state,
        waiting.controller_input.model_copy(
            update={
                "cursor": waiting.driver_turn.execution_state.protocol_visible.cursor,
                "pending_tool_request": None,
                "tool_result": None,
            }
        ),
        wake,
        status_tool_name="poll_async",
        status_argument_key="job_id",
    )
    assert authorized.driver_turn.decision.decision_type == ControllerDecisionType.DISPATCH_TOOL_RUNTIME

    poll_port = PollPort(status)
    result = runtime.dispatch_authorized_async_poll(authorized, poll_port)
    integrated = integrate_tool_result(
        authorized.driver_turn.execution_state,
        authorized.driver_turn.decision,
        result,
    )
    continued_input = authorized.controller_input.model_copy(
        update={
            "cursor": integrated.protocol_visible.cursor,
            "pending_tool_request": integrated.protocol_visible.pending_tool_request,
            "tool_result": result,
            "tool_execution_history": integrated.working.tool_execution_history,
            "async_wake_job_id": None,
            "async_poll_tool_name": None,
            "async_poll_argument_key": None,
        }
    )
    continued = runtime.turn(integrated, continued_input, dispatch_worker=True)

    assert continued.driver_turn.decision.decision_type == expected_decision
    assert len(brain.inputs) == brain_calls
    if brain.inputs:
        assert brain.inputs[0].last_tool_result == result


def test_async_production_code_has_no_named_node_resume_tokens():
    project = Path(__file__).resolve().parents[1]
    sources = (
        project / "core" / "runtime" / "async_poller.py",
        project / "core" / "runtime" / "portable_orchestration.py",
        project / "core" / "graph_async_resume.py",
    )

    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert 'as_node="tools"' not in text
    assert 'as_node="capture_tool_output"' not in text


class FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


@pytest.mark.parametrize(
    ("provider_status", "completed", "expected_status", "expected_decision"),
    [
        (
            "running",
            False,
            AsyncJobStatus.RUNNING,
            ControllerDecisionType.AWAIT_ASYNC_JOB,
        ),
        (
            "success",
            True,
            AsyncJobStatus.COMPLETED,
            ControllerDecisionType.DISPATCH_BRAIN,
        ),
    ],
)
def test_real_comfy_history_shape_crosses_portable_poll_boundary(
    monkeypatch,
    provider_status,
    completed,
    expected_status,
    expected_decision,
):
    monkeypatch.setattr(
        comfy_ops.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeHttpResponse(
            {
                "job-1": {
                    "outputs": {"9": {"images": [{"filename": "result.png"}]}},
                    "status": {
                        "completed": completed,
                        "status_str": provider_status,
                    },
                }
            }
        ),
    )
    history_tool = next(
        tool for tool in get_comfy_tools(".") if tool.name == "get_comfy_history"
    )
    production_value = history_tool.invoke({"prompt_id": "job-1"})
    assert isinstance(production_value, str)
    assert parse_tool_result(production_value) is None
    production_payload = unwrap_tool_output(production_value)
    assert production_payload["prompt_id"] == "job-1"
    assert "status_details" in production_payload

    brain = RecordingBrain()
    runtime = _runtime(brain)
    state, controller_input = _submitted_state_and_input()
    waiting = runtime.turn(state, controller_input, dispatch_worker=False)
    authorized = runtime.authorize_async_wake(
        waiting.driver_turn.execution_state,
        waiting.controller_input.model_copy(
            update={
                "cursor": waiting.driver_turn.execution_state.protocol_visible.cursor,
                "pending_tool_request": None,
                "tool_result": None,
            }
        ),
        AsyncExecutionWake(
            execution_id="portable-async",
            async_job_id="job-1",
        ),
        status_tool_name="get_comfy_history",
        status_argument_key="prompt_id",
    )
    request = authorized.driver_turn.decision.pending_tool_request
    result = runtime.dispatch_authorized_async_poll(
        authorized,
        SerializedToolRuntimePort([history_tool]),
    )

    assert result.request_id == request.request_id
    assert result.async_job_status == expected_status
    integrated = integrate_tool_result(
        authorized.driver_turn.execution_state,
        authorized.driver_turn.decision,
        result,
    )
    continued = runtime.turn(
        integrated,
        authorized.controller_input.model_copy(
            update={
                "cursor": integrated.protocol_visible.cursor,
                "pending_tool_request": integrated.protocol_visible.pending_tool_request,
                "tool_result": result,
                "tool_execution_history": integrated.working.tool_execution_history,
                "async_wake_job_id": None,
                "async_poll_tool_name": None,
                "async_poll_argument_key": None,
            }
        ),
        dispatch_worker=True,
    )
    assert continued.driver_turn.decision.decision_type == expected_decision
    if expected_decision == ControllerDecisionType.DISPATCH_BRAIN:
        assert len(brain.inputs) == 1
        assert brain.inputs[0].last_tool_result == result
    else:
        assert brain.inputs == []
