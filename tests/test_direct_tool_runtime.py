from datetime import datetime, timezone

from core.completion import CompletionService
from core.graph_worker_runtime import GraphWorkerRuntimePorts
from core.models import ReadFileResult, ToolResult as TransportToolResult
from core.protocol.enums import BrainOutcome, ControllerDecisionType, ExecutionPhase, WorkerRole
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
    ProtocolVisibleState,
    ToolExecutionRecord,
    ToolRequest,
    ToolResult,
    WorkingState,
)
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver
from core.runtime.portable_orchestration import PortableExecutionRuntime
from core.runtime.tool_result_integration import SerializedToolRuntimePort, integrate_tool_result
from tools.comfy_ops import get_comfy_tools


NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


class SerializedTool:
    def __init__(self, name, output):
        self.name = name
        self.output = output

    def invoke(self, _arguments):
        return self.output


def test_direct_runtime_preserves_capture_pagination_integrity_and_failure():
    output = ReadFileResult(
        success=False,
        message="partial failure",
        path="a.txt",
        content="bc",
        total_chars=6,
        offset=1,
        read_chars=2,
        is_truncated=True,
        error_code="READ_FAILED",
    ).to_tool_output()
    request = ToolRequest(
        request_id="read-1",
        tool_name="read_file",
        arguments={"path": "a.txt", "offset": 1, "limit": 2},
    )

    result = SerializedToolRuntimePort(
        [SerializedTool("read_file", output)], require_structured=False
    ).execute(request)

    assert result.request_id == request.request_id
    assert result.signature.startswith("read_file:")
    assert result.success is False
    assert result.error_code == "READ_FAILED"
    assert result.integrity.is_truncated is True
    assert result.integrity.original_bytes == 6
    assert result.integrity.captured_bytes == 2
    assert result.pagination is not None
    assert result.pagination.offset == 1
    assert result.pagination.limit is None
    assert result.pagination.returned_items == 2
    assert result.pagination.has_more is True


def test_direct_integration_preserves_artifacts_and_repeat_failure_accounting():
    request = ToolRequest(
        request_id="write-2",
        tool_name="write_file",
        arguments={"path": "result.txt", "content": "value"},
    )
    output = TransportToolResult(success=False, message="write failed").to_tool_output()
    result = SerializedToolRuntimePort(
        [SerializedTool("write_file", output)], require_structured=False
    ).execute(request)
    previous = result.model_copy(update={"request_id": "write-1"})
    step = ExecutionStep(step_id="step-1", title="Write output")
    plan = ExecutionPlan(plan_id="plan-1", revision=1, steps=(step,))
    state = ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=ExecutionIdentity(execution_id="run-1", protocol_version="1"),
            cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING),
            active_plan=plan,
            active_step=step,
            pending_tool_request=request,
        ),
        working=WorkingState(last_tool_result=previous, repeat_fail_count=1),
    )
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
        next_worker=WorkerRole.TOOL_RUNTIME,
        pending_tool_request=request,
    )

    integrated = integrate_tool_result(state, decision, result)

    assert integrated.working.repeat_fail_count == 2
    record = integrated.working.tool_execution_history[-1]
    assert record.artifacts[0].path == "result.txt"
    assert record.artifacts[0].action == "created"


class ScriptedController:
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def decide(self, _controller_input):
        return next(self.decisions)


class DownloadRequestBrain:
    def __init__(self, request):
        self.request = request
        self.inputs = []

    def node(self, state):
        self.inputs.append(state["execution_state"].working.last_tool_result)
        if len(self.inputs) == 1:
            result = BrainResult(
                outcome=BrainOutcome.TOOL_REQUEST, tool_request=self.request
            )
        else:
            result = BrainResult(outcome=BrainOutcome.CONTINUE)
        return {"brain_result": result}


class BytesResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b"production-image-bytes"


def test_terminal_async_continuation_executes_real_download_without_graph_workers(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "tools.comfy_ops.urllib.request.urlopen",
        lambda *_args, **_kwargs: BytesResponse(),
    )
    download_tool = next(
        tool
        for tool in get_comfy_tools(str(tmp_path))
        if tool.name == "download_comfy_output_image"
    )
    request = ToolRequest(
        request_id="download-1",
        tool_name="download_comfy_output_image",
        arguments={"filename": "result.png", "save_path": "result.png"},
        requested_by=WorkerRole.BRAIN,
    )
    identity = ExecutionIdentity(execution_id="post-async", protocol_version="1")
    step = ExecutionStep(
        step_id="3",
        title="Download image",
        primary_tool=request.tool_name,
    )
    plan = ExecutionPlan(plan_id="plan-1", revision=1, steps=(step,))
    cursor = ExecutionCursor(
        phase=ExecutionPhase.EXECUTING,
        current_worker=WorkerRole.CONTROLLER,
        step_id=step.step_id,
        plan_revision=plan.revision,
    )
    terminal_async = ToolResult(
        request_id="poll-1",
        success=True,
        message="completed",
        is_async_job=True,
        async_job_id="prompt-1",
        async_job_status="completed",
        async_terminal=True,
        async_observed_at_utc=NOW,
    )
    async_record = ToolExecutionRecord(
        execution_id=identity.execution_id,
        plan_id=plan.plan_id,
        plan_revision=plan.revision,
        step_id="2",
        tool_name="get_comfy_history",
        arguments={"prompt_id": "prompt-1"},
        result=terminal_async,
    )
    state = ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=identity,
            cursor=cursor,
            active_plan=plan,
            active_step=step,
        ),
        working=WorkingState(
            last_tool_result=terminal_async,
            tool_execution_history=(async_record,),
        ),
    )
    controller_input = ControllerInput(
        identity=identity,
        cursor=cursor,
        context=ExecutionContext(user_request="generate and download"),
        active_plan=plan,
        active_step=step,
        tool_result=terminal_async,
        tool_execution_history=(async_record,),
    )
    brain = DownloadRequestBrain(request)
    brain_cursor = cursor.model_copy(update={"current_worker": WorkerRole.BRAIN})
    tool_cursor = cursor.model_copy(update={"current_worker": WorkerRole.TOOL_RUNTIME})
    controller = ScriptedController((
        ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_BRAIN,
            next_worker=WorkerRole.BRAIN,
            cursor=brain_cursor,
        ),
        ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
            next_worker=WorkerRole.TOOL_RUNTIME,
            cursor=tool_cursor,
            pending_tool_request=request,
        ),
        ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_BRAIN,
            next_worker=WorkerRole.BRAIN,
            cursor=brain_cursor,
        ),
    ))
    direct = SerializedToolRuntimePort([download_tool], require_structured=False)
    worker_ports = GraphWorkerRuntimePorts(tool_runtime=direct)
    graph_tool_calls = []
    capture_calls = []
    worker_ports.bind_nodes(
        planner=lambda _state: {},
        brain=brain.node,
        tool=lambda _state: graph_tool_calls.append(True),
        capture=lambda _state: capture_calls.append(True),
    )
    runtime = PortableExecutionRuntime(
        driver=ExecutionDriver(
            coordinator=ControllerCoordinator(controller),
            planner=worker_ports,
            brain=worker_ports,
            tool_runtime=worker_ports,
            finalizer=worker_ports,
        ),
        completion_service=CompletionService(),
    )

    worker_ports.begin_turn({"execution_state": state})
    brain_turn = runtime.turn(state, controller_input)
    worker_ports.consume_update()
    assert brain.inputs[0] == terminal_async

    brain_result = brain_turn.driver_turn.worker_result
    tool_input = brain_turn.controller_input.model_copy(
        update={"cursor": brain_turn.driver_turn.execution_state.protocol_visible.cursor,
                "brain_result": brain_result, "tool_result": None}
    )
    worker_ports.begin_turn({"execution_state": brain_turn.driver_turn.execution_state})
    tool_turn = runtime.turn(brain_turn.driver_turn.execution_state, tool_input)
    integrated = worker_ports.consume_update()["execution_state"]
    result = tool_turn.driver_turn.worker_result

    assert result.request_id == request.request_id
    assert result.success is True
    assert (tmp_path / "result.png").read_bytes() == b"production-image-bytes"
    record = integrated.working.tool_execution_history[-1]
    assert record.tool_name == request.tool_name
    assert record.arguments == request.arguments
    assert record.result == result
    assert graph_tool_calls == []
    assert capture_calls == []

    continuation_input = tool_input.model_copy(
        update={"cursor": integrated.protocol_visible.cursor,
                "brain_result": None, "tool_result": result,
                "pending_tool_request": integrated.protocol_visible.pending_tool_request,
                "tool_execution_history": integrated.working.tool_execution_history}
    )
    worker_ports.begin_turn({"execution_state": integrated})
    continued = runtime.turn(integrated, continuation_input)
    assert continued.driver_turn.decision.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert graph_tool_calls == []
    assert capture_calls == []
