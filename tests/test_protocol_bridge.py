from langchain_core.messages import AIMessage, HumanMessage

from core.protocol.bridge import build_brain_input, build_controller_input, legacy_tool_result_to_model
from core.protocol.enums import AsyncJobStatus, BrainOutcome, ExecutionPhase, WorkerRole
from core.protocol.models import (
    AsyncJobPolicy,
    ArtifactRecord,
    BrainResult,
    ContentIntegrity,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    PaginationMetadata,
    PlannerResult,
    ProtocolVisibleState,
    ToolExecutionRecord,
    ToolResult,
    ToolRequest,
    WorkingState,
)


def test_build_controller_input_maps_protocol_outputs_from_legacy_state():
    state = {
        "run_id": "run-1",
        "protocol_version": "1.0",
        "steps": 2,
        "phase": "executing",
        "messages": [
            HumanMessage(content="create file"),
            AIMessage(content="Calling write_file", tool_calls=[{"name": "write_file", "args": {"path": "workspace/a.py", "content": "print('x')"}, "id": "call-1", "type": "tool_call"}]),
        ],
        "plan": "1. write file\n2. run",
        "plan_id": "p-1",
        "plan_revision": 3,
        "planner_result": PlannerResult(
            outcome="execution_plan",
            message="Plan generated successfully.",
            proposed_plan=ExecutionPlan(
                plan_id="p-1",
                revision=3,
                objective="1. write file\n2. run",
                steps=(ExecutionStep(step_id="s1", title="write"),),
            ),
        ),
        "brain_result": BrainResult(
            outcome=BrainOutcome.TOOL_REQUEST,
            message="Brain requested tool execution.",
            tool_request=ToolRequest(
                request_id="req-1",
                tool_name="write_file",
                arguments={"path": "workspace/a.py", "content": "print('x')"},
            ),
        ),
        "last_tool_signature": "write_file:{\"path\":\"workspace/a.py\"}",
        "last_tool_output": {
            "success": True,
            "message": "Wrote file",
            "data": {"path": "workspace/a.py"},
        },
    }

    controller_input = build_controller_input(state)

    assert controller_input.identity.execution_id == "run-1"
    assert controller_input.cursor.controller_iteration == 2
    assert controller_input.context.role == WorkerRole.CONTROLLER

    assert controller_input.planner_result is not None
    assert controller_input.planner_result.message == "Plan generated successfully."
    assert controller_input.planner_result.proposed_plan.plan_id == "p-1"
    assert controller_input.planner_result.proposed_plan.revision == 3

    assert controller_input.brain_result is not None
    assert controller_input.brain_result.outcome == BrainOutcome.TOOL_REQUEST
    assert controller_input.brain_result.tool_request is not None
    assert controller_input.brain_result.tool_request.request_id == "req-1"
    assert controller_input.brain_result.tool_request.tool_name == "write_file"

    assert controller_input.tool_result is not None
    assert controller_input.tool_result.success is True
    assert controller_input.tool_result.message == "Wrote file"


def test_build_controller_input_leaves_optional_worker_outputs_none_when_missing():
    state = {
        "messages": [HumanMessage(content="hello")],
        "steps": 0,
    }

    controller_input = build_controller_input(state)

    assert controller_input.planner_result is None
    assert controller_input.brain_result is None
    assert controller_input.tool_result is None


def test_build_controller_input_preserves_checkpointed_async_policy():
    policy = AsyncJobPolicy(
        visibility_grace_seconds=4,
        poll_interval_seconds=3,
        max_poll_interval_seconds=12,
        execution_timeout_seconds=60,
        max_poll_failures=2,
        max_submission_attempts=2,
    )
    controller_input = build_controller_input({
        "messages": [HumanMessage(content="generate image")],
        "async_job_policy": policy,
    })

    assert controller_input.async_policy == policy


def test_build_brain_input_transfers_tool_execution_history_from_working_state():
    execution_state = ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=ExecutionIdentity(
                execution_id="run-1",
                protocol_version="1.0",
            ),
            cursor=ExecutionCursor(
                phase=ExecutionPhase.EXECUTING,
                step_id="s1",
            ),
        ),
        working=WorkingState(
            tool_execution_history=(
                ToolExecutionRecord(
                    step_id="s1",
                    tool_name="list_files",
                    arguments={"path": "."},
                    result=ToolResult(
                        request_id="req-1",
                        signature='list_files:{"path": "."}',
                        success=True,
                        message="Listing for .",
                        rendered_output="Files under .:\n- a.py",
                        data={"entries": ["a.py"]},
                    ),
                ),
            )
        ),
    )

    state = {
        "messages": [HumanMessage(content="list files")],
        "execution_state": execution_state,
    }

    brain_input = build_brain_input(state)

    assert len(brain_input.tool_execution_history) == 1
    record = brain_input.tool_execution_history[0]
    assert record.result.request_id == "req-1"
    assert record.step_id == "s1"
    assert record.tool_name == "list_files"


def test_tool_result_content_integrity_and_artifacts():
    legacy_state = {
        "last_tool_signature": "read_file:{\"path\":\"test.txt\"}",
        "last_tool_output": {
            "success": True,
            "message": "Read file test.txt",
            "is_truncated": True,
            "offset": 0,
            "limit": 4000,
            "read_chars": 4000,
            "total_chars": 10000,
        },
    }
    res = legacy_tool_result_to_model(legacy_state)
    assert res is not None
    assert res.integrity.is_truncated is True
    assert res.pagination is not None
    assert res.pagination.has_more is True
    assert res.pagination.total_items == 10000
    assert res.pagination.offset == 0


def test_controller_input_evidence_predicates_and_consecutive_failures():
    rec1 = ToolExecutionRecord(
        step_id="step-1",
        tool_name="read_file",
        arguments={"path": "a.txt"},
        result=ToolResult(
            request_id="req-1",
            signature="read_file:{\"path\":\"a.txt\"}",
            success=False,
            message="Error reading file",
            error_code="FILE_READ_FAILED",
        ),
        artifacts=(
            ArtifactRecord(
                artifact_id="art-1",
                step_id="step-1",
                path="a.txt",
                action="modified",
            ),
        ),
    )
    rec2 = ToolExecutionRecord(
        step_id="step-1",
        tool_name="read_file",
        arguments={"path": "a.txt"},
        result=ToolResult(
            request_id="req-2",
            signature="read_file:{\"path\":\"a.txt\"}",
            success=False,
            message="Error reading file",
            error_code="FILE_READ_FAILED",
        ),
    )
    controller_input = build_controller_input({
        "execution_state": ExecutionState(
            protocol_visible=ProtocolVisibleState(
                identity=ExecutionIdentity(execution_id="ex-1", protocol_version="1.0"),
                cursor=ExecutionCursor(),
            ),
            working=WorkingState(
                tool_execution_history=(rec1, rec2),
            ),
        )
    })

    assert len(controller_input.get_step_records("step-1")) == 2
    assert controller_input.get_consecutive_failures("read_file:{\"path\":\"a.txt\"}") == 2
    assert controller_input.has_successful_artifact("a.txt") is False

    from core.protocol.controller import CortexController
    ctrl = CortexController(max_reasoning_steps=10)

    # Simulating a 3rd failure for the same tool signature
    rec3 = ToolExecutionRecord(
        step_id="step-1",
        tool_name="read_file",
        arguments={"path": "a.txt"},
        result=ToolResult(
            request_id="req-3",
            signature="read_file:{\"path\":\"a.txt\"}",
            success=False,
            message="Error reading file",
            error_code="FILE_READ_FAILED",
        ),
    )
    ci3 = build_controller_input({
        "execution_state": ExecutionState(
            protocol_visible=ProtocolVisibleState(
                identity=ExecutionIdentity(execution_id="ex-1", protocol_version="1.0"),
                cursor=ExecutionCursor(),
                pending_tool_request=ToolRequest(request_id="req-3", tool_name="read_file"),
            ),
            working=WorkingState(
                last_tool_result=rec3.result,
                tool_execution_history=(rec1, rec2, rec3),
            ),
        )
    })
    decision = ctrl.decide(ci3)
    assert decision.decision_type.value == "dispatch_planner"
    assert "Repeated tool failure threshold" in decision.reason


def test_controller_input_async_evidence_queries_use_latest_valid_observation():
    def async_record(request_id: str, job_id: str, status: AsyncJobStatus) -> ToolExecutionRecord:
        return ToolExecutionRecord(
            step_id="step-1",
            tool_name="get_comfy_history",
            result=ToolResult(
                request_id=request_id,
                success=status not in {AsyncJobStatus.FAILED, AsyncJobStatus.CANCELLED},
                message=f"Observed {status.value}",
                is_async_job=True,
                async_job_id=job_id,
                async_job_status=status,
                async_terminal=status in {
                    AsyncJobStatus.COMPLETED,
                    AsyncJobStatus.FAILED,
                    AsyncJobStatus.CANCELLED,
                },
            ),
        )

    submitted = async_record("req-1", "prompt-1", AsyncJobStatus.SUBMITTED)
    completed = async_record("req-2", "prompt-1", AsyncJobStatus.COMPLETED)
    stale_running = async_record("req-3", "prompt-1", AsyncJobStatus.RUNNING)
    running = async_record("req-4", "prompt-2", AsyncJobStatus.RUNNING)
    controller_input = build_controller_input({
        "execution_state": ExecutionState(
            protocol_visible=ProtocolVisibleState(
                identity=ExecutionIdentity(execution_id="ex-1", protocol_version="1.0"),
                cursor=ExecutionCursor(step_id="step-1"),
                active_step=ExecutionStep(step_id="step-1", title="Generate image"),
            ),
            working=WorkingState(
                tool_execution_history=(submitted, completed, stale_running, running),
            ),
        )
    })

    latest = controller_input.get_latest_async_result()

    assert latest is not None
    assert latest.async_job_id == "prompt-2"
    assert latest.async_job_status == AsyncJobStatus.RUNNING
    assert controller_input.get_nonterminal_async_job_id() == "prompt-2"
    assert controller_input.has_terminal_async_result("prompt-1") is True
    assert controller_input.has_terminal_async_result("prompt-2") is False


def test_controller_input_async_evidence_queries_respect_terminal_latest_result():
    terminal_record = ToolExecutionRecord(
        step_id="step-1",
        tool_name="get_comfy_history",
        result=ToolResult(
            request_id="req-1",
            success=True,
            message="ComfyUI workflow completed.",
            is_async_job=True,
            async_job_id="prompt-1",
            async_job_status=AsyncJobStatus.COMPLETED,
            async_terminal=True,
        ),
    )
    controller_input = build_controller_input({
        "execution_state": ExecutionState(
            protocol_visible=ProtocolVisibleState(
                identity=ExecutionIdentity(execution_id="ex-1", protocol_version="1.0"),
                cursor=ExecutionCursor(step_id="step-1"),
                active_step=ExecutionStep(step_id="step-1", title="Generate image"),
            ),
            working=WorkingState(tool_execution_history=(terminal_record,)),
        )
    })

    latest = controller_input.get_latest_async_result()

    assert latest is not None
    assert latest.async_terminal is True
    assert controller_input.get_nonterminal_async_job_id() is None


def test_format_action_completion_and_brain_evidence_with_records():
    from core.graph_response_formatters import format_action_completion_response
    rec_write = ToolExecutionRecord(
        step_id="step-1",
        tool_name="write_file",
        arguments={"path": "workspace/hello.py"},
        result=ToolResult(
            request_id="req-w",
            success=True,
            message="Wrote file",
        ),
        artifacts=(
            ArtifactRecord(
                artifact_id="art-1",
                step_id="step-1",
                path="workspace/hello.py",
                action="created",
            ),
        ),
    )
    completion = format_action_completion_response((rec_write,))
    assert completion is not None
    assert "Implemented the requested changes in workspace/hello.py" in completion
