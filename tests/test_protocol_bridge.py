from langchain_core.messages import AIMessage, HumanMessage

from core.protocol.bridge import build_brain_input, build_controller_input, legacy_tool_result_to_model
from core.protocol.enums import BrainOutcome, ExecutionPhase, WorkerRole
from core.protocol.models import (
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
