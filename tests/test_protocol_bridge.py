from langchain_core.messages import AIMessage, HumanMessage

from core.protocol.bridge import build_controller_input
from core.protocol.enums import BrainOutcome, WorkerRole


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
        "planner_message": "Plan generated successfully.",
        "brain_outcome": "tool_request",
        "brain_message": "Brain requested tool execution.",
        "tool_request": {
            "request_id": "req-1",
            "tool_name": "write_file",
            "arguments": {"path": "workspace/a.py", "content": "print('x')"},
            "requested_by": "brain",
        },
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
