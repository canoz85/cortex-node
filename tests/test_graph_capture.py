from langchain_core.messages import AIMessage, ToolMessage

from core.graph_capture import create_capture_tool_output_node
from core.models import ToolResult


def test_capture_tool_output_with_dict_payload_and_signature():
    capture_node = create_capture_tool_output_node()
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {"path": "a.py"}, "id": "call-1", "type": "tool_call"}],
    )
    tool_message = ToolMessage(
        content=ToolResult(success=True, message="ok", data={"exit_code": 0, "stdout": "x", "stderr": ""}).to_tool_output(),
        tool_call_id="call-1",
    )

    state = {"messages": [ai, tool_message], "last_tool_signature": "", "last_tool_success": True, "repeat_fail_count": 0}
    result = capture_node(state)

    assert result["last_tool_success"] is True
    assert result["last_tool_signature"].startswith("run_python:")
    assert isinstance(result["last_tool_output"], dict)


def test_capture_tool_output_increments_repeat_fail_count_for_same_failed_signature():
    capture_node = create_capture_tool_output_node()
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {"path": "a.py"}, "id": "call-1", "type": "tool_call"}],
    )
    tool_message = ToolMessage(
        content=ToolResult(success=False, message="failed", data={"exit_code": 1, "stdout": "", "stderr": "err"}).to_tool_output(),
        tool_call_id="call-1",
    )

    state = {
        "messages": [ai, tool_message],
        "last_tool_signature": 'run_python:{"path": "a.py"}',
        "last_tool_success": False,
        "repeat_fail_count": 2,
    }
    result = capture_node(state)

    assert result["last_tool_success"] is False
    assert result["repeat_fail_count"] == 3


def test_capture_tool_output_preserves_previous_when_last_message_not_tool():
    capture_node = create_capture_tool_output_node()
    state = {
        "messages": [AIMessage(content="no tool")],
        "last_tool_output": {"message": "old", "success": True},
        "last_tool_signature": "sig",
        "last_tool_success": True,
        "repeat_fail_count": 7,
    }

    result = capture_node(state)

    assert result["last_tool_output"] == {"message": "old", "success": True}
    assert result["last_tool_signature"] == "sig"
    assert result["last_tool_success"] is True
    assert result["repeat_fail_count"] == 7


def test_capture_tool_output_parses_content_block_tool_message():
    capture_node = create_capture_tool_output_node()
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "is_prime.py"}, "id": "call-2", "type": "tool_call"}],
    )
    tool_payload = ToolResult(success=True, message="Read file: is_prime.py", data={"path": "is_prime.py", "content": "print('ok')"}).to_tool_output()
    tool_message = ToolMessage(
        content=[{"type": "text", "text": tool_payload}],
        tool_call_id="call-2",
    )

    state = {"messages": [ai, tool_message], "last_tool_signature": "", "last_tool_success": True, "repeat_fail_count": 0}
    result = capture_node(state)

    assert result["last_tool_success"] is True
    assert result["last_tool_signature"].startswith("read_file:")


def test_capture_tool_output_parses_content_block_with_content_key():
    capture_node = create_capture_tool_output_node()
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "is_prime.py"}, "id": "call-3", "type": "tool_call"}],
    )
    tool_payload = ToolResult(success=True, message="Read file: is_prime.py", data={"path": "is_prime.py", "content": "print('ok')"}).to_tool_output()
    tool_message = ToolMessage(
        content=[{"type": "text", "content": tool_payload}],
        tool_call_id="call-3",
    )

    state = {"messages": [ai, tool_message], "last_tool_signature": "", "last_tool_success": True, "repeat_fail_count": 0}
    result = capture_node(state)

    assert result["last_tool_success"] is True
    assert result["last_tool_signature"].startswith("read_file:")
