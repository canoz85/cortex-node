from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.graph_tool_events import (
    current_turn_has_successful_tool_name,
    current_turn_has_successful_tool_result,
    successful_read_file_paths,
)
from core.models import ToolResult


def test_current_turn_success_detects_content_block_tool_message():
    tool_payload = ToolResult(success=True, message="Read file: is_prime.py", data={"path": "is_prime.py", "content": "x"}).to_tool_output()
    history = [
        HumanMessage(content="read is_prime.py"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "is_prime.py"}, "id": "c1", "type": "tool_call"}]),
        ToolMessage(content=[{"type": "text", "text": tool_payload}], tool_call_id="c1"),
    ]

    assert current_turn_has_successful_tool_result(history) is True


def test_successful_read_file_paths_collects_only_successful_reads():
    good_payload = ToolResult(success=True, message="Read file: workspace/is_prime.py", data={"path": "workspace/is_prime.py", "content": "x"}).to_tool_output()
    bad_payload = ToolResult(success=False, message="Error: file missing", data={"path": "workspace/missing.py", "content": ""}).to_tool_output()

    history = [
        HumanMessage(content="read workspace/is_prime.py"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "workspace/is_prime.py"}, "id": "r1", "type": "tool_call"}]),
        ToolMessage(content=good_payload, tool_call_id="r1"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "workspace/missing.py"}, "id": "r2", "type": "tool_call"}]),
        ToolMessage(content=bad_payload, tool_call_id="r2"),
    ]

    assert successful_read_file_paths(history) == ["workspace/is_prime.py"]


def test_current_turn_has_successful_tool_name_matches_specific_tool():
    list_payload = ToolResult(success=True, message="Listing for .", data={"path": ".", "entries": ["a.py"]}).to_tool_output()
    read_payload = ToolResult(success=False, message="Error: missing", data={"path": "b.py", "content": ""}).to_tool_output()
    history = [
        HumanMessage(content="list workspace"),
        AIMessage(content="", tool_calls=[{"name": "list_files", "args": {"path": "."}, "id": "c1", "type": "tool_call"}]),
        ToolMessage(content=list_payload, tool_call_id="c1"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "b.py"}, "id": "c2", "type": "tool_call"}]),
        ToolMessage(content=read_payload, tool_call_id="c2"),
    ]

    assert current_turn_has_successful_tool_name(history, "list_files") is True
    assert current_turn_has_successful_tool_name(history, "read_file") is False
