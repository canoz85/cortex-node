from langchain_core.messages import AIMessage
from langgraph.graph import END

from core.graph_routing import route_after_brain


def test_route_after_brain_returns_end_when_no_history():
    assert route_after_brain({"messages": [], "steps": 0}) == END


def test_route_after_brain_returns_tools_when_tool_calls_present():
    message = AIMessage(content="", tool_calls=[{"name": "list_files", "args": {}, "id": "1", "type": "tool_call"}])
    assert route_after_brain({"messages": [message], "steps": 1}) == "tools"


def test_route_after_brain_returns_end_when_step_limit_reached():
    message = AIMessage(content="", tool_calls=[{"name": "list_files", "args": {}, "id": "1", "type": "tool_call"}])
    assert route_after_brain({"messages": [message], "steps": 999}) == END
