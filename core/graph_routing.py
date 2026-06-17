from langgraph.graph import END

from core.graph_constants import MAX_REASONING_STEPS
from core.state import AgentState


def route_after_brain(state: AgentState):
    history = state.get("messages", [])
    if not history:
        return END

    if state.get("steps", 0) >= MAX_REASONING_STEPS:
        return END

    last_message = history[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "summarize_memory"