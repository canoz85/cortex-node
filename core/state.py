from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from core.models import TokenUsage


class AgentState(TypedDict, total=False):
    """Shared state for the CortexNode reasoning loop."""

    messages: Annotated[list[BaseMessage], add_messages]
    last_tool_output: str
    steps: int
    token_usage: TokenUsage
