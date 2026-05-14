from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from core.models import TokenUsage


class AgentState(TypedDict, total=False):
    """Shared state for the CortexNode reasoning loop."""

    messages: Annotated[list[BaseMessage], add_messages]
    last_tool_output: str
    last_tool_signature: str
    last_tool_success: bool
    repeat_fail_count: int
    tool_text_retry_used: bool
    steps: int
    token_usage: TokenUsage
    plan: str
