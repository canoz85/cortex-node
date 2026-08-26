from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from core.models import TokenUsage
from core.protocol.models import BrainResult, ControllerDecision, ExecutionState, PlannerResult, ToolResult


class AgentState(TypedDict, total=False):
    """Shared transport state container for the CortexNode reasoning loop.

    State Isolation Policy (Phase 4):
    - `execution_state` (ExecutionState) is the authoritative, immutable state container.
    - Workers (Planner, Brain, Controller, Capture) read protocol input models assembled
      from `execution_state` and `tool_execution_history` evidence log.
    - Legacy keys are preserved for transport compatibility with graph runners.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    last_tool_output: dict[str, Any] | str
    last_tool_result: ToolResult
    last_tool_signature: str
    last_tool_success: bool
    repeat_fail_count: int
    tool_text_retry_used: bool
    steps: int
    token_usage: TokenUsage
    plan: str
    planner_result: PlannerResult
    planner_domain: str
    planner_confidence: float
    planner_domain_enforced: bool
    rolling_summary: str
    retrieval_messages: list[BaseMessage]
    brain_result: BrainResult
    run_id: str
    execution_state: ExecutionState
    controller_decision: ControllerDecision

