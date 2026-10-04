from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from core.models import TokenUsage
from core.protocol.models import (
    AsyncJobPolicy,
    BrainResult,
    ControllerDecision,
    ExecutionState,
    FinalizationResult,
    PlannerResult,
    PlannerMemoryContext,
)


class AgentState(TypedDict, total=False):
    """Shared transport state container for the CortexNode reasoning loop.

    State ownership:
    - `execution_state` (ExecutionState) is the authoritative, immutable state container.
    - Workers (Planner, Brain, Controller, Capture) read protocol input models assembled
      from `execution_state` and `tool_execution_history` evidence log.
    - Graph fields carry conversation, observations, and display projections.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str | None
    token_usage: TokenUsage
    planner_result: PlannerResult
    retrieval_messages: list[BaseMessage]
    planner_memory_context: PlannerMemoryContext
    brain_result: BrainResult
    run_id: str
    async_job_policy: AsyncJobPolicy
    execution_state: ExecutionState
    controller_decision: ControllerDecision
    finalization_result: FinalizationResult
    finalization_error: str

