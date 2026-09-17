"""
Logging protocol event extraction.

This module converts LangGraph node updates into a single
typed logging event.

It is intentionally the ONLY place that knows how to read
LangGraph update dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from core.protocol.enums import BrainOutcome, ControllerDecisionType
from core.protocol.models import (
    ControllerDecision,
    ExecutionPlan,
    PlannerResult,
    BrainResult,
    FinalizationResult,
    ToolResult,
)



@dataclass(slots=True)
class NodeUpdate:
    """Normalized protocol event for logging/rendering."""

    from_node: str | None
    to_node: str

    @property
    def transition_reason(self) -> str | None:
        if self.planner_result is not None:
            return "Plan ready"

        if self.tool_result is not None:
            return "Tool completed"

        if self.brain_result is not None:
            match self.brain_result.outcome:
                case BrainOutcome.TOOL_REQUEST:
                    return "Tool requested"

                case BrainOutcome.STEP_COMPLETED:
                    return "Step completed"

                case BrainOutcome.FINAL_ANSWER:
                    return "Final answer"

                case BrainOutcome.REPLAN_REQUEST:
                    return "Replan requested"

        return None

    @property
    def transition(self) -> str | None:
        base = (
            self.to_node
            if self.from_node is None
            else f"{self.from_node} -> {self.to_node}"
        )

        reason = self.transition_reason
        if reason:
            return f"{base} ({reason})"

        return base
    
    planner_result: PlannerResult | None = None
    accepted_plan: ExecutionPlan | None = None
    brain_result: BrainResult | None = None
    accepted_completion_message: str = ""
    tool_result: ToolResult | None = None
    tool_name: str | None = None
    finalization_result: FinalizationResult | None = None

    ai_message: BaseMessage | None = None

    rolling_summary: str = ""
    has_summary_update: bool = False


def extract_node_update(
    *,
    from_node: str | None,
    to_node: str,
    value: dict[str, Any],
) -> NodeUpdate:
    """
    Convert a LangGraph node update into a protocol logging event.

    This function intentionally isolates all legacy update parsing.
    """

    # Do not render raw ToolMessage events.
    # capture_tool_output will emit the normalized ToolResult.
    if to_node == "tools":
        return None

    
    execution_state = value.get("execution_state")
    controller_decision = value.get("controller_decision")

    planner_result = value.get("planner_result")
    brain_result = value.get("brain_result")
    finalization_result = value.get("finalization_result")

    ai_message = None

    messages = value.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, BaseMessage):
            ai_message = last

    tool_result = None
    tool_name = None
    is_tool_result_update = (
        to_node == "capture_tool_output"
        or isinstance(ai_message, ToolMessage)
    )
    if is_tool_result_update and execution_state is not None:
        tool_result = execution_state.working.last_tool_result
        history = execution_state.working.tool_execution_history
        if history and history[-1].result == tool_result:
            tool_name = history[-1].tool_name

    if (
        isinstance(controller_decision, ControllerDecision)
        and controller_decision.decision_type
        == ControllerDecisionType.DISPATCH_TOOL_RUNTIME
        and controller_decision.pending_tool_request is not None
        and execution_state is not None
    ):
        history = execution_state.working.tool_execution_history
        if (
            history
            and history[-1].result.request_id
            == controller_decision.pending_tool_request.request_id
        ):
            tool_result = history[-1].result
            tool_name = history[-1].tool_name

    accepted_plan = None
    accepted_completion_message = ""
    if isinstance(controller_decision, ControllerDecision):
        if (
            controller_decision.reason == "Plan accepted."
            and controller_decision.accepted_plan is not None
        ):
            accepted_plan = controller_decision.accepted_plan
        if controller_decision.completion_evidence is not None:
            accepted_completion_message = (
                controller_decision.completion_evidence.summary
            )

    has_summary_update = "rolling_summary" in value

    return NodeUpdate(
        from_node=from_node,
        to_node=to_node,
        planner_result=planner_result,
        accepted_plan=accepted_plan,
        brain_result=brain_result,
        accepted_completion_message=accepted_completion_message,
        tool_result=tool_result,
        tool_name=tool_name,
        finalization_result=(
            finalization_result
            if isinstance(finalization_result, FinalizationResult)
            else None
        ),
        ai_message=ai_message,
        rolling_summary=str(value.get("rolling_summary") or ""),
        has_summary_update=has_summary_update,
    )
