"""Representation adapters from graph transport to typed runtime inputs.

ExecutionState is required and read unchanged. Controller owns lifecycle and
PlanningRequest construction; this module only projects context and observations.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_core.messages import HumanMessage

from core.graph_messages import conversational_messages
from .enums import WorkerRole
from .models import BrainInput, ControllerInput, ControllerDecision, ExecutionContext, ExecutionState

GraphState = Mapping[str, Any]
_DEFAULT_USER_REQUEST = "unspecified request"


def _require_execution_state(state: GraphState) -> ExecutionState:
    execution_state = state.get("execution_state")
    if not isinstance(execution_state, ExecutionState):
        raise ValueError("runtime input requires Controller-owned ExecutionState")
    return execution_state


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(part) for part in content if part is not None]
        return "\n".join(parts)
    if content is not None:
        return str(content)
    return str(message)


def _latest_user_request(messages: Sequence[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = _message_text(msg).strip()
            if text:
                return text
    return _DEFAULT_USER_REQUEST


def build_execution_context(
    state: GraphState,
    *,
    role: WorkerRole = WorkerRole.BRAIN,
) -> ExecutionContext:
    """Project bounded conversational context without authorizing work."""
    protocol = _require_execution_state(state).protocol_visible
    messages = state.get("messages", ())
    retrieval = state.get("retrieval_messages", ())

    resolved_user_request = (
        protocol.original_user_request or state.get("user_input")
        or _latest_user_request(messages)
    )
    retrieval_messages = tuple(_message_text(item) for item in retrieval)
    conversation = conversational_messages(list(messages))
    prior_conversation = list(conversation)
    for index in range(len(prior_conversation) - 1, -1, -1):
        if isinstance(prior_conversation[index], HumanMessage):
            prior_conversation.pop(index)
            break
    recent_history = tuple(_message_text(item) for item in prior_conversation[-32:])

    return ExecutionContext(
        user_request=resolved_user_request,
        retrieval_messages=retrieval_messages,
        recent_history=recent_history,
        clarification=protocol.clarification,
        role=role,
    )


def build_brain_input(state: GraphState) -> BrainInput:
    """Project accepted state and evidence for Brain."""

    execution_state = _require_execution_state(state)
    controller_decision = state.get("controller_decision")

    context = build_execution_context(state)
    protocol = execution_state.protocol_visible
    return BrainInput(
        identity=protocol.identity,
        cursor=protocol.cursor,
        context=context,
        active_plan=protocol.active_plan,
        active_step=protocol.active_step,
        last_tool_result=execution_state.working.last_tool_result,
        tool_execution_history=execution_state.working.tool_execution_history,
        coverage_assessment=execution_state.working.coverage_assessment,
        retry=protocol.retry,
        direct_response=(
            controller_decision.direct_response
            if isinstance(controller_decision, ControllerDecision)
            else False
        ),
    )


def build_controller_input(
    state: GraphState,
) -> ControllerInput:
    """Project durable state and incoming observations for Controller."""

    execution_state = _require_execution_state(state)

    protocol = execution_state.protocol_visible
    working = execution_state.working
    brain_result = state.get("brain_result")
    tool_result = working.last_tool_result
    planner_result = state.get("planner_result")

    context = build_execution_context(state, role=WorkerRole.CONTROLLER)
    return ControllerInput(
        identity=protocol.identity,
        cursor=protocol.cursor,
        context=context,
        user_input=state.get("user_input"),
        planner_memory_context=state.get("planner_memory_context"),
        active_plan=protocol.active_plan,
        active_step=protocol.active_step,
        pending_tool_request=protocol.pending_tool_request,
        tool_request_continuation=protocol.tool_request_continuation,
        planner_result=planner_result,
        brain_result=brain_result,
        tool_result=tool_result,
        retry=protocol.retry,
        async_policy=protocol.async_policy,
        cancel_requested=working.cancel_requested,
        tool_execution_history=working.tool_execution_history,
        coverage_assessment=working.coverage_assessment,
        accepted_requirements=protocol.accepted_requirements,
        planning_request=protocol.planning_request,
        planning_sequence=protocol.planning_sequence,
        planning_clarification=protocol.planning_clarification,
        completed_step_ids=protocol.completed_step_ids,
    )
