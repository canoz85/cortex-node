from langgraph.graph import END

from core.protocol.enums import (
    ControllerDecisionType,
)
from core.protocol.models import ControllerDecision, ExecutionState
from core.state import AgentState


def get_controller_decision(state: AgentState) -> ControllerDecision | None:
    decision = state.get("controller_decision")

    if isinstance(decision, ControllerDecision):
        return decision

    return None

def map_controller_decision(
    decision: ControllerDecision,
) -> str:
    """Translate a protocol ControllerDecision into a LangGraph node.

    This function is the only place that knows LangGraph node names.
    The protocol layer must never depend on graph topology.
    """

    match decision.decision_type:

        case ControllerDecisionType.DISPATCH_PLANNER:
            return "planner"

        case ControllerDecisionType.DISPATCH_BRAIN:
            return "brain"

        case ControllerDecisionType.DISPATCH_TOOL_RUNTIME:
            return "tools"

        case ControllerDecisionType.DISPATCH_SUMMARY:
             return END #return "summarize_memory" todo commented for debugging, we are not using summarize_memory node

        case ControllerDecisionType.AWAIT_ASYNC_JOB:
            return END

        case ControllerDecisionType.PAUSE:
            return END

        case ControllerDecisionType.CANCEL:
            return END

        case ControllerDecisionType.TERMINATE:
            return END

    raise ValueError(
        f"Unsupported controller decision: {decision.decision_type}"
    )


def apply_controller_decision_to_state(
    execution_state: ExecutionState,
    decision: ControllerDecision,
) -> ExecutionState:
    """
    Apply a ControllerDecision to the protocol ExecutionState.

    This function is protocol-only. It never mutates AgentState.
    """

    protocol_visible = execution_state.protocol_visible

    #
    # Cursor base (final synchronization happens after plan/step updates)
    #
    cursor = (
        decision.cursor
        if decision.cursor is not None
        else protocol_visible.cursor
    )

    #
    # Active plan
    #
    active_plan = (
        decision.accepted_plan
        if decision.accepted_plan is not None
        else protocol_visible.active_plan
    )

    #
    # Completed steps
    #
    completed_step_ids = protocol_visible.completed_step_ids
    completion_provenance = protocol_visible.completion_provenance

    if decision.completed_step_id is not None:
        if decision.completed_step_id not in completed_step_ids:
            completed_step_ids = (
                *completed_step_ids,
                decision.completed_step_id,
            )

    if decision.completion_evidence is not None:
        provenance = decision.completion_evidence
        if decision.completed_step_id != provenance.step_id:
            raise ValueError("completion provenance does not match completed step")
        scope = (
            provenance.execution_id,
            provenance.plan_id,
            provenance.plan_revision,
            provenance.step_id,
        )
        existing = next((
            item for item in completion_provenance
            if (item.execution_id, item.plan_id, item.plan_revision, item.step_id) == scope
        ), None)
        if existing is not None and existing != provenance:
            raise ValueError("accepted completion provenance cannot change")
        if existing is None:
            completion_provenance = (*completion_provenance, provenance)

    #
    # Pending tool request
    #
    if decision.clear_pending_tool_request:
        pending_tool_request = None
    else:
        pending_tool_request = (
            decision.pending_tool_request
            if decision.pending_tool_request is not None
            else protocol_visible.pending_tool_request
        )

    #
    # Active step
    #
    active_step = None

    should_clear_active_step = decision.clear_active_step

    if should_clear_active_step:
        active_step = None
    else:
        step_id = (
            decision.next_step_id
            or (
                protocol_visible.active_step.step_id
                if protocol_visible.active_step is not None
                else None
            )
        )

        if (
            active_plan is not None
            and step_id is not None
        ):
            active_step = next(
                (
                    step
                    for step in active_plan.steps
                    if step.step_id == step_id
                ),
                None,
            )

    synchronized_step_id = None
    if active_step is not None:
        synchronized_step_id = active_step.step_id
    elif not should_clear_active_step and decision.next_step_id is not None:
        synchronized_step_id = decision.next_step_id

    synchronized_cursor = cursor.model_copy(
        update={
            "phase": cursor.phase,
            "current_worker": decision.next_worker or cursor.current_worker,
            "step_id": synchronized_step_id,
            "plan_revision": (
                active_plan.revision
                if active_plan is not None
                else cursor.plan_revision
            ),
            "step_attempt": (
                active_step.attempt
                if active_step is not None
                else None
            ),
        }
    )
    #
    # Return updated immutable state
    #
    return execution_state.model_copy(
        update={
            "protocol_visible": protocol_visible.model_copy(
                update={
                    "status": (
                        decision.execution_status
                    ),
                    "cancellation_source": (
                        decision.cancellation_source
                        if decision.cancellation_source is not None
                        else protocol_visible.cancellation_source
                    ),
                    "cursor": synchronized_cursor,
                    "active_plan": active_plan,
                    "planning_request": (
                        None if decision.clear_planning_request or decision.terminal
                        else decision.planning_request or protocol_visible.planning_request
                    ),
                    "planning_sequence": (
                        decision.planning_request.sequence if decision.planning_request
                        else protocol_visible.planning_sequence
                    ),
                    "planning_clarification": (
                        None if decision.clear_planning_clarification or decision.terminal
                        else decision.planning_clarification or protocol_visible.planning_clarification
                    ),
                    "active_step": active_step,
                    "pending_tool_request": pending_tool_request,
                    "completed_step_ids": completed_step_ids,
                    "completion_provenance": completion_provenance,
                    "retry": (
                        decision.retry
                        if decision.retry is not None
                        else protocol_visible.retry
                    ),
                }
            ),
            "working": execution_state.working.model_copy(update={"last_tool_result": None})
            if decision.consume_tool_result else execution_state.working,
        }
    )
