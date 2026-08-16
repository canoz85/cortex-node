from dataclasses import dataclass

from langgraph.graph import END

from core.protocol.enums import BrainOutcome, ControllerDecisionType, ExecutionPhase, WorkerRole
from core.protocol.models import BrainInput, ControllerDecision, ControllerInput, ExecutionPlan, ExecutionState, PlannerResult

from core.graph_constants import MAX_REASONING_STEPS
from core.state import AgentState


@dataclass(frozen=True)
class TransitionDecision:
    next_node: str
    reason: str


@dataclass(frozen=True)
class BrainExecutionDecision:
    action_required: bool
    include_retrieval: bool
    reason: str
    execution_brief: str = ""  
    final_answer: bool = False
    tool_completed: bool = False

@dataclass(frozen=True)
class ActionRecoveryDecision:
    use_recovered_action_response: bool
    use_pseudo_fallback: bool
    finalize_generic_json: bool
    reason: str


@dataclass(frozen=True)
class RepeatedSignatureDecision:
    apply_guard: bool
    request_final_answer: bool
    repeat_reason: str
    reason: str

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

    if decision.completed_step_id is not None:
        if decision.completed_step_id not in completed_step_ids:
            completed_step_ids = (
                *completed_step_ids,
                decision.completed_step_id,
            )

    #
    # Pending tool request
    #
    is_terminating = (
        decision.decision_type == ControllerDecisionType.TERMINATE
        or decision.terminal
    )

    if decision.clear_pending_tool_request or is_terminating:
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

    should_clear_active_step = decision.clear_active_step or is_terminating

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

    synchronized_phase = cursor.phase
    if is_terminating:
        synchronized_phase = ExecutionPhase.TERMINATING
    elif decision.decision_type in {
        ControllerDecisionType.DISPATCH_BRAIN,
        ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
    }:
        synchronized_phase = ExecutionPhase.EXECUTING

    synchronized_step_id = None
    if active_step is not None:
        synchronized_step_id = active_step.step_id
    elif not should_clear_active_step and decision.next_step_id is not None:
        synchronized_step_id = decision.next_step_id

    synchronized_cursor = cursor.model_copy(
        update={
            "phase": synchronized_phase,
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
                    "cursor": synchronized_cursor,
                    "active_plan": active_plan,
                    "active_step": active_step,
                    "pending_tool_request": pending_tool_request,
                    "completed_step_ids": completed_step_ids,
                }
            ),
        }
    )

def decide_after_brain(
    state: AgentState,
    *,
    controller_input: ControllerInput | None = None,
) -> TransitionDecision:
    history = state.get("messages", [])
    if not history:
        return TransitionDecision(next_node=END, reason="empty_history")

    protocol_steps = (
        controller_input.cursor.controller_iteration
        if controller_input is not None
        else None
    )
    steps = protocol_steps if protocol_steps is not None else state.get("steps", 0)

    if steps >= MAX_REASONING_STEPS:
        return TransitionDecision(next_node=END, reason="max_steps")

    if controller_input is not None and controller_input.brain_result is not None:
        if controller_input.brain_result.outcome == BrainOutcome.TOOL_REQUEST:
            return TransitionDecision(next_node="tools", reason="protocol_tool_request")

    last_message = history[-1]
    if getattr(last_message, "tool_calls", None):
        return TransitionDecision(next_node="tools", reason="tool_calls_present")

    return TransitionDecision(next_node=END, reason="finalize_turn")
    #commented for now..
    #return TransitionDecision(next_node="summarize_memory", reason="finalize_turn")


def decide_brain_execution(
    brain_input: BrainInput,
) -> BrainExecutionDecision:
    """
    Decide whether the Brain should execute tools or respond directly.

    The Brain operates entirely from the protocol contract
    (BrainInput), not from legacy state.
    """

    if brain_input.active_plan is None:
        return BrainExecutionDecision(
            action_required=False,
            include_retrieval=False,
            reason="direct_response",
            execution_brief="",
            final_answer=False
        )

    if brain_input.active_step is None:
        return BrainExecutionDecision(
            action_required=False,
            include_retrieval=False,
            final_answer=True,
            reason="final_answer",
            execution_brief="",
        )

    if brain_input.last_tool_result is not None:
        return BrainExecutionDecision(
            action_required=True,
            include_retrieval=False,
            tool_completed=True,
            reason="tool_result",
            execution_brief="",
        )

    return BrainExecutionDecision(
        action_required=True,
        include_retrieval=False,
        reason="execution_plan",
        execution_brief=_build_brain_execution_brief(brain_input),
    )


def _build_brain_execution_brief(
    brain_input: BrainInput,
) -> str:
    """
    Build the execution instructions passed to the Brain LLM.

    The planner owns the plan.
    The controller owns progression through the plan.
    The Brain only performs the current step.
    """

    plan = brain_input.active_plan
    current_step = brain_input.active_step

    if plan is None or not plan.steps:
        return ""

    lines: list[str] = [
        "Execution plan:",
    ]

    for index, step in enumerate(plan.steps, start=1):
        marker = ">>" if current_step and step.step_id == current_step.step_id else "  "

        lines.append(
            f"{marker} {index}. {step.title} – {step.description}"
        )

    rendered = "\n".join(lines)

    if len(rendered) > 2000:
        rendered = rendered[:2000] + "..."

    return rendered


def decide_action_recovery(
    *,
    action_required: bool,
    recovered_action_response_exists: bool,
    pseudo_tool_response_detected: bool,
    generic_json_tool_response_detected: bool,
) -> ActionRecoveryDecision:
    if not action_required:
        return ActionRecoveryDecision(
            use_recovered_action_response=False,
            use_pseudo_fallback=False,
            finalize_generic_json=False,
            reason="non_action_route",
        )

    return ActionRecoveryDecision(
        use_recovered_action_response=recovered_action_response_exists,
        use_pseudo_fallback=(not recovered_action_response_exists and pseudo_tool_response_detected),
        finalize_generic_json=generic_json_tool_response_detected,
        reason="action_route_recovery",
    )


def should_retry_after_empty_response(is_empty: bool) -> bool:
    return is_empty


def should_fallback_after_empty_response(is_empty: bool) -> bool:
    return is_empty


def decide_repeated_signature(
    *,
    action_required: bool,
    has_last_tool_signature: bool,
    response_repeats_signature: bool,
    last_tool_success: bool,
    corrected_repeats_signature: bool,
) -> RepeatedSignatureDecision:
    if not (action_required and has_last_tool_signature):
        return RepeatedSignatureDecision(
            apply_guard=False,
            request_final_answer=False,
            repeat_reason="",
            reason="no_signature_guard",
        )

    if not response_repeats_signature:
        return RepeatedSignatureDecision(
            apply_guard=False,
            request_final_answer=False,
            repeat_reason="",
            reason="signature_not_repeated",
        )

    repeat_reason = "already succeeded" if last_tool_success else "already failed"
    request_final = bool(last_tool_success and corrected_repeats_signature)
    return RepeatedSignatureDecision(
        apply_guard=True,
        request_final_answer=request_final,
        repeat_reason=repeat_reason,
        reason=("repeat_signature_force_finalize" if request_final else "repeat_signature_request_correction"),
    )
