from dataclasses import dataclass

from langgraph.graph import END

from core.protocol.enums import BrainOutcome, ControllerDecisionType, ExecutionPhase, WorkerRole
from core.protocol.models import ControllerDecision, ControllerInput, ExecutionState

from core.graph_constants import MAX_REASONING_STEPS
from core.graph_node_helpers import planner_execution_brief
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
    planner_brief: str = ""  # Added default value for planner_brief

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

def apply_controller_decision_to_state(state: AgentState, decision: ControllerDecision) -> AgentState:
    if not isinstance(state, dict):
        state = dict(state)

    if decision.cursor is not None:
        phase = decision.cursor.phase
        if hasattr(phase, "value"):
            phase_value = phase.value
        else:
            phase_value = phase
        state["phase"] = phase_value

        current_worker = decision.cursor.current_worker
        state["current_worker"] = current_worker.value if hasattr(current_worker, "value") and current_worker is not None else current_worker
        state["steps"] = decision.cursor.controller_iteration if decision.cursor.controller_iteration is not None else state.get("steps", 0)

    if decision.requires_checkpoint:
        run_id = state.get("run_id") or "execution"
        state["checkpoint_id"] = f"{run_id}:checkpoint"

    execution_state = state.get("execution_state")
    if execution_state is None:
        execution_state = ExecutionState(
            protocol_visible={
                "identity": {"execution_id": state.get("run_id") or "execution", "protocol_version": "1.0"},
                "cursor": decision.cursor or {"phase": ExecutionPhase.INITIALIZING},
            },
            working={},
        )
    elif decision.cursor is not None:
        if isinstance(execution_state, ExecutionState):
            execution_state = execution_state.model_copy(
                update={
                    "protocol_visible": execution_state.protocol_visible.model_copy(
                        update={"cursor": decision.cursor}
                    )
                }
            )
        elif isinstance(execution_state, dict):
            protocol_visible = dict(execution_state.get("protocol_visible") or {})
            protocol_visible["cursor"] = decision.cursor
            execution_state = dict(execution_state)
            execution_state["protocol_visible"] = protocol_visible

    state["execution_state"] = execution_state
    return state


def decide_controller_decision(
    state: AgentState,
    *,
    controller_input: ControllerInput | None = None,
) -> ControllerDecision:
    history = state.get("messages", [])
    if not history:
        return ControllerDecision(
            decision_type=ControllerDecisionType.TERMINATE,
            reason="empty_history",
            terminal=True,
        )

    steps = state.get("steps", 0)
    if controller_input is not None and controller_input.cursor is not None:
        cursor_steps = controller_input.cursor.controller_iteration
        if cursor_steps is not None:
            steps = cursor_steps

    if steps >= MAX_REASONING_STEPS:
        return ControllerDecision(
            decision_type=ControllerDecisionType.TERMINATE,
            reason="max_steps",
            terminal=True,
        )

    if controller_input is not None and controller_input.planner_result is not None:
        return ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_BRAIN,
            reason="planner_result_ready",
            next_worker=WorkerRole.BRAIN,
            requires_checkpoint=False,
        )

    if controller_input is not None and controller_input.brain_result is not None:
        if controller_input.brain_result.outcome == BrainOutcome.TOOL_REQUEST:
            return ControllerDecision(
                decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
                reason="tool_request",
                next_worker=WorkerRole.TOOL_RUNTIME,
                requires_checkpoint=False,
            )
        if controller_input.brain_result.outcome == BrainOutcome.REPLAN_REQUEST:
            cursor = None
            if controller_input.cursor is not None:
                cursor = controller_input.cursor.model_copy(update={"phase": "replanning", "current_worker": WorkerRole.PLANNER})
            return ControllerDecision(
                decision_type=ControllerDecisionType.REQUEST_REPLAN,
                reason="replan_request",
                next_worker=WorkerRole.PLANNER,
                requires_checkpoint=True,
                cursor=cursor,
            )
        if controller_input.brain_result.outcome == BrainOutcome.FINAL_ANSWER:
            cursor = None
            if controller_input.cursor is not None:
                cursor = controller_input.cursor.model_copy(update={"phase": "terminating", "current_worker": WorkerRole.SUMMARY})
            return ControllerDecision(
                decision_type=ControllerDecisionType.DISPATCH_SUMMARY,
                reason="final_answer",
                next_worker=WorkerRole.SUMMARY,
                requires_checkpoint=False,
                cursor=cursor,
            )

    return ControllerDecision(
        decision_type=ControllerDecisionType.TERMINATE,
        reason="finalize_turn",
        terminal=True,
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
    #commented
    #return TransitionDecision(next_node="summarize_memory", reason="finalize_turn")


def decide_brain_execution(planner_route: str, plan_text: str) -> BrainExecutionDecision:
    if planner_route.startswith("action"):
        return BrainExecutionDecision(
            action_required=True,
            include_retrieval=True,
            reason="action_route",
            planner_brief=planner_execution_brief(planner_route, plan_text),
        )
    if planner_route.startswith("info"):
        return BrainExecutionDecision(
            action_required=True,
            include_retrieval=False,
            reason="info_route",
        )
    return BrainExecutionDecision(
        action_required=False,
        include_retrieval=False,
        reason="discussion_route",
    )


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
