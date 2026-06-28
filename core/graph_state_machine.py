from dataclasses import dataclass

from langgraph.graph import END

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


def decide_after_brain(state: AgentState) -> TransitionDecision:
    history = state.get("messages", [])
    if not history:
        return TransitionDecision(next_node=END, reason="empty_history")

    if state.get("steps", 0) >= MAX_REASONING_STEPS:
        return TransitionDecision(next_node=END, reason="max_steps")

    last_message = history[-1]
    if getattr(last_message, "tool_calls", None):
        return TransitionDecision(next_node="tools", reason="tool_calls_present")

    return TransitionDecision(next_node="summarize_memory", reason="finalize_turn")


def decide_brain_execution(route: str) -> BrainExecutionDecision:
    if route.startswith("action"):
        return BrainExecutionDecision(
            action_required=True,
            include_retrieval=True,
            reason="action_route",
        )
    if route.startswith("info"):
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
