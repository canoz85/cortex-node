"""P5 Controller-owned Planner outcome lifecycle."""

from core.graph_state_machine import apply_controller_decision_to_state
from core.protocol.controller import CortexController
from core.protocol.enums import (
    ControllerDecisionType, ExecutionPhase, ExecutionStatus, PlannerOutcome,
    PlanningFailureCategory, PlanningOperation,
)
from core.protocol.models import (
    ControllerInput, ExecutionContext, ExecutionCursor, ExecutionIdentity,
    ExecutionPlan, ExecutionState, ExecutionStep, PlannerResult, ProtocolVisibleState,
)


IDENTITY = ExecutionIdentity(execution_id="p5", protocol_version="1")


def context(*, text="Do the work", count=1):
    return ExecutionContext(user_request=text, user_message_count=count)


def initial_input(**updates):
    values = dict(identity=IDENTITY, cursor=ExecutionCursor(), context=context())
    values.update(updates)
    return ControllerInput(**values)


def authorize(ctrl):
    decision = ctrl.decide(initial_input())
    return decision, decision.planning_request


def result_input(dispatch, request, result, **updates):
    values = dict(
        identity=IDENTITY, cursor=dispatch.cursor, context=context(),
        planning_request=request, planning_sequence=request.sequence,
        planner_result=result,
    )
    values.update(updates)
    return ControllerInput(**values)


def test_create_plan_and_no_plan_have_distinct_terminal_semantics():
    ctrl = CortexController(20)
    dispatch, request = authorize(ctrl)
    plan = ExecutionPlan(plan_id="fresh", steps=(ExecutionStep(step_id="s", title="Work"),))
    accepted = ctrl.decide(result_input(dispatch, request, PlannerResult(
        outcome=PlannerOutcome.EXECUTION_PLAN, request_id=request.request_id, proposed_plan=plan,
    )))
    assert accepted.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert accepted.accepted_plan == plan
    assert accepted.clear_planning_request

    dispatch, request = authorize(ctrl)
    no_plan = ctrl.decide(result_input(dispatch, request, PlannerResult(
        outcome=PlannerOutcome.DIRECT_RESPONSE, request_id=request.request_id, message="No tools required",
    )))
    assert no_plan.decision_type == ControllerDecisionType.DISPATCH_SUMMARY
    assert no_plan.execution_status == ExecutionStatus.COMPLETED
    assert no_plan.accepted_plan is None and no_plan.terminal
    assert no_plan.clear_planning_request


def test_needs_input_pauses_and_new_user_input_authorizes_a_new_episode():
    ctrl = CortexController(20)
    dispatch, request = authorize(ctrl)
    paused = ctrl.decide(result_input(dispatch, request, PlannerResult(
        outcome=PlannerOutcome.CLARIFICATION_REQUIRED, request_id=request.request_id,
        message="Which target should be inspected?",
    )))
    assert paused.decision_type == ControllerDecisionType.PAUSE
    assert paused.cursor.phase == ExecutionPhase.WAITING
    assert paused.planning_clarification.source_request_id == request.request_id
    assert paused.clear_planning_request

    state = apply_controller_decision_to_state(
        ExecutionState(protocol_visible=ProtocolVisibleState(
            identity=IDENTITY, cursor=dispatch.cursor,
            planning_request=request, planning_sequence=request.sequence,
        )), paused,
    )
    restored = ExecutionState.model_validate_json(state.model_dump_json())
    marker = restored.protocol_visible.planning_clarification
    assert restored.protocol_visible.planning_request is None
    assert marker.prompt == "Which target should be inspected?"

    waiting = ctrl.decide(initial_input(
        cursor=restored.protocol_visible.cursor,
        planning_sequence=request.sequence, planning_clarification=marker,
    ))
    assert waiting.decision_type == ControllerDecisionType.PAUSE
    assert waiting.planning_request is None

    resumed = ctrl.decide(initial_input(
        cursor=restored.protocol_visible.cursor,
        context=context(text="The src directory", count=2),
        planning_sequence=request.sequence, planning_clarification=marker,
    ))
    new_request = resumed.planning_request
    assert resumed.decision_type == ControllerDecisionType.DISPATCH_PLANNER
    assert resumed.clear_planning_clarification
    assert new_request.request_id != request.request_id
    assert new_request.sequence == request.sequence + 1
    assert new_request.episode_id != request.episode_id
    assert new_request.context.user_request == "Do the work"
    assert new_request.context.clarification == "The src directory"


def test_retryable_failures_have_two_attempts_and_unplannable_does_not_retry():
    ctrl = CortexController(20)
    dispatch, request = authorize(ctrl)
    retry = ctrl.decide(result_input(dispatch, request, PlannerResult(
        outcome=PlannerOutcome.FAILED, request_id=request.request_id,
        failure_category=PlanningFailureCategory.INVALID_OUTPUT,
        message="bad schema",
    )))
    assert retry.decision_type == ControllerDecisionType.DISPATCH_PLANNER
    assert retry.planning_request.episode_id == request.episode_id
    assert retry.planning_request.attempt == 2
    assert retry.planning_request.sequence == 2
    assert not retry.clear_planning_request

    exhausted = ctrl.decide(result_input(
        retry, retry.planning_request, PlannerResult(
            outcome=PlannerOutcome.FAILED, request_id=retry.planning_request.request_id,
            failure_category=PlanningFailureCategory.PROVIDER_FAILURE,
            message="provider unavailable",
        ),
    ))
    assert exhausted.decision_type == ControllerDecisionType.TERMINATE
    assert exhausted.reason == "planning_retry_exhausted"

    dispatch, request = authorize(ctrl)
    unplannable = ctrl.decide(result_input(dispatch, request, PlannerResult(
        outcome=PlannerOutcome.FAILED, request_id=request.request_id,
        failure_category=PlanningFailureCategory.UNPLANNABLE,
        message="capability unavailable",
    )))
    assert unplannable.decision_type == ControllerDecisionType.TERMINATE
    assert unplannable.reason == "unplannable"


def test_planner_result_binding_rejects_stale_and_unbound_results():
    ctrl = CortexController(20)
    dispatch, request = authorize(ctrl)
    result = PlannerResult(
        outcome=PlannerOutcome.DIRECT_RESPONSE, request_id="different",
    )
    try:
        ctrl.decide(result_input(dispatch, request, result).model_copy(update={
            "planner_result": result,
        }))
    except ValueError as exc:
        assert "request identity" in str(exc)
    else:
        raise AssertionError("stale PlannerResult was accepted")

    try:
        ctrl.decide(initial_input(planner_result=result))
    except ValueError as exc:
        assert "pending PlanningRequest" in str(exc)
    else:
        raise AssertionError("unbound PlannerResult was accepted")


def test_revise_no_plan_is_rejected_and_keeps_accepted_plan():
    plan = ExecutionPlan(plan_id="p", revision=2, steps=(ExecutionStep(step_id="s", title="Work"),))
    ctrl = CortexController(20)
    base = initial_input(active_plan=plan)
    request = ctrl._build_planning_request(
        base, operation=PlanningOperation.REVISE,
        trigger="brain_requested", reason="strategy failed",
    )
    decision = ctrl.decide(initial_input(
        cursor=ExecutionCursor(phase=ExecutionPhase.REPLANNING, plan_revision=2),
        active_plan=plan, planning_request=request, planning_sequence=request.sequence,
        planner_result=PlannerResult(
            outcome=PlannerOutcome.DIRECT_RESPONSE, request_id=request.request_id,
        ),
    ))
    assert decision.decision_type == ControllerDecisionType.TERMINATE
    state = apply_controller_decision_to_state(
        ExecutionState(protocol_visible=ProtocolVisibleState(
            identity=IDENTITY, cursor=ExecutionCursor(plan_revision=2), active_plan=plan,
            planning_request=request, planning_sequence=request.sequence,
        )), decision,
    )
    assert state.protocol_visible.active_plan == plan
