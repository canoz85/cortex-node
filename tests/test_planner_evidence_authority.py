"""Route/outcome compatibility, without model judgment or runtime execution."""

import pytest

from core.planner import PlannerService
from core.planner_contract import PlannerProposal
from core.planner_feedback import planner_retry_feedback
from core.planner_normalization import normalize_planner_proposal
from core.protocol.controller import CortexController
from core.protocol.enums import (
    ControllerDecisionType, ExecutionStatus, PlannerOutcome, PlanningFailureCategory, WorkerRole,
)
from core.protocol.models import (
    ExecutionContext, PlannerMemoryContext, PlannerMemoryFact, PlannerResult, PlanningCapabilities,
)
from test_controller_planning import authorize, result_input
from test_planner_service import FakeProvider, FakeRouter, planner_input


@pytest.mark.parametrize("user_request", [
    "Inspect the current repository and determine whether Planner retry handling supports pagination correctly.",
    "Are there uncommitted changes in the current repository?",
    "Does config.py currently contain DEBUG=true?",
])
@pytest.mark.parametrize("preserved_route", [False, True])
def test_info_cannot_use_background_or_capability_metadata_as_runtime_observation(user_request, preserved_route):
    claim = "The current implementation does not handle pagination or continuation."
    request = planner_input(
        planner_route="info" if preserved_route else None,
        context=ExecutionContext(
            user_request=user_request, role=WorkerRole.PLANNER,
            retrieval_messages=(claim,), recent_history=(claim,),
            planner_memory_context=PlannerMemoryContext(project_facts=(PlannerMemoryFact(
                category="project_fact", text=claim, authority="tool_supported", source_turn_index=0,
            ),)),
        ),
    )
    provider = FakeProvider({"result": "NO_PLAN_REQUIRED", "message": claim})
    router = FakeRouter("info")
    result = PlannerService(provider=provider, router=router, mutating_tools={"write_file"}).run(request)

    assert result.outcome == PlannerOutcome.FAILED
    assert result.failure_category == PlanningFailureCategory.INVALID_OUTPUT
    assert result.planner_route == "info"
    assert result.direct_response_content is None
    assert claim not in result.message
    assert "read-only runtime evidence is required" in result.message
    assert len(provider.messages) == 1  # No extra call inside PlannerService.
    assert router.calls == ([] if preserved_route else [user_request])


def test_normalization_enforces_preserved_info_route_even_if_result_route_differs():
    request = planner_input(planner_route="info")
    proposal = PlannerProposal(result="NO_PLAN_REQUIRED", message="An unsupported live-state claim")
    result = normalize_planner_proposal(proposal, request, route="conversation")
    assert result.failure_category == PlanningFailureCategory.INVALID_OUTPUT
    assert result.direct_response_content is None


@pytest.mark.parametrize("request_route,result_route", [
    (None, "info"), ("info", "info"), ("info", "conversation"), ("info", None),
])
def test_controller_cannot_accept_info_direct_response_bypassing_normalization(request_route, result_route):
    ctrl = CortexController(24)
    dispatch, request = authorize(ctrl)
    request = request.model_copy(update={"planner_route": request_route})
    result = PlannerResult(
        request_id=request.request_id, planner_route=result_route,
        outcome=PlannerOutcome.DIRECT_RESPONSE,
        message="Unsupported current repository claim", direct_response_content="Unsupported current repository claim",
    )
    decision = ctrl.decide(result_input(dispatch, request, result))
    assert decision.decision_type == ControllerDecisionType.TERMINATE
    assert decision.execution_status == ExecutionStatus.FAILED
    assert decision.accepted_direct_response is None
    assert decision.reason.startswith("invalid_planner_direct_response:")


@pytest.mark.parametrize("valid_retry", [True, False])
def test_rejected_info_direct_answer_uses_existing_controller_retry_budget(valid_retry):
    ctrl = CortexController(24, planning_capabilities=PlanningCapabilities(available_tools=("read_file",)))
    dispatch, request = authorize(ctrl)
    provider = FakeProvider({"result": "NO_PLAN_REQUIRED", "message": "Unsupported implementation claim"})
    router = FakeRouter("info")
    service = PlannerService(provider=provider, router=router, mutating_tools=set())
    first = service.run(request)
    retry = ctrl.decide(result_input(dispatch, request, first))
    assert retry.decision_type == ControllerDecisionType.DISPATCH_PLANNER
    assert retry.accepted_direct_response is None
    next_request = retry.planning_request
    assert next_request.attempt == next_request.max_attempts == 2
    assert next_request.planner_route == "info"
    assert next_request.context == request.context
    assert next_request.feedback == planner_retry_feedback(first)
    assert "read-only runtime evidence is required" in next_request.feedback.message

    if valid_retry:
        provider.content = {"result": "PLAN_PROPOSED", "steps": [{
            "step_id": "inspect", "title": "Inspect current source",
            "description": "Read current source and explain the observed implementation evidence",
            "primary_tool": "read_file", "dependencies": [],
        }]}
    second = service.run(next_request)
    decision = ctrl.decide(result_input(retry, next_request, second))
    assert len(provider.messages) == 2
    assert len(router.calls) == 1
    assert decision.accepted_direct_response is None
    if valid_retry:
        assert decision.decision_type == ControllerDecisionType.DISPATCH_BRAIN
        assert decision.accepted_plan.available_tools == ("read_file",)
    else:
        assert decision.decision_type == ControllerDecisionType.TERMINATE
        assert decision.execution_status == ExecutionStatus.FAILED
        assert decision.reason == "planning_retry_exhausted"


@pytest.mark.parametrize("user_request,answer", [
    ("Name 20 Michael Jackson songs.",
     "Thriller, Billie Jean, Beat It, Smooth Criminal, Bad, Black or White, Man in the Mirror, "
     "The Way You Make Me Feel, Don't Stop 'Til You Get Enough, Rock with You, Off the Wall, "
     "Human Nature, Dirty Diana, Remember the Time, Heal the World, Earth Song, You Are Not Alone, "
     "Stranger in Moscow, They Don't Care About Us, Leave Me Alone."),
    ("Given these numbers: 3, 7, 9, which is largest?", "9 is largest."),
    ("Can the read_file capability return total_chars?", "Yes. total_chars is decoded character count, not byte size."),
])
def test_conversation_allows_knowledge_supplied_content_and_capability_contract_answers(user_request, answer):
    # A capability contract describes possible evidence, without observing any file.
    ctrl = CortexController(24, planning_capabilities=PlanningCapabilities(available_tools=("read_file",)))
    dispatch, request = authorize(ctrl)
    request = request.model_copy(update={"context": ExecutionContext(user_request=user_request)})
    provider = FakeProvider({"result": "NO_PLAN_REQUIRED", "message": answer})
    result = PlannerService(provider=provider, router=FakeRouter("conversation"), mutating_tools=set()).run(request)
    assert result.outcome == PlannerOutcome.DIRECT_RESPONSE
    assert result.direct_response_content == answer
    assert "total_chars" in provider.messages[0][0].content
    decision = ctrl.decide(result_input(dispatch, request, result))
    assert decision.execution_status == ExecutionStatus.COMPLETED
    assert decision.accepted_plan is None
    assert decision.accepted_direct_response.content == answer


@pytest.mark.parametrize("outcome,message,expected", [
    ("NEEDS_INPUT", "Which repository should be inspected?", PlannerOutcome.CLARIFICATION_REQUIRED),
    ("PLANNING_FAILED", "No authorized capability can inspect the requested device.", PlannerOutcome.FAILED),
])
def test_info_can_still_request_input_or_report_unavailable_capability(outcome, message, expected):
    result = normalize_planner_proposal({"result": outcome, "message": message}, planner_input(), route="info")
    assert result.outcome == expected
    if expected == PlannerOutcome.FAILED:
        assert result.failure_category == PlanningFailureCategory.UNPLANNABLE
