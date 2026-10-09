"""Align provider constraints and structural retry feedback with Controller authority."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from langchain_ollama import ChatOllama
from ollama._types import ChatRequest

from core.planner import PlannerMessage, PlannerService
from core.planner_contract import PlannerInvalidOutputError, PlannerProposal
from core.planner_feedback import PLANNING_FEEDBACK_HEADER, planner_retry_feedback
from core.planner_normalization import MAX_PROPOSED_STEPS, normalize_planner_proposal
from core.planner_provider import LangChainPlannerProvider, _extract_planner_proposal
from core.protocol.enums import ControllerDecisionType as Decision, PlannerOutcome, PlanningFailureCategory
from core.protocol.models import PlannerResult
from test_controller_planning import authorize, controller, result_input
from test_planner_service import FakeProvider, FakeRouter, planner_input


def proposal(count=1, tool="read_file"):
    return {"result": "PLAN_PROPOSED", "objective": "Inspect files", "steps": [
        {"step_id": f"s{i}", "title": "Inspect files", "description": "Inspect files and summarize",
         "primary_tool": tool, "dependencies": []}
        for i in range(count)
    ], "message": ""}


def ollama_provider(*replies):
    requests = []
    responses = iter(replies)

    def chat(**kwargs):
        requests.append(ChatRequest(**deepcopy(kwargs)).model_dump(exclude_none=True))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return iter([{"message": {"role": "assistant", "content": json.dumps(response)}, "done": True}])

    model = ChatOllama(model="gpt-oss:20b", temperature=0)
    model._client = SimpleNamespace(chat=chat)
    return LangChainPlannerProvider(planner_llm=model), requests


def test_prompt_exposes_shared_limit_and_repeated_tool_semantics_without_retry_feedback():
    provider = FakeProvider(proposal())
    planner = PlannerService(provider=provider, router=FakeRouter("info"), mutating_tools={"write_file"})
    result = planner.run(planner_input())
    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    messages = provider.messages[0]
    assert MAX_PROPOSED_STEPS == 4
    assert f"A plan may contain at most {MAX_PROPOSED_STEPS} steps." in messages[0].content
    assert "One logical step may invoke its primary tool repeatedly" in messages[0].content
    assert all(PLANNING_FEEDBACK_HEADER not in message.content for message in messages)
    assert "write_file" not in messages[0].available_tools


@pytest.mark.parametrize("count,expected", [(4, PlannerOutcome.EXECUTION_PLAN), (5, PlannerOutcome.FAILED)])
def test_normalization_keeps_four_step_limit(count, expected):
    result = normalize_planner_proposal(proposal(count), planner_input(), route="info")
    assert result.outcome == expected
    if count == 5:
        assert result.message == "Planner proposal is invalid: PLAN_PROPOSED exceeds the 4-step limit"
        assert result.failure_category == PlanningFailureCategory.INVALID_OUTPUT


@pytest.mark.parametrize("tool", ["read_file", "none", "write_file"])
def test_real_ollama_provider_preserves_authorized_enum_and_validates_output(tool):
    provider, requests = ollama_provider(proposal(tool=tool))
    messages = (PlannerMessage("system", "Plan", "exec", ("read_file", "list_files")),)
    if tool == "read_file":
        assert provider.generate(messages).steps[0].primary_tool == tool
    else:
        with pytest.raises(PlannerInvalidOutputError, match="currently authorized tools"):
            provider.generate(messages)
    assert len(requests) == 1
    schema = requests[0]["format"]
    assert schema["$defs"]["AuthorizedProposedStep"]["properties"]["primary_tool"]["enum"] == [
        "list_files", "read_file",
    ]
    assert schema["properties"]["steps"]["maxItems"] == MAX_PROPOSED_STEPS
    assert schema["additionalProperties"] is False


def test_no_authorized_tools_allows_direct_response_but_no_executable_steps():
    provider, requests = ollama_provider({"result": "NO_PLAN_REQUIRED", "message": "Hello"})
    result = provider.generate((PlannerMessage("system", "Plan"),))
    assert result.message == "Hello"
    schema = requests[0]["format"]
    assert schema["properties"]["steps"]["maxItems"] == 0
    assert "AuthorizedProposedStep" not in schema["$defs"]
    provider, requests = ollama_provider(proposal())
    with pytest.raises(PlannerInvalidOutputError, match="without authorized tools"):
        provider.generate((PlannerMessage("system", "Plan"),))
    assert len(requests) == 1


def test_extraction_revalidates_parsed_domain_proposals_and_normalizer_keeps_membership_guard():
    from core.planner_contract import authorized_planner_schema

    schema = authorized_planner_schema(("read_file",), max_steps=MAX_PROPOSED_STEPS)
    domain = PlannerProposal.model_validate(proposal(tool="none"))
    with pytest.raises(ValueError, match="literal"):
        _extract_planner_proposal({"raw": None, "parsed": domain}, schema)
    result = normalize_planner_proposal(domain, planner_input(), route="info")
    assert result.message == "Planner proposal is invalid: primary_tool 'none' is unknown"


@pytest.mark.parametrize("second_valid", [True, False])
def test_real_provider_retry_is_controller_authorized_bounded_and_preserves_context(second_valid):
    ctrl = controller()
    dispatch, request = authorize(ctrl)
    provider, requests = ollama_provider(proposal(5), proposal(4) if second_valid else proposal(tool="none"))
    router = FakeRouter("info")
    planner = PlannerService(provider=provider, router=router, mutating_tools={"write_file"})
    retrieval_calls = []

    def retrieve(query):
        retrieval_calls.append(query)
        return ("Retrieved background context",)

    first = planner.run(request, retrieve=retrieve)
    assert len(requests) == 1  # No provider-owned retry.
    assert first.failure_category == PlanningFailureCategory.INVALID_OUTPUT
    retry = ctrl.decide(result_input(dispatch, request, first))
    assert retry.decision_type == Decision.DISPATCH_PLANNER
    next_request = retry.planning_request
    assert next_request.attempt == 2 and next_request.max_attempts == request.max_attempts == 2
    assert next_request.episode_id == request.episode_id
    assert next_request.operation == request.operation
    assert next_request.capabilities == request.capabilities
    assert next_request.context.user_request == request.context.user_request
    assert next_request.context.recent_history == request.context.recent_history
    assert next_request.context.planner_memory_context == request.context.planner_memory_context
    assert next_request.planner_route == first.planner_route == "info"
    assert next_request.reason == request.reason == ""
    assert next_request.suggested_constraints == request.suggested_constraints == ()
    assert next_request.failure_json is None
    # The existing protocol shape supports this context, including serialization.
    assert type(next_request).model_validate_json(next_request.model_dump_json()) == next_request

    second = planner.run(next_request, retrieve=retrieve)
    decision = ctrl.decide(result_input(retry, next_request, second))
    assert decision.decision_type == (Decision.DISPATCH_BRAIN if second_valid else Decision.TERMINATE)
    if not second_valid:
        assert decision.reason == "planning_retry_exhausted"
        assert decision.planning_request is None
    assert len(requests) == 2
    assert len(router.calls) == 1
    assert retrieval_calls == []
    first_messages, retry_messages = (record["messages"] for record in requests)
    assert all(PLANNING_FEEDBACK_HEADER not in item["content"] for item in first_messages)
    feedback = [item for item in retry_messages if item["content"].startswith(PLANNING_FEEDBACK_HEADER)]
    assert len(feedback) == 1
    assert json.loads(feedback[0]["content"].split("\n")[1]) == {
        "source": "structural_validation", "code": "invalid_output",
        "message": "PLAN_PROPOSED exceeds the 4-step limit",
    }
    assert "return one corrected Planner proposal" in feedback[0]["content"]
    assert [item for item in retry_messages if item not in feedback] == first_messages
    assert requests[0]["format"] == requests[1]["format"]


def test_provider_failure_retry_remains_unchanged_without_structural_feedback():
    ctrl = controller()
    dispatch, request = authorize(ctrl)
    provider, requests = ollama_provider(RuntimeError("offline"), RuntimeError("offline"))
    planner = PlannerService(provider=provider, router=FakeRouter("info"), mutating_tools=set())
    first = planner.run(request)
    assert first.failure_category == PlanningFailureCategory.PROVIDER_FAILURE
    retry = ctrl.decide(result_input(dispatch, request, first))
    assert retry.planning_request.context == request.context
    second = planner.run(retry.planning_request)
    terminal = ctrl.decide(result_input(retry, retry.planning_request, second))
    assert terminal.reason == "planning_retry_exhausted"
    assert len(requests) == 2
    assert requests[0]["messages"] == requests[1]["messages"]


def test_normalizer_rejection_reaches_retry_provider_once_without_rag():
    ctrl = controller()
    dispatch, request = authorize(ctrl)
    provider = FakeProvider(proposal(5))
    planner = PlannerService(provider=provider, router=FakeRouter("info"), mutating_tools=set())
    first = planner.run(request)
    assert first.message == "Planner proposal is invalid: PLAN_PROPOSED exceeds the 4-step limit"
    retry = ctrl.decide(result_input(dispatch, request, first))
    provider.content = proposal()
    second = planner.run(retry.planning_request)
    assert second.outcome == PlannerOutcome.EXECUTION_PLAN
    assert len(provider.messages) == 2
    feedback = [message for message in provider.messages[1]
                if message.content.startswith(PLANNING_FEEDBACK_HEADER)]
    assert len(feedback) == 1
    assert "PLAN_PROPOSED exceeds the 4-step limit" in feedback[0].content
    assert provider.messages[1][-1].content == request.context.user_request


def test_retry_feedback_passes_structural_errors_and_excludes_exception_details():
    def failure(message):
        return PlannerResult(outcome=PlannerOutcome.FAILED, request_id="request",
                             failure_category=PlanningFailureCategory.INVALID_OUTPUT, message=message)

    diagnostic = planner_retry_feedback(failure(
        "Planner proposal is invalid: proposed step dependency graph is cyclic",
    ))
    assert "dependency graph is cyclic" in diagnostic.message
    diagnostic = planner_retry_feedback(failure(
        "Planner output is invalid (PlannerInvalidOutputError): Traceback:\nprivate rejected proposal",
    ))
    assert "Traceback" not in diagnostic.message and "private rejected proposal" not in diagnostic.message
    assert len(diagnostic.message) <= 240
