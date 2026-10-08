"""Bound Planner decoding through the real adapter, without live generation."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from ollama._types import ChatRequest

from core.graph import _default_chat_model_factory
from core.planner import PlannerMessage, PlannerService
from core.planner_contract import PlannerInvalidOutputError, PlannerProposal
from core.planner_limits import MAX_PLANNER_GENERATION_TOKENS, PLANNER_LENGTH_DIAGNOSTIC
from core.planner_provider import LangChainPlannerProvider, _extract_planner_proposal
from core.planner_routing import LangChainPlannerRouter
from core.protocol.enums import ControllerDecisionType as Decision, PlannerOutcome, PlanningFailureCategory
from test_controller_planning import authorize, controller, result_input
from test_planner_service import FakeRouter


PROPOSAL = {
    "result": "PLAN_PROPOSED", "objective": "Discover and inspect files", "message": "",
    "steps": [
        {"step_id": "discover", "title": "Discover files", "description": "Report discovered file paths",
         "primary_tool": "list_files", "dependencies": []},
        {"step_id": "inspect", "title": "Inspect files", "description": "Read files and summarize their purpose",
         "primary_tool": "read_file", "dependencies": ["discover"]},
    ],
}


def response(content="", reason="length", output_tokens=MAX_PLANNER_GENERATION_TOKENS):
    return {
        "message": {"role": "assistant", "content": content, "thinking": "private reasoning"},
        "done": True, "done_reason": reason, "model": "gpt-oss:20b",
        "prompt_eval_count": 3000, "eval_count": output_tokens,
    }


def scripted_model(*responses):
    replies = iter(responses)
    requests = []

    def chat(**kwargs):
        requests.append(ChatRequest(**deepcopy(kwargs)).model_dump(exclude_none=True))
        return iter([next(replies)])

    model = _default_chat_model_factory("gpt-oss:20b", 0)
    model._client = SimpleNamespace(chat=chat)
    return model, requests


def test_budget_reaches_serialized_ollama_options_and_keeps_router_and_brain_configuration():
    model, requests = scripted_model(
        response(json.dumps(PROPOSAL), "stop", MAX_PLANNER_GENERATION_TOKENS // 3),
        response('{"route":"info"}', "stop", 10),
    )
    original = model.model_dump()
    brain_model = _default_chat_model_factory("brain-model", 0)
    brain_before = brain_model._chat_params([HumanMessage(content="Inspect")])
    provider = LangChainPlannerProvider(planner_llm=model)
    result = provider.generate((PlannerMessage("system", "Plan", available_tools=("list_files", "read_file")),))
    assert len(result.steps) == 2
    assert requests[0]["options"] == {"temperature": 0.0, "num_predict": MAX_PLANNER_GENERATION_TOKENS}
    assert "think" not in requests[0] and requests[0]["stream"] is True
    assert "tools" not in requests[0] and isinstance(requests[0]["format"], dict)
    assert provider.planner_llm is not model
    assert model.model_dump() == original
    assert LangChainPlannerRouter(router_llm=model).route("Inspect").route == "info"
    assert requests[1]["options"] == {"temperature": 0.0}
    assert brain_model._chat_params([HumanMessage(content="Inspect")]) == brain_before
    assert brain_model.num_predict is None


@pytest.mark.parametrize("parsed", [None, PlannerProposal.model_validate(PROPOSAL)])
@pytest.mark.parametrize("reason_field", ["done_reason", "finish_reason"])
def test_length_response_is_explicit_invalid_output_even_if_parser_repairs_partial_json(parsed, reason_field):
    raw = AIMessage(content=json.dumps(PROPOSAL), response_metadata={reason_field: "length"})
    with pytest.raises(PlannerInvalidOutputError, match="permitted generation limit"):
        _extract_planner_proposal({"raw": raw, "parsed": parsed, "parsing_error": ValueError("private parser text")})


def test_visible_json_without_structured_extraction_never_becomes_text_fallback():
    raw = AIMessage(content=json.dumps(PROPOSAL), response_metadata={"done_reason": "stop"})
    with pytest.raises(PlannerInvalidOutputError, match="no valid proposal"):
        _extract_planner_proposal({"raw": raw, "parsed": None})


@pytest.mark.parametrize("second_succeeds", [True, False])
def test_length_then_second_attempt_uses_only_controller_budget_and_logs_attempts(
    second_succeeds, monkeypatch, tmp_path,
):
    path = tmp_path / "planner-budget.jsonl"
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", str(path))
    normal_output_tokens = MAX_PLANNER_GENERATION_TOKENS // 3
    model, requests = scripted_model(
        response(),
        response(json.dumps(PROPOSAL), "stop", normal_output_tokens) if second_succeeds else response(),
    )
    ctrl = controller()
    dispatch, request = authorize(ctrl)
    router = FakeRouter("info")
    planner = PlannerService(provider=LangChainPlannerProvider(planner_llm=model), router=router,
                             mutating_tools={"write_file"})
    first = planner.run(request)
    assert first.failure_category == PlanningFailureCategory.INVALID_OUTPUT
    assert first.proposed_plan is None and len(requests) == 1
    retry = ctrl.decide(result_input(dispatch, request, first))
    assert retry.decision_type == Decision.DISPATCH_PLANNER
    next_request = retry.planning_request
    assert next_request.attempt == next_request.max_attempts == 2
    assert next_request.feedback.message == PLANNER_LENGTH_DIAGNOSTIC
    assert next_request.context.retrieval_messages == request.context.retrieval_messages
    second = planner.run(next_request)
    accepted = ctrl.decide(result_input(retry, next_request, second))
    if second_succeeds:
        assert second.outcome == PlannerOutcome.EXECUTION_PLAN
        assert accepted.decision_type == Decision.DISPATCH_BRAIN
        assert accepted.next_step_id == "discover"
    else:
        assert second.failure_category == PlanningFailureCategory.INVALID_OUTPUT
        assert accepted.decision_type == Decision.TERMINATE
        assert accepted.reason == "planning_retry_exhausted"
    assert len(requests) == 2 and len(router.calls) == 1
    assert requests[0]["format"] == requests[1]["format"]
    assert all(r["options"]["num_predict"] == MAX_PLANNER_GENERATION_TOKENS for r in requests)
    prompt = requests[0]["messages"][0]["content"]
    assert requests[1]["messages"][0]["content"] == prompt
    assert "AVAILABLE CAPABILITIES" in prompt
    assert "total_chars counts decoded characters, not byte size" in prompt
    assert '"pagination"' in prompt and '"outputs"' in prompt and '"limits"' in prompt
    assert "write_file" not in prompt
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert [r["invocation"] for r in records] == [
        {"num_predict": MAX_PLANNER_GENERATION_TOKENS, "attempt": n} for n in (1, 2)
    ]
    assert records[0]["response"]["done_reason"] == "length"
    assert records[0]["response"]["content"] == ""
    assert records[0]["usage"]["output_tokens"] == MAX_PLANNER_GENERATION_TOKENS
    assert records[0]["usage"]["total_tokens"] == 3000 + MAX_PLANNER_GENERATION_TOKENS
    if second_succeeds:
        assert records[1]["usage"]["output_tokens"] == normal_output_tokens < MAX_PLANNER_GENERATION_TOKENS
    assert "private reasoning" not in path.read_text(encoding="utf-8")
