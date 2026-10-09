"""Offline checks of the live Planner harness's display and lifecycle boundary."""

from types import SimpleNamespace

import pytest
from core.debug import log_llm_exchange
from core.graph_authorization import require_planner_authorization
from core.planner_limits import PLANNER_LENGTH_DIAGNOSTIC
from core.protocol.enums import PlannerOutcome, PlanningFailureCategory
from core.protocol.models import ExecutionPlan, ExecutionStep, PlannerResult, PlanningCapabilities
from tests.live import test_planner_stability as harness


@pytest.mark.parametrize("outcome,category,label", [
    (PlannerOutcome.DIRECT_RESPONSE, None, "NO_PLAN_REQUIRED"),
    (PlannerOutcome.CLARIFICATION_REQUIRED, None, "NEEDS_INPUT"),
    (PlannerOutcome.FAILED, PlanningFailureCategory.UNPLANNABLE, "PLANNING_FAILED"),
    (PlannerOutcome.FAILED, PlanningFailureCategory.INVALID_OUTPUT, "INVALID_OUTPUT"),
    (PlannerOutcome.FAILED, PlanningFailureCategory.PROVIDER_FAILURE, "PROVIDER_FAILURE"),
])
def test_non_plan_results_print_production_outcome_and_full_message(outcome, category, label, capsys):
    message = "A full message\nwith a second line and Türkçe characters."
    result = PlannerResult(request_id="display", outcome=outcome, failure_category=category, message=message)
    harness._print_result(result)
    output = capsys.readouterr().out
    assert f"outcome: {label}" in output
    assert message in output
    if category:
        assert f"category: {category.value}" in output


def test_repeated_observations_preserve_controller_retry_and_stop_before_workers(tmp_path, monkeypatch, capsys):
    path = tmp_path / "exchanges.jsonl"
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", str(path))
    prompt = "An arbitrary prompt: 20 tane Michael Jackson şarkısı yaz"
    title = "Full step title"
    description = "A full description\n" + "Detail without truncation. " * 60
    requests = []

    def scripted_planner_node(state):
        request = require_planner_authorization(state)
        requests.append(request)
        if request.attempt == 1:
            assert request.feedback is None and request.planner_route is None
            result = PlannerResult(
                request_id=request.request_id, outcome=PlannerOutcome.FAILED, planner_route="info",
                failure_category=PlanningFailureCategory.INVALID_OUTPUT,
                message=f"Planner output is invalid (PlannerInvalidOutputError): {PLANNER_LENGTH_DIAGNOSTIC}",
            )
        else:
            assert request.attempt == request.max_attempts == 2
            assert request.feedback.message == PLANNER_LENGTH_DIAGNOSTIC
            assert request.planner_route == "info"
            result = PlannerResult(
                request_id=request.request_id, outcome=PlannerOutcome.EXECUTION_PLAN, planner_route="info",
                proposed_plan=ExecutionPlan(plan_id=f"{request.identity.execution_id}:plan", objective=prompt,
                    available_tools=("list_files",), steps=(ExecutionStep(
                        step_id="step1", title=title, description=description, primary_tool="list_files",
                    ),)),
            )
        log_llm_exchange(worker="planner", operation="plan", messages=[],
                         execution_id=request.identity.execution_id,
                         invocation={"attempt": request.attempt}, response={
                             "response_metadata": {"model": "offline-script", "done_reason":
                                                   "length" if request.attempt == 1 else "stop"},
                             "usage_metadata": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                         })
        return {"planner_result": result}

    capabilities = PlanningCapabilities(available_tools=("list_files",))
    observations = [harness._observe_run(scripted_planner_node, capabilities, prompt, path, run)
                    for run in (1, 2)]
    assert len(requests) == 4
    assert [request.context.user_request for request in requests] == [prompt] * 4
    assert requests[0].identity != requests[2].identity
    assert requests[0].episode_id != requests[2].episode_id
    for observation in observations:
        assert observation["controller_continuation"] == "dispatch_brain"
        assert len(observation["attempts"]) == 2
        assert [attempt["done_reason"] for attempt in observation["attempts"]] == ["length", "stop"]
        assert observation["usage"] == {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300}
    output = capsys.readouterr().out
    assert "ATTEMPT 1" in output and "ATTEMPT 2" in output and "FINAL PLANNER RESULT" in output
    assert title in output and description in output and prompt in output
    assert "depends_on: (none)" in output and "tool: list_files" in output
    assert "outcome: PLAN_PROPOSED" in output


def test_live_guard_rejects_fake_provider_and_model():
    model = harness._default_chat_model_factory("gpt-oss:20b", 0)
    service = harness.PlannerService(
        provider=SimpleNamespace(), router=harness.LangChainPlannerRouter(router_llm=model), mutating_tools=set(),
    )
    with pytest.raises(AssertionError, match="Fake Planner provider"):
        harness.assert_live_provider(service)
    service.provider = harness.LangChainPlannerProvider(planner_llm=SimpleNamespace())
    with pytest.raises(AssertionError, match="configured production ChatOllama"):
        harness.assert_live_provider(service)


def test_attempt_diagnostics_exclude_router_and_keep_missing_usage_unknown():
    request = SimpleNamespace(attempt=1, request_id="diagnostics")
    result = PlannerResult(request_id=request.request_id, outcome=PlannerOutcome.DIRECT_RESPONSE, message="Answer")
    record = {"worker": "planner", "operation": "plan", "invocation": {"attempt": 1},
              "response": {"model": "offline-script", "done_reason": "stop"},
              "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}
    router_record = {**record, "worker": "router"}
    metrics = harness._attempt_diagnostics(request, result, 1.5, [router_record, record])
    assert metrics["usage"] == record["usage"] and metrics["length_limit"] is False
    assert metrics["result"] == result.model_dump(mode="json")
    failure = PlannerResult(request_id=request.request_id, outcome=PlannerOutcome.FAILED,
                            failure_category=PlanningFailureCategory.PROVIDER_FAILURE)
    metrics = harness._attempt_diagnostics(request, failure, 1.5, [router_record])
    assert metrics["usage"] is None and metrics["length_limit"] is None
    with pytest.raises(AssertionError, match="Missing production Planner diagnostics"):
        harness._attempt_diagnostics(request, result, 1.5, [router_record])
