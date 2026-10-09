"""Opt-in observation of the production Planner for arbitrary user prompts."""

import json
import os
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from core.graph import (
    _build_tools, _default_chat_model_factory, _default_rag_factory,
    _default_tool_list_factory,
)
from core.graph_constants import MAX_REASONING_STEPS, MUTATING_TOOLS
from core.graph_planner import create_planner_node
from core.planner import PlannerService
from core.planner_provider import LangChainPlannerProvider
from core.planner_routing import LangChainPlannerRouter
from core.protocol.bridge import build_controller_input
from core.protocol.controller import CortexController, apply_controller_decision_to_state
from core.protocol.enums import ControllerDecisionType, PlannerOutcome, PlanningFailureCategory
from core.protocol.models import PlanningCapabilities
from main import _build_settings, parse_args
from tools.registry import ToolRegistry


def assert_live_provider(service):
    """Check production components without inspecting private HTTP internals."""
    assert type(service) is PlannerService, "Live harness requires production PlannerService"
    assert type(service.provider) is LangChainPlannerProvider, "Fake Planner provider"
    assert type(service.router) is LangChainPlannerRouter, "Fake Planner Router"
    for model in (service.provider.planner_llm, service.router.router_llm):
        assert type(model) is ChatOllama, "Live harness requires configured production ChatOllama"


def _outcome_name(result):
    """Display the production semantic/infrastructure outcome, without judging it."""
    if result.outcome == PlannerOutcome.FAILED:
        return "PLANNING_FAILED" if result.failure_category == PlanningFailureCategory.UNPLANNABLE else result.failure_category.value
    return {
        PlannerOutcome.EXECUTION_PLAN: "PLAN_PROPOSED",
        PlannerOutcome.DIRECT_RESPONSE: "NO_PLAN_REQUIRED",
        PlannerOutcome.CLARIFICATION_REQUIRED: "NEEDS_INPUT",
    }[result.outcome]


def _print_result(result):
    print(f"outcome: {_outcome_name(result)}")
    print(f"route: {result.planner_route}")
    if result.proposed_plan is not None:
        print(f"objective: {result.proposed_plan.objective}")
        print(f"steps: {len(result.proposed_plan.steps)}")
        for index, step in enumerate(result.proposed_plan.steps, 1):
            print(f"\n{index}. {step.title}")
            print(f"   id: {step.step_id}")
            print(f"   description: {step.description}")
            print(f"   tool: {step.primary_tool}")
            print(f"   depends_on: {', '.join(step.depends_on_step_ids) or '(none)'}")
    if result.failure_category is not None:
        print(f"category: {result.failure_category.value}")
    print(f"\nmessage: {result.message}", flush=True)


def _print_metrics(usage, elapsed, done_reasons):
    print("tokens: " + ("unknown" if usage is None else
          f"input={usage['input_tokens']} output={usage['output_tokens']} total={usage['total_tokens']}"))
    print(f"elapsed: {elapsed:.3f}s")
    print(f"done_reason: {', '.join(reason or 'unknown' for reason in done_reasons)}", flush=True)


def _read_exchanges(path, offset, execution_id):
    if not path.exists():
        return []
    with path.open("rb") as stream:
        stream.seek(offset)
        records = [json.loads(line) for line in stream if line.strip()]
    return [record for record in records if record["execution_id"] == execution_id]


def _attempt_diagnostics(request, result, elapsed, records):
    exchanges = [record for record in records
                 if record["worker"] == "planner" and record["operation"] == "plan"
                 and record.get("invocation", {}).get("attempt") == request.attempt]
    assert len(exchanges) <= 1, "Provider must invoke once per authorized attempt"
    exchange = exchanges[0] if exchanges else None
    if result.outcome != PlannerOutcome.FAILED:
        assert exchange is not None, "Missing production Planner diagnostics; possible fake provider"
    return {
        "attempt": request.attempt, "request_id": request.request_id,
        "outcome": _outcome_name(result), "result": result.model_dump(mode="json"),
        "elapsed_seconds": elapsed, "usage": exchange["usage"] if exchange else None,
        "provider_model": exchange["response"]["model"] if exchange else None,
        "done_reason": exchange["response"]["done_reason"] if exchange else None,
        "length_limit": exchange["response"]["done_reason"] == "length" if exchange else None,
    }


def _observe_run(planner_node, capabilities, prompt, log_path, run):
    # No history, output, route, feedback, or execution state crosses runs.
    execution_id = f"planner-live-{uuid4().hex}"
    controller = CortexController(MAX_REASONING_STEPS, planning_capabilities=capabilities)
    state = {"execution_state": controller.start_execution(execution_id),
             "messages": [HumanMessage(content=prompt)]}
    decision = controller.decide(build_controller_input(state))
    attempts = []
    print(f"\nRUN {run}", flush=True)
    while decision.decision_type == ControllerDecisionType.DISPATCH_PLANNER:
        planning_request = decision.planning_request
        state["execution_state"] = apply_controller_decision_to_state(state["execution_state"], decision)
        state["controller_decision"] = decision
        offset = log_path.stat().st_size if log_path.exists() else 0
        print(f"\nATTEMPT {planning_request.attempt}", flush=True)
        started = perf_counter()
        update = planner_node(state)
        elapsed = perf_counter() - started
        result = update["planner_result"]
        records = _read_exchanges(log_path, offset, execution_id)
        attempt = _attempt_diagnostics(planning_request, result, elapsed, records)
        attempts.append(attempt)
        _print_result(result)
        _print_metrics(attempt["usage"], elapsed, [attempt["done_reason"]])
        print(f"length_limit: {attempt['length_limit']}", flush=True)
        # Only pure Controller decisions: its retry budget, route and feedback
        # remain authoritative. No non-Planner worker is ever dispatched here.
        state.update(update)
        decision = controller.decide(build_controller_input(state))
        if decision.decision_type == ControllerDecisionType.DISPATCH_PLANNER:
            retry = decision.planning_request
            print(f"Controller requested Planner retry {retry.attempt}/{retry.max_attempts}.", flush=True)
    usage = None
    if all(attempt["usage"] is not None for attempt in attempts):
        usage = {key: sum(attempt["usage"][key] for attempt in attempts)
                 for key in ("input_tokens", "output_tokens", "total_tokens")}
    elapsed = sum(attempt["elapsed_seconds"] for attempt in attempts)
    print("\nFINAL PLANNER RESULT")
    _print_result(result)
    print(f"planner attempts: {len(attempts)}")
    _print_metrics(usage, elapsed, [attempt["done_reason"] for attempt in attempts])
    print(f"stopped before Controller continuation: {decision.decision_type.value}", flush=True)
    return {"run": run, "execution_id": execution_id, "attempts": attempts,
            "final_result": result.model_dump(mode="json"), "usage": usage,
            "elapsed_seconds": elapsed, "controller_continuation": decision.decision_type.value}


@pytest.mark.skipif(
    os.getenv("CORTEX_LIVE_PLANNER") != "1",
    reason="Set CORTEX_LIVE_PLANNER=1 to call the configured local Planner model",
)
def test_planner_stability_live(tmp_path, monkeypatch, request):
    # Application defaults and environment, with no application CLI arguments.
    with monkeypatch.context() as settings_patch:
        settings_patch.setattr(sys, "argv", ["main.py"])
        settings = _build_settings(parse_args())
    prompt = os.getenv("CORTEX_LIVE_PLANNER_PROMPT", "List files")
    assert prompt.strip(), "CORTEX_LIVE_PLANNER_PROMPT must not be blank"
    runs = int(os.getenv("CORTEX_LIVE_PLANNER_RUNS", "1"))
    assert runs > 0, "CORTEX_LIVE_PLANNER_RUNS must be positive"
    rag = _default_rag_factory(
        Path(settings["knowledge_dir"]).resolve(), settings["embedding_model"], settings["rag_top_k"],
    )
    tools = _build_tools(
        _default_tool_list_factory, str(Path(settings["workspace"]).resolve()),
        str(Path(settings["knowledge_dir"]).resolve()), rag, settings["model"],
    )
    capabilities = PlanningCapabilities(available_tools=tuple(sorted(ToolRegistry.from_tools(tools).names)))
    llm = _default_chat_model_factory(settings["model_planner"], 0)
    service = PlannerService(
        provider=LangChainPlannerProvider(planner_llm=llm),
        router=LangChainPlannerRouter(router_llm=llm), mutating_tools=MUTATING_TOOLS,
    )
    assert_live_provider(service)
    planner_node = create_planner_node(
        planner_service=service, rag_service=rag, rag_top_k=settings["rag_top_k"],
    )
    # Existing diagnostics include invalid/length responses. Honor explicit live
    # logging; otherwise retain exchanges in pytest's temporary directory.
    log_path = Path(request.config.getoption("--live-raw-llm-file") or tmp_path / "exchanges.jsonl")
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", str(log_path))
    metadata = {
        "provider": f"{type(service.provider).__module__}.{type(service.provider).__name__}",
        "model": llm.model, "base_url": llm.base_url,
        "temperature": service.provider.planner_llm.temperature,
        "num_predict": service.provider.planner_llm.num_predict,
        "workspace": settings["workspace"], "knowledge_dir": settings["knowledge_dir"],
        "runs": runs, "prompt": prompt, "raw_exchanges": str(log_path),
    }
    print("\nLIVE PRODUCTION PLANNER\n" + json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    observations = [_observe_run(planner_node, capabilities, prompt, log_path, run)
                    for run in range(1, runs + 1)]
    report_path = tmp_path / "planner-observations.json"
    report_path.write_text(json.dumps({**metadata, "runs": observations}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nREPORT {report_path}", flush=True)
    # All production outcomes are observations, including exhausted retries.
