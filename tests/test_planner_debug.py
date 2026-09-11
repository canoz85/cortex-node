"""Debug diagnostics must not affect Planner execution."""
import builtins

import pytest

from test_planner import FakeProvider, planner_input, service, VALID
from core.protocol.models import ExecutionPlan, ExecutionStep
from test_graph_planner import DummyPlannerLLM, DummyRAG, authorize
from langchain_core.messages import HumanMessage
from core.graph_planner import create_planner_node
from core.planner_debug import log_planner


@pytest.mark.parametrize("route,content,error", [
    ("action", VALID, None), ("conversation", VALID, None),
    ("action", "malformed", None), ("action", VALID, "route"),
    ("action", VALID, "generate"),
])
@pytest.mark.parametrize("revise", [False, True])
def test_service_debug_parity(capsys, route, content, error, revise):
    request = planner_input(active_plan=ExecutionPlan(
        plan_id="prior-plan", revision=1, objective="create a file",
        steps=(ExecutionStep(step_id="step-1", title="Inspect", description="Inspect", primary_tool="list_files"),),
    ) if revise else None)
    results, calls = [], []
    for enabled in (False, True):
        provider = FakeProvider(content, route=route, error_at=error)
        planner = service(provider)
        planner.show_raw_llm = enabled
        results.append(planner.run(request))
        calls.append((provider.requests, provider.messages))
        output = capsys.readouterr().out
        assert bool(output) == enabled
        if enabled:
            assert "[planner:request]" in output
            assert "[planner:normalized]" in output
            assert "fixture-request" in output
            if provider.messages and not error:
                assert "[planner:prompt][system]" in output
                assert "[planner:prompt][human]" in output
                assert "[planner:raw]" in output
    assert results[0] == results[1]
    assert calls[0] == calls[1]


@pytest.mark.parametrize("route,confidence", [("action", .95), ("action", .2)])
def test_adapter_flag_router_and_model_parity(capsys, route, confidence):
    state = authorize({"messages": [HumanMessage(content="create a file")]})
    results, calls = [], []
    for enabled in (False, True):
        llm, rag = DummyPlannerLLM(VALID, route=route, confidence=confidence), DummyRAG()
        node = create_planner_node(planner_llm=llm, router_llm=llm,
            rag_service=rag, rag_top_k=4, tools_set={"list_files", "write_file"},
            show_raw_llm=enabled)
        results.append(node(state))
        calls.append((llm.routes, llm.invocations, rag.calls))
        output = capsys.readouterr().out
        assert bool(output) == enabled
        if enabled:
            assert "[planner:router][structured]" in output
            assert '"selected"' in output
        assert len(llm.routes) == 1
        assert len(llm.invocations) == (1 if confidence == .95 else 0)
    assert results[0] == results[1]
    assert calls[0] == calls[1]


def test_redaction_only_changes_display(capsys):
    value = 'password="hidden-password" api_key=hidden-key Authorization: Bearer hidden-token https://user:pass@example.com'
    log_planner("raw", value)
    output = capsys.readouterr().out
    for secret in ("hidden-password", "hidden-key", "hidden-token", "user:pass"):
        assert secret not in output
    assert "[REDACTED]" in output
    assert "hidden-password" in value


def test_broken_stdout_does_not_change_result(monkeypatch):
    planner = service(FakeProvider())
    expected = planner.run(planner_input())
    planner.show_raw_llm = True
    def broken(*args, **kwargs):
        raise OSError("closed stdout")
    monkeypatch.setattr(builtins, "print", broken)
    assert planner.run(planner_input()) == expected

