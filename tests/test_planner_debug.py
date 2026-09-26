"""Exchange-based raw diagnostics for Planner invocations."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from core.planner_contract import PlannerProposal
from core.planner_provider import LangChainPlannerProvider
from core.planner_routing import planner_routing_decision


PROPOSAL = {
    "result": "PLAN_PROPOSED",
    "objective": "Inspect",
    "steps": [{
        "step_id": "inspect",
        "title": "Inspect",
        "description": "Inspect workspace",
        "primary_tool": "list_files",
        "dependencies": [],
    }],
}


class Structured:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.value


class LLM:
    def __init__(self, value):
        self.structured = Structured(value)

    def with_structured_output(self, _schema, **_kwargs):
        return self.structured


def _records(monkeypatch, name):
    path = Path(".tmp") / name
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        path.unlink()
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", str(path))
    return path


def test_router_writes_one_exchange_record(monkeypatch, capsys):
    raw = AIMessage(
        content='{"route":"info"}',
        response_metadata={"model": "router", "done_reason": "stop"},
        usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    )
    llm = LLM({"raw": raw, "parsed": type("Route", (), {"route": "info"})(),
               "parsing_error": None})
    path = _records(monkeypatch, "planner-route.jsonl")

    decision = planner_routing_decision("list files", router_llm=llm, show_raw_llm=True)

    assert decision.route == "info"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert (records[0]["worker"], records[0]["operation"]) == ("planner", "route")
    assert [item["role"] for item in records[0]["messages"]] == ["system", "human"]
    assert records[0]["response"]["model"] == "router"
    assert records[0]["usage"]["total_tokens"] == 5
    assert "[raw-llm:planner:route]" in capsys.readouterr().out


def test_planner_provider_writes_one_plan_exchange(monkeypatch, capsys):
    raw = AIMessage(content=json.dumps(PROPOSAL), response_metadata={"model": "planner"})
    llm = LLM({
        "raw": raw,
        "parsed": PlannerProposal.model_validate(PROPOSAL),
        "parsing_error": None,
    })
    path = _records(monkeypatch, "planner-plan.jsonl")
    provider = LangChainPlannerProvider(planner_llm=llm, show_raw_llm=False)
    from core.planner import PlannerMessage

    result = provider.generate((
        PlannerMessage("system", "Plan safely", "exec-1"),
        PlannerMessage("human", "inspect", "exec-1"),
    ))

    assert result.objective == "Inspect"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["execution_id"] == "exec-1"
    assert (record["worker"], record["operation"]) == ("planner", "plan")
    assert capsys.readouterr().out == ""
