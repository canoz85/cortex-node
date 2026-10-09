"""Structured Planner provider exchanges and isolated diagnostics."""

import json
import pytest
from core.planner_contract import PlannerProposal
from core.planner_provider import LangChainPlannerProvider
from core.planner_routing import LangChainPlannerRouter, RouterDecisionSchema
from langchain_core.messages import AIMessage
from types import SimpleNamespace


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


def _records(monkeypatch, tmp_path, name):
    path = tmp_path / name
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", str(path))
    return path


def test_router_writes_one_exchange_record(monkeypatch, tmp_path, capsys):
    raw = AIMessage(
        content='{"route":"info"}',
        response_metadata={"model": "router", "done_reason": "stop"},
        usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    )
    llm = LLM({"raw": raw, "parsed": RouterDecisionSchema(route="info"),
               "parsing_error": None})
    path = _records(monkeypatch, tmp_path, "planner-route.jsonl")

    decision = LangChainPlannerRouter(router_llm=llm, show_raw_llm=True).route("list files")

    assert decision.route == "info"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert (records[0]["worker"], records[0]["operation"]) == ("planner", "route")
    assert [item["role"] for item in records[0]["messages"]] == ["system", "human"]
    assert records[0]["response"]["model"] == "router"
    assert records[0]["usage"]["total_tokens"] == 5
    assert "[raw-llm:planner:route]" in capsys.readouterr().out


def test_planner_provider_writes_one_plan_exchange(monkeypatch, tmp_path, capsys):
    raw = AIMessage(content=json.dumps(PROPOSAL), response_metadata={"model": "planner"})
    llm = LLM({
        "raw": raw,
        "parsed": PlannerProposal.model_validate(PROPOSAL),
        "parsing_error": None,
    })
    path = _records(monkeypatch, tmp_path, "planner-plan.jsonl")
    provider = LangChainPlannerProvider(planner_llm=llm, show_raw_llm=False)
    from core.planner import PlannerMessage

    result = provider.generate((
        PlannerMessage("system", "Plan safely", "exec-1", ("list_files",)),
        PlannerMessage("human", "inspect", "exec-1"),
    ))

    assert result.objective == "Inspect"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["execution_id"] == "exec-1"
    assert (record["worker"], record["operation"]) == ("planner", "plan")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("failure,expected", [
    ("parser", "INVALID_OUTPUT"),
    ("missing_parsed", "INVALID_OUTPUT"),
    ("provider", "PROVIDER_FAILURE"),
    ("router", "PROVIDER_FAILURE"),
])
def test_infrastructure_failures_are_normalized_at_provider_boundary(failure, expected):
    from core.planner import PlannerService
    from core.protocol.enums import PlannerOutcome
    from tests.test_planner_service import FakeRouter, planner_input

    class Model:
        def with_structured_output(self, schema, *, method, include_raw):
            assert issubclass(schema, PlannerProposal)
            assert method == "json_schema" and include_raw

            def invoke(messages):
                assert [m.type for m in messages] == ["system", "system", "human"]
                if failure == "provider":
                    raise RuntimeError("provider unavailable")
                return {"raw": AIMessage(content='{"result":"NO_PLAN_REQUIRED","message":"Valid raw JSON"}'),
                        "parsed": None,
                        "parsing_error": ValueError("schema failed") if failure == "parser" else None}

            return SimpleNamespace(invoke=invoke)

    service = PlannerService(
        provider=LangChainPlannerProvider(planner_llm=Model()),
        router=FakeRouter(error=RuntimeError("router unavailable") if failure == "router" else None),
        mutating_tools={"write_file"},
    )
    result = service.run(planner_input())
    assert result.outcome == PlannerOutcome.FAILED
    assert result.failure_category.value == expected
    assert result.proposed_plan is None
