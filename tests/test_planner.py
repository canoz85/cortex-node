"""Framework-neutral P1 service and legacy-normalization tests."""

import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from core.planner import PlannerRouting, PlannerService, filter_planner_tools
from core.planner_normalization import normalize_planner_output
from core.protocol.enums import PlannerOutcome, StepStatus, WorkerRole, PlanningOperation, ReplanTrigger
from core.protocol.models import (
    ExecutionContext, ExecutionIdentity, ExecutionPlan, ExecutionStep, PlanningRequest, PlanningCapabilities, RetryMetadata,
)


VALID = "1. Inspect – Use `list_files` to inspect.\n2. Write - Use `write_file` to create."


def planner_input(**updates):
    plan = updates.pop("active_plan", None)
    return PlanningRequest(
        request_id="fixture-request", sequence=1, created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        operation=PlanningOperation.REVISE if plan else PlanningOperation.CREATE,
        base_plan=plan, base_plan_id=plan.plan_id if plan else None,
        base_revision=plan.revision if plan else None,
        trigger=ReplanTrigger.BRAIN_REQUESTED if plan else None,
        reason="fixture revision" if plan else "",
        capabilities=PlanningCapabilities(available_tools=("list_files", "write_file", "current_time", "agent_info", "token_usage")),
        identity=ExecutionIdentity(execution_id="p1", protocol_version="1"),
        context=ExecutionContext(user_request="create a file", role=WorkerRole.PLANNER),
        **updates,
    )


class FakeProvider:
    def __init__(self, content=VALID, *, route="action", error_at=None):
        self.content = content
        self.routing = PlannerRouting(route, "workspace", 0.95, "fixture")
        self.error_at = error_at
        self.requests = []
        self.messages = []

    def route(self, user_request):
        self.requests.append(user_request)
        if self.error_at == "route":
            raise RuntimeError("offline")
        return self.routing

    def generate(self, messages):
        self.messages.append(messages)
        if self.error_at == "generate":
            raise RuntimeError("offline")
        return self.content


def service(provider):
    return PlannerService(
        provider=provider, tools_set={"list_files", "write_file"},
        domain_tool_map={"workspace": {"list_files", "write_file"}},
        mutating_tools={"write_file"}, system_capabilities_text="fixture capabilities",
    )


def test_fake_provider_produces_current_plan_contract():
    provider = FakeProvider()
    result = service(provider).run(planner_input(), retrieve=lambda _: ("retrieved",))
    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    assert result.message == "Plan generated successfully."
    assert result.planning_rationale == "Route 'action' selected with confidence 0.95."
    plan = result.proposed_plan
    assert (plan.plan_id, plan.revision, plan.objective) == ("p1:plan", 1, VALID)
    assert plan.steps == (
        ExecutionStep(step_id="step-1", title="Inspect", description="Use `list_files` to inspect.", primary_tool="list_files"),
        ExecutionStep(step_id="step-2", title="Write", description="Use `write_file` to create.", primary_tool="write_file", depends_on_step_ids=("step-1",)),
    )
    assert provider.requests == ["create a file"]
    assert len(provider.messages) == 1
    assert [(m.role, m.content) for m in (provider.messages[0][1], provider.messages[0][-1])] == [
        ("system", "retrieved"), ("human", "create a file"),
    ]


@pytest.mark.parametrize("content", ["", " \n\t", "not a numbered plan", "1. missing separator", "1) Inspect - Use list_files", "1. Inspect — Use list_files"])
def test_empty_or_completely_malformed_output_fails_without_retry(content):
    provider = FakeProvider(content)
    result = service(provider).run(planner_input())
    assert result.outcome == PlannerOutcome.FAILED
    assert result.proposed_plan is None
    assert "normalization" in result.message
    assert len(provider.messages) == 1


@pytest.mark.parametrize("error_at", ["route", "generate"])
def test_provider_exception_is_failed_without_retry(error_at):
    provider = FakeProvider(error_at=error_at)
    result = service(provider).run(planner_input())
    assert result.outcome == PlannerOutcome.FAILED
    assert result.proposed_plan is None
    assert result.message == "Planner provider failed (RuntimeError)."
    assert len(provider.requests) == 1
    assert len(provider.messages) == (1 if error_at == "generate" else 0)


@pytest.mark.parametrize("content", [VALID, "invalid"])
def test_service_preserves_supplied_plan_context_completion_and_retry(content):
    original = planner_input(
        active_plan=ExecutionPlan(plan_id="accepted", revision=3, steps=(
            ExecutionStep(step_id="step-1", title="Already done", status=StepStatus.COMPLETED, attempt=2),
        )),
        completed_step_ids=("step-1",),
        retry=RetryMetadata(step_id="step-2", retry_count=1, max_retries=2),
    )
    before = original.model_dump(mode="json")
    result = service(FakeProvider(content)).run(original)
    assert original.model_dump(mode="json") == before
    if content == VALID:
        # P1 intentionally retains replacement behavior; reconciliation is P2/P3+.
        assert result.proposed_plan.plan_id == "accepted"
        assert result.proposed_plan.revision == 4
        assert all(s.status == StepStatus.PENDING and s.attempt == 0 for s in result.proposed_plan.steps)


@pytest.mark.parametrize("route", ["conversation", "clarify_domain"])
def test_direct_result_skips_generation_and_retrieval(route):
    provider = FakeProvider(route=route)
    def unexpected_retrieval(_):
        pytest.fail("direct route retrieved context")
    result = service(provider).run(planner_input(), retrieve=unexpected_retrieval)
    assert result.outcome == PlannerOutcome.DIRECT_RESPONSE
    assert result.proposed_plan is None
    assert result.message == "No execution plan required."
    assert provider.messages == []


def test_context_retrieval_failure_is_explicit():
    def retrieve(_):
        raise OSError("unavailable")
    provider = FakeProvider()
    result = service(provider).run(planner_input(), retrieve=retrieve)
    assert result.outcome == PlannerOutcome.FAILED
    assert "context retrieval" in result.message
    assert provider.messages == []


def test_legacy_partial_lines_are_skipped_exactly_as_before():
    content = "intro\n1. Inspect – Use `list_files`.\n2. malformed\n3. Verify - Use `read_file`.\nfooter"
    result = normalize_planner_output(content, planner_input(), route="action", confidence=1)
    assert result.proposed_plan.objective == content
    assert [s.step_id for s in result.proposed_plan.steps] == ["step-1", "step-3"]
    assert result.proposed_plan.steps[1].depends_on_step_ids == ("step-1",)


def test_legacy_missing_unknown_and_multiple_tools_are_not_newly_validated():
    result = normalize_planner_output(
        "1. Unsupported - No tool available.\n2. Work - Use invented and Use write_file.",
        planner_input(), route="action", confidence=1,
    )
    assert [s.primary_tool for s in result.proposed_plan.steps] == [None, "invented"]


def test_legacy_numbering_and_step_limit_remain_unchanged():
    content = "\n".join("7. Work - Use list_files." for _ in range(5))
    result = normalize_planner_output(content, planner_input(), route="action", confidence=1)
    assert len(result.proposed_plan.steps) == 5
    assert all(s.step_id == "step-7" for s in result.proposed_plan.steps)


def test_unknown_domain_filter_keeps_registry_and_ubiquitous_tools_without_mutating_inputs():
    tools, domains, mutating = {"custom", "write_file"}, {"workspace": {"list_files"}}, {"write_file"}
    result = filter_planner_tools(tools, route="info", domain="unknown",
                                  domain_tool_map=domains, mutating_tools=mutating)
    assert result == {"custom", "current_time", "agent_info", "token_usage"}
    assert tools == {"custom", "write_file"}
    assert domains == {"workspace": {"list_files"}}
    assert mutating == {"write_file"}


def test_service_runs_with_graph_and_model_imports_blocked():
    script = '''
import importlib.abc
import sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("langchain", "langgraph", "ollama", "core.graph", "core.planner_provider")):
            raise AssertionError("framework import: " + fullname)
sys.meta_path.insert(0, BlockFrameworks())
from core.planner import PlannerService, PlannerRouting
from core.protocol.models import PlanningRequest, PlanningCapabilities, ExecutionIdentity, ExecutionContext
from core.protocol.enums import WorkerRole, PlannerOutcome, PlanningOperation
from datetime import datetime, timezone
class Fake:
    def route(self, request):
        return PlannerRouting("action", "general", 1.0, "fixture")
    def generate(self, messages):
        return "1. Inspect - Use list_files."
service = PlannerService(provider=Fake(), tools_set={"list_files"}, domain_tool_map={},
                         mutating_tools=set(), system_capabilities_text="fixture")
result = service.run(PlanningRequest(request_id="test", sequence=1, created_at_utc=datetime.now(timezone.utc),
    operation=PlanningOperation.CREATE, capabilities=PlanningCapabilities(available_tools=("list_files",)),
    identity=ExecutionIdentity(execution_id="test", protocol_version="1"),
    context=ExecutionContext(user_request="inspect", role=WorkerRole.PLANNER)))
assert result.outcome == PlannerOutcome.EXECUTION_PLAN
'''
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script], cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
