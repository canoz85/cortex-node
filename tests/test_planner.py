"""Framework-neutral structured Planner P3 contract tests."""
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from core.planner import PlannerRouting, PlannerService, filter_planner_tools
from core.planner_contract import PlannerInvalidOutputError, PlannerProposal, ProposedStep
from core.planner_normalization import normalize_planner_proposal
from core.protocol.enums import (PlannerOutcome, PlanningFailureCategory, PlanningOperation,
                                 ReplanTrigger, StepStatus, WorkerRole)
from core.protocol.models import (ExecutionContext, ExecutionIdentity, ExecutionPlan,
    ExecutionStep, PlanningCapabilities, PlanningRequest, RetryMetadata)

VALID = {"result":"PLAN_PROPOSED", "objective":"Inspect then write", "steps":[
    {"step_id":"inspect", "title":"Inspect", "description":"Inspect workspace", "primary_tool":"list_files", "dependencies":[]},
    {"step_id":"write", "title":"Write", "description":"Create file", "primary_tool":"write_file", "dependencies":["inspect"]},
]}

def planner_input(**updates):
    plan = updates.pop("active_plan", None)
    return PlanningRequest(request_id="fixture-request", sequence=1,
        created_at_utc=datetime(2026,1,1,tzinfo=timezone.utc),
        operation=PlanningOperation.REVISE if plan else PlanningOperation.CREATE,
        base_plan=plan, base_plan_id=plan.plan_id if plan else None,
        base_revision=plan.revision if plan else None,
        trigger=ReplanTrigger.BRAIN_REQUESTED if plan else None,
        reason="fixture revision" if plan else "",
        capabilities=PlanningCapabilities(available_tools=("list_files","read_file","write_file","current_time","agent_info","token_usage"), unavailable_tools=("blocked_tool",)),
        identity=ExecutionIdentity(execution_id="p3", protocol_version="1"),
        context=ExecutionContext(user_request="create a file", role=WorkerRole.PLANNER), **updates)

class FakeProvider:
    def __init__(self, content=VALID, *, route="action", error_at=None):
        self.content=content; self.routing=PlannerRouting(route,"workspace",.95,"fixture")
        self.error_at=error_at; self.requests=[]; self.messages=[]
    def route(self, text):
        self.requests.append(text)
        if self.error_at=="route": raise RuntimeError("offline")
        return self.routing
    def generate(self, messages):
        self.messages.append(messages)
        if self.error_at=="generate": raise RuntimeError("offline")
        if self.error_at=="invalid": raise PlannerInvalidOutputError("bad schema")
        return self.content

def service(provider):
    return PlannerService(provider=provider, tools_set={"list_files","write_file"},
        domain_tool_map={"workspace":{"list_files","read_file","write_file"}}, mutating_tools={"write_file"},
        system_capabilities_text="fixture capabilities")

def test_valid_dependent_plan():
    result=service(FakeProvider()).run(planner_input(), retrieve=lambda _:("retrieved",))
    assert result.outcome==PlannerOutcome.EXECUTION_PLAN
    assert result.proposed_plan.steps[1].depends_on_step_ids==("inspect",)
    assert result.proposed_plan.steps[0].primary_tool=="list_files"

def test_valid_independent_steps():
    value={**VALID,"steps":[{**VALID["steps"][0]},{**VALID["steps"][1],"dependencies":[]}]}
    result=normalize_planner_proposal(value,planner_input(),route="action",confidence=1)
    assert [s.depends_on_step_ids for s in result.proposed_plan.steps]==[(),()]

@pytest.mark.parametrize("mutate,needle", [
    (lambda v:v["steps"].__setitem__(1,{**v["steps"][1],"step_id":"inspect"}),"unique"),
    (lambda v:v["steps"][1].update(dependencies=["missing"]),"unknown dependency"),
    (lambda v:(v["steps"][0].update(dependencies=["write"]),v["steps"][1].update(dependencies=["inspect"])),"cyclic"),
    (lambda v:v["steps"][0].update(primary_tool="blocked_tool"),"unavailable"),
    (lambda v:v["steps"][0].update(primary_tool="invented"),"unknown"),
    (lambda v:v.update(steps=[]),"at least one"),
])
def test_invalid_plan_constraints_rejected(mutate,needle):
    import copy
    value=copy.deepcopy(VALID); mutate(value)
    result=normalize_planner_proposal(value,planner_input(),route="action",confidence=1)
    assert result.outcome==PlannerOutcome.FAILED
    assert result.failure_category==PlanningFailureCategory.INVALID_OUTPUT
    assert needle in result.message

@pytest.mark.parametrize("content",["numbered prose",{},{"result":"PLAN_PROPOSED","steps":[{"step_id":"x"}]}])
def test_malformed_structured_output_is_invalid(content):
    result=service(FakeProvider(content)).run(planner_input())
    assert result.failure_category==PlanningFailureCategory.INVALID_OUTPUT

@pytest.mark.parametrize("error_at",["route","generate"])
def test_provider_exception_is_provider_failure(error_at):
    result=service(FakeProvider(error_at=error_at)).run(planner_input())
    assert result.failure_category==PlanningFailureCategory.PROVIDER_FAILURE

def test_provider_parse_failure_is_invalid_output():
    assert service(FakeProvider(error_at="invalid")).run(planner_input()).failure_category==PlanningFailureCategory.INVALID_OUTPUT

@pytest.mark.parametrize("payload,outcome",[
    ({"result":"NO_PLAN_REQUIRED","message":"direct"},PlannerOutcome.DIRECT_RESPONSE),
    ({"result":"NEEDS_INPUT","message":"need path"},PlannerOutcome.CLARIFICATION_REQUIRED),
    ({"result":"PLANNING_FAILED","failure_category":"UNPLANNABLE","message":"impossible"},PlannerOutcome.FAILED),
])
def test_explicit_result_variants(payload,outcome):
    result=service(FakeProvider(payload,route="conversation")).run(planner_input())
    assert result.outcome==outcome
    if outcome==PlannerOutcome.FAILED: assert result.failure_category==PlanningFailureCategory.UNPLANNABLE

def test_revise_preserves_request_and_versions_candidate():
    base=ExecutionPlan(plan_id="accepted",revision=3,steps=(ExecutionStep(step_id="done",title="Done",status=StepStatus.COMPLETED),))
    request=planner_input(active_plan=base,completed_step_ids=("done",),completed_steps=base.steps,
        retry=RetryMetadata(step_id="failed",retry_count=1,max_retries=2))
    before=request.model_dump_json(); provider=FakeProvider()
    result=service(provider).run(request)
    assert request.model_dump_json()==before
    assert (result.proposed_plan.plan_id,result.proposed_plan.revision)==("accepted",4)
    context=provider.messages[0][-2].content
    assert '"completed_step_ids": ["done"]' in context and '"base_revision": 3' in context

def test_normal_path_does_not_invoke_legacy_parser(monkeypatch):
    import core.planner_normalization as normalization
    monkeypatch.setattr(normalization,"legacy_numbered_execution_steps",lambda _:pytest.fail("legacy invoked"))
    assert service(FakeProvider()).run(planner_input()).outcome==PlannerOutcome.EXECUTION_PLAN

def discovery_dependent_proposal():
    return {"result":"PLAN_PROPOSED","objective":"Discover and process resources","steps":[
        {"step_id":"discover","title":"Discover resources","description":"Discover the resources to process","primary_tool":"list_files","dependencies":[]},
        {"step_id":"process","title":"Process discovered resources","description":"Read each resource discovered by the dependency step","primary_tool":"read_file","dependencies":["discover"]},
    ]}

def test_discovery_dependent_plan_is_accepted():
    result=service(FakeProvider(discovery_dependent_proposal())).run(planner_input())
    assert result.outcome==PlannerOutcome.EXECUTION_PLAN
    assert result.proposed_plan.steps[1].depends_on_step_ids==("discover",)

def test_one_step_can_represent_repeated_primary_tool_invocations():
    result=service(FakeProvider(discovery_dependent_proposal())).run(planner_input())
    processing=result.proposed_plan.steps[1]
    assert processing.primary_tool=="read_file"
    assert "each resource" in processing.description
    assert len(result.proposed_plan.steps)==2

def test_prompt_defers_dynamic_arguments_and_batching_to_brain():
    provider=FakeProvider(discovery_dependent_proposal())
    result=service(provider).run(planner_input())
    prompt=provider.messages[0][0].content
    assert result.outcome==PlannerOutcome.EXECUTION_PLAN
    assert "Concrete tool arguments may be derived by the Brain from evidence" in prompt
    assert "not necessarily one tool invocation" in prompt
    assert "may invoke the step's primary tool multiple times" in prompt
    assert "Runtime-discoverable inputs do not make a request unplannable" in prompt
    assert "tool arguments" not in ProposedStep.model_fields

def test_genuinely_missing_capability_can_remain_unplannable():
    proposal={"result":"PLANNING_FAILED","failure_category":"UNPLANNABLE",
              "message":"A required capability is unavailable."}
    result=service(FakeProvider(proposal)).run(planner_input())
    assert result.outcome==PlannerOutcome.FAILED
    assert result.failure_category==PlanningFailureCategory.UNPLANNABLE

def test_unknown_domain_filter_does_not_mutate_inputs():
    tools={"custom","write_file"}; domains={"workspace":{"list_files"}}; mutating={"write_file"}
    assert filter_planner_tools(tools,route="info",domain="unknown",domain_tool_map=domains,mutating_tools=mutating)=={"custom","current_time","agent_info","token_usage"}
    assert tools=={"custom","write_file"}

def test_service_runs_with_framework_imports_blocked():
    script='''
import importlib.abc,sys
sys.path.insert(0,"tests")
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith(("langchain","langgraph","ollama","core.graph","core.planner_provider")): raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from test_planner import FakeProvider,planner_input,service
from core.protocol.enums import PlannerOutcome
assert service(FakeProvider()).run(planner_input()).outcome==PlannerOutcome.EXECUTION_PLAN
'''
    completed=subprocess.run([sys.executable,"-B","-c",script],cwd=Path(__file__).resolve().parents[1],
        env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"},capture_output=True,text=True)
    assert completed.returncode==0,completed.stdout+completed.stderr
