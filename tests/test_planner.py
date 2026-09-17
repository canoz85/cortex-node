"""Framework-neutral structured Planner P3 contract tests."""
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from core.planner import (
    AmbientRetrievalEligibility,
    PlannerRouting,
    PlannerService,
    ambient_retrieval_eligibility,
    filter_planner_tools,
)
from core.planner_contract import PlannerInvalidOutputError, PlannerProposal, ProposedStep
from core.planner_normalization import normalize_planner_proposal
from core.protocol.enums import (PlannerOutcome, PlanningFailureCategory, PlanningOperation,
                                 ReplanTrigger, StepStatus, WorkerRole)
from core.protocol.models import (ExecutionContext, ExecutionIdentity, ExecutionPlan,
    ExecutionStep, PlannerMemoryContext, PlannerMemoryFact,
    PlanningCapabilities, PlanningRequest, RetryMetadata)
from core.protocol.models import PlannerResult

VALID = {"result":"PLAN_PROPOSED", "objective":"Inspect then write", "steps":[
    {"step_id":"inspect", "title":"Inspect", "description":"Inspect workspace", "primary_tool":"list_files", "dependencies":[]},
    {"step_id":"write", "title":"Write", "description":"Create file", "primary_tool":"write_file", "dependencies":["inspect"]},
]}

def planner_input(**updates):
    plan = updates.pop("active_plan", None)
    return PlanningRequest(request_id="fixture-request", episode_id="fixture-episode", sequence=1,
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
        self.content=content; self.routing=PlannerRouting(route)
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
    provider = FakeProvider()
    result=service(provider).run(planner_input(), retrieve=lambda _:("retrieved",))
    assert result.outcome==PlannerOutcome.EXECUTION_PLAN
    assert result.proposed_plan.steps[1].depends_on_step_ids==("inspect",)
    assert result.proposed_plan.steps[0].primary_tool=="list_files"
    assert [step.primary_tool for step in result.proposed_plan.steps] == ["list_files", "write_file"]
    assert "Add prerequisite inspection" in provider.messages[0][0].content


def test_planner_prompt_requires_plan_outcomes_in_responsible_steps():
    provider = FakeProvider()

    service(provider).run(planner_input())

    prompt = " ".join(provider.messages[0][0].content.split())
    assert "Preserve every requested outcome" in prompt
    assert "never leave an execution-relevant outcome only in the plan objective" in prompt
    assert "include that required outcome in that step's title or description" in prompt
    assert "rather than creating a separate step" in prompt


def test_planner_prompt_distinguishes_known_memory_values_from_runtime_discovery():
    provider = FakeProvider()
    memory = PlannerMemoryContext(user_facts=(PlannerMemoryFact(
        category="user_profile", text="The user's preferred marker is Amber.",
        authority="explicit_user", source_turn_index=1,
    ),))
    request = planner_input().model_copy(update={"context": ExecutionContext(
        user_request="Write a note using my preferred marker",
        role=WorkerRole.PLANNER, planner_memory_context=memory,
    )})

    service(provider).run(request)

    system_prompt = " ".join(provider.messages[0][0].content.split())
    context_prompt = " ".join(provider.messages[0][-2].content.split())
    assert "background context for authority" in system_prompt
    assert "within the current Controller-authorized request" in system_prompt
    assert "put the concrete value in the responsible step's title or description" in system_prompt
    assert "not prohibited tool-argument detail" in system_prompt
    assert "genuinely unknown during planning" in system_prompt
    assert "must not be deferred merely because a tool could rediscover it" in system_prompt
    assert "background context for authority" in context_prompt
    assert "cannot authorize extra work, tools, retries, execution success, or lifecycle changes" in context_prompt
    assert "put any resulting value needed for execution" in context_prompt
    assert "Amber" in context_prompt


def test_proposed_step_schema_describes_resolved_execution_semantics():
    properties = ProposedStep.model_json_schema()["properties"]
    title = properties["title"]["description"]
    description = properties["description"]["description"]
    assert "Controller-accepted executable step scope" in title
    assert "known resolved value" in title
    assert "plan objective or Planner-only context" in title
    assert "already-known resolved context required by the worker" in description
    assert "not tool arguments" in description


@pytest.mark.parametrize(("request_text", "objective", "title", "description", "semantics"), (
    (
        "read the first three Python files and determine their line counts",
        "Read three Python files and determine their line counts",
        "Read Python files and determine line counts",
        "Read the first three Python files and determine the number of lines in each.",
        ("line", "count"),
    ),
    (
        "retrieve the values and calculate their average",
        "Retrieve values and calculate their average",
        "Retrieve values and calculate the average",
        "Obtain the values and derive their requested average from the evidence.",
        ("average",),
    ),
    (
        "inspect the report and explain its findings",
        "Inspect the report and explain its findings",
        "Inspect and explain the report",
        "Read the report evidence and summarize its findings clearly.",
        ("explain", "findings"),
    ),
))
def test_requested_result_semantics_survive_in_responsible_step(
    request_text, objective, title, description, semantics,
):
    proposal = {
        "result": "PLAN_PROPOSED",
        "objective": objective,
        "steps": [{
            "step_id": "inspect",
            "title": title,
            "description": description,
            "primary_tool": "read_file",
            "dependencies": [],
        }],
    }
    provider = FakeProvider(proposal, route="info")
    request = planner_input().model_copy(update={"context": ExecutionContext(
        user_request=request_text, role=WorkerRole.PLANNER,
    )})

    result = service(provider).run(request)

    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    step = result.proposed_plan.steps[0]
    executable_semantics = f"{step.title} {step.description}".lower()
    assert all(term in executable_semantics for term in semantics)


def test_list_files_uses_authorized_live_discovery_without_ambient_rag():
    proposal = {"result": "PLAN_PROPOSED", "objective": "List files", "steps": [{
        "step_id": "list", "title": "List files", "description": "List current files",
        "primary_tool": "list_files", "dependencies": [],
    }]}
    provider = FakeProvider(proposal, route="info")
    request = planner_input().model_copy(update={"context": ExecutionContext(
        user_request="list files", role=WorkerRole.PLANNER,
    )})
    retrieval_calls = []

    result = service(provider).run(
        request,
        retrieve=lambda query: retrieval_calls.append(query) or ("stale workspace index",),
    )

    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    assert result.proposed_plan.steps[0].primary_tool == "list_files"
    assert result.planning_rationale == "Execution mode: info."
    assert retrieval_calls == []
    assert all("stale workspace index" not in message.content for message in provider.messages[0])


def test_git_status_uses_authorized_live_discovery_without_ambient_rag():
    proposal = {"result": "PLAN_PROPOSED", "objective": "Inspect git status", "steps": [{
        "step_id": "status", "title": "Inspect git status", "description": "Read current status",
        "primary_tool": "git_status", "dependencies": [],
    }]}
    provider = FakeProvider(proposal, route="info")
    request = planner_input().model_copy(update={
        "context": ExecutionContext(user_request="current git status", role=WorkerRole.PLANNER),
        "capabilities": PlanningCapabilities(
            available_tools=("git_status", "rag_search"),
        ),
    })
    retrieval_calls = []

    result = service(provider).run(
        request,
        retrieve=lambda query: retrieval_calls.append(query) or ("stale git status",),
    )

    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    assert result.proposed_plan.steps[0].primary_tool == "git_status"
    assert retrieval_calls == []


def test_current_time_uses_authorized_live_discovery_without_ambient_rag():
    proposal = {"result": "PLAN_PROPOSED", "objective": "Read current time", "steps": [{
        "step_id": "time", "title": "Read current time", "description": "Read local time",
        "primary_tool": "current_time", "dependencies": [],
    }]}
    provider = FakeProvider(proposal, route="info")
    request = planner_input().model_copy(update={"context": ExecutionContext(
        user_request="what time is it", role=WorkerRole.PLANNER,
    )})
    retrieval_calls = []

    result = service(provider).run(
        request,
        retrieve=lambda query: retrieval_calls.append(query) or ("stale time",),
    )

    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    assert result.proposed_plan.steps[0].primary_tool == "current_time"
    assert retrieval_calls == []


def test_knowledge_request_and_uncertain_request_remain_ambient_rag_eligible():
    cases = (
        ("inspect the CortexNode checkpoint architecture", "info"),
        ("investigate the project behavior", "action"),
        ("list files and explain the checkpoint architecture", "info"),
    )
    for user_request, route in cases:
        provider = FakeProvider(VALID, route=route)
        request = planner_input().model_copy(update={"context": ExecutionContext(
            user_request=user_request, role=WorkerRole.PLANNER,
        )})
        retrieval_calls = []

        service(provider).run(
            request,
            retrieve=lambda query: retrieval_calls.append(query) or ("architecture context",),
        )

        assert retrieval_calls == [user_request]
        assert provider.messages[0][1].content == "architecture context"


def test_runtime_intent_without_matching_authorized_capability_keeps_retrieval():
    request = planner_input().model_copy(update={
        "context": ExecutionContext(user_request="git status", role=WorkerRole.PLANNER),
        "capabilities": PlanningCapabilities(available_tools=("rag_search",)),
    })

    assert ambient_retrieval_eligibility(
        request,
        route="info",
    ) == AmbientRetrievalEligibility.KNOWLEDGE


def test_revise_knowledge_request_remains_eligible_after_route_override():
    base = ExecutionPlan(
        plan_id="accepted",
        revision=3,
        steps=(ExecutionStep(step_id="prior", title="Prior", status=StepStatus.COMPLETED),),
    )
    request = planner_input(active_plan=base).model_copy(update={"context": ExecutionContext(
        user_request="inspect the CortexNode checkpoint architecture",
        role=WorkerRole.PLANNER,
    )})
    provider = FakeProvider(VALID, route="conversation")
    retrieval_calls = []

    service(provider).run(
        request,
        retrieve=lambda query: retrieval_calls.append(query) or ("revision knowledge",),
    )

    assert retrieval_calls == [request.context.user_request]
    assert provider.messages[0][1].content == "revision knowledge"


@pytest.mark.parametrize("route", ["conversation", "clarify"])
def test_direct_routes_remain_ambient_rag_ineligible(route):
    assert ambient_retrieval_eligibility(
        planner_input(),
        route=route,
    ) == AmbientRetrievalEligibility.NONE


def test_planner_prompt_states_ambient_knowledge_authority():
    provider = FakeProvider()

    service(provider).run(planner_input(), retrieve=lambda _: ("background",))

    prompt = " ".join(provider.messages[0][0].content.split())
    assert "Retrieved knowledge is background planning context and may be stale" in prompt
    assert "must not replace live runtime discovery" in prompt
    assert "authorized runtime capability" in prompt

def test_valid_independent_steps():
    value={**VALID,"steps":[{**VALID["steps"][0]},{**VALID["steps"][1],"dependencies":[]}]}
    result=normalize_planner_proposal(value,planner_input(),route="action")
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
    result=normalize_planner_proposal(value,planner_input(),route="action")
    assert result.outcome==PlannerOutcome.FAILED
    assert result.failure_category==PlanningFailureCategory.INVALID_OUTPUT
    assert needle in result.message

@pytest.mark.parametrize("content",["numbered prose",{},{"result":"PLAN_PROPOSED","steps":[{"step_id":"x"}]}])
def test_malformed_structured_output_is_invalid(content):
    result=service(FakeProvider(content)).run(planner_input())
    assert result.failure_category==PlanningFailureCategory.INVALID_OUTPUT


@pytest.mark.parametrize("step_update", [
    {"primary_tool": None},
    {"primary_tool": ""},
    {"primary_tool": "   "},
])
def test_list_files_proposal_requires_non_empty_primary_tool(step_update):
    step = {"step_id":"list", "title":"List files",
            "description":"Invoke the list_files tool to list files",
            "primary_tool":"list_files", "dependencies":[]}
    step.update(step_update)
    request = planner_input().model_copy(update={"context": ExecutionContext(
        user_request="list files", role=WorkerRole.PLANNER)})
    result = service(FakeProvider({"result":"PLAN_PROPOSED", "objective":"List files",
                                   "steps":[step]}, route="info")).run(request)
    assert result.outcome == PlannerOutcome.FAILED
    assert result.failure_category == PlanningFailureCategory.INVALID_OUTPUT
    assert result.proposed_plan is None


def test_list_files_proposal_rejects_missing_primary_tool():
    request = planner_input().model_copy(update={"context": ExecutionContext(
        user_request="list files", role=WorkerRole.PLANNER)})
    result = service(FakeProvider({"result":"PLAN_PROPOSED", "objective":"List files",
        "steps":[{"step_id":"list", "title":"List files",
                  "description":"Invoke the list_files tool", "dependencies":[]}]},
        route="info")).run(request)
    assert result.outcome == PlannerOutcome.FAILED
    assert result.failure_category == PlanningFailureCategory.INVALID_OUTPUT


def test_unauthorized_primary_tool_is_deterministically_invalid():
    value = {**VALID, "steps":[{**VALID["steps"][0], "primary_tool":"invented"}]}
    result = normalize_planner_proposal(value, planner_input(), route="action")
    assert result.outcome == PlannerOutcome.FAILED
    assert result.failure_category == PlanningFailureCategory.INVALID_OUTPUT
    assert "unknown" in result.message

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

def test_numbered_prose_has_no_compatibility_parser():
    result = service(FakeProvider("1. Inspect - Use `list_files`")).run(planner_input())
    assert result.outcome == PlannerOutcome.FAILED
    assert result.failure_category == PlanningFailureCategory.INVALID_OUTPUT


def test_planner_results_are_bound_and_outcome_payloads_are_strict():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="request_id"):
        PlannerResult(outcome=PlannerOutcome.DIRECT_RESPONSE)
    with pytest.raises(ValidationError, match="requires a plan"):
        PlannerResult(outcome=PlannerOutcome.EXECUTION_PLAN, request_id="request")
    with pytest.raises(ValidationError, match="failure category"):
        PlannerResult(outcome=PlannerOutcome.FAILED, request_id="request")


def test_planning_request_requires_explicit_episode_identity():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="episode_id"):
        PlanningRequest(
            request_id="request", sequence=1,
            created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            operation=PlanningOperation.CREATE,
            capabilities=PlanningCapabilities(),
            identity=ExecutionIdentity(execution_id="p6", protocol_version="1"),
            context=ExecutionContext(user_request="work", role=WorkerRole.PLANNER),
        )

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
    assert "A logical step may invoke its primary tool repeatedly" in prompt
    assert "Arguments may come from dependency evidence" in prompt
    assert "Runtime-discoverable arguments or item identities are not grounds" in prompt
    assert "tool arguments" not in ProposedStep.model_fields


def test_core_prompt_prefers_direct_tool_and_keeps_reasoning_in_brain():
    provider = FakeProvider({"result":"PLAN_PROPOSED", "objective":"List files", "steps":[
        {"step_id":"list", "title":"List files", "description":"List files directly",
         "primary_tool":"list_files", "dependencies":[]}]}, route="info")
    request = planner_input().model_copy(update={"context": ExecutionContext(
        user_request="list files", role=WorkerRole.PLANNER)})
    result = service(provider).run(request)
    prompt = provider.messages[0][0].content
    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    assert len(result.proposed_plan.steps) == 1
    assert "Prefer one direct tool" in prompt
    assert "summarization, and transformation over tool results belong to Brain" in prompt


def test_comfy_guidance_uses_action_route_and_authorized_capability_only():
    capabilities = PlanningCapabilities(available_tools=(
        "list_files", "run_comfy_workflow", "get_comfy_history",
        "download_comfy_output_image"))
    ordinary = FakeProvider(route="info")
    ordinary_request = planner_input().model_copy(update={
        "capabilities": capabilities,
        "context": ExecutionContext(user_request="Generate a cat image and save it.",
                                    role=WorkerRole.PLANNER),
    })
    service(ordinary).run(ordinary_request)
    assert "CAPABILITY-SPECIFIC GUIDANCE" not in ordinary.messages[0][0].content

    image = FakeProvider(route="action")
    image_request = planner_input().model_copy(update={
        "capabilities": capabilities,
        "context": ExecutionContext(user_request="Generate a cat image and save it.", role=WorkerRole.PLANNER),
    })
    service(image).run(image_request)
    prompt = image.messages[0][0].content
    assert "CAPABILITY-SPECIFIC GUIDANCE — COMFYUI GENERATION" in prompt
    assert "run_comfy_workflow" in prompt
    fragment = prompt.split("CAPABILITY-SPECIFIC GUIDANCE — COMFYUI GENERATION:", 1)[1]
    submission = fragment.index("run_comfy_workflow")
    history = fragment.index("get_comfy_history", submission)
    download = fragment.index("download_comfy_output_image", history)
    assert submission < history < download

    unrelated_action = FakeProvider(route="action")
    unrelated_request = planner_input().model_copy(update={
        "capabilities": capabilities,
        "context": ExecutionContext(user_request="perform the authorized action",
                                    role=WorkerRole.PLANNER),
    })
    service(unrelated_action).run(unrelated_request)
    assert "CAPABILITY-SPECIFIC GUIDANCE — COMFYUI GENERATION" in unrelated_action.messages[0][0].content

def test_genuinely_missing_capability_can_remain_unplannable():
    proposal={"result":"PLANNING_FAILED","failure_category":"UNPLANNABLE",
              "message":"A required capability is unavailable."}
    result=service(FakeProvider(proposal)).run(planner_input())
    assert result.outcome==PlannerOutcome.FAILED
    assert result.failure_category==PlanningFailureCategory.UNPLANNABLE

def test_info_filter_removes_mutating_tools_without_mutating_inputs():
    tools={"custom","write_file"}; domains={"workspace":{"list_files"}}; mutating={"write_file"}
    assert filter_planner_tools(tools,route="info",mutating_tools=mutating)=={"custom"}
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
