"""P2 authorization, durable request context, and result-consumption regressions."""
from datetime import datetime, timezone
import json

import pytest
from langchain_core.messages import HumanMessage

from core.graph_controller import create_controller_node
from core.graph_planner import create_planner_node
from core.graph_state_machine import apply_controller_decision_to_state
from core.planner import PlannerService, PlannerRouting
from core.protocol.bridge import build_controller_input, build_planner_input
from core.protocol.controller import CortexController
from core.protocol.enums import (
    BrainOutcome, ControllerDecisionType, ExecutionPhase, ExecutionStatus,
    PlannerOutcome, PlanningOperation, ReplanTrigger, StepStatus, WorkerRole,
)
from core.protocol.models import (
    BrainResult, ExecutionCursor, ExecutionIdentity,
    ExecutionPlan, ExecutionState, ExecutionStep, PlanningCapabilities, PlanningRequest,
    ProtocolVisibleState, ReplanRequest, RetryMetadata, ToolExecutionRecord, ToolRequest,
    ToolResult, WorkingState,
)


NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)
IDENTITY = ExecutionIdentity(execution_id="planning-p2", protocol_version="1")
CAPABILITIES = PlanningCapabilities(
    available_tools=("list_files", "read_file"), unavailable_tools=("write_file",),
    constraints=("Workspace access is read-only.",),
)


def controller():
    return CortexController(max_reasoning_steps=24, now_utc=lambda: NOW, planning_capabilities=CAPABILITIES)


def initial_state():
    return {"execution_state": ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=IDENTITY, cursor=ExecutionCursor(),
    )), "messages": [HumanMessage(content="Inspect the workspace")]}


def active_state(trigger=ReplanTrigger.BRAIN_REQUESTED):
    done = ExecutionStep(step_id="done", title="Listed workspace", status=StepStatus.COMPLETED, primary_tool="list_files")
    interrupted = ExecutionStep(step_id="active", title="Read files", status=StepStatus.ACTIVE, attempt=2, primary_tool="read_file")
    plan = ExecutionPlan(plan_id="accepted", revision=3, objective="Inspect", steps=(done, interrupted))
    records = (ToolExecutionRecord(
        execution_id=IDENTITY.execution_id, plan_id=plan.plan_id, plan_revision=3,
        step_id="done", tool_name="list_files", result=ToolResult(
            request_id="listed", success=True, message="Listed", data={"files": ["a.txt"]}),
    ), *tuple(ToolExecutionRecord(
        execution_id=IDENTITY.execution_id, plan_id=plan.plan_id, plan_revision=3,
        step_id="active", tool_name="read_file", arguments={"path": "a.txt"},
        result=ToolResult(request_id=f"failed-{i}", signature="read:a.txt", success=False,
                          message="Access denied", error_code="DENIED", data={"path": "a.txt"}),
    ) for i in range(3)))
    tool_trigger = trigger == ReplanTrigger.REPEATED_TOOL_FAILURE
    state = {
        "messages": [HumanMessage(content="Inspect the workspace")],
        "execution_state": ExecutionState(protocol_visible=ProtocolVisibleState(
            identity=IDENTITY, cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING,
                current_worker=WorkerRole.TOOL_RUNTIME if tool_trigger else WorkerRole.BRAIN,
                step_id="active", plan_revision=3, step_attempt=2),
            active_plan=plan, active_step=interrupted, completed_step_ids=("done",),
            planning_sequence=1, retry=RetryMetadata(step_id="active", retry_count=1, max_retries=3),
            pending_tool_request=ToolRequest(request_id="failed-2", tool_name="read_file") if tool_trigger else None,
        ), working=WorkingState(tool_execution_history=records,
            last_tool_result=records[-1].result if tool_trigger else None)),
    }
    if not tool_trigger:
        state["brain_result"] = BrainResult(outcome=BrainOutcome.REPLAN_REQUEST,
            replan_request=ReplanRequest(reason="Cannot read selected files", failed_step_id="active",
                                         constraints=("Try a metadata-only approach",)))
    return state


def test_initial_controller_authorizes_create_without_revision_facts():
    state = initial_state()
    decision = controller().decide(build_controller_input(state))
    request = decision.planning_request
    assert decision.decision_type == ControllerDecisionType.DISPATCH_PLANNER
    assert request.operation == PlanningOperation.CREATE
    assert request.identity == IDENTITY
    assert request.context.user_request == "Inspect the workspace"
    assert request.context.recent_history == ("Inspect the workspace",)
    assert request.capabilities == CAPABILITIES
    assert request.created_at_utc == NOW
    assert request.sequence == 1
    assert request.base_plan is request.interrupted_step is request.trigger is None
    assert request.completed_steps == request.completed_step_ids == request.evidence_json == ()
    authorized = apply_controller_decision_to_state(state["execution_state"], decision)
    assert build_planner_input({"execution_state": authorized}) == request
    assert state["execution_state"].protocol_visible.planning_request is None


@pytest.mark.parametrize("trigger", list(ReplanTrigger))
def test_both_sources_share_helper_and_capture_revision_facts(trigger):
    class ObservingController(CortexController):
        def _build_planning_request(self, *args, **kwargs):
            self.constructions += 1
            return super()._build_planning_request(*args, **kwargs)
    ctrl = ObservingController(max_reasoning_steps=24, planning_capabilities=CAPABILITIES)
    ctrl.constructions = 0
    state = active_state(trigger)
    before = state["execution_state"].model_dump(mode="json")
    original = state["execution_state"].protocol_visible
    decision = ctrl.decide(build_controller_input(state))
    request = decision.planning_request
    assert ctrl.constructions == 1
    assert isinstance(request, PlanningRequest)
    assert request.operation == PlanningOperation.REVISE
    assert request.trigger == trigger
    assert request.base_plan == original.active_plan
    assert (request.base_plan_id, request.base_revision) == ("accepted", 3)
    assert request.completed_step_ids == ("done",)
    assert request.completed_steps == (original.active_plan.steps[0],)
    assert request.interrupted_step == original.active_step
    assert request.retry == original.retry
    assert request.sequence == 2
    assert request.capabilities == CAPABILITIES
    evidence = [json.loads(record) for record in request.evidence_json]
    assert evidence[0]["result"]["data"] == {"files": ["a.txt"]}
    assert evidence[-1]["arguments"] == {"path": "a.txt"}
    assert evidence[-1]["result"]["error_code"] == "DENIED"
    if trigger == ReplanTrigger.BRAIN_REQUESTED:
        assert request.reason == "Cannot read selected files"
        assert request.suggested_constraints == ("Try a metadata-only approach",)
    else:
        assert request.reason == "Access denied"
        assert json.loads(request.failure_json)["request_id"] == "failed-2"
        assert request.suggested_constraints == ()
    assert state["execution_state"].model_dump(mode="json") == before
    applied = apply_controller_decision_to_state(state["execution_state"], decision)
    assert applied.protocol_visible.active_step is None
    assert applied.protocol_visible.pending_tool_request is None
    assert applied.protocol_visible.active_plan.steps[1].status == StepStatus.FAILED
    assert applied.protocol_visible.completed_step_ids == ("done",)
    assert applied.protocol_visible.retry == RetryMetadata(max_retries=3)


class FakeProvider:
    def __init__(self, content="1. Inspect - Use list_files.", route="info"):
        self.content = content
        self.routing = PlannerRouting(route, "workspace", 0.99, "fixture")
        self.messages = []

    def route(self, text):
        return self.routing

    def generate(self, messages):
        self.messages.append(messages)
        return self.content


def planner(provider):
    return PlannerService(provider=provider, tools_set={"write_file", "invented"},
        domain_tool_map={"workspace": {"list_files", "read_file", "write_file"}},
        mutating_tools={"write_file"}, system_capabilities_text="fixture")


@pytest.mark.parametrize("trigger", list(ReplanTrigger))
def test_revise_prompt_contains_facts_and_enforced_ceiling(trigger):
    state = active_state(trigger)
    request = controller().decide(build_controller_input(state)).planning_request
    before = request.model_dump_json()
    provider = FakeProvider(route="conversation")
    result = planner(provider).run(request)
    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    assert result.proposed_plan.revision == 4
    assert result.proposed_plan.plan_id == "accepted"
    prompt = provider.messages[0][0].content
    tools = prompt.split("AVAILABLE TOOLS FOR THIS REQUEST", 1)[1].split("PLANNING RULES:", 1)[0]
    assert "- list_files" in tools and "- read_file" in tools
    assert "- write_file" not in tools and "- agent_info" not in tools
    context = provider.messages[0][-2].content
    assert "Do not repeat completed work" in context
    payload = json.loads(context.split("\n", 1)[1])
    assert payload["operation"] == "revise"
    assert payload["base_revision"] == 3
    assert payload["completed_steps"][0]["step_id"] == "done"
    assert payload["interrupted_step"]["step_id"] == "active"
    assert payload["trigger"] == trigger.value
    assert payload["reason"] == request.reason
    assert payload["capabilities"]["unavailable_tools"] == ["write_file"]
    assert payload["suggested_constraints"] == list(request.suggested_constraints)
    assert payload["evidence"][-1]["result"]["error_code"] == "DENIED"
    assert request.model_dump_json() == before


@pytest.mark.parametrize("content", ["1. Inspect - Use list_files.", "malformed"])
@pytest.mark.parametrize("trigger", list(ReplanTrigger))
def test_worker_consumption_through_controller_planner_controller(trigger, content):
    state = active_state(trigger)
    ctrl_node = create_controller_node(controller=controller())
    handoff = ctrl_node(state)
    planning_state = {**state, **handoff}
    accepted_before = planning_state["execution_state"].protocol_visible.active_plan
    assert planning_state["execution_state"].working.last_tool_result is None
    assert planning_state.get("brain_result") is None
    assert build_controller_input(planning_state).tool_result is None
    provider = FakeProvider(content)
    class RAG:
        def format_context(self, **kwargs):
            return ""
    node = create_planner_node(planner_service=planner(provider), rag_service=RAG(), rag_top_k=1, tools_set=set())
    output = node(planning_state)
    next_state = {**planning_state, **output}
    ci = build_controller_input(next_state)
    assert ci.planner_result is not None and ci.brain_result is ci.tool_result is None
    final = ctrl_node(next_state)
    assert final["planner_result"] is None
    assert final["execution_state"].protocol_visible.planning_request is None
    assert final["execution_state"].working.last_tool_result is None
    assert final["execution_state"].working.tool_execution_history == state["execution_state"].working.tool_execution_history
    if content == "malformed":
        assert final["execution_state"].protocol_visible.active_plan == accepted_before
        assert final["execution_state"].protocol_visible.status == ExecutionStatus.FAILED


@pytest.mark.parametrize("revise", [False, True])
def test_request_serializes_and_resume_reuses_authorization(revise):
    state = active_state() if revise else initial_state()
    ctrl = controller()
    decision = ctrl.decide(build_controller_input(state))
    applied = apply_controller_decision_to_state(state["execution_state"], decision)
    restored = ExecutionState.model_validate_json(applied.model_dump_json())
    assert restored == applied
    request = build_planner_input({"execution_state": restored})
    resumed = ctrl.decide(build_controller_input({"execution_state": restored}))
    assert resumed.planning_request == request
    assert resumed.planning_request.request_id == decision.planning_request.request_id
    assert resumed.planning_request.sequence == decision.planning_request.sequence


def test_adapter_rejects_missing_and_stale_authorizations():
    state = initial_state()
    with pytest.raises(ValueError, match="Controller-authorized"):
        build_planner_input(state)
    provider = FakeProvider()
    node = create_planner_node(planner_service=planner(provider), rag_service=None, rag_top_k=1, tools_set=set())
    with pytest.raises(ValueError, match="Controller-authorized"):
        node(state)
    assert provider.messages == []
    original = active_state()
    decision = controller().decide(build_controller_input(original))
    applied = apply_controller_decision_to_state(original["execution_state"], decision)
    stale = applied.model_copy(update={"protocol_visible": applied.protocol_visible.model_copy(update={
        "active_plan": applied.protocol_visible.active_plan.model_copy(update={"revision": 4})})})
    with pytest.raises(ValueError, match="base revision"):
        build_planner_input({"execution_state": stale})


def test_create_cannot_revise_and_request_snapshots_are_detached():
    state = active_state()
    context = build_controller_input(state)
    with pytest.raises(ValueError, match="no accepted plan"):
        controller()._dispatch_planner(context, "invalid")
    request = controller().decide(context).planning_request
    assert request.base_plan is not context.active_plan
    assert request.interrupted_step is not context.active_step
    with pytest.raises(ValueError):
        request.reason = "change"
    context.tool_execution_history[-1].arguments["path"] = "changed"
    assert json.loads(request.evidence_json[-1])["arguments"]["path"] == "a.txt"


def test_single_worker_result_invariant_is_still_enforced():
    state = active_state(ReplanTrigger.REPEATED_TOOL_FAILURE)
    ci = build_controller_input(state).model_copy(update={"brain_result": BrainResult(outcome=BrainOutcome.CONTINUE)})
    with pytest.raises(ValueError, match="only one worker result"):
        controller().decide(ci)


def test_request_operation_and_capability_contracts_reject_inconsistent_values():
    request = controller().decide(build_controller_input(active_state())).planning_request
    payload = request.model_dump(mode="json")
    with pytest.raises(ValueError, match="CREATE cannot"):
        PlanningRequest.model_validate({**payload, "operation": "create"})
    with pytest.raises(ValueError, match="base identity"):
        PlanningRequest.model_validate({**payload, "base_revision": 9})
    with pytest.raises(ValueError, match="disjoint"):
        PlanningCapabilities(available_tools=("read_file",), unavailable_tools=("read_file",))


def test_controller_rejects_stale_request_sequence_on_resume():
    state = initial_state()
    decision = controller().decide(build_controller_input(state))
    applied = apply_controller_decision_to_state(state["execution_state"], decision)
    ci = build_controller_input({"execution_state": applied}).model_copy(update={"planning_sequence": 2})
    with pytest.raises(ValueError, match="identity/sequence"):
        controller().decide(ci)
