"""Controller planning authorization and lifecycle regressions."""

import ast
import json
import pytest
from core.graph_controller import create_controller_node
from core.graph_planner import create_planner_node
from core.planner import PlannerRoute, PlannerService
from core.protocol.bridge import build_brain_input, build_controller_input
from core.protocol.controller import CortexController, apply_controller_decision_to_state
from core.protocol.enums import (
    BrainOutcome,
    ControllerDecisionType,
    ExecutionPhase,
    ExecutionStatus,
    PlannerOutcome,
    PlanningFailureCategory,
    PlanningOperation,
    ReplanTrigger,
    StepStatus,
    WorkerRole,
)
from core.protocol.models import (
    BrainResult,
    ControllerInput,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    PlannerResult,
    PlanningCapabilities,
    PlanningRequest,
    ProtocolVisibleState,
    ReplanRequest,
    RetryMetadata,
    ToolExecutionRecord,
    ToolRequest,
    ToolResult,
    WorkingState,
)
from core.runtime.execution_driver import WorkerDispatchError
from datetime import datetime, timezone
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pathlib import Path


NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)
IDENTITY = ExecutionIdentity(execution_id="controller-planning", protocol_version="1")
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
    assert request.context.recent_history == ()
    assert request.capabilities == CAPABILITIES
    assert request.created_at_utc == NOW
    assert request.sequence == 1
    assert request.base_plan is request.interrupted_step is request.trigger is None
    assert request.completed_steps == request.completed_step_ids == request.evidence_json == ()
    authorized = apply_controller_decision_to_state(state["execution_state"], decision)
    assert authorized.protocol_visible.planning_request == request
    assert state["execution_state"].protocol_visible.planning_request is None


@pytest.mark.parametrize("trigger", list(ReplanTrigger))
def test_controller_authorizes_revision_with_failure_and_progress_context(trigger):
    ctrl = controller()
    state = active_state(trigger)
    before = state["execution_state"].model_dump(mode="json")
    original = state["execution_state"].protocol_visible
    decision = ctrl.decide(build_controller_input(state))
    request = decision.planning_request
    assert isinstance(request, PlanningRequest)
    assert request.operation == PlanningOperation.REVISE
    assert request.trigger == trigger
    assert request.base_plan == original.active_plan
    assert request.base_plan is not original.active_plan
    assert (request.base_plan_id, request.base_revision) == ("accepted", 3)
    assert request.completed_step_ids == ("done",)
    assert request.completed_steps == (original.active_plan.steps[0],)
    assert request.interrupted_step == original.active_step
    assert request.retry == original.retry
    assert request.progress == ctrl.decide(build_controller_input(state)).planning_request.progress
    assert len(request.progress.action_groups) == 2
    repeated = request.progress.action_groups[-1]
    assert repeated.signature == "read:a.txt"
    assert repeated.occurrence_count == 3
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
    state["execution_state"].working.tool_execution_history[-1].arguments["path"] = "changed"
    assert json.loads(request.evidence_json[-1])["arguments"]["path"] == "a.txt"
    applied = apply_controller_decision_to_state(state["execution_state"], decision)
    assert applied.protocol_visible.active_step is None
    assert applied.protocol_visible.pending_tool_request is None
    assert applied.protocol_visible.active_plan.steps[1].status == StepStatus.FAILED
    assert applied.protocol_visible.completed_step_ids == ("done",)
    assert applied.protocol_visible.retry == RetryMetadata(max_retries=3)


class FakePlannerRouter:
    def __init__(self, route: str = "action"):
        self.route_value = route

    def route(self, user_request: str):
        return PlannerRoute(route=self.route_value)

class FakeProvider:
    def __init__(self, content=None, route="info"):
        if content is None:
            content = {"result": "PLAN_PROPOSED", "objective": "Inspect", "steps": [
                {"step_id": "inspect", "title": "Inspect", "description": "Inspect workspace", "primary_tool": "list_files", "dependencies": []}
            ]}
        self.content = content
        self.messages = []

    def generate(self, messages):
        self.messages.append(messages)
        return self.content


def planner(provider):
    return PlannerService(provider=provider, router=FakePlannerRouter(), mutating_tools={"write_file"})


def test_revise_context_projection_with_scripted_provider():
    trigger = ReplanTrigger.REPEATED_TOOL_FAILURE
    state = active_state(trigger)
    request = controller().decide(build_controller_input(state)).planning_request
    before = request.model_dump_json()
    provider = FakeProvider(route="conversation")
    result = planner(provider).run(request)
    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    assert result.proposed_plan.revision == 4
    assert result.proposed_plan.plan_id == "accepted"
    prompt = provider.messages[0][0].content
    tools = prompt.split("AVAILABLE CAPABILITIES FOR THIS REQUEST", 1)[1].split("STEP SEMANTICS:", 1)[0]
    assert '"name":"list_files"' in tools and '"name":"read_file"' in tools
    assert '"name":"write_file"' not in tools and '"name":"agent_info"' not in tools
    context = provider.messages[0][-2].content
    assert "Do not repeat completed work" in context
    payload = json.loads(context.split("\n", 1)[1])
    assert payload["operation"] == "revise"
    assert payload["base_plan"]["revision"] == 3
    assert payload["base_plan"]["steps"][0]["step_id"] == "done"
    assert payload["trigger"] == trigger.value
    assert payload["reason"] == request.reason
    assert "capabilities" not in payload
    assert payload["suggested_constraints"] == list(request.suggested_constraints)
    progress = payload["previous_execution_progress"]
    failed = progress["action_groups"][-1]
    assert failed["latest_error_code"] == "DENIED"
    assert failed["signature"] == "read:a.txt"
    assert failed["occurrence_count"] == 3
    assert failed["failure_count"] == 3
    assert failed["semantic_conclusion"] == "unknown"
    assert "raw_evidence" not in payload
    assert request.model_dump_json() == before


def test_revision_runtime_flow_with_scripted_provider():
    state = active_state(ReplanTrigger.REPEATED_TOOL_FAILURE)
    ctrl_node = create_controller_node(controller=controller())
    handoff = ctrl_node(state)
    planning_state = {**state, **handoff}
    accepted_before = planning_state["execution_state"].protocol_visible.active_plan
    assert planning_state["execution_state"].working.last_tool_result is None
    assert planning_state.get("brain_result") is None
    assert build_controller_input(planning_state).tool_result is None
    provider = FakeProvider()
    class RAG:
        def format_context(self, **kwargs):
            return ""
    node = create_planner_node(planner_service=planner(provider), rag_service=RAG(), rag_top_k=1)
    output = node(planning_state)
    next_state = {**planning_state, **output}
    ci = build_controller_input(next_state)
    assert ci.planner_result is not None and ci.brain_result is ci.tool_result is None
    final = ctrl_node(next_state)
    assert final["planner_result"] is None
    assert final["execution_state"].protocol_visible.planning_request is None
    assert final["execution_state"].working.last_tool_result is None
    assert final["execution_state"].working.tool_execution_history == state["execution_state"].working.tool_execution_history
    accepted = final["execution_state"].protocol_visible.active_plan
    assert (accepted.plan_id, accepted.revision) == (accepted_before.plan_id, 4)
    assert accepted.steps[0].step_id == "done"
    assert accepted.steps[0].status == StepStatus.COMPLETED
    assert accepted.steps[-1].primary_tool == "list_files"


@pytest.mark.parametrize("revise", [False, True])
def test_request_serializes_and_resume_reuses_authorization(revise):
    state = active_state() if revise else initial_state()
    ctrl = controller()
    decision = ctrl.decide(build_controller_input(state))
    applied = apply_controller_decision_to_state(state["execution_state"], decision)
    restored = ExecutionState.model_validate_json(applied.model_dump_json())
    assert restored == applied
    request = restored.protocol_visible.planning_request
    assert request.progress == decision.planning_request.progress
    resumed = ctrl.decide(build_controller_input({"execution_state": restored}))
    assert resumed.planning_request == request
    assert resumed.planning_request.request_id == decision.planning_request.request_id
    assert resumed.planning_request.sequence == decision.planning_request.sequence


def test_adapter_rejects_missing_and_stale_authorizations():
    state = initial_state()
    provider = FakeProvider()
    node = create_planner_node(planner_service=planner(provider), rag_service=None, rag_top_k=1)
    with pytest.raises(WorkerDispatchError, match="Controller authorization"):
        node(state)
    assert provider.messages == []
    original = active_state()
    decision = controller().decide(build_controller_input(original))
    applied = apply_controller_decision_to_state(original["execution_state"], decision)
    stale = applied.model_copy(update={"protocol_visible": applied.protocol_visible.model_copy(update={
        "active_plan": applied.protocol_visible.active_plan.model_copy(update={"revision": 4})})})
    with pytest.raises(WorkerDispatchError, match="revision") :
        node({"execution_state": stale, "controller_decision": decision})


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


LIFECYCLE_IDENTITY = ExecutionIdentity(execution_id="controller-lifecycle", protocol_version="1")


def context(*, text="Do the work"):
    return ExecutionContext(user_request=text)


def initial_input(**updates):
    values = dict(identity=LIFECYCLE_IDENTITY, cursor=ExecutionCursor(), context=context())
    values.update(updates)
    return ControllerInput(**values)


def authorize(ctrl):
    decision = ctrl.decide(initial_input())
    return decision, decision.planning_request


def result_input(dispatch, request, result, **updates):
    values = dict(
        identity=LIFECYCLE_IDENTITY, cursor=dispatch.cursor, context=context(),
        planning_request=request, planning_sequence=request.sequence,
        planner_result=result,
    )
    values.update(updates)
    return ControllerInput(**values)


def test_create_plan_and_no_plan_have_distinct_terminal_semantics():
    ctrl = CortexController(20, planning_capabilities=PlanningCapabilities(available_tools=("read_file",)))
    dispatch, request = authorize(ctrl)
    plan = ExecutionPlan(plan_id=f"{request.identity.execution_id}:plan", available_tools=("read_file",),
                         steps=(ExecutionStep(step_id="s", title="Work", description="Inspect files",
                                              primary_tool="read_file"),))
    accepted = ctrl.decide(result_input(dispatch, request, PlannerResult(
        outcome=PlannerOutcome.EXECUTION_PLAN, request_id=request.request_id, proposed_plan=plan,
    )))
    assert accepted.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert accepted.accepted_plan == plan
    assert accepted.clear_planning_request

    dispatch, request = authorize(ctrl)
    no_plan = ctrl.decide(result_input(dispatch, request, PlannerResult(
        outcome=PlannerOutcome.DIRECT_RESPONSE, request_id=request.request_id, message="No tools required",
    )))
    assert no_plan.decision_type == ControllerDecisionType.TERMINATE
    assert no_plan.execution_status == ExecutionStatus.COMPLETED
    assert no_plan.accepted_plan is None and no_plan.terminal
    assert no_plan.direct_response is False
    assert no_plan.clear_planning_request


def test_clarification_pause_resume_controller_lifecycle():
    capabilities = PlanningCapabilities(available_tools=("read_file",))
    ctrl = CortexController(20, planning_capabilities=capabilities)
    dispatch, request = authorize(ctrl)
    paused = ctrl.decide(result_input(dispatch, request, PlannerResult(
        outcome=PlannerOutcome.CLARIFICATION_REQUIRED, request_id=request.request_id,
        planner_route="info",
        message="Which target should be inspected?",
    )))
    assert paused.decision_type == ControllerDecisionType.PAUSE
    assert paused.cursor.phase == ExecutionPhase.WAITING
    assert paused.planning_clarification.request.request_id == request.request_id
    assert paused.clear_planning_request

    state = apply_controller_decision_to_state(
        ExecutionState(protocol_visible=ProtocolVisibleState(
            identity=LIFECYCLE_IDENTITY, cursor=dispatch.cursor,
            planning_request=request, planning_sequence=request.sequence,
        )), paused,
    )
    restored = ExecutionState.model_validate_json(state.model_dump_json())
    marker = restored.protocol_visible.planning_clarification
    assert restored.protocol_visible.planning_request is None
    assert marker.prompt == "Which target should be inspected?"
    assert marker.request.planner_route == "info"

    waiting = ctrl.decide(initial_input(
        cursor=restored.protocol_visible.cursor,
        planning_sequence=request.sequence, planning_clarification=marker,
    ))
    assert waiting.decision_type == ControllerDecisionType.PAUSE
    assert waiting.planning_request is None

    # Recreating the Controller with a broader registry cannot expand the
    # authorization retained by the paused execution.
    resumed = CortexController(20, planning_capabilities=PlanningCapabilities(
        available_tools=("read_file", "write_file"),
    )).decide(initial_input(
        cursor=restored.protocol_visible.cursor,
        context=context(text="The src directory"), user_input="The src directory",
        planning_sequence=request.sequence, planning_clarification=marker,
    ))
    new_request = resumed.planning_request
    assert resumed.decision_type == ControllerDecisionType.DISPATCH_PLANNER
    assert resumed.clear_planning_clarification
    assert new_request.request_id != request.request_id
    assert new_request.sequence == request.sequence + 1
    assert new_request.episode_id != request.episode_id
    assert new_request.context.user_request == "Do the work"
    assert new_request.context.clarification_question == marker.prompt
    assert new_request.context.clarification == "The src directory"
    assert new_request.planner_route == "info"
    assert new_request.capabilities == capabilities

    repeated_pause = ctrl.decide(result_input(resumed, new_request, PlannerResult(
        outcome=PlannerOutcome.CLARIFICATION_REQUIRED, request_id=new_request.request_id,
        planner_route="info", message="Which file within that directory?",
    )))
    assert repeated_pause.decision_type == ControllerDecisionType.PAUSE
    assert repeated_pause.planning_clarification.request.identity == request.identity
    assert repeated_pause.planning_clarification.prompt == "Which file within that directory?"
    assert repeated_pause.planning_clarification.request.capabilities == capabilities

    plan = ExecutionPlan(
        plan_id="clarified",
        available_tools=("read_file",),
        steps=(ExecutionStep(
            step_id="inspect",
            title="Inspect the clarified target",
            description="Read the clarified target and report findings",
            primary_tool="read_file",
        ),),
    )
    continued = ctrl.decide(result_input(
        resumed,
        new_request,
        PlannerResult(
            outcome=PlannerOutcome.EXECUTION_PLAN,
            request_id=new_request.request_id,
            proposed_plan=plan,
        ),
        context=new_request.context,
        planning_sequence=new_request.sequence,
    ))
    assert continued.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert continued.accepted_plan == plan

    planning_state = apply_controller_decision_to_state(restored, resumed)
    accepted_state = apply_controller_decision_to_state(planning_state, continued)
    brain_input = build_brain_input({
        "messages": [
            HumanMessage(content="Do the work"),
            HumanMessage(content="The src directory"),
        ],
        "execution_state": accepted_state,
    })
    assert accepted_state.protocol_visible.original_user_request == "Do the work"
    assert accepted_state.protocol_visible.clarification == "The src directory"
    assert brain_input.context.user_request == "Do the work"
    assert brain_input.context.clarification == "The src directory"
    assert brain_input.context.user_request != "The src directory"


def test_retryable_failures_have_two_attempts_and_unplannable_does_not_retry():
    ctrl = CortexController(20)
    dispatch, request = authorize(ctrl)
    retry = ctrl.decide(result_input(dispatch, request, PlannerResult(
        outcome=PlannerOutcome.FAILED, request_id=request.request_id,
        planner_route="info",
        failure_category=PlanningFailureCategory.INVALID_OUTPUT,
        message="bad schema",
    )))
    assert retry.decision_type == ControllerDecisionType.DISPATCH_PLANNER
    assert retry.planning_request.episode_id == request.episode_id
    assert retry.planning_request.attempt == 2
    assert retry.planning_request.sequence == 2
    assert retry.planning_request.planner_route == "info"
    assert retry.planning_request.capabilities == request.capabilities
    assert not retry.clear_planning_request

    exhausted = ctrl.decide(result_input(
        retry, retry.planning_request, PlannerResult(
            outcome=PlannerOutcome.FAILED, request_id=retry.planning_request.request_id,
            failure_category=PlanningFailureCategory.PROVIDER_FAILURE,
            message="provider unavailable",
        ),
    ))
    assert exhausted.decision_type == ControllerDecisionType.TERMINATE
    assert exhausted.reason == "planning_retry_exhausted"

    dispatch, request = authorize(ctrl)
    unplannable = ctrl.decide(result_input(dispatch, request, PlannerResult(
        outcome=PlannerOutcome.FAILED, request_id=request.request_id,
        failure_category=PlanningFailureCategory.UNPLANNABLE,
        message="capability unavailable",
    )))
    assert unplannable.decision_type == ControllerDecisionType.TERMINATE
    assert unplannable.reason == "unplannable"


def test_revise_clarification_preserves_operation_base_and_failure_context():
    ctrl = controller()
    base = build_controller_input(active_state(ReplanTrigger.REPEATED_TOOL_FAILURE))
    plan = base.active_plan
    dispatch = ctrl.decide(base)
    request = dispatch.planning_request
    paused = ctrl.decide(base.model_copy(update={
        "cursor": dispatch.cursor, "tool_result": None,
        "planning_request": request, "planning_sequence": request.sequence,
        "planner_result": PlannerResult(
        outcome=PlannerOutcome.CLARIFICATION_REQUIRED,
        request_id=request.request_id, planner_route="info", message="Which other file?",
    )}))
    resumed = ctrl.decide(base.model_copy(update={
        "cursor": paused.cursor, "tool_result": None,
        "planning_sequence": request.sequence,
        "planning_clarification": paused.planning_clarification, "user_input": "README.md",
    })).planning_request
    assert resumed.operation == PlanningOperation.REVISE
    assert (resumed.base_plan_id, resumed.base_revision) == (plan.plan_id, plan.revision)
    assert resumed.reason == request.reason
    assert resumed.failure_json == request.failure_json
    assert resumed.context.user_request == request.context.user_request
    assert resumed.context.clarification_question == "Which other file?"
    assert resumed.context.clarification == "README.md"
    assert resumed.planner_route == "info"
    assert resumed.capabilities == request.capabilities


def test_planner_result_binding_rejects_stale_and_unbound_results():
    ctrl = CortexController(20)
    dispatch, request = authorize(ctrl)
    result = PlannerResult(
        outcome=PlannerOutcome.DIRECT_RESPONSE, request_id="different",
    )
    try:
        ctrl.decide(result_input(dispatch, request, result).model_copy(update={
            "planner_result": result,
        }))
    except ValueError as exc:
        assert "request identity" in str(exc)
    else:
        raise AssertionError("stale PlannerResult was accepted")

    try:
        ctrl.decide(initial_input(planner_result=result))
    except ValueError as exc:
        assert "pending PlanningRequest" in str(exc)
    else:
        raise AssertionError("unbound PlannerResult was accepted")


def test_revise_no_plan_is_rejected_and_keeps_accepted_plan():
    ctrl = controller()
    state = active_state()
    base = build_controller_input(state)
    plan = base.active_plan
    dispatch = ctrl.decide(base)
    request = dispatch.planning_request
    decision = ctrl.decide(base.model_copy(update={
        "cursor": dispatch.cursor, "brain_result": None,
        "planning_request": request, "planning_sequence": request.sequence,
        "planner_result": PlannerResult(
            outcome=PlannerOutcome.DIRECT_RESPONSE, request_id=request.request_id,
        ),
    }))
    assert decision.decision_type == ControllerDecisionType.TERMINATE
    planning_state = apply_controller_decision_to_state(state["execution_state"], dispatch)
    rejected_state = apply_controller_decision_to_state(planning_state, decision)
    assert rejected_state.protocol_visible.active_plan == planning_state.protocol_visible.active_plan


def test_planning_requests_and_controller_decisions_have_one_production_constructor_owner():
    creators = set()
    for path in Path('core').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'PlanningRequest', 'ControllerDecision'}:
                creators.add(path.as_posix())
    assert creators == {'core/protocol/controller.py'}


def test_revise_keeps_typed_execution_facts_outside_recent_history():
    identity = ExecutionIdentity(execution_id="current", protocol_version="1")
    done = ExecutionStep(step_id="done", title="Discover", status=StepStatus.COMPLETED)
    active = ExecutionStep(step_id="active", title="Process", status=StepStatus.ACTIVE)
    plan = ExecutionPlan(plan_id="plan", revision=3, objective="Inspect", steps=(done, active))
    record = ToolExecutionRecord(
        execution_id="current", plan_id="plan", plan_revision=3, step_id="done",
        tool_name="list_files",
        result=ToolResult(request_id="listed", success=True, message="listed", data={"files": ["a.py"]}),
    )
    state = {
        "messages": [
            HumanMessage(content="Inspect workspace"),
            ToolMessage(content="serialized tool transcript", tool_call_id="listed"),
            AIMessage(content="Brain lifecycle progress"),
        ],
        "brain_result": BrainResult(
            outcome=BrainOutcome.REPLAN_REQUEST,
            replan_request=ReplanRequest(reason="Need another approach", failed_step_id="active"),
        ),
        "execution_state": ExecutionState(
            protocol_visible=ProtocolVisibleState(
                identity=identity,
                cursor=ExecutionCursor(
                    phase=ExecutionPhase.EXECUTING, step_id="active", plan_revision=3,
                ),
                active_plan=plan, active_step=active, completed_step_ids=("done",),
                retry=RetryMetadata(step_id="active", retry_count=1, max_retries=3),
            ),
            working=WorkingState(tool_execution_history=(record,)),
        ),
    }
    request = CortexController(24).decide(build_controller_input(state)).planning_request
    assert request.operation.value == "revise"
    assert request.context.recent_history == ()
    assert request.base_plan == plan
    assert request.completed_step_ids == ("done",)
    assert request.interrupted_step == active
    assert request.trigger == ReplanTrigger.BRAIN_REQUESTED
    assert request.reason == "Need another approach"
    assert request.retry.retry_count == 1
    assert len(request.evidence_json) == 1 and "listed" in request.evidence_json[0]
