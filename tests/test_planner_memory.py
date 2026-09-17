"""Slice 4 bounded Planner context without memory execution authority."""

from datetime import datetime, timezone

import pytest
from langchain_core.messages import HumanMessage

from core.graph_planner import create_planner_node
from core.memory import (
    ConversationContinuity, ConversationMemory, FactCategory, MemoryFact,
    MemorySource, OpenQuestion, QuestionStatus, SourceKind, TurnStatus,
)
from core.planner import planning_request_context
from core.planner import PlannerRouting, PlannerService
from core.planner_contract import PlannerProposal, PlannerProposalResultType
from core.planner_memory import PlannerMemoryLimits, project_planner_memory
from core.protocol.bridge import build_brain_input, build_controller_input
from core.protocol.enums import (
    ExecutionPhase, PlannerOutcome, PlanningOperation, ReplanTrigger, WorkerRole,
)
from core.protocol.models import (
    ControllerDecision, ControllerDecisionType, ExecutionContext, ExecutionCursor,
    ExecutionIdentity, ExecutionPlan, ExecutionState, ExecutionStep,
    PlannerResult, PlanningCapabilities, PlanningRequest, ProtocolVisibleState,
)


def source(kind, turn):
    if kind in (SourceKind.HUMAN, SourceKind.INFERRED):
        return MemorySource(kind=kind, turn_index=turn, turn_id=f"turn-{turn}")
    if kind == SourceKind.TOOL:
        return MemorySource(kind=kind, turn_index=turn, execution_id=f"exec-{turn}",
                            tool_request_id=f"request-{turn}", tool_result_ref=f"result-{turn}",
                            tool_success=True)
    return MemorySource(kind=kind, turn_index=turn, execution_id=f"exec-{turn}",
                        plan_id="plan", plan_revision=1, step_id="step",
                        accepted_result_ref=f"accepted-{turn}")


def fact(text, kind, turn, *, project=False, key=None):
    return MemoryFact(
        category=FactCategory.PROJECT_REPOSITORY if project else FactCategory.USER_PREFERENCE,
        scope_key=key or f"key-{turn}", text=text, source=source(kind, turn),
    )


def memory():
    return ConversationMemory(
        facts=(fact("Use concise answers", SourceKind.HUMAN, 2),
               fact("Repository uses Python", SourceKind.ACCEPTED_RESULT, 3, project=True),
               fact("Tool observed files", SourceKind.TOOL, 3, project=True, key="tool-observation"),
               fact("Maybe uses Rust", SourceKind.INFERRED, 4, project=True, key="language-guess")),
        continuity=ConversationContinuity(
            topic="Inspect repository", status=TurnStatus.COMPLETED,
            outcome_note="Three files inspected", pending_follow_up="Check tests",
            source_turn_index=4, source_turn_id="turn-4", source_execution_id="exec-4",
        ),
        questions=(
            OpenQuestion(question_id="open", text="Which test next?",
                         source_turn_index=4, source_turn_id="turn-4"),
            OpenQuestion(question_id="resolved", text="Which file?",
                         source_turn_index=1, source_turn_id="turn-1",
                         status=QuestionStatus.RESOLVED, resolved_turn_index=2),
        ),
    )


def test_empty_projection_and_authority_classes_are_explicit():
    assert project_planner_memory(ConversationMemory()).user_facts == ()
    original = memory()
    before = original.model_dump_json()
    projected = project_planner_memory(original)
    assert [(f.text, f.authority) for f in projected.user_facts] == [
        ("Use concise answers", "explicit_user"),
    ]
    assert [(f.text, f.authority) for f in projected.project_facts] == [
        ("Repository uses Python", "controller_accepted"),
        ("Tool observed files", "tool_supported"),
        ("Maybe uses Rust", "inferred"),
    ]
    assert projected.continuity.topic == "Inspect repository"
    assert projected.continuity.terminal_status == "completed"
    assert [q.text for q in projected.open_questions] == ["Which test next?"]
    assert original.model_dump_json() == before


def test_counts_and_record_limits_keep_whole_records():
    original = memory()
    projected = project_planner_memory(original, limits=PlannerMemoryLimits(
        max_user_facts=0, max_project_facts=1, max_open_questions=0,
        max_record_chars=240, max_continuity_chars=400, max_total_chars=3000,
    ))
    assert projected.user_facts == () and projected.open_questions == ()
    assert [f.text for f in projected.project_facts] == ["Repository uses Python"]
    short = project_planner_memory(original, limits=PlannerMemoryLimits(
        max_record_chars=10, max_continuity_chars=20,
    ))
    assert short.user_facts == () and short.project_facts == ()
    assert short.continuity is None
    assert short.open_questions == ()


def test_expired_continuity_is_omitted_without_dropping_durable_facts():
    original = memory().model_copy(update={
        "continuity": memory().continuity.model_copy(update={"expires_after_turn": 5}),
    })
    projected = project_planner_memory(original, current_turn_index=6)
    assert projected.continuity is None
    assert projected.user_facts[0].text == "Use concise answers"


def test_total_budget_prefers_strong_sources_then_recent_within_rank():
    original = ConversationMemory(facts=(
        fact("old accepted", SourceKind.ACCEPTED_RESULT, 1, project=True, key="a"),
        fact("new accepted", SourceKind.ACCEPTED_RESULT, 5, project=True, key="b"),
        fact("inferred", SourceKind.INFERRED, 8, project=True, key="c"),
    ))
    first = project_planner_memory(original, limits=PlannerMemoryLimits(max_total_chars=230))
    second = project_planner_memory(original, limits=PlannerMemoryLimits(max_total_chars=230))
    assert first == second
    assert len(first.model_dump_json()) <= 230
    assert [f.text for f in first.project_facts] == ["new accepted"]


def authorized_state(*, revise=False, projection=None):
    identity = ExecutionIdentity(execution_id="current", protocol_version="1")
    plan = ExecutionPlan(plan_id="plan", revision=1, objective="Old task", steps=(
        ExecutionStep(step_id="done", title="Done"),
        ExecutionStep(step_id="active", title="Active"),
    )) if revise else None
    operation = PlanningOperation.REVISE if revise else PlanningOperation.CREATE
    cursor = ExecutionCursor(
        phase=ExecutionPhase.REPLANNING if revise else ExecutionPhase.PLANNING,
        current_worker=WorkerRole.PLANNER,
        plan_revision=1 if revise else None,
    )
    request = PlanningRequest(
        request_id="request", episode_id="episode", sequence=1,
        created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        operation=operation, identity=identity,
        context=ExecutionContext(user_request="Use long answers instead", role=WorkerRole.PLANNER,
                                 recent_history=("Earlier question",)),
        capabilities=PlanningCapabilities(available_tools=("list_files",)),
        base_plan=plan, base_plan_id=plan.plan_id if plan else None,
        base_revision=plan.revision if plan else None,
        trigger=ReplanTrigger.BRAIN_REQUESTED if revise else None,
        reason="Current failure: permission denied" if revise else "",
        completed_step_ids=("done",) if revise else (),
        interrupted_step=plan.steps[1] if revise else None,
    )
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_PLANNER,
        next_worker=WorkerRole.PLANNER, cursor=cursor,
        planning_request=request, requires_checkpoint=True,
        requires_replan=revise,
    )
    state = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=identity, cursor=cursor, active_plan=plan,
        planning_request=request, planning_sequence=1,
    ))
    return {
        "execution_state": state, "controller_decision": decision,
        "planner_memory_context": projection or project_planner_memory(memory()),
        "messages": [HumanMessage(content="Use long answers instead")],
    }, request


class SpyPlanner:
    def __init__(self):
        self.requests = []

    def run(self, request, *, retrieve):
        self.requests.append(request)
        return PlannerResult(outcome=PlannerOutcome.DIRECT_RESPONSE, request_id=request.request_id)


@pytest.mark.parametrize("revise", [False, True])
def test_authorized_planner_receives_ephemeral_context_without_protocol_mutation(revise):
    state, original_request = authorized_state(revise=revise)
    spy = SpyPlanner()
    node = create_planner_node(planner_service=spy, rag_service=None, rag_top_k=1,
                               tools_set={"list_files"})
    result = node(state)
    assert result["planner_result"].request_id == original_request.request_id
    received = spy.requests[0]
    assert received.context.planner_memory_context == state["planner_memory_context"]
    assert received.context.user_request == "Use long answers instead"
    assert received.context.recent_history == ("Earlier question",)
    assert received.operation == original_request.operation
    assert original_request.context.planner_memory_context is None
    assert state["execution_state"].protocol_visible.planning_request == original_request
    assert "Use concise answers" not in state["execution_state"].protocol_visible.model_dump_json()
    rendered = planning_request_context(received)
    assert '"planner_memory_context"' in rendered
    assert "current user_request is the active instruction" in rendered
    assert "supersedes conflicting remembered user facts" in rendered
    assert "never an active Controller execution" in rendered
    assert "Controller progress, failure evidence" in rendered
    if revise:
        assert "Current failure: permission denied" in rendered
        assert received.completed_step_ids == ("done",)


def test_projection_does_not_enter_brain_or_finalizer_context():
    state, _ = authorized_state()
    assert build_brain_input(state).context.planner_memory_context is None
    assert build_controller_input(state).context.planner_memory_context is None
    assert state["execution_state"].protocol_visible.planning_request.context.planner_memory_context is None
    with pytest.raises(ValueError, match="Planner-only"):
        ExecutionContext(user_request="x", role=WorkerRole.BRAIN,
                         planner_memory_context=state["planner_memory_context"])


@pytest.mark.parametrize("route", ["conversation", "action"])
def test_current_request_and_memory_use_one_existing_planner_generation(route):
    state, request = authorized_state()
    worker_request = request.model_copy(update={
        "context": request.context.model_copy(update={
            "planner_memory_context": state["planner_memory_context"],
        }),
    })

    class Provider:
        routes = 0
        generations = 0
        messages = None

        def route(self, user_request):
            self.routes += 1
            assert user_request == "Use long answers instead"
            return PlannerRouting(route)

        def generate(self, messages):
            self.generations += 1
            self.messages = messages
            return PlannerProposal(result=PlannerProposalResultType.NO_PLAN_REQUIRED)

    provider = Provider()
    service = PlannerService(provider=provider, tools_set={"list_files"},
                             domain_tool_map={}, mutating_tools=set(),
                             system_capabilities_text="")
    service.run(worker_request)
    assert provider.routes == provider.generations == 1
    assert provider.messages[-1].role == "human"
    assert provider.messages[-1].content == "Use long answers instead"
    context_message = provider.messages[-2].content
    assert '"recent_history": ["Earlier question"]' in context_message
    assert '"planner_memory_context"' in context_message
    assert "Use concise answers" in context_message
