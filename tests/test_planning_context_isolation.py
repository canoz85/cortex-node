"""Cross-turn Planner context ownership regressions."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.graph_messages import (
    ACCEPTED_FINALIZER_PROVENANCE, CONVERSATION_PROVENANCE_KEY,
)
from core.graph_runner import run_prompt
from core.protocol.bridge import build_controller_input
from core.protocol.controller import CortexController
from core.protocol.enums import (
    BrainOutcome, ExecutionPhase, ExecutionStatus, ReplanTrigger, StepStatus,
)
from core.protocol.models import (
    BrainResult, ExecutionContext, ExecutionCursor, ExecutionIdentity,
    ExecutionPlan, ExecutionState, ExecutionStep, ExecutionSummary,
    FinalizationResult, ProtocolVisibleState, ReplanRequest, RetryMetadata,
    ToolExecutionRecord, ToolResult, WorkingState,
)


def finalization(answer="Workspace inspected."):
    return FinalizationResult(
        execution_summary=ExecutionSummary(
            execution_id="prior", status=ExecutionStatus.COMPLETED,
            summary_text="Completed.",
        ),
        final_answer=answer,
    )


class Events:
    def __init__(self, events):
        self.events = events
        self.initial_state = None

    def stream(self, initial_state):
        self.initial_state = initial_state
        yield from self.events


def test_runner_returns_only_users_and_accepted_finalizer_answers():
    events = Events([
        {"brain": {"messages": [AIMessage(
            content="Brain requested tool execution.",
            tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "call"}],
        )]}},
        {"tools": {"messages": [ToolMessage(content="full file contents", tool_call_id="call")]}},
        {"brain": {"messages": [AIMessage(content="Finalization requested.")]}},
        {"controller": {
            "finalization_result": finalization(),
            "messages": [AIMessage(content="Workspace inspected.")],
        }},
    ])
    history, _ = run_prompt(events, "Inspect workspace")
    assert [message.content for message in history] == ["Inspect workspace", "Workspace inspected."]
    assert isinstance(history[0], HumanMessage)
    assert history[1].additional_kwargs[CONVERSATION_PROVENANCE_KEY] == ACCEPTED_FINALIZER_PROVENANCE


def test_new_create_request_has_conversation_but_no_prior_execution_artifacts():
    accepted = AIMessage(
        content="Previous final answer",
        additional_kwargs={CONVERSATION_PROVENANCE_KEY: ACCEPTED_FINALIZER_PROVENANCE},
    )
    polluted = [
        HumanMessage(content="Previous request"),
        HumanMessage(content="Explicit clarification"),
        AIMessage(content="Brain requested tool execution.", tool_calls=[{
            "name": "read_file", "args": {"path": "a.py"}, "id": "call",
        }]),
        ToolMessage(content="full read_file contents", tool_call_id="call"),
        AIMessage(content="Finalization requested."),
        accepted,
    ]
    app = Events([])
    history, _ = run_prompt(app, "hi", history=polluted, run_id="new-execution")
    request = CortexController(24).decide(build_controller_input(app.initial_state)).planning_request
    assert request.operation.value == "create"
    assert request.context.recent_history == (
        "Previous request", "Explicit clarification", "Previous final answer", "hi",
    )
    assert [message.content for message in history] == list(request.context.recent_history)


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
    assert request.context.recent_history == ("Inspect workspace",)
    assert request.base_plan == plan
    assert request.completed_step_ids == ("done",)
    assert request.interrupted_step == active
    assert request.trigger == ReplanTrigger.BRAIN_REQUESTED
    assert request.reason == "Need another approach"
    assert request.retry.retry_count == 1
    assert len(request.evidence_json) == 1 and "listed" in request.evidence_json[0]
