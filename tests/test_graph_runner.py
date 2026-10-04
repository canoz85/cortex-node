from core.graph_messages import ACCEPTED_FINALIZER_PROVENANCE, CONVERSATION_PROVENANCE_KEY
from core.graph_runner import PendingClarification, run_prompt
from core.protocol.bridge import build_controller_input
from core.protocol.controller import CortexController
from core.protocol.enums import (
    BrainOutcomeKind,
    ControllerDecisionType,
    ExecutionPhase,
    ExecutionStatus,
    PlannerOutcome,
    WorkerRole,
)
from core.protocol.models import (
    BrainOutcome,
    ControllerDecision,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionState,
    ExecutionSummary,
    FinalizationResult,
    PlannerResult,
    PlanningClarification,
    ProtocolVisibleState,
    ToolExecutionRecord,
    ToolRequest,
    ToolResult as ProtocolToolResult,
    WorkingState,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class FakeApp:
    def __init__(self, events: list[dict]):
        self._events = events
        self.initial_state: dict | None = None

    def stream(self, initial_state):
        self.initial_state = initial_state
        for event in self._events:
            yield event


def test_run_prompt_transports_injected_pause_and_execution_reference_on_reply(capsys):
    from core.protocol.controller import CortexController
    from core.protocol.models import ControllerInput, ExecutionContext
    request = CortexController(20).decide(ControllerInput(
        identity=ExecutionIdentity(execution_id="mqtt-run", protocol_version="1"),
        cursor=ExecutionCursor(),
        context=ExecutionContext(user_request="Connect to the MQTT broker"),
    )).planning_request
    marker = PlanningClarification(
        prompt="Which MQTT password should I use?",
        request=request,
    )
    cursor = ExecutionCursor(
        phase=ExecutionPhase.WAITING,
        current_worker=WorkerRole.CONTROLLER,
    )
    execution_state = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=ExecutionIdentity(execution_id="mqtt-run", protocol_version="1"),
        cursor=cursor,
        planning_clarification=marker,
        planning_sequence=1,
    ))
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.PAUSE,
        reason="needs_input",
        next_worker=WorkerRole.CONTROLLER,
        cursor=cursor,
        planning_clarification=marker,
        requires_checkpoint=True,
    )
    app = FakeApp([{"controller": {
        "execution_state": execution_state,
        "controller_decision": decision,
        "planner_result": PlannerResult(
            outcome=PlannerOutcome.CLARIFICATION_REQUIRED,
            request_id="request-1",
            message=marker.prompt,
        ),
    }}])
    sink: list[PendingClarification] = []

    history, _ = run_prompt(
        app,
        marker.request.context.user_request,
        run_id="mqtt-run",
        clarification_sink=sink,
    )

    assert len(sink) == 1
    assert sink[0].execution_state.protocol_visible.identity.execution_id == "mqtt-run"
    assert marker.prompt in capsys.readouterr().out

    resumed_app = FakeApp([])
    run_prompt(
        resumed_app,
        "secret-value",
        history=history,
        pending_clarification=sink[0],
    )
    assert resumed_app.initial_state["execution_state"] is execution_state
    assert resumed_app.initial_state["run_id"] == "mqtt-run"
    assert [message.content for message in resumed_app.initial_state["messages"]][-2:] == [
        marker.request.context.user_request,
        "secret-value",
    ]


def test_run_prompt_renders_portable_controller_tool_result_concisely(capsys):
    request = ToolRequest(
        request_id="call-1",
        tool_name="list_files",
        arguments={"path": "."},
    )
    brain_result = BrainOutcome(
        outcome=BrainOutcomeKind.TOOL_REQUESTED,
        tool_request=request,
    )
    request_message = AIMessage(
        content="",
        tool_calls=[{
            "name": "list_files", "args": {"path": "."},
            "id": "call-1", "type": "tool_call",
        }],
    )
    evidence = {"entries": [f"large-entry-{index}" for index in range(100)]}
    result = ProtocolToolResult(
        request_id="call-1",
        success=True,
        message="Listing for .",
        rendered_output="stdout that must stay private",
        data=evidence,
    )
    record = ToolExecutionRecord(
        step_id="step-1",
        tool_name="list_files",
        arguments={"path": "."},
        result=result,
    )
    execution_state = ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=ExecutionIdentity(execution_id="live-path", protocol_version="1"),
            cursor=ExecutionCursor(),
        ),
        working=WorkingState(
            last_tool_result=result,
            tool_execution_history=(record,),
        ),
    )
    raw_payload = "<tool_result_json>" + result.model_dump_json()
    app = FakeApp([
        {"controller": {
            "brain_result": brain_result,
            "messages": [request_message],
        }},
        {"controller": {
            "execution_state": execution_state,
            "messages": [ToolMessage(content=raw_payload, tool_call_id="call-1")],
        }},
    ])

    run_prompt(app, "list files")

    output = capsys.readouterr().out
    assert output.count("[brain]") == 1
    assert "Calling list_files" in output
    assert "Calling list_files with" not in output
    assert output.count("[tool:list_files]") == 1
    assert "Listing for ." in output
    assert "<tool_result_json>" not in output
    assert "large-entry-99" not in output
    assert "stdout that must stay private" not in output
    assert execution_state.working.last_tool_result is result
    assert execution_state.working.tool_execution_history[-1].result is result


"""Cross-turn Planner context ownership regressions."""


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


def test_runner_projects_session_history_from_injected_finalization_events():
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
        "Previous request", "Explicit clarification", "Previous final answer",
    )
    assert [message.content for message in history] == [*request.context.recent_history, request.context.user_request]
