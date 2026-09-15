"""Brain graph adapter, prompt contract, and execution flow integration tests."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from core.brain import build_brain_output_protocol
from core.finalizer import Finalizer
from core.graph import build_app
from core.graph_brain import create_brain_node
from core.graph_capture import create_capture_tool_output_node
from core.graph_constants import CASUAL_SYSTEM_PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE
from core.graph_controller import create_controller_node
from core.models import ToolResult as TransportToolResult
from core.protocol.bridge import (
    _legacy_brain_result_to_model, brain_result_to_legacy, build_brain_input,
    build_controller_input,
)
from core.protocol.enums import BrainOutcomeKind as Kind, ControllerDecisionType, ExecutionPhase, ExecutionStatus, PlannerOutcome, StepStatus, WorkerRole
from core.protocol.models import (
    BrainOutcome, BrainUsage, ControllerDecision, ExecutionCursor, ExecutionIdentity, ExecutionPlan,
    ExecutionState, ExecutionStep, PlannerResult, ProtocolVisibleState,
    RetryMetadata, StepCompletionEvidence, ToolRequest, ToolResult, WorkingState,
)


def execution_state():
    step = ExecutionStep(step_id="s1", title="Read the file", status=StepStatus.ACTIVE)
    return ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=ExecutionIdentity(execution_id="adapter-test", protocol_version="1.0"),
            cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING, step_id="s1", plan_revision=1, current_worker=WorkerRole.BRAIN),
            active_plan=ExecutionPlan(plan_id="p1", steps=(step,)), active_step=step,
        ),
        working=WorkingState(last_tool_result=ToolResult(request_id="old", success=True, message="Read")),
    )


def authorize_brain(state):
    execution = state["execution_state"]
    return {**state, "controller_decision": ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_BRAIN,
        next_worker=WorkerRole.BRAIN,
        cursor=execution.protocol_visible.cursor,
    )}


def node(**kwargs):
    return create_brain_node(
        brain_llm=None, tool_brain_llm=None, agent_system_prompt="active",
        casual_system_prompt="casual", tools_set={"read_file"},
        show_raw_llm=False, **kwargs,
    )


def test_adapter_only_translates_input_output_and_consumed_tool_evidence():
    outcome = BrainOutcome(
        outcome=Kind.TOOL_REQUESTED, step_id="s1",
        tool_request=ToolRequest(request_id="domain-id", tool_name="read_file", arguments={"path": "a"}),
        usage=BrainUsage(prompt_tokens=5, completion_tokens=3),
    )
    invocations = []

    class Service:
        def run(self, value):
            invocations.append(value)
            return outcome

    original = authorize_brain({"execution_state": execution_state(), "messages": [HumanMessage(content="read file")], "steps": 2})
    update = node(brain_service=Service())(original)
    assert invocations == [build_brain_input(original)]
    assert update["brain_result"] is outcome
    assert update["steps"] == 3
    assert update["token_usage"].total_tokens == 8
    assert update["messages"][0].tool_calls == [{"name": "read_file", "args": {"path": "a"}, "id": "domain-id", "type": "tool_call"}]
    assert update["execution_state"].working.last_tool_result is None
    assert original["execution_state"].working.last_tool_result is not None
    assert update["execution_state"].protocol_visible.active_step == original["execution_state"].protocol_visible.active_step
    assembled = build_controller_input({**original, **update})
    assert assembled.brain_result is outcome
    assert assembled.tool_result is None


@pytest.mark.parametrize("outcome", [
    BrainOutcome(outcome=Kind.STEP_COMPLETED, step_id="s1", completion_evidence=StepCompletionEvidence(step_id="s1", summary="Read")),
    BrainOutcome(outcome=Kind.FINAL_ANSWER_READY, message="Finalization requested."),
    BrainOutcome(outcome=Kind.INVALID_OUTPUT, error_code="invalid", message="Bad output"),
])
def test_bridge_preserves_typed_payloads_without_reparsing_messages(outcome):
    assert _legacy_brain_result_to_model(brain_result_to_legacy(outcome)) == outcome
    value = build_controller_input({
        "execution_state": execution_state(), "messages": [AIMessage(content="STEP COMPLETED: misleading")],
        "brain_result": outcome,
    })
    assert value.brain_result is outcome


def test_all_brain_prompts_align_with_the_outcome_contract():
    assert "Output capability is specified" not in SYSTEM_PROMPT_TEMPLATE
    assert "BRAIN OUTCOME CONTRACT" in CASUAL_SYSTEM_PROMPT_TEMPLATE
    for prompt in (SYSTEM_PROMPT_TEMPLATE, CASUAL_SYSTEM_PROMPT_TEMPLATE):
        assert "STEP COMPLETED" not in prompt
        assert "STEP FAILED" not in prompt
        assert "starting with" not in prompt


@pytest.mark.parametrize("kind", [Kind.TOOL_REQUESTED, Kind.STEP_COMPLETED, Kind.REPLAN_REQUESTED, Kind.STEP_FAILED])
def test_prompt_examples_are_complete_json_envelopes(kind):
    protocol = build_brain_output_protocol(supports_native_tool_calls=False, tools_enabled=True)
    examples = [json.loads(line) for line in protocol.splitlines() if line.startswith("{")]
    assert any(example["kind"] == kind.name or Kind.__members__[example["kind"]] == kind for example in examples)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("supports_native_tool_calls", [True, False])
def test_current_graph_runs_typed_brain_tool_completion_and_final_answer(direct, supports_native_tool_calls):
    calls = []
    tool_calls = []
    tool_node_bindings = []
    snapshots = []
    bindings = []

    class Model:
        def __init__(self, tool_enabled=False):
            self.tool_enabled = tool_enabled

        def bind_tools(self, tools):
            assert supports_native_tool_calls
            bindings.append(tools)
            return Model(tool_enabled=True)

        def invoke(self, messages):
            rendered = "\n".join(message.content for message in messages)
            calls.append((self.tool_enabled, rendered))
            if any(message.content.startswith("Active step:") for message in messages):
                if "Execution evidence v1:" not in rendered:
                    if supports_native_tool_calls:
                        return AIMessage(content="", tool_calls=[{
                            "name": "read_file", "args": {"path": "a.py"}, "id": "native-call-id",
                        }])
                    return AIMessage(content='{"name":"read_file","arguments":{"path":"a.py"}}')
                if supports_native_tool_calls:
                    return AIMessage(content="", tool_calls=[{
                        "name": "brain_step_completed",
                        "args": {"message": "File read"},
                        "id": "native-completion-id",
                    }])
                return AIMessage(content=json.dumps({
                    "kind": "STEP_COMPLETED", "step_id": "s1", "message": "File read",
                }))
            return AIMessage(content="File contents reported")

    @tool
    def read_file(path: str) -> str:
        """Read a file inside the workspace."""
        return "unused"

    def graph_nodes_factory(**kwargs):
        class Renderer:
            def render(self, _request, _summary):
                return "File contents reported"

        controller = create_controller_node(
            finalizer=Finalizer(answer_renderer=Renderer()),
            worker_ports=kwargs["worker_ports"],
        )

        def observe_controller(state):
            snapshots.append(state)
            return controller(state)
        observe_controller._portable_dispatch = True

        def planner(_state):
            request_id = _state["execution_state"].protocol_visible.planning_request.request_id
            if direct:
                return {"planner_result": PlannerResult(outcome=PlannerOutcome.DIRECT_RESPONSE, request_id=request_id)}
            return {"planner_result": PlannerResult(
                outcome=PlannerOutcome.EXECUTION_PLAN,
                request_id=request_id,
                proposed_plan=ExecutionPlan(plan_id="p1", steps=(ExecutionStep(step_id="s1", title="Read the file"),)),
            )}

        brain = create_brain_node(**{name: kwargs[name] for name in (
            "brain_llm", "tool_brain_llm", "agent_system_prompt",
            "casual_system_prompt", "tools_set", "show_raw_llm",
            "supports_native_tool_calls",
        )})
        return observe_controller, planner, brain, create_capture_tool_output_node(), lambda _state: {}

    def tool_factory(_tools):
        tool_node_bindings.extend(_tools)
        def invoke(state):
            request = state["execution_state"].protocol_visible.pending_tool_request
            transported = state["messages"][-1].tool_calls[0]
            assert transported["id"] == request.request_id
            assert transported["args"] == request.arguments
            tool_calls.append(request)
            return {"messages": [ToolMessage(
                content=TransportToolResult(success=True, message="Read", data={"path": "a.py", "content": "print('a')"}).to_tool_output(),
                tool_call_id=request.request_id,
            )]}
        return invoke

    app = build_app(
        rag_factory=lambda *_args: object(), tool_list_factory=lambda *_args: [read_file],
        chat_model_factory=lambda *_args: Model(), graph_nodes_factory=graph_nodes_factory,
        tool_node_factory=tool_factory, project_root=Path("."),
        supports_native_tool_calls=supports_native_tool_calls,
    )
    result = app.invoke({
        "messages": [HumanMessage(content="Read a.py")], "steps": 0,
        "execution_state": ExecutionState(protocol_visible=ProtocolVisibleState(
            identity=ExecutionIdentity(execution_id="brain-graph-test", protocol_version="1"), cursor=ExecutionCursor(),
            retry=RetryMetadata(max_retries=1),
        )),
    })
    protocol = result["execution_state"].protocol_visible
    assert protocol.status == ExecutionStatus.COMPLETED
    assert protocol.cursor.phase == ExecutionPhase.COMPLETED
    assert protocol.cursor.step_id is None
    assert protocol.active_step is None
    assert result["messages"][-1].content == "File contents reported"
    assert len(tool_calls) == (0 if direct else 1)
    assert [enabled for enabled, _ in calls] == (
        [] if direct else [supports_native_tool_calls, supports_native_tool_calls]
    )
    assert len(bindings) == (1 if supports_native_tool_calls else 0)
    assert tool_node_bindings == [read_file]
    if supports_native_tool_calls:
        assert bindings[0][0] is read_file
        assert {item["function"]["name"] for item in bindings[0][1:]} == {
            "brain_step_completed", "brain_step_failed", "brain_replan_requested",
        }
    assert all("BRAIN OUTCOME CONTRACT" in prompt for _, prompt in calls)
    if not direct:
        if supports_native_tool_calls:
            assert "Return exactly one native call" in calls[0][1]
            assert "brain_step_completed" in calls[0][1]
            assert '"kind":"TOOL_REQUESTED"' not in calls[0][1]
        else:
            assert "Tool format: JSON" in calls[0][1]
            assert '"kind":"TOOL_REQUESTED"' in calls[0][1]
            assert '"parameters"' in calls[0][1]
            assert '"path"' in calls[0][1]
        assert "Execution evidence v1:" in calls[-1][1]
        assert protocol.completion_provenance[0].tool_request_ids == (tool_calls[0].request_id,)
        assert protocol.completed_step_ids == ("s1",)
        assert protocol.retry.retry_count == 0
        assert protocol.active_plan.steps[0].status == StepStatus.COMPLETED
        assert app.builder.edges == {("__start__", "controller")}
        assert set(app.get_graph().nodes) == {
            "__start__", "controller", "__end__",
        }
    assert set(app.builder.branches) == {"controller"}
