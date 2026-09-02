from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

from core.graph_capture import create_capture_tool_output_node
from core.graph_controller import create_controller_node
from core.protocol.controller import CortexController
from core.graph import build_app
from core.graph_routing import route_after_controller
from core.graph_runner import run_prompt
from core.models import ToolResult as TransportToolResult
from core.protocol.enums import (
    AsyncJobStatus,
    BrainOutcome,
    CancellationSource,
    ControllerDecisionType,
    ExecutionPhase,
    ExecutionStatus,
    PlannerOutcome,
    StepStatus,
    WorkerRole,
)
from core.protocol.models import (
    AsyncJobPolicy,
    BrainResult,
    ControllerDecision,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    PlannerResult,
    ProtocolVisibleState,
    ToolExecutionRecord,
    ToolRequest,
    ToolResult,
    WorkingState,
)
from core.runtime.async_poller import LocalAsyncPollingRuntime
from core.state import AgentState


OBSERVED_AT = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)


class FakeStatusTool:
    name = "get_comfy_history"

    def __init__(self, output: str):
        self.output = output
        self.invocations: list[dict] = []

    def invoke(self, arguments: dict):
        self.invocations.append(arguments)
        return self.output


def _execution_state() -> ExecutionState:
    active_step = ExecutionStep(
        step_id="step-1",
        title="Generate image",
        primary_tool="run_comfy_workflow",
        status=StepStatus.ACTIVE,
        attempt=1,
    )
    plan = ExecutionPlan(
        plan_id="plan-1",
        revision=1,
        objective="Generate an image",
        steps=(active_step,),
    )
    submitted = ToolExecutionRecord(
        step_id="step-1",
        tool_name="run_comfy_workflow",
        arguments={"workflow_json": {"1": {}}},
        result=ToolResult(
            request_id="submit-1",
            signature="run_comfy_workflow:{}",
            success=True,
            message="Submitted.",
            is_async_job=True,
            async_job_id="prompt-1",
            async_job_status=AsyncJobStatus.SUBMITTED,
            async_terminal=False,
            async_observed_at_utc=OBSERVED_AT,
        ),
    )
    return ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=ExecutionIdentity(
                execution_id="run-1",
                protocol_version="1.0",
            ),
            cursor=ExecutionCursor(
                phase=ExecutionPhase.WAITING,
                step_id="step-1",
                current_worker=WorkerRole.CONTROLLER,
            ),
            active_plan=plan,
            active_step=active_step,
        ),
        working=WorkingState(tool_execution_history=(submitted,)),
    )


def _await_decision() -> ControllerDecision:
    return ControllerDecision(
        decision_type=ControllerDecisionType.AWAIT_ASYNC_JOB,
        reason="Awaiting async job prompt-1.",
        next_worker=WorkerRole.CONTROLLER,
        cursor=ExecutionCursor(
            phase=ExecutionPhase.WAITING,
            step_id="step-1",
            current_worker=WorkerRole.CONTROLLER,
        ),
        async_job_id="prompt-1",
        resume_after_utc=OBSERVED_AT,
        requires_checkpoint=True,
    )


def _status_output(status: AsyncJobStatus) -> str:
    return TransportToolResult(
        success=status not in {AsyncJobStatus.FAILED, AsyncJobStatus.CANCELLED},
        message=f"Observed {status.value}.",
        is_async_job=True,
        async_job_id="prompt-1",
        async_job_status=status,
        async_terminal=status in {
            AsyncJobStatus.COMPLETED,
            AsyncJobStatus.FAILED,
            AsyncJobStatus.CANCELLED,
        },
        async_observed_at_utc=OBSERVED_AT,
    ).to_tool_output()


def _compiled_resume_graph(brain_invocations: list[ExecutionState]):
    workflow = StateGraph(AgentState)
    workflow.add_node("tools", lambda _state: {})
    workflow.add_node("capture_tool_output", create_capture_tool_output_node())
    workflow.add_node(
        "controller",
        create_controller_node(
            CortexController(
                max_reasoning_steps=10,
                now_utc=lambda: OBSERVED_AT,
            )
        ),
    )

    def brain_node(state: AgentState):
        execution_state = state["execution_state"]
        brain_invocations.append(execution_state)
        consumed_execution_state = execution_state.model_copy(
            update={
                "working": execution_state.working.model_copy(
                    update={"last_tool_result": None}
                )
            }
        )
        return {
            "brain_result": BrainResult(
                outcome=BrainOutcome.FINAL_ANSWER,
                message="Finished.",
                final_answer="Finished.",
            ),
            "execution_state": consumed_execution_state,
        }

    workflow.add_node("brain", brain_node)
    workflow.set_entry_point("tools")
    workflow.add_edge("tools", "capture_tool_output")
    workflow.add_edge("capture_tool_output", "controller")
    workflow.add_conditional_edges("controller", route_after_controller)
    workflow.add_edge("brain", "controller")
    return workflow.compile(checkpointer=InMemorySaver())


@pytest.mark.parametrize(
    ("status", "expected_brain_invocations"),
    [
        (AsyncJobStatus.RUNNING, 0),
        (AsyncJobStatus.COMPLETED, 1),
    ],
)
def test_local_polling_resumes_from_capture_without_brain_hot_loop(
    status,
    expected_brain_invocations,
):
    brain_invocations: list[ExecutionState] = []
    compiled_graph = _compiled_resume_graph(brain_invocations)
    config = {"configurable": {"thread_id": f"thread-{status.value}"}}
    initial_state = {
        "messages": [],
        "execution_state": _execution_state(),
        "controller_decision": _await_decision(),
    }
    compiled_graph.update_state(config, initial_state, as_node="controller")

    status_tool = FakeStatusTool(_status_output(status))
    runtime = LocalAsyncPollingRuntime(
        compiled_graph=compiled_graph,
        tools=[status_tool],
        sleep=lambda _seconds: None,
        now_utc=lambda: OBSERVED_AT,
    )

    events = list(
        runtime.poll_and_resume(
            config=config,
            decision=_await_decision(),
        )
    )

    assert status_tool.invocations == [{"prompt_id": "prompt-1"}]
    assert len(brain_invocations) == expected_brain_invocations
    assert events[0].keys() == {"capture_tool_output"}

    snapshot = compiled_graph.get_state(config)
    final_state = snapshot.values["execution_state"]
    history = final_state.working.tool_execution_history
    assert [record.result.async_job_status for record in history] == [
        AsyncJobStatus.SUBMITTED,
        status,
    ]

    final_decision = snapshot.values["controller_decision"]
    if status == AsyncJobStatus.RUNNING:
        assert final_decision.decision_type == ControllerDecisionType.AWAIT_ASYNC_JOB
    else:
        assert final_decision.decision_type == ControllerDecisionType.DISPATCH_SUMMARY

    stale_events = list(
        runtime.poll_and_resume(
            config=config,
            decision=_await_decision(),
        )
    )
    assert stale_events == []
    assert status_tool.invocations == [{"prompt_id": "prompt-1"}]


def test_run_prompt_reenters_runtime_only_for_await_decision():
    await_decision = _await_decision()
    summary_decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_SUMMARY,
        reason="Finished.",
        next_worker=WorkerRole.SUMMARY,
        execution_status=ExecutionStatus.COMPLETED,
        cursor=ExecutionCursor(
            phase=ExecutionPhase.COMPLETED,
            current_worker=WorkerRole.SUMMARY,
        ),
        terminal=True,
    )

    class FakeRuntime:
        def __init__(self):
            self.calls: list[tuple[dict, ControllerDecision]] = []

        def poll_and_resume(self, *, config, decision):
            self.calls.append((config, decision))
            return iter([
                {"brain": {"messages": [AIMessage(content="Finished.")]}},
                {"controller": {"controller_decision": summary_decision}},
            ])

    class FakeCheckpointedApp:
        def __init__(self):
            self.async_runtime = FakeRuntime()
            self.config = None
            self.initial_state = None

        def stream(self, initial_state, config=None):
            self.config = config
            self.initial_state = initial_state
            return iter([
                {"controller": {"controller_decision": await_decision}},
            ])

    app = FakeCheckpointedApp()

    policy = AsyncJobPolicy(poll_interval_seconds=3)
    history, _summary = run_prompt(
        app,
        "Generate an image",
        run_id="known-run-id",
        async_job_policy=policy,
    )

    assert len(app.async_runtime.calls) == 1
    poll_config, poll_decision = app.async_runtime.calls[0]
    assert poll_config == app.config
    assert poll_decision == await_decision
    assert app.config["configurable"]["thread_id"] == "known-run-id"
    assert (
        app.initial_state["execution_state"].protocol_visible.async_policy
        == policy
    )
    assert isinstance(history[-1], AIMessage)
    assert history[-1].content == "Finished."


def test_local_polling_resumes_existing_terminal_evidence_without_repolling():
    brain_invocations: list[ExecutionState] = []
    compiled_graph = _compiled_resume_graph(brain_invocations)
    config = {"configurable": {"thread_id": "thread-terminal-recovery"}}
    execution_state = _execution_state()
    completed_record = ToolExecutionRecord(
        step_id="step-1",
        tool_name="get_comfy_history",
        arguments={"prompt_id": "prompt-1"},
        result=ToolResult(
            request_id="poll-completed",
            success=True,
            message="Completed.",
            is_async_job=True,
            async_job_id="prompt-1",
            async_job_status=AsyncJobStatus.COMPLETED,
            async_terminal=True,
            async_observed_at_utc=OBSERVED_AT,
        ),
    )
    execution_state = execution_state.model_copy(
        update={
            "working": execution_state.working.model_copy(
                update={
                    "last_tool_result": None,
                    "tool_execution_history": (
                        *execution_state.working.tool_execution_history,
                        completed_record,
                    ),
                }
            )
        }
    )
    compiled_graph.update_state(
        config,
        {
            "messages": [],
            "execution_state": execution_state,
            "controller_decision": _await_decision(),
        },
        as_node="controller",
    )
    status_tool = FakeStatusTool(_status_output(AsyncJobStatus.COMPLETED))

    class ResourceCoordinator:
        def __init__(self):
            self.prepare_for_llm_calls = 0

        def prepare_for_llm(self):
            self.prepare_for_llm_calls += 1

    resource_coordinator = ResourceCoordinator()
    runtime = LocalAsyncPollingRuntime(
        compiled_graph=compiled_graph,
        tools=[status_tool],
        sleep=lambda _seconds: None,
        now_utc=lambda: OBSERVED_AT,
        resource_coordinator=resource_coordinator,
    )

    list(runtime.poll_and_resume(config=config, decision=_await_decision()))

    assert status_tool.invocations == []
    assert resource_coordinator.prepare_for_llm_calls == 1
    assert len(brain_invocations) == 1
    snapshot = compiled_graph.get_state(config)
    assert len(
        snapshot.values["execution_state"].working.tool_execution_history
    ) == 2


def test_local_polling_observes_only_the_provider_status_call():
    class RecordingResourceObserver:
        def __init__(self):
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        @contextmanager
        def observe_operation(self, *, component, operation, fields=None):
            self.calls.append((component, operation, dict(fields or {})))
            yield

    brain_invocations: list[ExecutionState] = []
    compiled_graph = _compiled_resume_graph(brain_invocations)
    config = {"configurable": {"thread_id": "thread-observed-poll"}}
    compiled_graph.update_state(
        config,
        {
            "messages": [],
            "execution_state": _execution_state(),
            "controller_decision": _await_decision(),
        },
        as_node="controller",
    )
    status_tool = FakeStatusTool(_status_output(AsyncJobStatus.RUNNING))
    observer = RecordingResourceObserver()
    runtime = LocalAsyncPollingRuntime(
        compiled_graph=compiled_graph,
        tools=[status_tool],
        sleep=lambda _seconds: None,
        now_utc=lambda: OBSERVED_AT,
        resource_observer=observer,
    )

    list(runtime.poll_and_resume(config=config, decision=_await_decision()))

    assert observer.calls == [
        (
            "async_provider",
            "get_comfy_history",
            {
                "provider": "comfyui",
                "async_job_id": "prompt-1",
                "thread_id": "thread-observed-poll",
            },
        )
    ]


def test_local_cancellation_interrupts_wait_without_polling_provider():
    brain_invocations: list[ExecutionState] = []
    compiled_graph = _compiled_resume_graph(brain_invocations)
    config = {"configurable": {"thread_id": "thread-local-cancel"}}
    compiled_graph.update_state(
        config,
        {
            "messages": [],
            "execution_state": _execution_state(),
            "controller_decision": _await_decision(),
        },
        as_node="controller",
    )
    status_tool = FakeStatusTool(_status_output(AsyncJobStatus.RUNNING))
    runtime = LocalAsyncPollingRuntime(
        compiled_graph=compiled_graph,
        tools=[status_tool],
        sleep=lambda _seconds: None,
        now_utc=lambda: OBSERVED_AT,
    )
    runtime.request_cancel("thread-local-cancel")

    list(runtime.poll_and_resume(config=config, decision=_await_decision()))

    assert status_tool.invocations == []
    assert brain_invocations == []
    snapshot = compiled_graph.get_state(config)
    final_state = snapshot.values["execution_state"]
    assert final_state.working.cancel_requested is True
    assert final_state.protocol_visible.status == ExecutionStatus.CANCELLED
    assert final_state.protocol_visible.cursor.phase == ExecutionPhase.CANCELLED
    assert final_state.protocol_visible.cancellation_source == CancellationSource.LOCAL
    assert len(final_state.working.tool_execution_history) == 1
    assert (
        snapshot.values["controller_decision"].decision_type
        == ControllerDecisionType.CANCEL
    )


def test_build_app_runs_submission_await_poll_capture_and_resume_end_to_end():
    status_tool = FakeStatusTool(_status_output(AsyncJobStatus.COMPLETED))
    brain_calls: list[str] = []
    submission_calls: list[str] = []

    class FakeSubmissionTool:
        name = "run_comfy_workflow"

    class FakeChatModel:
        def bind_tools(self, _tools):
            return self

    def graph_nodes_factory(**_kwargs):
        controller_node = create_controller_node(
            CortexController(
                max_reasoning_steps=10,
                now_utc=lambda: OBSERVED_AT,
            )
        )
        capture_node = create_capture_tool_output_node()

        def planner_node(_state):
            step = ExecutionStep(
                step_id="step-1",
                title="Generate image",
                primary_tool="run_comfy_workflow",
            )
            return {
                "planner_result": PlannerResult(
                    outcome=PlannerOutcome.EXECUTION_PLAN,
                    proposed_plan=ExecutionPlan(
                        plan_id="plan-1",
                        revision=1,
                        objective="Generate image",
                        steps=(step,),
                    ),
                )
            }

        def brain_node(state: AgentState):
            execution_state = state["execution_state"]
            history = execution_state.working.tool_execution_history
            brain_calls.append("after-evidence" if history else "submit")

            if not history:
                request = ToolRequest(
                    request_id="submit-request",
                    tool_name="run_comfy_workflow",
                    arguments={"workflow_json": {"1": {}}},
                )
                return {
                    "brain_result": BrainResult(
                        outcome=BrainOutcome.TOOL_REQUEST,
                        tool_request=request,
                    ),
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[{
                                "name": request.tool_name,
                                "args": request.arguments,
                                "id": "model-call-id",
                                "type": "tool_call",
                            }],
                        )
                    ],
                }

            consumed_state = execution_state.model_copy(
                update={
                    "working": execution_state.working.model_copy(
                        update={"last_tool_result": None}
                    )
                }
            )
            return {
                "brain_result": BrainResult(
                    outcome=BrainOutcome.FINAL_ANSWER,
                    message="Finished.",
                    final_answer="Finished.",
                ),
                "execution_state": consumed_state,
                "messages": [AIMessage(content="Finished.")],
            }

        def summary_node(_state):
            return {}

        return (
            controller_node,
            planner_node,
            brain_node,
            capture_node,
            summary_node,
        )

    def tool_node_factory(_tools):
        def submission_node(state: AgentState):
            submission_calls.append("submitted")
            tool_call = state["messages"][-1].tool_calls[0]
            return {
                "messages": [
                    ToolMessage(
                        content=_status_output(AsyncJobStatus.SUBMITTED),
                        tool_call_id=tool_call["id"],
                    )
                ]
            }

        return submission_node

    app = build_app(
        workspace_dir="workspace",
        knowledge_dir="knowledge",
        rag_factory=lambda *_args: object(),
        tool_list_factory=lambda *_args: [FakeSubmissionTool(), status_tool],
        chat_model_factory=lambda *_args: FakeChatModel(),
        graph_nodes_factory=graph_nodes_factory,
        tool_node_factory=tool_node_factory,
        project_root=Path("."),
    )
    app.async_runtime._sleep = lambda _seconds: None

    history, _summary = run_prompt(app, "Generate an image")

    assert status_tool.invocations == [{"prompt_id": "prompt-1"}]
    assert submission_calls == ["submitted"]
    assert brain_calls == ["submit", "after-evidence"]
    assert isinstance(history[-1], AIMessage)
    assert history[-1].content == "Finished."
