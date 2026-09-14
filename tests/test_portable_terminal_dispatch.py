import ast
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from core.completion import CompletionService
from core.finalizer import Finalizer
from core.graph_controller import create_controller_node
from core.graph_routing import route_after_controller
from core.protocol.enums import (
    ControllerDecisionType,
    ExecutionPhase,
    ExecutionStatus,
    WorkerRole,
)
from core.protocol.models import (
    ControllerDecision,
    ControllerInput,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    ProtocolVisibleState,
    ToolExecutionRecord,
    ToolResult,
    WorkingState,
)
from core.state import AgentState
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver
from core.runtime.portable_orchestration import PortableExecutionRuntime


class NeverWorker:
    def run(self, _value):
        raise AssertionError("non-terminal worker invoked")

    def execute(self, _value):
        raise AssertionError("tool worker invoked")


class TerminalController:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def decide(self, _value):
        self.calls += 1
        return self.decision


class RecordingFinalizer:
    def __init__(self, *, failure=None):
        self.requests = []
        self.failure = failure

    def finalize(self, request):
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return Finalizer().finalize(request)


def terminal_fixture():
    identity = ExecutionIdentity(execution_id="portable-terminal", protocol_version="1")
    step = ExecutionStep(step_id="done", title="Done")
    plan = ExecutionPlan(plan_id="accepted-plan", revision=3, steps=(step,))
    cursor = ExecutionCursor(
        phase=ExecutionPhase.EXECUTING,
        current_worker=WorkerRole.BRAIN,
        plan_revision=3,
        controller_iteration=4,
    )
    record = ToolExecutionRecord(
        execution_id=identity.execution_id,
        plan_id=plan.plan_id,
        plan_revision=plan.revision,
        step_id=step.step_id,
        tool_name="read_file",
        arguments={"path": "a.py"},
        result=ToolResult(request_id="read-1", success=True, message="read"),
    )
    state = ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=identity,
            cursor=cursor,
            active_plan=plan,
            completed_step_ids=(step.step_id,),
        ),
        working=WorkingState(tool_execution_history=(record,)),
    )
    terminal_cursor = cursor.model_copy(update={
        "phase": ExecutionPhase.FAILED,
        "current_worker": WorkerRole.CONTROLLER,
    })
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.TERMINATE,
        next_worker=WorkerRole.CONTROLLER,
        cursor=terminal_cursor,
        execution_status=ExecutionStatus.FAILED,
        reason="terminal reason",
        failure_reason="terminal reason",
        terminal=True,
    )
    value = ControllerInput(
        identity=identity,
        cursor=cursor,
        context=ExecutionContext(user_request="Do work"),
        active_plan=plan,
        completed_step_ids=(step.step_id,),
        tool_execution_history=(record,),
    )
    return state, decision, value


def runtime_for(controller, finalizer):
    never = NeverWorker()
    return PortableExecutionRuntime(
        driver=ExecutionDriver(
            coordinator=ControllerCoordinator(controller),
            planner=never,
            brain=never,
            tool_runtime=never,
            finalizer=finalizer,
        ),
        completion_service=CompletionService(),
    )


def test_portable_turn_owns_exactly_once_terminal_dispatch_without_reapplication():
    state, decision, value = terminal_fixture()
    controller = TerminalController(decision)
    finalizer = RecordingFinalizer()

    turn = runtime_for(controller, finalizer).turn(
        state, value, dispatch_worker=False
    )

    assert controller.calls == 1
    assert len(finalizer.requests) == 1
    assert turn.terminal_dispatch_error is None
    assert turn.driver_turn.worker_result is not None
    assert turn.driver_turn.execution_state.protocol_visible.cursor == decision.cursor
    assert turn.driver_turn.execution_state.protocol_visible.cursor.controller_iteration == 4
    request = finalizer.requests[0]
    assert request.identity == state.protocol_visible.identity
    assert request.status == ExecutionStatus.FAILED
    assert request.accepted_plan == state.protocol_visible.active_plan
    assert request.completed_step_ids == ("done",)
    assert request.terminal_reason == "terminal reason"
    assert request.tool_execution_history == state.working.tool_execution_history


def test_portable_terminal_failure_is_distinct_and_not_a_success_result():
    state, decision, value = terminal_fixture()
    controller = TerminalController(decision)
    finalizer = RecordingFinalizer(failure=RuntimeError("renderer unavailable"))

    turn = runtime_for(controller, finalizer).turn(state, value, dispatch_worker=False)

    assert controller.calls == 1
    assert len(finalizer.requests) == 1
    assert turn.driver_turn.worker_result is None
    assert isinstance(turn.terminal_dispatch_error, RuntimeError)


def test_graph_controller_does_not_call_terminal_dispatch():
    source = Path("core/graph_controller.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr in {"dispatch", "dispatch_authorized", "finalize"}
        for call in calls
    )


def test_terminal_checkpoint_resume_preserves_result_without_redispatch():
    finalizer = RecordingFinalizer()
    node = create_controller_node(finalizer=finalizer)
    workflow = StateGraph(AgentState)
    workflow.add_node("controller", node)
    workflow.set_entry_point("controller")
    workflow.add_conditional_edges("controller", route_after_controller)
    app = workflow.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "terminal-resume"}}
    identity = ExecutionIdentity(execution_id="terminal-resume", protocol_version="1")
    initial = ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=identity,
            cursor=ExecutionCursor(phase=ExecutionPhase.INITIALIZING),
        ),
        working=WorkingState(cancel_requested=True),
    )

    completed = app.invoke({
        "execution_state": initial,
        "user_request": "Cancel",
    }, config)
    resumed = app.invoke(None, config)

    assert len(finalizer.requests) == 1
    assert resumed["finalization_result"] == completed["finalization_result"]
    assert resumed["execution_state"].protocol_visible.cursor.controller_iteration == (
        completed["execution_state"].protocol_visible.cursor.controller_iteration
    )
