import ast
from pathlib import Path

import pytest

from core.finalizer import Finalizer as FinalizerService
from core.protocol.controller import CortexController
from core.protocol.enums import (
    BrainOutcome,
    ControllerDecisionType,
    ExecutionPhase,
    ExecutionStatus,
    PlannerOutcome,
    StepStatus,
    WorkerRole,
)
from core.protocol.models import (
    BrainResult,
    ControllerDecision,
    ControllerInput,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    PlannerResult,
    ProtocolVisibleState,
    StepCompletionEvidence,
    ToolRequest,
    ToolResult,
)
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver, WorkerDispatchError


IDENTITY = ExecutionIdentity(execution_id="driver-test", protocol_version="1.0")
CONTEXT = ExecutionContext(user_request="Read one file")


def initial_state() -> ExecutionState:
    return ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=IDENTITY,
            cursor=ExecutionCursor(phase=ExecutionPhase.INITIALIZING),
        )
    )


def controller_input(state, result=None) -> ControllerInput:
    protocol = state.protocol_visible
    updates = {}
    if isinstance(result, PlannerResult):
        updates["planner_result"] = result
    elif isinstance(result, BrainResult):
        updates["brain_result"] = result
    elif isinstance(result, ToolResult):
        updates["tool_result"] = result
    elif result is not None:
        raise TypeError("unsupported worker result")
    return ControllerInput(
        identity=protocol.identity,
        cursor=protocol.cursor,
        context=CONTEXT,
        active_plan=protocol.active_plan,
        active_step=protocol.active_step,
        pending_tool_request=protocol.pending_tool_request,
        retry=protocol.retry,
        async_policy=protocol.async_policy,
        tool_execution_history=state.working.tool_execution_history,
        planning_request=protocol.planning_request,
        planning_sequence=protocol.planning_sequence,
        completed_step_ids=protocol.completed_step_ids,
        **updates,
    )


class AuthorizationTrace:
    def __init__(self):
        self.current = None
        self.invocations = []

    def authorize(self, decision):
        assert self.current is None
        self.current = decision

    def invoked(self, worker, expected_type):
        decision = self.current
        assert decision is not None
        assert decision.decision_type == expected_type
        self.invocations.append((worker, decision.decision_type))
        self.current = None


class TracedController:
    def __init__(self, controller, trace):
        self.controller = controller
        self.trace = trace

    def decide(self, value):
        decision = self.controller.decide(value)
        self.trace.authorize(decision)
        return decision


class Planner:
    def __init__(self, trace):
        self.trace = trace

    def run(self, request):
        self.trace.invoked("planner", ControllerDecisionType.DISPATCH_PLANNER)
        return PlannerResult(
            outcome=PlannerOutcome.EXECUTION_PLAN,
            request_id=request.request_id,
            proposed_plan=ExecutionPlan(
                plan_id="driver-plan",
                objective="Read one file",
                steps=(ExecutionStep(
                    step_id="read",
                    title="Read target",
                    primary_tool="read_file",
                ),),
            ),
        )


class Brain:
    def __init__(self, trace):
        self.trace = trace
        self.calls = 0

    def run(self, value):
        self.trace.invoked("brain", ControllerDecisionType.DISPATCH_BRAIN)
        self.calls += 1
        if self.calls == 1:
            return BrainResult(
                outcome=BrainOutcome.TOOL_REQUEST,
                step_id="read",
                tool_request=ToolRequest(
                    request_id="driver-read",
                    tool_name="read_file",
                    arguments={"path": "target.txt"},
                ),
            )
        if self.calls == 2:
            assert value.last_tool_result is not None
            assert value.last_tool_result.request_id == "driver-read"
            return BrainResult(
                outcome=BrainOutcome.STEP_COMPLETED,
                step_id="read",
                message="Target read.",
            )
        assert value.active_step is None
        return BrainResult(
            outcome=BrainOutcome.FINAL_ANSWER,
            message="Finalization requested.",
        )


class ToolRuntime:
    def __init__(self, trace):
        self.trace = trace

    def execute(self, request):
        self.trace.invoked("tool", ControllerDecisionType.DISPATCH_TOOL_RUNTIME)
        return ToolResult(
            request_id=request.request_id,
            signature='read_file:{"path": "target.txt"}',
            success=True,
            message="Read target.txt",
            data={"path": "target.txt", "content": "hello"},
        )


class FinalizerWorker:
    def __init__(self, trace):
        self.trace = trace
        self.delegate = FinalizerService()

    def finalize(self, request):
        self.trace.invoked("finalizer", ControllerDecisionType.DISPATCH_SUMMARY)
        return self.delegate.finalize(request)
def test_complete_non_graph_lifecycle_is_controller_authorized():
    trace = AuthorizationTrace()
    controller = TracedController(
        CortexController(max_reasoning_steps=10),
        trace,
    )
    brain = Brain(trace)
    driver = ExecutionDriver(
        coordinator=ControllerCoordinator(controller),
        planner=Planner(trace),
        brain=brain,
        tool_runtime=ToolRuntime(trace),
        finalizer=FinalizerWorker(trace),
    )

    state = initial_state()
    result = None
    decisions = []
    iterations = []
    while True:
        turn = driver.turn(state, controller_input(state, result))
        state, result = turn.execution_state, turn.worker_result
        decisions.append(turn.decision.decision_type)
        iterations.append(state.protocol_visible.cursor.controller_iteration)
        if turn.decision.terminal:
            break

    assert decisions == [
        ControllerDecisionType.DISPATCH_PLANNER,
        ControllerDecisionType.DISPATCH_BRAIN,
        ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
        ControllerDecisionType.DISPATCH_BRAIN,
        ControllerDecisionType.DISPATCH_BRAIN,
        ControllerDecisionType.DISPATCH_SUMMARY,
    ]
    assert trace.invocations == [
        ("planner", ControllerDecisionType.DISPATCH_PLANNER),
        ("brain", ControllerDecisionType.DISPATCH_BRAIN),
        ("tool", ControllerDecisionType.DISPATCH_TOOL_RUNTIME),
        ("brain", ControllerDecisionType.DISPATCH_BRAIN),
        ("brain", ControllerDecisionType.DISPATCH_BRAIN),
        ("finalizer", ControllerDecisionType.DISPATCH_SUMMARY),
    ]
    assert iterations == [None, 1, 1, 2, 3, 3]
    assert trace.current is None
    assert state.protocol_visible.active_plan.steps[0].status == StepStatus.COMPLETED
    assert result.execution_summary.completed_step_ids == ("read",)


class FixedController:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def decide(self, _value):
        self.calls += 1
        return self.decision


class RecordingPort:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def run(self, value):
        self.calls.append(value)
        return self.result

    def execute(self, value):
        self.calls.append(value)
        return self.result

    def finalize(self, value):
        self.calls.append(value)
        return self.result


def fixed_driver(state, decision, *, results=()):
    ports = [RecordingPort(result) for result in results]
    while len(ports) < 4:
        ports.append(RecordingPort())
    controller = FixedController(decision)
    driver = ExecutionDriver(
        coordinator=ControllerCoordinator(controller),
        planner=ports[0], brain=ports[1], tool_runtime=ports[2], finalizer=ports[3],
    )
    return driver, controller, ports


class CapturingFinalizer:
    def __init__(self):
        self.requests = []

    def finalize(self, request):
        self.requests.append(request)
        return FinalizerService().finalize(request)


def accepted_evidence(
    summary,
    *,
    step_id="s1",
    revision=1,
    tool_request_ids=("request-1",),
):
    return StepCompletionEvidence(
        execution_id=IDENTITY.execution_id,
        plan_id="accepted-plan",
        plan_revision=revision,
        step_id=step_id,
        summary=summary,
        tool_request_ids=tool_request_ids,
        evidence_id=f"evidence-{step_id}-{revision}",
    )


def capture_terminal_request(completion_provenance=(), *, brain_result=None):
    state = ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=IDENTITY,
            cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING),
            completion_provenance=completion_provenance,
        )
    )
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_SUMMARY,
        next_worker=WorkerRole.SUMMARY,
        execution_status=ExecutionStatus.COMPLETED,
        cursor=state.protocol_visible.cursor.model_copy(
            update={
                "phase": ExecutionPhase.COMPLETED,
                "current_worker": WorkerRole.SUMMARY,
            }
        ),
        terminal=True,
    )
    finalizer = CapturingFinalizer()
    driver = ExecutionDriver(
        coordinator=ControllerCoordinator(FixedController(decision)),
        planner=RecordingPort(),
        brain=RecordingPort(),
        tool_runtime=RecordingPort(),
        finalizer=finalizer,
    )

    driver.turn(state, controller_input(state, brain_result))

    assert len(finalizer.requests) == 1
    return finalizer.requests[0]


def test_finalizer_request_projects_accepted_semantics_and_bound_evidence():
    evidence = accepted_evidence("Semantic result X", revision=4)

    request = capture_terminal_request((evidence,))

    assert len(request.accepted_step_results) == 1
    result = request.accepted_step_results[0]
    assert result.semantic_content == "Semantic result X"
    assert result.completion_evidence == evidence
    assert result.completion_evidence.plan_revision == 4


def test_finalizer_projection_excludes_rejected_attempt_and_uses_later_acceptance():
    rejected = BrainResult(
        outcome=BrainOutcome.STEP_COMPLETED,
        step_id="s1",
        message="Rejected X",
        completion_evidence=StepCompletionEvidence(
            step_id="s1",
            summary="Rejected X",
        ),
    )

    rejected_request = capture_terminal_request(brain_result=rejected)
    assert rejected_request.accepted_step_results == ()

    accepted = accepted_evidence("Accepted retry X2")
    retry_request = capture_terminal_request((accepted,), brain_result=rejected)
    assert tuple(
        result.semantic_content for result in retry_request.accepted_step_results
    ) == ("Accepted retry X2",)


def test_finalizer_projection_preserves_acceptance_order_and_revision_identity():
    first = accepted_evidence("First", step_id="s1", revision=2)
    second = accepted_evidence("Second", step_id="s2", revision=5)

    request = capture_terminal_request((first, second))

    assert tuple(
        result.semantic_content for result in request.accepted_step_results
    ) == ("First", "Second")
    assert tuple(
        result.completion_evidence.plan_revision
        for result in request.accepted_step_results
    ) == (2, 5)


def test_finalizer_projection_includes_valid_empty_tool_provenance():
    evidence = accepted_evidence(
        "Reasoning-only completion",
        tool_request_ids=(),
    )

    request = capture_terminal_request((evidence,))

    assert (
        request.accepted_step_results[0].semantic_content
        == "Reasoning-only completion"
    )
    assert request.accepted_step_results[0].completion_evidence.tool_request_ids == ()


@pytest.mark.parametrize(
    ("decision_type", "next_worker", "message"),
    (
        (
            ControllerDecisionType.DISPATCH_PLANNER,
            WorkerRole.BRAIN,
            "does not authorize planner",
        ),
        (
            ControllerDecisionType.DISPATCH_BRAIN,
            WorkerRole.PLANNER,
            "does not authorize brain",
        ),
        (
            ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
            WorkerRole.BRAIN,
            "does not authorize tool_runtime",
        ),
    ),
)
def test_mismatched_dispatch_fails_before_any_worker_runs(
    decision_type,
    next_worker,
    message,
):
    state = initial_state()
    decision = ControllerDecision(
        decision_type=decision_type,
        next_worker=next_worker,
        cursor=state.protocol_visible.cursor,
    )
    driver, _, ports = fixed_driver(state, decision)

    with pytest.raises(WorkerDispatchError, match=message):
        driver.turn(state, controller_input(state))
    assert all(port.calls == [] for port in ports)


def test_unknown_or_unsupported_dispatch_fails_closed():
    state = initial_state()
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.REQUEST_REPLAN,
        cursor=state.protocol_visible.cursor,
    )
    driver, _, ports = fixed_driver(state, decision)

    with pytest.raises(WorkerDispatchError, match="Unsupported"):
        driver.turn(state, controller_input(state))
    assert all(port.calls == [] for port in ports)


def test_finalizer_cannot_run_without_terminal_authorization():
    state = initial_state()
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_SUMMARY,
        next_worker=WorkerRole.SUMMARY,
        cursor=state.protocol_visible.cursor,
    )
    driver, _, ports = fixed_driver(state, decision)

    with pytest.raises(WorkerDispatchError, match="Unsupported"):
        driver.turn(state, controller_input(state))
    assert ports[3].calls == []


def test_tool_result_request_id_must_match_authorized_request():
    state = initial_state()
    request = ToolRequest(
        request_id="expected",
        tool_name="read_file",
        arguments={"path": "target.txt"},
    )
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
        next_worker=WorkerRole.TOOL_RUNTIME,
        cursor=state.protocol_visible.cursor.model_copy(
            update={"current_worker": WorkerRole.TOOL_RUNTIME}
        ),
        pending_tool_request=request,
    )
    authorized_state = state.model_copy(update={
        "protocol_visible": state.protocol_visible.model_copy(update={
            "pending_tool_request": request,
        }),
    })
    wrong = ToolResult(request_id="wrong", success=True, message="wrong")
    driver, _, ports = fixed_driver(authorized_state, decision, results=(None, None, wrong))

    with pytest.raises(WorkerDispatchError, match="request identity"):
        driver.turn(authorized_state, controller_input(authorized_state))
    assert len(ports[2].calls) == 1
    assert all(port.calls == [] for index, port in enumerate(ports) if index != 2)


def test_one_turn_uses_coordinator_once_and_preserves_typed_brain_result():
    state = initial_state()
    cursor = state.protocol_visible.cursor.model_copy(update={
        "current_worker": WorkerRole.BRAIN,
        "controller_iteration": 1,
    })
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_BRAIN,
        next_worker=WorkerRole.BRAIN,
        cursor=cursor,
    )
    brain_result = BrainResult(outcome=BrainOutcome.CONTINUE)
    driver, controller, ports = fixed_driver(
        state, decision, results=(None, brain_result),
    )

    turn = driver.turn(state, controller_input(state))

    assert controller.calls == 1
    assert turn.worker_result is brain_result
    assert isinstance(turn.worker_result, BrainResult)
    assert state.protocol_visible.cursor.controller_iteration is None
    assert turn.execution_state.protocol_visible.cursor.controller_iteration == 1
    assert len(ports[1].calls) == 1
    assert ports[0].calls == ports[2].calls == ports[3].calls == []


def test_driver_and_worker_ports_have_no_graph_framework_imports():
    root = Path(__file__).parents[1]
    for relative in (
        "core/runtime/execution_driver.py",
        "core/runtime/worker_ports.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        assert not any(name.startswith(("langgraph", "langchain")) for name in imports)
        assert "core.state" not in imports
