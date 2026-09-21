from core.finalizer import Finalizer
from core.graph_controller import create_controller_node
from core.graph_worker_runtime import GraphWorkerRuntimePorts
from core.logging.node_update import extract_node_update
from core.logging.renderer import render_node_update
from core.graph_routing import route_after_controller
from core.protocol.enums import (
    BrainOutcomeKind,
    ControllerDecisionType,
    ExecutionPhase,
    ExecutionStatus,
    PlannerOutcome,
    StepStatus,
    WorkerRole,
)
from core.protocol.models import (
    BrainOutcome,
    ControllerDecision,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    PlannerResult,
    ProtocolVisibleState,
    RetryMetadata,
    WorkingState,
)


def _state(**updates):
    state = {
        "execution_state": ExecutionState(
            protocol_visible=ProtocolVisibleState(
                identity=ExecutionIdentity(execution_id="shadow", protocol_version="1"),
                cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING),
                retry=RetryMetadata(max_retries=0),
            )
        )
    }
    state.update(updates)
    return state


def _completed_plan_state():
    step = ExecutionStep(step_id="s1", title="Work", status=StepStatus.COMPLETED)
    return ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=ExecutionIdentity(execution_id="shadow", protocol_version="1"),
        cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING),
        active_plan=ExecutionPlan(plan_id="p1", revision=2, steps=(step,)),
        completed_step_ids=("s1",),
    ))


class ObservingFinalizer:
    def __init__(self):
        self.requests = []

    def finalize(self, request):
        self.requests.append(request)
        class Renderer:
            def render(self, _request, _summary):
                return "Finalizer answer"
        return Finalizer(answer_renderer=Renderer()).finalize(request)


def test_authorized_success_finalizer_owns_answer_summary_and_preserves_route(capsys):
    observer = ObservingFinalizer()
    state = _state(
        execution_state=_completed_plan_state(),
        brain_result=BrainOutcome(
            outcome=BrainOutcomeKind.FINAL_ANSWER,
            message="Finalization requested.",
        ),
    )

    update = create_controller_node(finalizer=observer)(state)

    assert update["controller_decision"].terminal is True
    assert observer.requests[0].status == ExecutionStatus.COMPLETED
    assert observer.requests[0].accepted_plan.plan_id == "p1"
    assert observer.requests[0].completed_step_ids == ("s1",)
    assert update["finalization_result"].execution_summary.failed_step_ids == ()
    assert update["execution_state"].protocol_visible.summary == update["finalization_result"].execution_summary
    assert update["final_answer"] == "Finalizer answer"
    assert update["messages"][0].content == "Finalizer answer"
    assert route_after_controller(update) == "__end__"
    assert update["execution_state"].protocol_visible.status == ExecutionStatus.COMPLETED
    render_node_update(extract_node_update(
        from_node="brain",
        to_node="controller",
        value=update,
    ))
    presentation = capsys.readouterr().out
    assert "[finalizer]" in presentation
    assert "[brain]" not in presentation


def test_failed_terminal_transition_is_finalized_from_controller_facts():
    observer = ObservingFinalizer()
    node = create_controller_node(finalizer=observer)
    initial = _state()
    authorized = node(initial)
    request = authorized["execution_state"].protocol_visible.planning_request
    update = node({**initial, **authorized,
        "planner_result": PlannerResult(
            outcome=PlannerOutcome.FAILED, request_id=request.request_id,
            failure_category="UNPLANNABLE", message="Planning failed")})

    request = observer.requests[0]
    assert update["controller_decision"].decision_type == ControllerDecisionType.TERMINATE
    assert request.status == ExecutionStatus.FAILED
    assert request.terminal_reason == "Planning failed"
    assert update["finalization_result"].execution_summary.status == ExecutionStatus.FAILED


def test_direct_response_answer_is_produced_by_finalizer():
    observer = ObservingFinalizer()
    prior = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_BRAIN,
        next_worker=WorkerRole.BRAIN,
        direct_response=True,
    )
    update = create_controller_node(finalizer=observer)(_state(
        controller_decision=prior,
        brain_result=BrainOutcome(
            outcome=BrainOutcomeKind.FINAL_ANSWER,
            message="Finalization requested.",
        ),
    ))

    assert observer.requests[0].direct_response is True
    assert update["finalization_result"].execution_summary.completed_step_ids == ()
    assert update["final_answer"] == "Finalizer answer"


def test_cancelled_terminal_transition_is_finalized():
    observer = ObservingFinalizer()
    state = _state()
    state["execution_state"] = state["execution_state"].model_copy(update={
        "working": WorkingState(cancel_requested=True),
    })

    update = create_controller_node(finalizer=observer)(state)

    assert observer.requests[0].status == ExecutionStatus.CANCELLED
    assert update["finalization_result"].execution_summary.status == ExecutionStatus.CANCELLED


def test_equivalent_terminal_executions_produce_equivalent_summaries():
    def run_once():
        return create_controller_node(finalizer=ObservingFinalizer())(_state(
            execution_state=_completed_plan_state(),
            brain_result=BrainOutcome(
                outcome=BrainOutcomeKind.FINAL_ANSWER,
                message="Finalization requested.",
            ),
        ))["finalization_result"].execution_summary

    assert run_once() == run_once()


def test_finalizer_failure_is_observable_and_never_falls_back_to_brain_answer():
    class FailingFinalizer:
        def finalize(self, _request):
            raise RuntimeError("shadow unavailable")

    update = create_controller_node(finalizer=FailingFinalizer())(_state(
        execution_state=_completed_plan_state(),
        brain_result=BrainOutcome(
            outcome=BrainOutcomeKind.FINAL_ANSWER,
            message="Finalization requested.",
        ),
    ))

    assert update["final_answer"] == "Execution finished, but finalization failed."
    assert "shadow unavailable" in update["finalization_error"]
    assert "finalization_result" not in update
    assert update["execution_state"].protocol_visible.status == ExecutionStatus.COMPLETED
    assert update["controller_decision"].decision_type == ControllerDecisionType.DISPATCH_SUMMARY
    assert route_after_controller(update) == "__end__"


def test_non_terminal_controller_decision_does_not_invoke_finalizer():
    observer = ObservingFinalizer()
    node = create_controller_node(finalizer=observer)
    initial = _state()
    authorized = node(initial)
    request = authorized["execution_state"].protocol_visible.planning_request
    update = node({**initial, **authorized,
        "planner_result": PlannerResult(
            outcome=PlannerOutcome.EXECUTION_PLAN,
            request_id=request.request_id,
            proposed_plan=ExecutionPlan(
                plan_id="p1",
                steps=(ExecutionStep(step_id="s1", title="Work"),),
            ),
        ),
    })

    assert observer.requests == []
    assert "finalization_result" not in update
    assert update["controller_decision"].terminal is False


def test_successful_planning_retry_is_consumed_before_brain_without_redispatch():
    planner_request_ids = []
    brain_calls = []

    def planner(state):
        request = state["execution_state"].protocol_visible.planning_request
        planner_request_ids.append(request.request_id)
        if len(planner_request_ids) == 1:
            result = PlannerResult(
                outcome=PlannerOutcome.FAILED,
                request_id=request.request_id,
                failure_category="PROVIDER_FAILURE",
                message="Router structured output failed.",
            )
        else:
            result = PlannerResult(
                outcome=PlannerOutcome.EXECUTION_PLAN,
                request_id=request.request_id,
                proposed_plan=ExecutionPlan(
                    plan_id="retry-plan",
                    steps=(ExecutionStep(step_id="s1", title="Get current time"),),
                ),
            )
        return {"planner_result": result}

    def brain(state):
        brain_calls.append(state["execution_state"].protocol_visible.cursor.current_worker)
        return {"brain_result": BrainOutcome(
            outcome=BrainOutcomeKind.INVALID_OUTPUT,
            error_code="test-stop",
            message="Brain was dispatched.",
        )}

    ports = GraphWorkerRuntimePorts()
    ports.bind_nodes(planner=planner, brain=brain)
    node = create_controller_node(worker_ports=ports)

    first = node(_state())
    first_state = {**_state(), **first}
    retry = node(first_state)
    retry_state = {**first_state, **retry}

    successful_request_id = retry["planner_result"].request_id
    assert retry["planner_result"].outcome == PlannerOutcome.EXECUTION_PLAN
    assert planner_request_ids == [
        first["planner_result"].request_id,
        successful_request_id,
    ]

    handoff = node(retry_state)

    assert planner_request_ids.count(successful_request_id) == 1
    assert handoff["controller_decision"].decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert handoff["controller_decision"].next_worker == WorkerRole.BRAIN
    assert handoff["execution_state"].protocol_visible.cursor.current_worker == WorkerRole.BRAIN
    assert brain_calls == [WorkerRole.BRAIN]
