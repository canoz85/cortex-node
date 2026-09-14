import pytest
from datetime import datetime, timezone
from langchain_core.messages import AIMessage

from core.graph import _wrap_tool_node_for_protocol_request_id
from core.graph_authorization import (
    require_brain_authorization,
    require_planner_authorization,
    require_tool_authorization,
)
from core.protocol.enums import (
    ControllerDecisionType,
    ExecutionPhase,
    PlanningOperation,
    WorkerRole,
)
from core.protocol.models import (
    ControllerDecision,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    PlanningCapabilities,
    PlanningRequest,
    ProtocolVisibleState,
    ToolRequest,
)
from core.runtime.execution_driver import WorkerDispatchError


IDENTITY = ExecutionIdentity(execution_id="authorization-test", protocol_version="1")


def planner_state():
    request = PlanningRequest(
        request_id="plan-request",
        episode_id="plan-episode",
        identity=IDENTITY,
        operation=PlanningOperation.CREATE,
        sequence=1,
        context=ExecutionContext(user_request="Plan work"),
        capabilities=PlanningCapabilities(),
        created_at_utc=datetime.now(timezone.utc),
    )
    cursor = ExecutionCursor(
        phase=ExecutionPhase.PLANNING,
        current_worker=WorkerRole.PLANNER,
    )
    execution = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=IDENTITY,
        cursor=cursor,
        planning_request=request,
        planning_sequence=1,
    ))
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_PLANNER,
        next_worker=WorkerRole.PLANNER,
        cursor=cursor,
        planning_request=request,
    )
    return {"execution_state": execution, "controller_decision": decision}, request


def active_state(worker):
    step = ExecutionStep(step_id="step-1", title="Work")
    plan = ExecutionPlan(plan_id="plan-1", revision=2, steps=(step,))
    cursor = ExecutionCursor(
        phase=ExecutionPhase.EXECUTING,
        current_worker=worker,
        step_id=step.step_id,
        plan_revision=plan.revision,
        controller_iteration=3,
    )
    execution = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=IDENTITY,
        cursor=cursor,
        active_plan=plan,
        active_step=step,
    ))
    return execution, cursor


def brain_state():
    execution, cursor = active_state(WorkerRole.BRAIN)
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_BRAIN,
        next_worker=WorkerRole.BRAIN,
        cursor=cursor,
    )
    return {"execution_state": execution, "controller_decision": decision}


def tool_state():
    execution, cursor = active_state(WorkerRole.TOOL_RUNTIME)
    request = ToolRequest(
        request_id="tool-request",
        tool_name="read_file",
        arguments={"path": "a.py"},
    )
    execution = execution.model_copy(update={
        "protocol_visible": execution.protocol_visible.model_copy(update={
            "pending_tool_request": request,
        }),
    })
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
        next_worker=WorkerRole.TOOL_RUNTIME,
        cursor=cursor,
        pending_tool_request=request,
    )
    message = AIMessage(content="", tool_calls=[{
        "id": request.request_id,
        "name": request.tool_name,
        "args": request.arguments,
        "type": "tool_call",
    }])
    return {
        "execution_state": execution,
        "controller_decision": decision,
        "messages": [message],
    }, request


def test_authorized_planner_and_brain_validation_preserves_iteration():
    planner, request = planner_state()
    assert require_planner_authorization(planner) == request
    brain = brain_state()
    before = brain["execution_state"].protocol_visible.cursor.controller_iteration
    assert require_brain_authorization(brain) is brain["execution_state"]
    assert brain["execution_state"].protocol_visible.cursor.controller_iteration == before


@pytest.mark.parametrize("mutation", ["missing", "kind", "worker", "cursor", "request"])
def test_planner_missing_or_mismatched_authorization_fails_closed(mutation):
    state, _ = planner_state()
    decision = state["controller_decision"]
    if mutation == "missing":
        state.pop("controller_decision")
    elif mutation == "kind":
        state["controller_decision"] = decision.model_copy(update={
            "decision_type": ControllerDecisionType.DISPATCH_BRAIN,
        })
    elif mutation == "worker":
        state["controller_decision"] = decision.model_copy(update={"next_worker": WorkerRole.BRAIN})
    elif mutation == "cursor":
        state["controller_decision"] = decision.model_copy(update={
            "cursor": decision.cursor.model_copy(update={"event_index": 9}),
        })
    else:
        state["controller_decision"] = decision.model_copy(update={"planning_request": None})
    with pytest.raises(WorkerDispatchError):
        require_planner_authorization(state)


@pytest.mark.parametrize("mutation", ["missing", "kind", "worker", "step", "revision"])
def test_brain_missing_mismatched_or_stale_authorization_fails_closed(mutation):
    state = brain_state()
    decision = state["controller_decision"]
    if mutation == "missing":
        state.pop("controller_decision")
    elif mutation == "kind":
        state["controller_decision"] = decision.model_copy(update={
            "decision_type": ControllerDecisionType.DISPATCH_PLANNER,
        })
    elif mutation == "worker":
        state["controller_decision"] = decision.model_copy(update={"next_worker": WorkerRole.PLANNER})
    else:
        cursor = decision.cursor.model_copy(update={
            "step_id": "stale" if mutation == "step" else decision.cursor.step_id,
            "plan_revision": 99 if mutation == "revision" else decision.cursor.plan_revision,
        })
        state["controller_decision"] = decision.model_copy(update={"cursor": cursor})
    with pytest.raises(WorkerDispatchError):
        require_brain_authorization(state)


def test_authorized_tool_invocation_uses_exact_typed_request():
    state, request = tool_state()
    calls = []
    wrapped = _wrap_tool_node_for_protocol_request_id(
        lambda value: calls.append(value) or {"ok": True}
    )
    assert require_tool_authorization(state) == request
    assert wrapped(state) == {"ok": True}
    assert calls == [state]


@pytest.mark.parametrize("mutation", ["missing", "request_id", "tool", "args", "step", "revision"])
def test_raw_or_stale_tool_transport_cannot_bypass_authorization(mutation):
    state, _ = tool_state()
    if mutation == "missing":
        state.pop("controller_decision")
    elif mutation in {"request_id", "tool", "args"}:
        call = dict(state["messages"][-1].tool_calls[0])
        key, value = {
            "request_id": ("id", "wrong"),
            "tool": ("name", "write_file"),
            "args": ("args", {"path": "wrong"}),
        }[mutation]
        call[key] = value
        state["messages"] = [AIMessage(content="", tool_calls=[call])]
    else:
        decision = state["controller_decision"]
        cursor = decision.cursor.model_copy(update={
            "step_id": "stale" if mutation == "step" else decision.cursor.step_id,
            "plan_revision": 99 if mutation == "revision" else decision.cursor.plan_revision,
        })
        state["controller_decision"] = decision.model_copy(update={"cursor": cursor})
    invoked = []
    wrapped = _wrap_tool_node_for_protocol_request_id(lambda value: invoked.append(value))
    with pytest.raises((WorkerDispatchError, RuntimeError)):
        wrapped(state)
    assert invoked == []
