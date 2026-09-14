"""Fail-closed validation for LangGraph's temporary worker-node bridge."""

from __future__ import annotations

from core.protocol.enums import ControllerDecisionType, ExecutionStatus, WorkerRole
from core.protocol.models import ControllerDecision, ExecutionState, ToolRequest
from core.runtime.execution_driver import WorkerDispatchError


def require_worker_authorization(
    state,
    *,
    decision_type: ControllerDecisionType,
    worker: WorkerRole,
) -> tuple[ExecutionState, ControllerDecision]:
    execution_state = state.get("execution_state")
    decision = state.get("controller_decision")
    if not isinstance(execution_state, ExecutionState):
        raise WorkerDispatchError("worker requires authoritative ExecutionState")
    if not isinstance(decision, ControllerDecision):
        raise WorkerDispatchError("worker requires Controller authorization")
    if decision.decision_type != decision_type or decision.next_worker != worker:
        raise WorkerDispatchError(f"Controller decision does not authorize {worker.value}")

    protocol = execution_state.protocol_visible
    if protocol.status != ExecutionStatus.NON_TERMINAL:
        raise WorkerDispatchError("worker authorization is not non-terminal")
    if decision.cursor is None or decision.cursor != protocol.cursor:
        raise WorkerDispatchError("worker authorization cursor is stale")
    if protocol.cursor.current_worker != worker:
        raise WorkerDispatchError("execution cursor does not authorize worker")

    plan = protocol.active_plan
    if plan is not None and protocol.cursor.plan_revision != plan.revision:
        raise WorkerDispatchError("worker authorization plan revision is stale")
    step = protocol.active_step
    if step is not None:
        if protocol.cursor.step_id != step.step_id:
            raise WorkerDispatchError("worker authorization step is stale")
        if plan is None or not any(item.step_id == step.step_id for item in plan.steps):
            raise WorkerDispatchError("active step is not in the authoritative plan")
    elif protocol.cursor.step_id is not None:
        raise WorkerDispatchError("worker authorization references a missing step")
    return execution_state, decision


def require_planner_authorization(state):
    execution_state, decision = require_worker_authorization(
        state,
        decision_type=ControllerDecisionType.DISPATCH_PLANNER,
        worker=WorkerRole.PLANNER,
    )
    request = execution_state.protocol_visible.planning_request
    if request is None or decision.planning_request != request:
        raise WorkerDispatchError("Planner authorization request is missing or stale")
    if request.identity != execution_state.protocol_visible.identity:
        raise WorkerDispatchError("Planner authorization execution identity mismatch")
    return request


def require_brain_authorization(state) -> ExecutionState:
    execution_state, _ = require_worker_authorization(
        state,
        decision_type=ControllerDecisionType.DISPATCH_BRAIN,
        worker=WorkerRole.BRAIN,
    )
    return execution_state


def require_tool_authorization(state) -> ToolRequest:
    execution_state, decision = require_worker_authorization(
        state,
        decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
        worker=WorkerRole.TOOL_RUNTIME,
    )
    protocol = execution_state.protocol_visible
    request = protocol.pending_tool_request
    if request is None or decision.pending_tool_request != request:
        raise WorkerDispatchError("Tool authorization request is missing or stale")
    if protocol.active_plan is None or protocol.active_step is None:
        raise WorkerDispatchError("Tool authorization requires an active plan and step")
    if protocol.cursor.plan_revision != protocol.active_plan.revision:
        raise WorkerDispatchError("Tool authorization plan revision mismatch")
    if protocol.cursor.step_id != protocol.active_step.step_id:
        raise WorkerDispatchError("Tool authorization step mismatch")
    return request


__all__ = [
    "require_brain_authorization",
    "require_planner_authorization",
    "require_tool_authorization",
    "require_worker_authorization",
]
