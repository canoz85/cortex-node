"""Shared structural admissibility checks; never judge semantic plan quality."""

from collections.abc import Sized

from core.protocol.enums import PlanningOperation, StepStatus
from core.protocol.models import ExecutionPlan, ExecutionStep, PlanningRequest
from pydantic import ValidationError
from tools.registry import get_tool_definition


MAX_PROPOSED_STEPS = 4


class PlanValidationError(ValueError):
    """A candidate violates a mechanically checkable planning invariant."""


def authorized_completed_steps(
    request: PlanningRequest, accepted_plan: ExecutionPlan | None = None,
) -> tuple[ExecutionStep, ...]:
    """Validate completed dependency context against the authorized base only."""
    if request.operation == PlanningOperation.CREATE:
        return ()
    base = accepted_plan if accepted_plan is not None else request.base_plan
    if (base is None or request.base_plan is None
            or request.base_plan_id != base.plan_id
            or request.base_revision != base.revision
            or request.base_plan_id != request.base_plan.plan_id
            or request.base_revision != request.base_plan.revision):
        raise PlanValidationError("stale_or_mismatched_base_revision")
    ids = request.completed_step_ids
    by_id = {step.step_id: step for step in base.steps}
    base_snapshot = {step.step_id: step for step in request.base_plan.steps}
    snapshot = {step.step_id: step for step in request.completed_steps}
    completed_ids = {step.step_id for step in base.steps if step.status == StepStatus.COMPLETED}
    if (len(by_id) != len(base.steps) or len(base_snapshot) != len(request.base_plan.steps)
            or len(ids) != len(set(ids))
            or len(snapshot) != len(request.completed_steps)
            or set(ids) != completed_ids or set(snapshot) != completed_ids
            or any(snapshot[step_id] != by_id[step_id] for step_id in completed_ids)
            or any(snapshot[step_id] != base_snapshot.get(step_id) for step_id in completed_ids)):
        raise PlanValidationError("completed_work_snapshot_mismatch")
    return tuple(by_id[step_id] for step_id in ids)


def first_ready_step(
    plan: ExecutionPlan, *, completed_steps: tuple[ExecutionStep, ...] = (),
) -> ExecutionStep | None:
    """First pending step with completed prerequisites, in stored plan order."""
    completed = {step.step_id for step in (*plan.steps, *completed_steps)
                 if step.status == StepStatus.COMPLETED}
    return next((step for step in plan.steps
                 if step.status == StepStatus.PENDING
                 and step.step_id not in completed
                 and set(step.depends_on_step_ids) <= completed), None)


def validate_proposed_step_count(steps: Sized) -> None:
    if not steps:
        raise PlanValidationError("PLAN_PROPOSED requires at least one step")
    if len(steps) > MAX_PROPOSED_STEPS:
        raise PlanValidationError(f"PLAN_PROPOSED exceeds the {MAX_PROPOSED_STEPS}-step limit")


def authorized_plan_tools(request: PlanningRequest, *, route: str | None = None) -> set[str]:
    """Consume capability facts and the existing info-route effect restriction."""
    available = set(request.capabilities.available_tools)
    if (request.planner_route or route) == "info":
        available = {name for name in available
                     if not (definition := get_tool_definition(name)) or not definition.mutating}
    return available - set(request.capabilities.unavailable_tools)


def _validate_graph(
    step_ids: tuple[str, ...], dependencies: dict[str, tuple[str, ...]], *,
    completed_steps: tuple[ExecutionStep, ...] = (),
) -> None:
    # Completed definitions are supplied by Controller, never by model claims.
    all_dependencies = {step.step_id: step.depends_on_step_ids for step in completed_steps}
    all_dependencies.update(dependencies)
    known = set(step_ids) | {step.step_id for step in completed_steps}
    for step_id, refs in all_dependencies.items():
        if len(refs) != len(set(refs)):
            raise PlanValidationError(f"step '{step_id}' contains duplicate dependencies")
        for ref in refs:
            if ref not in known:
                raise PlanValidationError(f"step '{step_id}' references unknown dependency '{ref}'")
            if ref == step_id:
                raise PlanValidationError(f"step '{step_id}' cannot depend on itself")
    visiting, visited = set(), set()

    def visit(step_id):
        if step_id in visiting:
            raise PlanValidationError("proposed step dependency graph is cyclic")
        if step_id in visited:
            return
        visiting.add(step_id)
        for ref in all_dependencies[step_id]:
            visit(ref)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in all_dependencies:
        visit(step_id)


def validate_execution_plan(
    plan: ExecutionPlan, request: PlanningRequest, *, route: str | None = None,
    reconciled: bool = False,
) -> ExecutionPlan:
    """Check a proposal or reconciled candidate against its planning authorization.

    Historical completed steps do not consume a revised unfinished-step budget.
    Normalization also checks the provider's original proposal length. Reconciliation
    alone preserves completed definitions and assigns accepted revision numbers.
    Free-text constraints are projected as guidance, not parsed as executable policy.
    """
    try:
        plan = ExecutionPlan.model_validate(plan.model_dump())
    except ValidationError as exc:
        raise PlanValidationError("invalid_execution_plan_model") from exc
    completed = authorized_completed_steps(request)
    completed_ids = {step.step_id for step in completed}
    if request.operation == PlanningOperation.CREATE:
        # Plan IDs are opaque; request identity/sequence bind CREATE to execution.
        if plan.revision != 1 or any((request.base_plan, request.base_plan_id,
                                      request.base_revision, request.completed_steps,
                                      request.completed_step_ids)):
            raise PlanValidationError("initial_plan_scope_mismatch")
    elif (plan.plan_id != request.base_plan_id
          or (reconciled and plan.revision != request.base_revision + 1)):
        raise PlanValidationError("revision_plan_scope_mismatch")
    steps = tuple(step for step in plan.steps if step.step_id not in completed_ids)
    validate_proposed_step_count(steps)
    ids = tuple(step.step_id for step in plan.steps)
    if len(ids) != len(set(ids)):
        raise PlanValidationError("proposed step ids must be unique")
    if any(not step_id.strip() or step_id != step_id.strip() for step_id in ids):
        raise PlanValidationError("proposed step ids must be non-empty and normalized")
    _validate_graph(ids, {step.step_id: step.depends_on_step_ids for step in plan.steps},
                    completed_steps=completed)
    if plan.available_tools is None:
        raise PlanValidationError("plan_capability_ceiling_missing")
    available = authorized_plan_tools(request, route=route)
    unavailable = set(request.capabilities.unavailable_tools)
    ceiling = set(plan.available_tools)
    if not ceiling <= available or ceiling & unavailable:
        raise PlanValidationError("plan_capability_exceeds_controller_ceiling")
    for step in steps:
        if not step.title.strip() or not step.description.strip():
            raise PlanValidationError("proposed step titles and descriptions must be non-empty")
        tool = step.primary_tool
        if not tool or not tool.strip():
            raise PlanValidationError("primary_tool cannot be empty")
        if tool in unavailable:
            raise PlanValidationError(f"primary_tool '{tool}' is unavailable")
        if tool not in available:
            raise PlanValidationError(f"primary_tool '{tool}' is unknown")
        if tool not in ceiling:
            raise PlanValidationError(f"primary_tool '{tool}' is outside plan capability ceiling")
        if step.status != StepStatus.PENDING or step.attempt != 0:
            raise PlanValidationError("planner_authored_lifecycle_state")
    if first_ready_step(plan, completed_steps=completed) is None:
        raise PlanValidationError("plan_has_no_ready_unfinished_step")
    return plan
