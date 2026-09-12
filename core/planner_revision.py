"""Deterministic acceptance and reconciliation of Planner REVISE proposals."""

from __future__ import annotations

from dataclasses import dataclass

from core.protocol.enums import PlanningOperation, StepStatus
from core.protocol.models import ExecutionPlan, ExecutionStep, PlanningRequest


@dataclass(frozen=True)
class RevisionRejection(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def _definition(step: ExecutionStep) -> tuple:
    """The planner-owned, material definition of a step."""
    return (
        step.step_id,
        step.title,
        step.description,
        step.primary_tool,
        step.completion_requirement,
        step.depends_on_step_ids,
    )


def _remaining_shape(plan: ExecutionPlan, completed_ids: set[str]) -> tuple[tuple, ...]:
    return tuple(_definition(step) for step in plan.steps if step.step_id not in completed_ids)


def reconcile_revision(
    request: PlanningRequest,
    accepted_plan: ExecutionPlan,
    proposal: ExecutionPlan,
) -> ExecutionPlan:
    """Return the next accepted revision, or reject without mutating any input.

    Completed definitions come only from the accepted base snapshot.  Candidate
    revision/status/attempt values are never accepted as lifecycle authority.
    """
    if request.operation != PlanningOperation.REVISE:
        raise RevisionRejection("revision_acceptance_requires_revise")
    if request.identity.execution_id == "":  # defensive; the model forbids this
        raise RevisionRejection("revision_execution_mismatch")
    if (request.base_plan is None
            or request.base_plan_id != accepted_plan.plan_id
            or request.base_revision != accepted_plan.revision):
        raise RevisionRejection("stale_or_mismatched_base_revision")
    if proposal.plan_id != accepted_plan.plan_id:
        raise RevisionRejection("revision_plan_id_mismatch")

    completed_ids = set(request.completed_step_ids)
    base_by_id = {step.step_id: step for step in accepted_plan.steps}
    accepted_completed_ids = {
        step.step_id for step in accepted_plan.steps if step.status == StepStatus.COMPLETED
    }
    if (completed_ids != accepted_completed_ids
            or completed_ids != {step.step_id for step in request.completed_steps}
            or any(step.step_id not in base_by_id or base_by_id[step.step_id] != step
                   for step in request.completed_steps)
            or any(base_by_id[step_id].status != StepStatus.COMPLETED
                   for step_id in completed_ids)):
        raise RevisionRejection("completed_work_snapshot_mismatch")

    proposed_by_id = {step.step_id: step for step in proposal.steps}
    if len(proposed_by_id) != len(proposal.steps):
        raise RevisionRejection("duplicate_revised_step_id")
    for step_id in completed_ids & proposed_by_id.keys():
        if _definition(proposed_by_id[step_id]) != _definition(base_by_id[step_id]):
            raise RevisionRejection("completed_step_changed")

    remaining = tuple(step for step in proposal.steps if step.step_id not in completed_ids)
    if any(step.status != StepStatus.PENDING or step.attempt != 0 for step in remaining):
        raise RevisionRejection("planner_authored_lifecycle_state")
    if _remaining_shape(proposal, completed_ids) == _remaining_shape(accepted_plan, completed_ids):
        raise RevisionRejection("ineffective_revision")

    completed = tuple(base_by_id[step_id] for step_id in request.completed_step_ids)
    reconciled_steps = (*completed, *remaining)
    ids = {step.step_id for step in reconciled_steps}
    if len(ids) != len(reconciled_steps):
        raise RevisionRejection("duplicate_revised_step_id")
    if any(set(step.depends_on_step_ids) - ids for step in reconciled_steps):
        raise RevisionRejection("revision_discards_required_dependency")

    return ExecutionPlan(
        plan_id=accepted_plan.plan_id,
        revision=accepted_plan.revision + 1,
        objective=proposal.objective,
        steps=reconciled_steps,
    )
