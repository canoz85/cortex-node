"""Deterministic normalization for structured Planner proposals."""
from pydantic import ValidationError

from core.planner_contract import PlannerProposal, PlannerProposalResultType
from core.protocol.enums import PlannerOutcome, PlanningFailureCategory, PlanningOperation, StepStatus
from core.protocol.models import ExecutionPlan, ExecutionStep, PlanningRequest, PlannerResult

MAX_PROPOSED_STEPS = 4
def planner_failure(
    request_id: str, category: PlanningFailureCategory, message: str,
) -> PlannerResult:
    return PlannerResult(
        outcome=PlannerOutcome.FAILED, request_id=request_id,
        message=message, failure_category=category,
    )


def _validate_graph(step_ids: tuple[str, ...], dependencies: dict[str, tuple[str, ...]]) -> None:
    known = set(step_ids)
    for step_id, refs in dependencies.items():
        if len(refs) != len(set(refs)):
            raise ValueError(f"step '{step_id}' contains duplicate dependencies")
        for ref in refs:
            if ref not in known:
                raise ValueError(f"step '{step_id}' references unknown dependency '{ref}'")
            if ref == step_id:
                raise ValueError(f"step '{step_id}' cannot depend on itself")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visiting:
            raise ValueError("proposed step dependency graph is cyclic")
        if step_id in visited:
            return
        visiting.add(step_id)
        for ref in dependencies[step_id]:
            visit(ref)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in step_ids:
        visit(step_id)


def normalize_planner_proposal(
    content: object, planner_input: PlanningRequest, *, route: str, confidence: float,
    effective_tools: frozenset[str] | None = None,
) -> PlannerResult:
    """Validate a proposal and convert it to the existing PlannerResult path."""
    rationale = f"Route '{route}' selected with confidence {confidence:.2f}."
    try:
        proposal = PlannerProposal.model_validate(content)
    except (ValidationError, TypeError, ValueError, AttributeError) as exc:
        return planner_failure(planner_input.request_id, PlanningFailureCategory.INVALID_OUTPUT,
                               f"Planner output is invalid ({type(exc).__name__}).")

    if proposal.result == PlannerProposalResultType.NO_PLAN_REQUIRED:
        return PlannerResult(outcome=PlannerOutcome.DIRECT_RESPONSE, request_id=planner_input.request_id,
                             message=proposal.message or "No execution plan required.",
                             planning_rationale=rationale)
    if proposal.result == PlannerProposalResultType.NEEDS_INPUT:
        return PlannerResult(outcome=PlannerOutcome.CLARIFICATION_REQUIRED, request_id=planner_input.request_id,
                             message=proposal.message or "Planner needs additional input.",
                             planning_rationale=rationale)
    if proposal.result == PlannerProposalResultType.PLANNING_FAILED:
        category = PlanningFailureCategory(proposal.failure_category.value)
        return planner_failure(planner_input.request_id, category,
                               proposal.message or "Planner could not produce a plan.")

    try:
        if not proposal.steps:
            raise ValueError("PLAN_PROPOSED requires at least one step")
        if len(proposal.steps) > MAX_PROPOSED_STEPS:
            raise ValueError(f"PLAN_PROPOSED exceeds the {MAX_PROPOSED_STEPS}-step limit")
        ids = tuple(step.step_id.strip() for step in proposal.steps)
        if len(ids) != len(set(ids)):
            raise ValueError("proposed step ids must be unique")
        dependencies = {step.step_id.strip(): tuple(ref.strip() for ref in step.dependencies)
                        for step in proposal.steps}
        _validate_graph(ids, dependencies)
        available = set(effective_tools if effective_tools is not None
                        else planner_input.capabilities.available_tools)
        unavailable = set(planner_input.capabilities.unavailable_tools)
        for step in proposal.steps:
            if not step.title.strip() or not step.description.strip():
                raise ValueError("proposed step titles and descriptions must be non-empty")
            if step.primary_tool is not None:
                tool = step.primary_tool.strip()
                if not tool:
                    raise ValueError("primary_tool cannot be empty")
                if tool in unavailable:
                    raise ValueError(f"primary_tool '{tool}' is unavailable")
                if tool not in available:
                    raise ValueError(f"primary_tool '{tool}' is unknown")
        revising = planner_input.operation == PlanningOperation.REVISE
        plan = ExecutionPlan(
            plan_id=planner_input.base_plan_id if revising else f"{planner_input.identity.execution_id}:plan",
            revision=planner_input.base_revision + 1 if revising else 1,
            objective=proposal.objective.strip() or planner_input.context.user_request,
            steps=tuple(ExecutionStep(
                step_id=step.step_id.strip(), title=step.title.strip(),
                description=step.description.strip(),
                primary_tool=step.primary_tool.strip() if step.primary_tool else None,
                status=StepStatus.PENDING, attempt=0,
                depends_on_step_ids=dependencies[step.step_id.strip()],
            ) for step in proposal.steps),
        )
        return PlannerResult(outcome=PlannerOutcome.EXECUTION_PLAN,
                             request_id=planner_input.request_id, proposed_plan=plan,
                             message=proposal.message or "Plan generated successfully.",
                             planning_rationale=rationale)
    except (ValueError, TypeError, AttributeError) as exc:
        return planner_failure(planner_input.request_id, PlanningFailureCategory.INVALID_OUTPUT,
                               f"Planner proposal is invalid: {exc}")
