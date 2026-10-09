"""Deterministic normalization for structured Planner proposals."""
from pydantic import ValidationError

from core.planner_contract import PlannerProposal, PlannerProposalResultType
from core.protocol.enums import PlannerOutcome, PlanningFailureCategory, PlanningOperation, StepStatus
from core.protocol.models import ExecutionPlan, ExecutionStep, PlanningRequest, PlannerResult
from core.planner_validation import (
    MAX_PROPOSED_STEPS, PlanValidationError, authorized_plan_tools,
    validate_direct_response_route, validate_execution_plan, validate_proposed_step_count,
)

def planner_failure(
    request_id: str, category: PlanningFailureCategory, message: str, *,
    route: str | None = None,
) -> PlannerResult:
    return PlannerResult(
        outcome=PlannerOutcome.FAILED, request_id=request_id,
        planner_route=route, message=message, failure_category=category,
    )


def normalize_planner_proposal(
    content: object, planner_input: PlanningRequest, *, route: str,
) -> PlannerResult:
    """Validate a proposal and convert it to the existing PlannerResult path."""
    try:
        proposal = PlannerProposal.model_validate(
            content.model_dump() if isinstance(content, PlannerProposal) else content,
        )
    except (ValidationError, TypeError, ValueError, AttributeError) as exc:
        return planner_failure(planner_input.request_id, PlanningFailureCategory.INVALID_OUTPUT,
                               f"Planner output is invalid ({type(exc).__name__}).", route=route)

    try:
        if proposal.result == PlannerProposalResultType.NO_PLAN_REQUIRED:
            validate_direct_response_route(planner_input, route=route)
            semantic_content = proposal.message.strip()
            return PlannerResult(outcome=PlannerOutcome.DIRECT_RESPONSE, request_id=planner_input.request_id,
                                 planner_route=route, message=semantic_content,
                                 direct_response_content=semantic_content)
        if proposal.result == PlannerProposalResultType.NEEDS_INPUT:
            return PlannerResult(outcome=PlannerOutcome.CLARIFICATION_REQUIRED, request_id=planner_input.request_id,
                                 planner_route=route, message=proposal.message.strip())
        if proposal.result == PlannerProposalResultType.PLANNING_FAILED:
            return planner_failure(planner_input.request_id, PlanningFailureCategory.UNPLANNABLE,
                                   proposal.message, route=route)

        validate_proposed_step_count(proposal.steps)
        dependencies = {step.step_id.strip(): tuple(ref.strip() for ref in step.dependencies)
                        for step in proposal.steps}
        available = authorized_plan_tools(planner_input, route=route)
        revising = planner_input.operation == PlanningOperation.REVISE
        plan = ExecutionPlan(
            plan_id=planner_input.base_plan_id if revising else f"{planner_input.identity.execution_id}:plan",
            revision=planner_input.base_revision + 1 if revising else 1,
            objective=proposal.objective.strip() or planner_input.context.user_request,
            available_tools=tuple(sorted(available)),
            steps=tuple(ExecutionStep(
                step_id=step.step_id.strip(),
                title=step.title.strip(),
                description=step.description.strip(),
                primary_tool=step.primary_tool.strip(),
                status=StepStatus.PENDING,
                attempt=0,
                depends_on_step_ids=dependencies[step.step_id.strip()],
            ) for step in proposal.steps),
        )
        plan = validate_execution_plan(plan, planner_input, route=route)
        return PlannerResult(outcome=PlannerOutcome.EXECUTION_PLAN,
                             request_id=planner_input.request_id, planner_route=route,
                             proposed_plan=plan,
                             message=proposal.message or "Plan generated successfully.")
    except ValidationError as exc:
        return planner_failure(planner_input.request_id, PlanningFailureCategory.INVALID_OUTPUT,
                               f"Planner proposal conversion is invalid ({type(exc).__name__}).", route=route)
    except PlanValidationError as exc:
        return planner_failure(planner_input.request_id, PlanningFailureCategory.INVALID_OUTPUT,
                               f"Planner proposal is invalid: {exc}", route=route)
