"""Legacy numbered-plan compatibility, to be removed in P3.

This deliberately preserves the old parser, including skipping unmatched lines.
It is not a structured proposal contract and must not gain repair heuristics.
"""

import re

from core.protocol.enums import PlannerOutcome, StepStatus, PlanningOperation
from core.protocol.models import ExecutionPlan, ExecutionStep, PlanningRequest, PlannerResult


DIRECT_RESPONSE_ROUTES = frozenset({"conversation", "clarify_domain"})
STEP_RE = re.compile(r"^\s*(\d+)\.\s+(.*?)\s+[–-]\s+(.*)$")
TOOL_RE = re.compile(r"Use\s+`?([A-Za-z_][A-Za-z0-9_]*)`?", re.IGNORECASE)


def planner_failure(category: str, error: Exception) -> PlannerResult:
    return PlannerResult(
        outcome=PlannerOutcome.FAILED,
        message=f"Planner {category} failed ({type(error).__name__}).",
    )


def _legacy_execution_steps(text: str) -> tuple[ExecutionStep, ...]:
    steps = []
    previous_step_id = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = STEP_RE.match(line)
        if not match:
            continue
        number, title, description = match.groups()
        step_id = f"step-{number}"
        tool_match = TOOL_RE.search(description)
        steps.append(ExecutionStep(
            step_id=step_id,
            title=title.strip(),
            description=description.strip(),
            primary_tool=tool_match.group(1) if tool_match else None,
            status=StepStatus.PENDING,
            attempt=0,
            depends_on_step_ids=(previous_step_id,) if previous_step_id else (),
        ))
        previous_step_id = step_id
    if not steps:
        raise ValueError("Planner produced no executable steps.")
    return tuple(steps)


def normalize_planner_output(
    content: object, planner_input: PlanningRequest, *, route: str, confidence: float,
) -> PlannerResult:
    """Convert current provider content to the existing result, without state writes."""
    try:
        rationale = f"Route '{route}' selected with confidence {confidence:.2f}."
        if route in DIRECT_RESPONSE_ROUTES:
            return PlannerResult(
                outcome=PlannerOutcome.DIRECT_RESPONSE,
                message="No execution plan required.",
                planning_rationale=rationale,
            )
        # Preserve the previous provider-content conversion, not a new prose format.
        text = str(content)
        steps = _legacy_execution_steps(text)
        revising = planner_input.operation == PlanningOperation.REVISE
        plan = ExecutionPlan(
            plan_id=planner_input.base_plan_id if revising else f"{planner_input.identity.execution_id}:plan",
            revision=planner_input.base_revision + 1 if revising else 1,
            objective=text,
            steps=steps,
        )
        return PlannerResult(
            outcome=PlannerOutcome.EXECUTION_PLAN,
            proposed_plan=plan,
            message="Plan generated successfully.",
            planning_rationale=rationale,
        )
    except (ValueError, TypeError, AttributeError) as exc:
        return planner_failure("normalization", exc)
