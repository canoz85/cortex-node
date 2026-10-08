"""Bounded structural defect data, kept outside retrieval and execution evidence."""

import re

from core.planner_limits import PLANNER_LENGTH_DIAGNOSTIC
from core.protocol.enums import PlanningFailureCategory
from core.protocol.models import PlannerResult, PlanningFeedback


PLANNING_FEEDBACK_HEADER = (
    "PLANNING FEEDBACK (Controller-owned previous-candidate defects; "
    "not execution authority; return one corrected Planner proposal):\n"
)


def planner_retry_feedback(result: PlannerResult) -> PlanningFeedback | None:
    if result.failure_category != PlanningFailureCategory.INVALID_OUTPUT:
        return None
    reason = result.message.removeprefix("Planner proposal is invalid: ").removeprefix(
        "Planner output is invalid (PlannerInvalidOutputError): "
    )
    safe = reason in {
        PLANNER_LENGTH_DIAGNOSTIC,
        "PLAN_PROPOSED requires at least one step",
        "proposed step ids must be unique",
        "proposed step dependency graph is cyclic",
        "proposed step titles and descriptions must be non-empty",
        "primary_tool cannot be empty",
        "primary_tool must be one of the currently authorized tools",
        "Planner proposal is missing required fields",
        "Required Planner semantic fields must be non-empty strings",
        "Planner proposal contains unsupported fields",
        "No executable steps are allowed without authorized tools",
    } or any(re.fullmatch(pattern, reason) for pattern in (
        r"PLAN_PROPOSED exceeds the \d+-step limit",
        r"primary_tool '[^'\r\n]{1,80}' is (?:unknown|unavailable)",
        r"step '[^'\r\n]{1,80}' contains duplicate dependencies",
        r"step '[^'\r\n]{1,80}' references unknown dependency '[^'\r\n]{1,80}'",
        r"step '[^'\r\n]{1,80}' cannot depend on itself",
    ))
    if not safe:
        # Provider parser errors may contain full responses or exception details.
        reason = "Planner output did not match the required proposal schema."
    return PlanningFeedback(message=reason[:240])
