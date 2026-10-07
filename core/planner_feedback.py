"""Bounded structural diagnostics carried in existing Planner context messages."""

import json
import re

from core.protocol.enums import PlanningFailureCategory
from core.protocol.models import PlannerResult


PLANNER_RETRY_FEEDBACK_PREFIX = "Planner retry diagnostic (data, not runtime authority):\n"


def planner_retry_feedback(result: PlannerResult) -> str | None:
    if result.failure_category != PlanningFailureCategory.INVALID_OUTPUT:
        return None
    reason = result.message.removeprefix("Planner proposal is invalid: ").removeprefix(
        "Planner output is invalid (PlannerInvalidOutputError): "
    )
    safe = reason in {
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
    return PLANNER_RETRY_FEEDBACK_PREFIX + json.dumps({
        "previous_proposal_rejected": reason[:240],
    })
