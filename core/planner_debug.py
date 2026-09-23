"""Planner stdout diagnostics, gated by the same show_raw_llm flag as Brain."""

import json
import re

from core.protocol.models import PlanningRequest, PlannerResult
from core.debug import save_raw_llm

def summarize_raw_llm(raw: object) -> dict:
    response_metadata = getattr(raw, "response_metadata", {}) or {}
    usage_metadata = getattr(raw, "usage_metadata", {}) or {}

    content = getattr(raw, "content", None)

    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            pass

    return {
        "content": content,
        "model": response_metadata.get("model"),
        "done_reason": response_metadata.get("done_reason"),
        "usage": usage_metadata,
    }

def log_planner_result(
    result: PlannerResult,
    *,
    enabled: bool = True,
    execution_id: str | None = None,
) -> None:
    log_planner(
        "normalized",
        result.model_dump(mode="json"),
        enabled=enabled,
        execution_id=execution_id,
    )

    # if result.proposed_plan is not None:
    #     log_planner(
    #         "execution_plan",
    #         format_execution_plan(result.proposed_plan),
    #         enabled=enabled,
    #         execution_id=execution_id,
    #     )

def log_planner_request(
    planner_input: PlanningRequest,
    *,
    enabled: bool = True,
    execution_id: str | None = None,
) -> None:
    request_debug = planner_input.model_dump(
        mode="json",
        include={
            "request_id",
            "operation",
            "base_plan_id",
            "base_revision",
            "completed_step_ids",
            "interrupted_step",
            "trigger",
            "reason",
            "capabilities",
        },
    )

    log_planner(
        "request",
        {
            "user_request": planner_input.context.user_request,
            **request_debug,
        },
        enabled=enabled,
        execution_id=execution_id,
    )

def log_planner(
    section: str,
    value: object,
    *,
    enabled: bool = True,
    execution_id: str | None = None,
) -> None:
    # Diagnostics must never break Planner execution.
    try:
        # Serialize temporarily only for sanitization.
        if isinstance(value, str):
            text = value
            structured = False
        else:
            text = json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
            structured = True

        text = re.sub(
            r'(?i)\bBearer\s+[^\s"\']+',
            'Bearer [REDACTED]',
            text,
        )

        text = re.sub(
            r'(?i)(["\']?(?:api[_-]?key|password|passwd|secret|access[_-]?token|'
            r'refresh[_-]?token|authorization|credential)["\']?\s*[:=]\s*)'
            r'(?:"[^"\n]*"|\'[^\'\n]*\'|[^\s,;}]+)',
            r'\1[REDACTED]',
            text,
        )

        text = re.sub(
            r'(?i)(https?://)[^\s/@]+:[^\s/@]+@',
            r'\1[REDACTED]@',
            text,
        )

        text = re.sub(
            r'-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----',
            '[REDACTED]',
            text,
            flags=re.DOTALL,
        )

        # Restore structured JSON when possible.
        file_value: object = text

        if structured:
            try:
                file_value = json.loads(text)
            except json.JSONDecodeError:
                file_value = text

        save_raw_llm(
            "planner",
            section,
            file_value,
            execution_id=execution_id,
        )

        if not enabled:
            return

        console_text = (
            text
            if isinstance(file_value, str)
            else json.dumps(
                file_value,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

        print(f"[planner:{section}]\n{console_text}")

    except Exception:
        # Diagnostics must not turn a successful invocation into a failure.
        pass

def format_execution_plan(plan) -> str:
    lines = [f"Objective: {plan.objective}"]

    for step in plan.steps:
        lines.extend(
            (
                "",
                f"{step.step_id}. {step.title}",
                f"   tool: {step.primary_tool}",
                f"   depends_on: {list(step.depends_on_step_ids)}",
            )
        )

    return "\n".join(lines)