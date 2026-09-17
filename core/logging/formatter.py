# core/logging/formatter.py

from __future__ import annotations

from langchain_core.messages import BaseMessage

from core.graph_messages import normalize_message_content
from core.protocol.models import ToolResult
from core.tool_output import parse_tool_result


MAX_DEFAULT_TOOL_RESULT_CHARS = 500


def _compact_tool_message(message: str) -> str:
    text = message.strip()
    if len(text) <= MAX_DEFAULT_TOOL_RESULT_CHARS:
        return text
    return text[:MAX_DEFAULT_TOOL_RESULT_CHARS].rstrip() + "..."


def format_ai_message(message: BaseMessage | None) -> str:
    """Return normalized AI message text."""
    if message is None:
        return ""

    return normalize_message_content(message)


def format_planner_plan(planner_result) -> str:
    """Return planner text for console rendering."""
    if planner_result is None:
        return ""

    if planner_result.proposed_plan is not None:
        return planner_result.proposed_plan.objective

    return planner_result.message


def format_accepted_plan(plan) -> str:
    """Return a compact Controller-accepted plan for default rendering."""
    if plan is None:
        return ""

    lines: list[str] = []
    for index, step in enumerate(plan.steps, start=1):
        if lines:
            lines.append("")
        lines.append(f"{index}. {step.title}")
        if step.primary_tool:
            lines.append(f"   tool: {step.primary_tool}")
    return "\n".join(lines)


def format_tool_call_preview(message: BaseMessage | None) -> str:
    """Return a readable tool-call preview."""
    if message is None:
        return ""

    tool_calls = list(getattr(message, "tool_calls", None) or [])

    if not tool_calls:
        return normalize_message_content(message)

    lines: list[str] = []

    for tool_call in tool_calls:
        name = tool_call.get("name", "<unknown>")
        args = tool_call.get("args", {})

        #lines.append(f"Calling {name} with {args}")
        lines.append(f"Calling {name}")


    return "\n".join(lines)


def format_tool_result(tool_result: ToolResult | str | None) -> str:
    """Return a human-readable tool result."""

    if tool_result is None:
        return ""

    if isinstance(tool_result, ToolResult):
        message = _compact_tool_message(tool_result.message)
        if message:
            return message
        return "Completed." if tool_result.success else "Failed."

    parsed = parse_tool_result(tool_result)

    if parsed is not None:
        message = _compact_tool_message(parsed.message)
        if message:
            return message
        return "Completed." if parsed.success else "Failed."

    return "Completed."
