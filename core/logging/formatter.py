# core/logging/formatter.py

from __future__ import annotations

from langchain_core.messages import BaseMessage

from core.graph_messages import normalize_message_content
from core.protocol.models import ToolResult
from core.tool_output import parse_tool_result


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

        lines.append(f"Calling {name} with {args}")

    return "\n".join(lines)


def format_tool_result(tool_result: ToolResult | str | None) -> str:
    """Return a human-readable tool result."""

    if tool_result is None:
        return ""

    if isinstance(tool_result, ToolResult):
        return tool_result.message

    parsed = parse_tool_result(tool_result)

    if parsed is not None:
        return parsed.message

    return str(tool_result)