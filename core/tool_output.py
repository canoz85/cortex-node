from typing import Any

from core.models import ToolResult


def parse_tool_result(raw: Any) -> ToolResult | None:
    """Return a ToolResult if the raw value contains a valid structured payload."""
    return ToolResult.try_parse(raw)


def unwrap_tool_output(raw: Any) -> dict[str, Any] | list[Any] | str | None:
    """Unwrap summary-plus-JSON tool output into Python values."""
    return ToolResult.unwrap_tool_output(raw)