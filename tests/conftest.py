import json

from core.models import ToolResult


def get_tool(tools: list, name: str):
    for t in tools:
        if getattr(t, "name", "") == name:
            return t
    raise AssertionError(f"Tool not found: {name}")


def parse_result(raw: str) -> dict:
    _, payload = ToolResult.split_tool_output(raw)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Could not parse tool output as JSON: {raw}") from exc
    assert isinstance(parsed, dict), f"Expected dict payload, got: {type(parsed).__name__}"
    return parsed
