def format_tool_result_response(tool_result: dict) -> str:
    """Format tool output from canonical display text produced at tool serialization."""
    if not isinstance(tool_result, dict):
        return str(tool_result)

    display = tool_result.get("display")
    if isinstance(display, str) and display.strip():
        return display
    message = str(tool_result.get("message", "") or "")
    return message or "Tool completed successfully."
