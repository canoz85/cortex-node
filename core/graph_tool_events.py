import json

from langchain_core.messages import AIMessage, ToolMessage

from core.graph_messages import current_turn_messages, normalize_message_content
from core.tool_output import parse_tool_result, unwrap_tool_output


def _infer_tool_name(unwrapped: dict | list | str | None) -> str:
    if not isinstance(unwrapped, dict):
        return ""
    if "characters_written" in unwrapped:
        return "write_file"
    if "content" in unwrapped and "path" in unwrapped:
        return "read_file"
    if "entries" in unwrapped:
        return "list_files"
    if "path" in unwrapped and unwrapped.get("message", "").startswith("Directory ready"):
        return "make_directory"
    data = unwrapped.get("data")
    if isinstance(data, dict) and {"exit_code", "stdout", "stderr"}.issubset(data.keys()):
        return "run_python"
    return ""


def current_turn_tool_events(history: list) -> list[dict]:
    """Collect tool-call events for the latest user turn."""
    current_turn = current_turn_messages(history)
    if not current_turn:
        return []

    tool_call_lookup: dict[str, dict] = {}
    for message in current_turn:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", None) or []:
            call_id = call.get("id")
            if call_id:
                tool_call_lookup[call_id] = call

    events: list[dict] = []
    for message in current_turn:
        if not isinstance(message, ToolMessage):
            continue
        raw_content = normalize_message_content(message)
        parsed = parse_tool_result(raw_content)
        unwrapped = unwrap_tool_output(raw_content)
        tool_call = tool_call_lookup.get(getattr(message, "tool_call_id", ""), {})
        tool_name = str(tool_call.get("name", "") or _infer_tool_name(unwrapped))
        success = parsed.success if parsed is not None else bool(isinstance(unwrapped, dict) and unwrapped.get("success") is True)
        signature = f"{tool_name or 'unknown'}:{json.dumps(tool_call.get('args', {}), sort_keys=True)}" if tool_call else ""
        
        events.append(
            {
                "name": tool_name or "unknown",
                "args": tool_call.get("args", {}),
                "signature": signature,
                "success": success,
                "result": parsed,
                "unwrapped": unwrapped,
                "raw": raw_content,
            }
        )
    return events