import json

from langchain_core.messages import AIMessage, ToolMessage

from core.graph_messages import current_turn_messages, normalize_message_content
from core.tool_output import parse_tool_result, unwrap_tool_output


def _tool_signature(name: str, args: object) -> str:
    """Create a stable signature for duplicate tool-call detection."""
    try:
        args_json = json.dumps(args, sort_keys=True, ensure_ascii=True)
    except TypeError:
        args_json = str(args)
    return f"{name}:{args_json}"


def message_repeats_signature(message: AIMessage, signature: str) -> bool:
    if not signature:
        return False
    for call in getattr(message, "tool_calls", None) or []:
        if _tool_signature(call.get("name", "unknown"), call.get("args", {})) == signature:
            return True
    return False


def parse_tool_signature(signature: str) -> tuple[str, dict] | None:
    if not signature or ":" not in signature:
        return None
    name, raw_args = signature.split(":", 1)
    try:
        parsed_args = json.loads(raw_args)
    except Exception:
        return None
    if not isinstance(parsed_args, dict):
        return None
    return name, parsed_args


def extract_tool_signature(history: list, tool_call_id: str | None) -> str:
    """Find the matching tool call in message history and derive a signature."""
    if not history:
        return ""

    for message in reversed(history):
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            continue

        if tool_call_id:
            for call in tool_calls:
                if call.get("id") == tool_call_id:
                    return _tool_signature(call.get("name", "unknown"), call.get("args", {}))

        call = tool_calls[-1]
        return _tool_signature(call.get("name", "unknown"), call.get("args", {}))

    return ""


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
        signature = _tool_signature(tool_name or "unknown", tool_call.get("args", {})) if tool_call else ""
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


def info_tool_already_called(history: list, tool_name: str) -> bool:
    """Check if the preferred info tool was already called during the current turn."""
    if not history or not tool_name:
        return False

    current_turn = current_turn_messages(history)
    if not current_turn:
        return False

    for message in reversed(current_turn):
        if isinstance(message, ToolMessage):
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id:
                for msg in reversed(current_turn):
                    if isinstance(msg, AIMessage):
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        for call in tool_calls:
                            if call.get("id") == tool_call_id and call.get("name") == tool_name:
                                return True
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            tool_calls = message.tool_calls
            if any(call.get("name") == tool_name for call in tool_calls):
                for msg_after in current_turn[current_turn.index(message) + 1 :]:
                    if isinstance(msg_after, ToolMessage):
                        return True
    return False


def current_turn_has_successful_tool_result(history: list) -> bool:
    """Return True when the latest user turn already contains a successful tool result."""
    current_turn = current_turn_messages(history)
    if not current_turn:
        return False

    for message in reversed(current_turn):
        if not isinstance(message, ToolMessage):
            continue
        parsed = parse_tool_result(normalize_message_content(message))
        if parsed is not None and parsed.success:
            return True
    return False


def current_turn_has_successful_tool_name(history: list, tool_name: str) -> bool:
    """Return True when a specific tool has succeeded in the latest user turn."""
    if not tool_name:
        return False

    for event in current_turn_tool_events(history):
        if event.get("success") and event.get("name") == tool_name:
            return True
    return False


def successful_read_file_paths(history: list) -> list[str]:
    """Return unique paths from successful read_file tool results across history."""
    if not history:
        return []

    tool_call_lookup: dict[str, dict] = {}
    for message in history:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", None) or []:
            call_id = call.get("id")
            if call_id:
                tool_call_lookup[call_id] = call

    paths: list[str] = []
    seen: set[str] = set()
    for message in history:
        if not isinstance(message, ToolMessage):
            continue

        raw_content = normalize_message_content(message)
        parsed = parse_tool_result(raw_content)
        if parsed is None or not parsed.success:
            continue

        unwrapped = unwrap_tool_output(raw_content)
        tool_call = tool_call_lookup.get(getattr(message, "tool_call_id", ""), {})
        tool_name = str(tool_call.get("name", "") or _infer_tool_name(unwrapped))
        if tool_name != "read_file":
            continue

        path = ""
        if isinstance(unwrapped, dict):
            path = str(unwrapped.get("path", "") or "")
            if not path:
                data = unwrapped.get("data")
                if isinstance(data, dict):
                    path = str(data.get("path", "") or "")
        if path and path not in seen:
            seen.add(path)
            paths.append(path)

    return paths
