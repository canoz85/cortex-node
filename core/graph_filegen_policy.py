import re
from uuid import uuid4

from langchain_core.messages import AIMessage

from core.graph_constants import MODIFYING_TOOL_NAMES, VERIFICATION_TOOL_NAMES
from core.graph_tool_events import current_turn_tool_events, parse_tool_signature
from core.state import AgentState


def last_read_file_snapshot(state: AgentState) -> tuple[str, str] | None:
    """Return the latest read_file path/content snapshot from state if available."""
    last_tool_output = state.get("last_tool_output", "")
    if not isinstance(last_tool_output, dict):
        return None

    path = str(last_tool_output.get("path", "") or "").strip()
    content = str(last_tool_output.get("content", "") or "")
    if not path:
        return None
    return path, content


def response_has_unchanged_write(message: AIMessage, snapshot: tuple[str, str] | None) -> bool:
    """Detect write_file calls that rewrite exactly the same file content."""
    if snapshot is None:
        return False

    snapshot_path, snapshot_content = snapshot
    for call in getattr(message, "tool_calls", None) or []:
        if call.get("name") != "write_file":
            continue
        args = call.get("args", {}) or {}
        write_path = str(args.get("path", "") or "").strip()
        write_content = str(args.get("content", "") or "")
        if write_path == snapshot_path and write_content == snapshot_content:
            return True
    return False


def should_finalize_action_turn(history: list, planner_route: str) -> bool:
    events = current_turn_tool_events(history)
    if not events:
        return False

    successful_names = {str(event.get("name", "")) for event in events if event.get("success")}
    if planner_route == "action:file_generation":
        return bool(successful_names & MODIFYING_TOOL_NAMES) and bool(successful_names & VERIFICATION_TOOL_NAMES)
    return bool(successful_names & MODIFYING_TOOL_NAMES)


def last_failed_verification_rewrite_info(history: list, state: AgentState) -> tuple[str, str] | None:
    """Return (path, content) for the last written file that failed subsequent run_python verification."""
    if state.get("last_tool_success") is not False:
        return None

    parsed_signature = parse_tool_signature(str(state.get("last_tool_signature", "")))
    if not parsed_signature:
        return None

    tool_name, tool_args = parsed_signature
    if tool_name != "run_python":
        return None

    path = str(tool_args.get("path", "")).strip()
    if not path.endswith(".py"):
        return None

    events = current_turn_tool_events(history)
    for event in reversed(events):
        if not event.get("success") or str(event.get("name", "")) != "write_file":
            continue
        args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
        unwrapped = event.get("unwrapped") if isinstance(event.get("unwrapped"), dict) else {}
        written_path = str(args.get("path") or unwrapped.get("path") or "").strip()
        if written_path == path:
            content = str(args.get("content") or "")
            return path, content
    return None


def message_repeats_write_content(message: AIMessage, path: str, content: str) -> bool:
    if not path:
        return False
    for call in getattr(message, "tool_calls", None) or []:
        if call.get("name") != "write_file":
            continue
        args = call.get("args", {}) or {}
        if str(args.get("path", "")).strip() == path and str(args.get("content", "")) == content:
            return True
    return False


def next_file_generation_verification_call(history: list) -> dict | None:
    """If a Python file was just written successfully, deterministically verify it next."""
    events = current_turn_tool_events(history)
    if not events:
        return None

    last_run_python_index = max(
        (index for index, event in enumerate(events) if str(event.get("name", "")) == "run_python"),
        default=-1,
    )

    last_successful_write_index = -1
    last_written_python_path = ""
    for index, event in enumerate(events):
        if not event.get("success") or str(event.get("name", "")) != "write_file":
            continue
        args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
        unwrapped = event.get("unwrapped") if isinstance(event.get("unwrapped"), dict) else {}
        path = str(args.get("path") or unwrapped.get("path") or "").strip()
        if path.endswith(".py"):
            last_successful_write_index = index
            last_written_python_path = path

    if last_successful_write_index == -1 or last_successful_write_index <= last_run_python_index:
        return None

    return {
        "name": "run_python",
        "args": {"path": last_written_python_path},
        "id": f"pseudo-{uuid4()}",
        "type": "tool_call",
    }


def next_file_generation_repair_call(state: AgentState) -> dict | None:
    """After a failed verification, inspect the generated file before retrying execution."""
    if state.get("last_tool_success") is not False:
        return None

    parsed_signature = parse_tool_signature(str(state.get("last_tool_signature", "")))
    if not parsed_signature:
        return None

    tool_name, tool_args = parsed_signature
    if tool_name != "run_python":
        return None

    path = str(tool_args.get("path", "")).strip()
    if not path.endswith(".py"):
        return None

    last_tool_output = state.get("last_tool_output", "")
    if last_tool_missing_required_args(last_tool_output):
        return None

    return {
        "name": "read_file",
        "args": {"path": path},
        "id": f"pseudo-{uuid4()}",
        "type": "tool_call",
    }


def file_generation_verification_failures(history: list) -> tuple[int, str]:
    """Return number of failed run_python calls in current turn and the latest stderr."""
    events = current_turn_tool_events(history)
    fail_count = 0
    latest_error = ""
    for event in events:
        if str(event.get("name", "")) != "run_python":
            continue
        if event.get("success"):
            continue
        fail_count += 1
        result = event.get("result")
        if result is not None and isinstance(result.data, dict):
            latest_error = str(result.data.get("stderr", "") or "")
    return fail_count, latest_error


def _apply_args_scope_repair(content: str) -> str:
    """Repair a common args-scope bug where validation lines escape main()."""
    if not content:
        return content

    repaired = content
    repaired = re.sub(
        r"(?m)^data = read_and_validate_json\(args\.file_path\)$",
        "    data = read_and_validate_json(args.file_path)",
        repaired,
    )
    repaired = re.sub(
        r"(?m)^if data is not None:$",
        "    if data is not None:",
        repaired,
    )
    repaired = re.sub(
        r"(?m)^[ \t]{0,4}print\((f?[\"\']Sensor data loaded successfully: \{data\}[\"\']|.*)\)$",
        "        print(\\1)",
        repaired,
    )
    return repaired


def next_args_scope_autofix_call(history: list) -> dict | None:
    """Create a deterministic write_file repair call when args NameError is detected."""
    events = current_turn_tool_events(history)
    if not events:
        return None

    latest_failed_run_index = -1
    latest_failed_run: dict | None = None
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        if str(event.get("name", "")) != "run_python" or event.get("success"):
            continue
        result = event.get("result")
        if result is None or not isinstance(result.data, dict):
            continue
        stderr = str(result.data.get("stderr", "") or "")
        lowered = stderr.lower()
        if "nameerror" in lowered and "args" in lowered and "not defined" in lowered:
            latest_failed_run_index = index
            latest_failed_run = event
            break

    if latest_failed_run is None:
        return None

    run_args = latest_failed_run.get("args", {}) if isinstance(latest_failed_run.get("args"), dict) else {}
    failed_path = str(run_args.get("path", "")).strip()
    if not failed_path.endswith(".py"):
        return None

    latest_read_index = -1
    latest_read_event: dict | None = None
    for index in range(len(events) - 1, latest_failed_run_index, -1):
        event = events[index]
        if not event.get("success") or str(event.get("name", "")) != "read_file":
            continue
        unwrapped = event.get("unwrapped") if isinstance(event.get("unwrapped"), dict) else {}
        read_path = str(unwrapped.get("path", "")).strip()
        if read_path != failed_path:
            continue
        latest_read_index = index
        latest_read_event = event
        break

    if latest_read_event is None:
        return None

    for index in range(latest_read_index + 1, len(events)):
        event = events[index]
        if not event.get("success") or str(event.get("name", "")) != "write_file":
            continue
        args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
        unwrapped = event.get("unwrapped") if isinstance(event.get("unwrapped"), dict) else {}
        written_path = str(args.get("path") or unwrapped.get("path") or "").strip()
        if written_path == failed_path:
            return None

    unwrapped = latest_read_event.get("unwrapped") if isinstance(latest_read_event.get("unwrapped"), dict) else {}
    read_content = str(unwrapped.get("content", ""))
    repaired = _apply_args_scope_repair(read_content)
    if repaired == read_content:
        return None
    return {
        "name": "write_file",
        "args": {"path": failed_path, "content": repaired, "overwrite": True},
        "id": f"pseudo-{uuid4()}",
        "type": "tool_call",
    }


def last_tool_missing_required_args(tool_output: dict[str, object] | str) -> bool:
    if isinstance(tool_output, str):
        stderr = tool_output
    elif isinstance(tool_output, dict):
        data = tool_output.get("data")
        if not isinstance(data, dict):
            return False
        stderr = str(data.get("stderr", "") or "")
    else:
        return False
    return "the following arguments are required" in stderr.lower()


def last_tool_stderr(tool_output: dict[str, object] | str) -> str:
    if not isinstance(tool_output, dict):
        return ""
    data = tool_output.get("data")
    if not isinstance(data, dict):
        return ""
    return str(data.get("stderr", "") or "").strip()


def last_tool_has_args_nameerror(tool_output: dict[str, object] | str) -> bool:
    stderr = last_tool_stderr(tool_output)
    if not stderr:
        return False
    lowered = stderr.lower()
    return "nameerror" in lowered and "args" in lowered and "not defined" in lowered
