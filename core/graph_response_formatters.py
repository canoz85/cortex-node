import json

from langchain_core.messages import AIMessage

from core.graph_tool_events import current_turn_tool_events


def _shell_preview_for_run_python(args: dict) -> str:
    path = str(args.get("path", "")).strip()
    pieces = ["python"]
    if path:
        pieces.append(path)

    extra_args = args.get("v__args")
    if isinstance(extra_args, list):
        pieces.extend(str(item) for item in extra_args)

    arg_string = str(args.get("args", "") or "").strip()
    if arg_string:
        pieces.append(arg_string)
    return " ".join(piece for piece in pieces if piece)


def format_action_completion_response(history: list) -> str | None:
    events = current_turn_tool_events(history)
    if not events:
        return None

    successful_events = [event for event in events if event.get("success")]
    if not successful_events:
        return None

    written_files: list[str] = []
    created_dirs: list[str] = []
    verification_commands: list[str] = []
    verification_outputs: list[str] = []

    for event in successful_events:
        name = str(event.get("name", ""))
        args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
        result = event.get("result")
        unwrapped = event.get("unwrapped") if isinstance(event.get("unwrapped"), dict) else {}
        if name == "write_file":
            path = str(args.get("path") or unwrapped.get("path") or "").strip()
            if path and path not in written_files:
                written_files.append(path)
        elif name == "make_directory":
            path = str(args.get("path") or unwrapped.get("path") or "").strip()
            if path and path not in created_dirs:
                created_dirs.append(path)
        elif name == "run_python":
            command_preview = _shell_preview_for_run_python(args)
            if command_preview == "python" and isinstance(unwrapped, dict):
                path = str(args.get("path") or "").strip()
                if path:
                    command_preview = f"python {path}"
            if command_preview and command_preview not in verification_commands:
                verification_commands.append(command_preview)
            if result is not None and isinstance(result.data, dict):
                stdout = str(result.data.get("stdout", "")).strip()
                if stdout and stdout != "<empty>" and stdout not in verification_outputs:
                    verification_outputs.append(stdout)

    if not written_files and not created_dirs and not verification_commands:
        return None

    parts: list[str] = []
    if written_files:
        if len(written_files) == 1:
            parts.append(f"Implemented the requested changes in {written_files[0]}.")
        else:
            parts.append(f"Implemented the requested changes in: {', '.join(written_files)}.")
    if created_dirs:
        parts.append(f"Created directories: {', '.join(created_dirs)}.")
    if verification_commands:
        parts.append(f"Verified with: {', '.join(verification_commands)}.")
    if verification_outputs:
        parts.append(f"Verification output: {verification_outputs[-1]}")
    return " ".join(parts).strip()


def format_info_tool_response(tool_name: str, tool_result: dict) -> str:
    """Format info tool ToolResult into a readable response using message + data fields."""
    if not isinstance(tool_result, dict):
        return str(tool_result)

    message = tool_result.get("message", "")
    data = tool_result.get("data", {})

    if tool_name == "current_time" and isinstance(data, dict):
        formatted = data.get("formatted", "")
        return f"The current time is: {formatted}"

    if tool_name == "token_usage" and isinstance(data, dict):
        prompt = data.get("prompt_tokens", 0)
        completion = data.get("completion_tokens", 0)
        total = data.get("total_tokens", 0)
        return f"Token usage: {prompt} prompt tokens, {completion} completion tokens ({total} total)"

    if tool_name == "agent_info" and isinstance(data, dict):
        model = data.get("model", "unknown")
        context = data.get("context_window", "unknown")
        workspace = data.get("workspace", "unknown")
        return f"Agent info: Model={model}, Context window={context}, Workspace={workspace}"

    return message


def format_tool_call_preview(message: AIMessage) -> str:
    """Return a human-readable preview for AI messages that contain tool calls."""
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return "Preparing next action."

    previews = []
    for call in tool_calls:
        name = call.get("name", "tool")
        args = call.get("args", {})
        if args:
            previews.append(f"Calling {name} with {json.dumps(args, ensure_ascii=True)}")
        else:
            previews.append(f"Calling {name}")
    return "\n".join(previews)
