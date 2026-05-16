import ast
import json
import re
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from core.models import TokenUsage, ToolResult
from core.rag import WorkspaceRAG
from core.state import AgentState
from core.tool_output import parse_tool_result, unwrap_tool_output
from tools.exec_ops import get_exec_tools
from tools.file_ops import get_file_tools
from tools.git_ops import get_git_tools
from tools.info_ops import get_info_tools, update_token_usage
from tools.rag_ops import get_rag_tools
from tools.scada_ops import get_scada_tools

MAX_REASONING_STEPS = 24
RECENT_MESSAGE_WINDOW = 12
MAX_SUMMARY_CHARS = 1800
ANSI_GREEN = "\033[32m"
ANSI_RESET = "\033[0m"

SYSTEM_PROMPT_TEMPLATE = """You are CortexNode, a local-first autonomous software engineering agent.
You can reason, use tools, and iterate until the task is complete.
Runtime info:
- Model: {model}
- Context window: ~128k tokens
- Sandbox workspace: {workspace_dir}
- Knowledge folder: {knowledge_dir}
- Max reasoning steps per prompt: {max_steps}
Constraints:
- Operate only inside the sandbox workspace directory.
- Prefer Python solutions with clear, testable code.
- For time/date requests, use the current_time tool instead of generating guessed values.
- Use retrieved knowledge when the request matches one of the indexed examples or rules.
- Use rag_search if you need targeted context from the knowledge folder.
- After writing code, run it to verify behavior when possible.
- Do not print pseudo tool calls like write_file(...). If an action is needed, emit actual tool calls.
- If a tool call fails, do not repeat the same tool with identical arguments; choose a different next action.
- Keep responses concise and action-oriented.
"""

PSEUDO_TOOL_CALL_PATTERN = re.compile(
    r"\b(?:list_files|read_file|write_file|make_directory|run_python|git_status|git_diff|git_log|git_show|agent_info|token_usage|current_time|scada_status|rag_search|rag_refresh_index)\s*\(",
    re.IGNORECASE,
)

PSEUDO_JSON_TOOL_CALL_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"(?:list_files|read_file|write_file|make_directory|run_python|git_status|git_diff|git_log|git_show|agent_info|token_usage|current_time|scada_status|rag_search|rag_refresh_index)"\s*,\s*"arguments"\s*:',
    re.IGNORECASE,
)

MAX_PSEUDO_RETRIES = 2

ACTION_INTENT_PATTERN = re.compile(
    r"\b(create|write|edit|update|modify|generate|implement|fix|refactor|run|execute|test|build|add|remove|delete)\b",
    re.IGNORECASE,
)
TOKEN_USAGE_INTENT_PATTERN = re.compile(
    r"\b(token|tokens|usage|consumed|consume|spent|prompt tokens|completion tokens)\b",
    re.IGNORECASE,
)
CURRENT_TIME_INTENT_PATTERN = re.compile(
    r"\b(time|date|datetime|today|now|current time|current date)\b",
    re.IGNORECASE,
)
AGENT_INFO_INTENT_PATTERN = re.compile(
    r"\b(model|runtime|context window|max steps|workspace|agent info|configuration)\b",
    re.IGNORECASE,
)
CASUAL_CHAT_PATTERN = re.compile(
    r"\b(hi|hello|hey|thanks|thank you|how are you|what'?s up|good morning|good afternoon|good evening|my name is|i am|call me|what is my name|who am i|nice to meet you|bye|goodbye|see you)\b",
    re.IGNORECASE,
)
CODE_DISCUSSION_PATTERN = re.compile(
    r"\b(code|coding|bug|debug|error|issue|python|javascript|typescript|file|function|class|test|build|repo|repository|project|app|script|refactor|stack trace|traceback|api|database|sql|json|yaml|docker|git|langgraph|ollama|tool|workspace)\b",
    re.IGNORECASE,
)
CODING_DISCUSSION_QUESTION_PATTERN = re.compile(
    r"^(how|why|what|when|where|can|could|would|should|do)\b|\b(explain|help me|walk me through|show me how)\b",
    re.IGNORECASE,
)
FILE_GENERATION_PATTERN = re.compile(
    r"\b(cli|command line|script|tool|file|module|program|utility|app|sensor|json file)\b",
    re.IGNORECASE,
)
MODIFYING_TOOL_NAMES = {"write_file", "make_directory"}
VERIFICATION_TOOL_NAMES = {"run_python"}


def _tool_signature(name: str, args: object) -> str:
    """Create a stable signature for duplicate tool-call detection."""
    try:
        args_json = json.dumps(args, sort_keys=True, ensure_ascii=True)
    except TypeError:
        args_json = str(args)
    return f"{name}:{args_json}"


def _message_repeats_signature(message: AIMessage, signature: str) -> bool:
    if not signature:
        return False
    for call in getattr(message, "tool_calls", None) or []:
        if _tool_signature(call.get("name", "unknown"), call.get("args", {})) == signature:
            return True
    return False


def _last_read_file_snapshot(state: AgentState) -> tuple[str, str] | None:
    """Return the latest read_file path/content snapshot from state if available."""
    last_tool_output = state.get("last_tool_output", "")
    if not isinstance(last_tool_output, dict):
        return None

    path = str(last_tool_output.get("path", "") or "").strip()
    content = str(last_tool_output.get("content", "") or "")
    if not path:
        return None
    return path, content


def _response_has_unchanged_write(message: AIMessage, snapshot: tuple[str, str] | None) -> bool:
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


def _looks_like_pseudo_tool_text(content: str) -> bool:
    text = content or ""
    return bool(PSEUDO_TOOL_CALL_PATTERN.search(text) or PSEUDO_JSON_TOOL_CALL_PATTERN.search(text))


def _is_effectively_empty_response(message: AIMessage) -> bool:
    if getattr(message, "tool_calls", None):
        return False

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        return len(content) == 0
    return not bool(content)


def _is_pseudo_tool_response(message: AIMessage) -> bool:
    if getattr(message, "tool_calls", None):
        return False
    return _looks_like_pseudo_tool_text(str(getattr(message, "content", "")))


def _normalize_message_content(message: AIMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _recover_pseudo_tool_response(message: AIMessage, allowed_tool_names: set[str]) -> AIMessage:
    content = _normalize_message_content(message)
    recovered_tool_call = _extract_json_pseudo_tool_call(content, allowed_tool_names)
    if recovered_tool_call is None:
        recovered_tool_call = _extract_function_pseudo_tool_call(content, allowed_tool_names)
    if recovered_tool_call is not None:
        return AIMessage(content="Recovered pseudo tool-call text into executable tool call.", tool_calls=[recovered_tool_call])
    return message


def _escape_newlines_inside_strings(text: str) -> str:
    """Escape literal newlines occurring inside quoted string values."""
    if not text:
        return text

    result: list[str] = []
    in_string = False
    quote_char = ""
    escaped = False

    for char in text:
        if escaped:
            result.append(char)
            escaped = False
            continue

        if char == "\\":
            result.append(char)
            escaped = True
            continue

        if in_string:
            if char == quote_char:
                in_string = False
                quote_char = ""
                result.append(char)
                continue
            if char == "\n":
                result.append("\\n")
                continue
            if char == "\r":
                result.append("\\r")
                continue
            result.append(char)
            continue

        if char in {'"', "'"}:
            in_string = True
            quote_char = char
        result.append(char)

    return "".join(result)


def _extract_json_pseudo_tool_call(content: str, allowed_tool_names: set[str]) -> dict | None:
    """Best-effort parse for pseudo tool text shaped like JSON.

    Expected shape:
    {"name": "write_file", "arguments": {...}}
    """
    if not content:
        return None

    candidates: list[str] = []
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    fenced_match = re.search(r"```[a-zA-Z0-9_-]*\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    inline_match = re.search(r"(\{\s*\"name\"\s*:\s*\".*?\"\s*,\s*\"arguments\"\s*:.*\})", content, flags=re.DOTALL)
    if inline_match:
        candidates.append(inline_match.group(1).strip())

    for candidate in candidates:
        sanitized_candidate = _escape_newlines_inside_strings(candidate)
        try:
            parsed = json.loads(sanitized_candidate)
        except Exception:
            pythonish_candidate = re.sub(r"\btrue\b", "True", sanitized_candidate, flags=re.IGNORECASE)
            pythonish_candidate = re.sub(r"\bfalse\b", "False", pythonish_candidate, flags=re.IGNORECASE)
            pythonish_candidate = re.sub(r"\bnull\b", "None", pythonish_candidate, flags=re.IGNORECASE)
            try:
                parsed = ast.literal_eval(pythonish_candidate)
            except Exception:
                continue

        if not isinstance(parsed, dict):
            continue

        name = parsed.get("name")
        if not isinstance(name, str) or name not in allowed_tool_names:
            continue

        arguments = parsed.get("arguments", parsed.get("args", {}))
        if not isinstance(arguments, dict):
            continue

        return {
            "name": name,
            "args": arguments,
            "id": f"pseudo-{uuid4()}",
            "type": "tool_call",
        }

    return None


def _extract_function_pseudo_tool_call(content: str, allowed_tool_names: set[str]) -> dict | None:
    """Best-effort parse for pseudo tool text shaped like a Python function call."""
    if not content:
        return None

    match = re.search(
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)",
        content,
        flags=re.DOTALL,
    )
    if not match:
        return None

    tool_name = match.group(1)
    if tool_name not in allowed_tool_names:
        return None

    try:
        expression = ast.parse(f"_tool_proxy({match.group(2)})", mode="eval")
    except SyntaxError:
        return None

    call = expression.body
    if not isinstance(call, ast.Call):
        return None
    if call.args:
        return None

    arguments: dict[str, object] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        try:
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except Exception:
            return None

    return {
        "name": tool_name,
        "args": arguments,
        "id": f"pseudo-{uuid4()}",
        "type": "tool_call",
    }


def _finalize_action_response(response: AIMessage, allowed_tool_names: set[str]) -> AIMessage:
    if getattr(response, "tool_calls", None):
        return response

    if _is_pseudo_tool_response(response):
        recovered = _recover_pseudo_tool_response(response, allowed_tool_names)
        if getattr(recovered, "tool_calls", None):
            return recovered
        raw_preview = _normalize_message_content(response).strip()
        raw_preview = re.sub(r"\s+", " ", raw_preview)
        raw_preview = raw_preview[:240]
        return AIMessage(
            content=(
                "Action-required run stopped because the model returned pseudo tool syntax that could not be recovered into an executable tool call. "
                f"Pseudo text preview: {raw_preview}"
            )
        )

    if _is_effectively_empty_response(response):
        return AIMessage(
            content=(
                "Action-required run stopped because the model returned an empty response instead of a tool call."
            )
        )

    return AIMessage(
        content=(
            "Action-required run stopped because the model returned plain text instead of an executable tool call."
        )
    )


def _latest_user_message(history: list) -> str:
    for message in reversed(history):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _current_turn_messages(history: list) -> list:
    """Return messages from the latest user turn onward."""
    if not history:
        return []

    for index in range(len(history) - 1, -1, -1):
        if isinstance(history[index], HumanMessage):
            return history[index:]
    return history


def _requires_action(user_text: str) -> bool:
    text = (user_text or "").strip()
    if not text:
        return False
    return bool(ACTION_INTENT_PATTERN.search(text))


def _preferred_info_tool(user_text: str) -> str | None:
    text = (user_text or "").strip()
    if not text:
        return None
    if TOKEN_USAGE_INTENT_PATTERN.search(text):
        return "token_usage"
    if CURRENT_TIME_INTENT_PATTERN.search(text):
        return "current_time"
    if AGENT_INFO_INTENT_PATTERN.search(text):
        return "agent_info"
    return None


def _is_casual_chat(user_text: str) -> bool:
    """Return True for social/identity chat that should skip planning/tool routing."""
    text = (user_text or "").strip()
    if not text:
        return False
    if _preferred_info_tool(text) or _requires_action(text):
        return False
    if CODE_DISCUSSION_PATTERN.search(text):
        return False
    if CASUAL_CHAT_PATTERN.search(text):
        return True

    word_count = len(text.split())
    if word_count <= 6 and text.endswith("?"):
        return True
    return False


def _is_file_generation_request(user_text: str) -> bool:
    text = (user_text or "").strip()
    if not text or not _requires_action(text):
        return False
    return bool(FILE_GENERATION_PATTERN.search(text))


def _planner_route(user_text: str) -> str:
    text = (user_text or "").strip()
    if not text:
        return "conversation"
    if _preferred_info_tool(text):
        return "info"
    if _is_casual_chat(text):
        return "casual"
    if CODE_DISCUSSION_PATTERN.search(text) and CODING_DISCUSSION_QUESTION_PATTERN.search(text):
        return "coding_discussion"
    if _is_file_generation_request(text):
        return "action:file_generation"
    if _requires_action(text):
        return "action"
    if CODE_DISCUSSION_PATTERN.search(text):
        return "coding_discussion"
    return "conversation"


def _current_turn_tool_events(history: list) -> list[dict]:
    """Collect tool-call events for the latest user turn."""
    current_turn = _current_turn_messages(history)
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
        raw_content = str(message.content)
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


def _format_action_completion_response(history: list) -> str | None:
    events = _current_turn_tool_events(history)
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


def _should_finalize_action_turn(history: list, planner_route: str) -> bool:
    events = _current_turn_tool_events(history)
    if not events:
        return False

    successful_names = {str(event.get("name", "")) for event in events if event.get("success")}
    if planner_route == "action:file_generation":
        return bool(successful_names & MODIFYING_TOOL_NAMES) and bool(successful_names & VERIFICATION_TOOL_NAMES)
    return bool(successful_names & MODIFYING_TOOL_NAMES)


def _parse_tool_signature(signature: str) -> tuple[str, dict] | None:
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


def _last_failed_verification_rewrite_info(history: list, state: AgentState) -> tuple[str, str] | None:
    """Return (path, content) for the last written file that failed subsequent run_python verification."""
    if state.get("last_tool_success") is not False:
        return None

    parsed_signature = _parse_tool_signature(str(state.get("last_tool_signature", "")))
    if not parsed_signature:
        return None

    tool_name, tool_args = parsed_signature
    if tool_name != "run_python":
        return None

    path = str(tool_args.get("path", "")).strip()
    if not path.endswith(".py"):
        return None

    events = _current_turn_tool_events(history)
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


def _message_repeats_write_content(message: AIMessage, path: str, content: str) -> bool:
    if not path:
        return False
    for call in getattr(message, "tool_calls", None) or []:
        if call.get("name") != "write_file":
            continue
        args = call.get("args", {}) or {}
        if str(args.get("path", "")).strip() == path and str(args.get("content", "")) == content:
            return True
    return False


def _next_file_generation_verification_call(history: list) -> dict | None:
    """If a Python file was just written successfully, deterministically verify it next."""
    events = _current_turn_tool_events(history)
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


def _next_file_generation_repair_call(state: AgentState) -> dict | None:
    """After a failed verification, inspect the generated file before retrying execution."""
    if state.get("last_tool_success") is not False:
        return None

    parsed_signature = _parse_tool_signature(str(state.get("last_tool_signature", "")))
    if not parsed_signature:
        return None

    tool_name, tool_args = parsed_signature
    if tool_name != "run_python":
        return None

    path = str(tool_args.get("path", "")).strip()
    if not path.endswith(".py"):
        return None

    last_tool_output = state.get("last_tool_output", "")
    if _last_tool_missing_required_args(last_tool_output):
        return None

    return {
        "name": "read_file",
        "args": {"path": path},
        "id": f"pseudo-{uuid4()}",
        "type": "tool_call",
    }


def _file_generation_verification_failures(history: list) -> tuple[int, str]:
    """Return number of failed run_python calls in current turn and the latest stderr."""
    events = _current_turn_tool_events(history)
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


def _next_args_scope_autofix_call(history: list) -> dict | None:
    """Create a deterministic write_file repair call when args NameError is detected."""
    events = _current_turn_tool_events(history)
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

    return None


def _last_tool_missing_required_args(tool_output: dict[str, object] | str) -> bool:
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


def _last_tool_stderr(tool_output: dict[str, object] | str) -> str:
    if not isinstance(tool_output, dict):
        return ""
    data = tool_output.get("data")
    if not isinstance(data, dict):
        return ""
    return str(data.get("stderr", "") or "").strip()


def _last_tool_has_args_nameerror(tool_output: dict[str, object] | str) -> bool:
    stderr = _last_tool_stderr(tool_output)
    if not stderr:
        return False
    lowered = stderr.lower()
    return "nameerror" in lowered and "args" in lowered and "not defined" in lowered


def _info_tool_already_called(history: list, tool_name: str) -> bool:
    """Check if the preferred info tool was already called during the current turn."""
    if not history or not tool_name:
        return False

    current_turn = _current_turn_messages(history)
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


def _current_turn_has_successful_tool_result(history: list) -> bool:
    """Return True when the latest user turn already contains a successful tool result."""
    current_turn = _current_turn_messages(history)
    if not current_turn:
        return False

    for message in reversed(current_turn):
        if not isinstance(message, ToolMessage):
            continue
        parsed = parse_tool_result(str(message.content))
        if parsed is not None and parsed.success:
            return True
    return False


def _format_info_tool_response(tool_name: str, tool_result: dict) -> str:
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


def _format_tool_call_preview(message: AIMessage) -> str:
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


def _direct_discussion_response(
    planner_llm: ChatOllama,
    system_prompt: str,
    retrieval_messages: list[SystemMessage],
    rolling_summary: str,
    recent_history: list,
) -> AIMessage:
    messages = [
        SystemMessage(content=system_prompt),
        *retrieval_messages,
        *_rolling_summary_message(rolling_summary),
        *recent_history,
        SystemMessage(
            content=(
                "This turn is discussion-only. Answer directly in concise prose. "
                "Do not call tools, do not propose tool syntax, and do not create or modify files."
            )
        ),
    ]
    response = planner_llm.invoke(messages)
    content = _normalize_message_content(response).strip()
    if getattr(response, "tool_calls", None) or _looks_like_pseudo_tool_text(content) or not content:
        fallback = planner_llm.invoke(
            [
                *messages,
                SystemMessage(
                    content=(
                        "Your previous reply was not a direct discussion answer. "
                        "Reply with plain prose only, no code blocks and no tool-like syntax."
                    )
                ),
            ]
        )
        fallback_content = _normalize_message_content(fallback).strip()
        if fallback_content and not _looks_like_pseudo_tool_text(fallback_content):
            return AIMessage(content=fallback_content)
        return AIMessage(
            content=(
                "Describe the error message, the JSON input, and the code path that fails, and I will help isolate the parsing bug directly."
            )
        )
    return AIMessage(content=content)


def _retrieval_message(rag_service: WorkspaceRAG, query: str, top_k: int) -> list[SystemMessage]:
    context = rag_service.format_context(query=query, top_k=top_k)
    if not context:
        return []
    return [SystemMessage(content=context)]


def _recent_messages(history: list, limit: int = RECENT_MESSAGE_WINDOW) -> list:
    if limit <= 0:
        return []
    return history[-limit:]


def _clip_summary(text: str, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _rolling_summary_message(summary: str) -> list[SystemMessage]:
    compact = (summary or "").strip()
    if not compact:
        return []
    return [
        SystemMessage(
            content=(
                "Rolling summary from earlier turns (authoritative context):\n"
                f"{compact}"
            )
        )
    ]


def _update_rolling_summary(
    planner_llm: ChatOllama,
    existing_summary: str,
    recent_history: list,
) -> str:
    if not recent_history and not existing_summary:
        return ""

    summarization_prompt = (
        "Update the rolling conversation summary for a coding agent. "
        "Keep only durable facts and unresolved items. "
        "Output plain text with these headings exactly: Goal, Constraints, Decisions, Done, Next, Open Questions, Facts. "
        "Use short bullet-like lines, no markdown code blocks, no verbosity. "
        f"Hard limit: {MAX_SUMMARY_CHARS} characters."
    )

    summary_messages = [
        SystemMessage(content=summarization_prompt),
        SystemMessage(content=f"Existing summary:\n{existing_summary or '(none)'}"),
        *recent_history,
    ]
    response = planner_llm.invoke(summary_messages)
    text = str(getattr(response, "content", "") or "").strip()
    if not text:
        return _clip_summary(existing_summary or "")
    return _clip_summary(text)


def _extract_tool_signature(history: list, tool_call_id: str | None) -> str:
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


def build_app(
    workspace_dir: str = "workspace",
    model: str = "qwen2.5:7b",
    knowledge_dir: str = "knowledge",
    embedding_model: str = "nomic-embed-text",
    rag_top_k: int = 4,
):
    knowledge_root = Path(knowledge_dir).resolve()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        model=model,
        workspace_dir=workspace_dir,
        knowledge_dir=str(knowledge_root),
        max_steps=MAX_REASONING_STEPS,
    )

    rag_service = WorkspaceRAG(
        knowledge_root,
        embed_model=embedding_model,
        top_k=rag_top_k,
    )

    tools = [
        *get_file_tools(workspace_dir),
        *get_exec_tools(workspace_dir),
        *get_git_tools(workspace_dir),
        *get_info_tools(model=model, workspace_dir=workspace_dir),
        *get_rag_tools(rag_service),
        *get_scada_tools(workspace_dir),
    ]
    tool_name_set = {getattr(tool, "name", "") for tool in tools}

    llm = ChatOllama(model=model, temperature=0).bind_tools(tools)
    planner_llm = ChatOllama(model=model, temperature=0)  # No tools for planning

    def planner_node(state: AgentState):
        """First pass: analyze prompt and create a plan WITHOUT taking actions."""
        history = state.get("messages", [])
        recent_history = _recent_messages(history)
        previous_summary = state.get("rolling_summary", "")
        updated_summary = _update_rolling_summary(
            planner_llm=planner_llm,
            existing_summary=previous_summary,
            recent_history=recent_history,
        )

        latest_user_prompt = _latest_user_message(history)
        planner_route = _planner_route(latest_user_prompt)
        preferred_info_tool = _preferred_info_tool(latest_user_prompt)
        
        if planner_route == "info":
            plan_text = f"Info query detected: call {preferred_info_tool} tool and report the result."
        elif planner_route == "casual":
            plan_text = (
                "Casual conversation detected: respond directly without tools. "
                "Use conversation context for personal facts already shared and keep the reply brief."
            )
        elif planner_route == "coding_discussion":
            plan_text = (
                "Coding discussion detected: answer directly unless a targeted tool becomes necessary. "
                "Use conversation context, retrieved knowledge, and keep the reply concise."
            )
        elif planner_route == "conversation":
            plan_text = "Conversation detected: respond directly and briefly without tools unless the user asks for concrete action."
        else:
            retrieval_messages = _retrieval_message(rag_service, latest_user_prompt, rag_top_k)
            summary_message = _rolling_summary_message(updated_summary)
            
            planning_system = """You are a strategic planner. Analyze the user's request and create a clear step-by-step plan.
DO NOT take any actions yet. Just output:
1. What needs to be done (list of 2-4 key tasks)
2. File/tool sequence required
3. Expected outcome

Be concise. Format as a numbered list."""
            
            pre_messages = [
                SystemMessage(content=planning_system),
                *retrieval_messages,
                *summary_message,
            ]
            
            plan_response = planner_llm.invoke([*pre_messages, *recent_history])
            plan_text = str(plan_response.content)
        
        return {
            "plan": plan_text,
            "planner_route": planner_route,
            "rolling_summary": updated_summary,
            "steps": 0,  # Reset step counter for execution phase
            "last_tool_success": True,
            "repeat_fail_count": 0,
            "tool_text_retry_used": False,
        }

    def brain_node(state: AgentState):
        history = state.get("messages", [])
        recent_history = _recent_messages(history)
        rolling_summary = state.get("rolling_summary", "")
        planner_route = str(state.get("planner_route", ""))
        latest_user_prompt = _latest_user_message(history)
        action_required = _requires_action(latest_user_prompt)
        preferred_info_tool = _preferred_info_tool(latest_user_prompt)
        file_generation_requested = planner_route == "action:file_generation" or _is_file_generation_request(latest_user_prompt)
        successful_tool_result_in_turn = _current_turn_has_successful_tool_result(history)
        action_completion_summary = _format_action_completion_response(history)
        retrieval_messages = _retrieval_message(rag_service, _latest_user_message(history), rag_top_k)

        if planner_route in {"casual", "coding_discussion", "conversation"} and not preferred_info_tool:
            response = _direct_discussion_response(
                planner_llm=planner_llm,
                system_prompt=system_prompt,
                retrieval_messages=retrieval_messages,
                rolling_summary=rolling_summary,
                recent_history=recent_history,
            )
            meta = getattr(response, "response_metadata", {}) or {}
            usage = TokenUsage.from_response_metadata(meta)
            update_token_usage(usage.model_dump())
            return {
                "messages": [response],
                "steps": state.get("steps", 0) + 1,
                "token_usage": usage,
                "tool_text_retry_used": False,
            }

        if file_generation_requested:
            failed_verifications, latest_verification_error = _file_generation_verification_failures(history)
            if (
                failed_verifications >= 2
                and not _should_finalize_action_turn(history, planner_route)
                and not _last_tool_missing_required_args(latest_verification_error)
            ):
                response = AIMessage(
                    content=(
                        "Action-required run stopped after repeated verification failures while generating the file. "
                        f"Latest verification error: {latest_verification_error} "
                        "I prevented further read/write/run looping. Please retry and I will apply a different repair strategy immediately."
                    )
                )
                meta = getattr(response, "response_metadata", {}) or {}
                usage = TokenUsage.from_response_metadata(meta)
                update_token_usage(usage.model_dump())
                return {
                    "messages": [response],
                    "steps": state.get("steps", 0) + 1,
                    "token_usage": usage,
                    "tool_text_retry_used": False,
                }

            args_scope_fix_call = _next_args_scope_autofix_call(history)
            if args_scope_fix_call is not None:
                response = AIMessage(content="Applying deterministic args-scope repair before re-verification.", tool_calls=[args_scope_fix_call])
                meta = getattr(response, "response_metadata", {}) or {}
                usage = TokenUsage.from_response_metadata(meta)
                update_token_usage(usage.model_dump())
                return {
                    "messages": [response],
                    "steps": state.get("steps", 0) + 1,
                    "token_usage": usage,
                    "tool_text_retry_used": False,
                }

            repair_tool_call = _next_file_generation_repair_call(state)
            if repair_tool_call is not None:
                response = AIMessage(content="Inspecting the generated Python file before another verification attempt.", tool_calls=[repair_tool_call])
                meta = getattr(response, "response_metadata", {}) or {}
                usage = TokenUsage.from_response_metadata(meta)
                update_token_usage(usage.model_dump())
                return {
                    "messages": [response],
                    "steps": state.get("steps", 0) + 1,
                    "token_usage": usage,
                    "tool_text_retry_used": False,
                }

            verification_tool_call = _next_file_generation_verification_call(history)
            if verification_tool_call is not None:
                response = AIMessage(content="Proceeding to verify the generated Python file.", tool_calls=[verification_tool_call])
                meta = getattr(response, "response_metadata", {}) or {}
                usage = TokenUsage.from_response_metadata(meta)
                update_token_usage(usage.model_dump())
                return {
                    "messages": [response],
                    "steps": state.get("steps", 0) + 1,
                    "token_usage": usage,
                    "tool_text_retry_used": False,
                }

        if action_required and action_completion_summary and _should_finalize_action_turn(history, planner_route):
            response = AIMessage(content=action_completion_summary)
            meta = getattr(response, "response_metadata", {}) or {}
            usage = TokenUsage.from_response_metadata(meta)
            update_token_usage(usage.model_dump())
            return {
                "messages": [response],
                "steps": state.get("steps", 0) + 1,
                "token_usage": usage,
                "tool_text_retry_used": False,
            }
        
        info_tool_already_called = preferred_info_tool and _info_tool_already_called(history, preferred_info_tool)
        
        if info_tool_already_called:
            last_tool_result = state.get("last_tool_output", "")
            formatted_response = _format_info_tool_response(preferred_info_tool, last_tool_result)
            response = AIMessage(content=formatted_response)
        else:
            pre_messages = [
                SystemMessage(content=system_prompt),
                *retrieval_messages,
                *_rolling_summary_message(rolling_summary),
            ]
            skip_action_enforcement = False
            response_llm = planner_llm if planner_route in {"casual", "coding_discussion", "conversation"} else llm
            allow_tool_recovery = planner_route not in {"casual", "coding_discussion", "conversation"}
            if preferred_info_tool:
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "This request must be answered by calling the "
                            f"{preferred_info_tool} tool first. "
                            "Do not answer from memory. Use the tool and then respond from its output."
                        )
                    )
                )
            if file_generation_requested and not successful_tool_result_in_turn:
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "This is a concrete file-generation task inside the sandbox workspace. "
                            "Your next response should start with executable tool calls only. "
                            "Prefer write_file or make_directory for implementation, then run_python to verify when possible. "
                            "Do not explain planned code before taking action."
                        )
                    )
                )
            if file_generation_requested and state.get("last_tool_success") is False and _last_tool_missing_required_args(state.get("last_tool_output", "")):
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "The CLI verification failed because required command-line arguments were missing. "
                            "Do not rerun the same bare command. Instead, create a minimal sample input file in the workspace if needed, "
                            "then call run_python again with the required argument values using v__args."
                        )
                    )
                )
            if planner_route in {"coding_discussion", "conversation"}:
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "This turn is a discussion request, not an implementation request. "
                            "Answer directly and do not create files, run tools, or execute code unless the user explicitly asks for concrete actions."
                        )
                    )
                )
            if state.get("last_tool_success") is False and state.get("last_tool_signature"):
                signature = state.get("last_tool_signature", "")
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "The previous tool call failed. "
                            f"Failed signature: {signature}. "
                            "Do not repeat that same tool call with identical arguments. "
                            "Choose a different corrective next action."
                        )
                    )
                )
                stderr = _last_tool_stderr(state.get("last_tool_output", ""))
                if stderr:
                    pre_messages.append(
                        SystemMessage(
                            content=(
                                "Latest Python/tool error to fix before re-verification:\n"
                                f"{stderr[:1200]}"
                            )
                        )
                    )
                if _last_tool_has_args_nameerror(state.get("last_tool_output", "")):
                    pre_messages.append(
                        SystemMessage(
                            content=(
                                "The failure indicates args scope is broken. "
                                "Repair the script so parse_args() result is defined and used inside main(), "
                                "and avoid referencing args at module top level. "
                                "Write a corrected file version before running run_python again."
                            )
                        )
                    )
            elif action_required and state.get("last_tool_success") is True and state.get("last_tool_signature"):
                signature = state.get("last_tool_signature", "")
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "The previous tool call already succeeded. "
                            f"Successful signature: {signature}. "
                            "Do not repeat that same tool call with identical arguments. "
                            "Choose the next distinct step, such as verification, creating required sample input, or giving the final answer if the task is done."
                        )
                    )
                )
            elif action_required and successful_tool_result_in_turn:
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "A tool has already succeeded during this user turn. "
                            "If that successful result satisfies the request, provide the final concise answer now "
                            "instead of calling more tools. Only call another tool if a specific remaining gap still exists."
                        )
                    )
                )

            response = response_llm.invoke([*pre_messages, *recent_history])

            retry_used = state.get("tool_text_retry_used", False)
            pseudo_retry_count = 0
            while allow_tool_recovery and _is_pseudo_tool_response(response) and pseudo_retry_count < MAX_PSEUDO_RETRIES:
                response = response_llm.invoke(
                    [
                        *pre_messages,
                        *recent_history,
                        response,
                        SystemMessage(
                            content=(
                                "Your previous response included pseudo tool invocation text. "
                                "Do not output code blocks that look like tool calls. "
                                "If actions are needed, emit real tool calls only. "
                                "If no action is needed, provide only the final concise answer."
                            )
                        ),
                    ]
                )
                retry_used = True
                pseudo_retry_count += 1

            if allow_tool_recovery and _is_pseudo_tool_response(response):
                response = _recover_pseudo_tool_response(response, tool_name_set)
                if not getattr(response, "tool_calls", None):
                    response = AIMessage(
                        content=(
                            "I produced pseudo tool-call text instead of executable tool calls, so no action was taken. "
                            "Please retry with a task phrased as file changes inside the sandbox workspace."
                        )
                    )

            if _is_effectively_empty_response(response):
                response = response_llm.invoke(
                    [
                        *pre_messages,
                        *recent_history,
                        SystemMessage(
                            content=(
                                "Your previous response was empty. "
                                "Respond again with one of the following: "
                                "(1) concrete tool calls to progress the task, or "
                                "(2) a concise final answer."
                            )
                        ),
                    ]
                )

            if _is_effectively_empty_response(response):
                response = AIMessage(
                    content=(
                        "I could not produce a valid action or answer from the model. "
                        "Please retry the prompt or check model availability in Ollama."
                    )
                )

            if planner_route in {"coding_discussion", "conversation"} and getattr(response, "tool_calls", None):
                response = planner_llm.invoke(
                    [
                        *pre_messages,
                        *recent_history,
                        SystemMessage(
                            content=(
                                "You started taking actions for a discussion-only request. "
                                "Do not call tools. Provide a concise direct answer only."
                            )
                        ),
                    ]
                )

            if (
                action_required
                and state.get("last_tool_signature")
                and _message_repeats_signature(response, str(state.get("last_tool_signature", "")))
            ):
                repeated_signature = str(state.get("last_tool_signature", ""))
                repeat_reason = (
                    "already succeeded"
                    if state.get("last_tool_success") is True
                    else "already failed"
                )
                response = llm.invoke(
                    [
                        *pre_messages,
                        *recent_history,
                        response,
                        SystemMessage(
                            content=(
                                "You repeated the exact same tool call again. "
                                f"Repeated signature: {repeated_signature}. "
                                f"That signature {repeat_reason}. "
                                "Do not emit that same tool call again. "
                                "Choose the next distinct step now, such as fixing the file, reading the error output, creating sample input, verifying with different arguments, or giving the final answer if the task is complete."
                            )
                        ),
                    ]
                )

            if file_generation_requested and state.get("last_tool_success") is True:
                read_snapshot = _last_read_file_snapshot(state)
                unchanged_retry_count = 0
                while _response_has_unchanged_write(response, read_snapshot) and unchanged_retry_count < 2:
                    response = llm.invoke(
                        [
                            *pre_messages,
                            *recent_history,
                            response,
                            SystemMessage(
                                content=(
                                    "Your proposed write_file call rewrites the file with identical content after a failed verification. "
                                    "That is a no-op and will loop. "
                                    "Produce a corrected write_file call that changes the file to address the failing error, then verify again. "
                                    "Do not emit an unchanged write_file call."
                                )
                            ),
                        ]
                    )
                    response = _finalize_action_response(response, tool_name_set)
                    unchanged_retry_count += 1

                if _response_has_unchanged_write(response, read_snapshot):
                    response = AIMessage(
                        content=(
                            "Action-required run stopped because repair attempts kept rewriting identical file content after a failed verification. "
                            "Please retry so I can apply a different repair strategy."
                        )
                    )
                    skip_action_enforcement = True

            failed_rewrite_context = _last_failed_verification_rewrite_info(history, state)
            if failed_rewrite_context and getattr(response, "tool_calls", None):
                failed_path, failed_write_content = failed_rewrite_context
                if failed_write_content and _message_repeats_write_content(response, failed_path, failed_write_content):
                    last_tool_output = state.get("last_tool_output", "")
                    failure_details = ""
                    if isinstance(last_tool_output, dict):
                        data = last_tool_output.get("data")
                        if isinstance(data, dict):
                            failure_details = str(data.get("stderr", "") or data.get("stdout", "") or "")
                    response = llm.invoke(
                        [
                            *pre_messages,
                            *recent_history,
                            response,
                            SystemMessage(
                                content=(
                                    "You rewrote the same file content that already failed verification. "
                                    f"File: {failed_path}. "
                                    "Do not write the same content again. Produce a corrected write_file call that changes the file to address this failure before any further verification. "
                                    f"Failure details: {failure_details}"
                                )
                            ),
                        ]
                    )

            if file_generation_requested and not successful_tool_result_in_turn and not getattr(response, "tool_calls", None):
                response = llm.invoke(
                    [
                        *pre_messages,
                        *recent_history,
                        response,
                        SystemMessage(
                            content=(
                                "This file-generation request must continue with executable tool calls now. "
                                "Return tool calls only. Start by creating or updating files in the sandbox, then verify the result."
                            )
                        ),
                    ]
                )
                response = _finalize_action_response(response, tool_name_set)

            if preferred_info_tool and not getattr(response, "tool_calls", None):
                response = llm.invoke(
                    [
                        *pre_messages,
                        *recent_history,
                        response,
                        SystemMessage(
                            content=(
                                "You ignored the required info tool. "
                                f"Call {preferred_info_tool} now. "
                                "Do not answer from memory or provide a prose-only response."
                            )
                        ),
                    ]
                )
                response = _finalize_action_response(response, tool_name_set)

            if action_required and not skip_action_enforcement and not successful_tool_result_in_turn and not getattr(response, "tool_calls", None):
                response = llm.invoke(
                    [
                        *pre_messages,
                        *recent_history,
                        response,
                        SystemMessage(
                            content=(
                                "The user requested concrete actions. "
                                "Return at least one executable tool call now. "
                                "Do not return a prose-only response."
                            )
                        ),
                    ]
                )
                response = _finalize_action_response(response, tool_name_set)

        meta = getattr(response, "response_metadata", {}) or {}
        usage = TokenUsage.from_response_metadata(meta)
        update_token_usage(usage.model_dump())
        return {
            "messages": [response],
            "steps": state.get("steps", 0) + 1,
            "token_usage": usage,
            "tool_text_retry_used": False,
        }

    def capture_tool_output_node(state: AgentState):
        history = state.get("messages", [])
        if not history:
            return {
                "last_tool_output": "",
                "last_tool_signature": "",
                "last_tool_success": True,
                "repeat_fail_count": 0,
            }

        last_message = history[-1]
        if isinstance(last_message, ToolMessage):
            raw_content = str(last_message.content)
            parsed = parse_tool_result(raw_content)
            unwrapped = unwrap_tool_output(raw_content)
            success = parsed.success if parsed is not None else bool(isinstance(unwrapped, dict) and unwrapped.get("success") is True)
            current_signature = _extract_tool_signature(history[:-1], getattr(last_message, "tool_call_id", None))

            previous_signature = state.get("last_tool_signature", "")
            previous_success = state.get("last_tool_success", True)
            previous_repeat_count = state.get("repeat_fail_count", 0)
            if not success and current_signature and previous_signature == current_signature and not previous_success:
                repeat_fail_count = previous_repeat_count + 1
            elif not success and current_signature:
                repeat_fail_count = 1
            else:
                repeat_fail_count = 0

            if isinstance(unwrapped, dict):
                return {
                    "last_tool_output": unwrapped,
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            if isinstance(unwrapped, list):
                return {
                    "last_tool_output": {"message": str(unwrapped), "data": unwrapped, "success": success},
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            if isinstance(unwrapped, str):
                return {
                    "last_tool_output": {"message": unwrapped, "data": None, "success": success},
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            return {
                "last_tool_output": raw_content,
                "last_tool_signature": current_signature,
                "last_tool_success": success,
                "repeat_fail_count": repeat_fail_count,
            }
        return {
            "last_tool_output": state.get("last_tool_output", ""),
            "last_tool_signature": state.get("last_tool_signature", ""),
            "last_tool_success": state.get("last_tool_success", True),
            "repeat_fail_count": state.get("repeat_fail_count", 0),
        }

    def route_after_brain(state: AgentState):
        history = state.get("messages", [])
        if not history:
            return END

        if state.get("steps", 0) >= MAX_REASONING_STEPS:
            return END

        last_message = history[-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("brain", brain_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("capture_tool_output", capture_tool_output_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "brain")
    workflow.add_conditional_edges("brain", route_after_brain)
    workflow.add_edge("tools", "capture_tool_output")
    workflow.add_edge("capture_tool_output", "brain")

    return workflow.compile()


def run_prompt(
    app,
    prompt: str,
    history: list | None = None,
    rolling_summary: str = "",
) -> tuple[list, str]:
    """Run a single prompt and return updated history with rolling summary."""
    prior_messages = history or []
    initial_state: AgentState = {
        "messages": [*prior_messages, HumanMessage(content=prompt)],
        "steps": 0,
        "plan": "",
        "planner_route": "",
        "rolling_summary": rolling_summary,
        "last_tool_output": "",
        "last_tool_signature": "",
        "last_tool_success": True,
        "repeat_fail_count": 0,
        "tool_text_retry_used": False,
    }

    final_messages = list(initial_state["messages"])
    last_usage: dict = {}
    latest_step_count = 0
    saw_pseudo_stop = False
    latest_summary = ""
    saw_action_stop = False
    events = app.stream(initial_state)
    for event in events:
        for node_name, value in event.items():
            # if isinstance(value, dict) and "token_usage" in value:
            #     last_usage = value["token_usage"]

            if isinstance(value, dict):
                latest_step_count = int(value.get("steps", latest_step_count) or 0)
                if "rolling_summary" in value:
                    latest_summary = str(value.get("rolling_summary") or "")

                if node_name == "planner" and value.get("plan"):
                    planner_route = str(value.get("planner_route") or "")
                    header = "[planner]"
                    if planner_route:
                        header = f"[planner:{planner_route}]"
                    print(f"\n{ANSI_GREEN}{header}{ANSI_RESET}")
                    print(str(value.get("plan")))

            messages = value.get("messages") if isinstance(value, dict) else None
            if not messages:
                continue

            message = messages[-1]
            final_messages.append(message)
            print(f"\n[{node_name}]")
            if getattr(message, "tool_calls", None):
                print(_format_tool_call_preview(message))
            else:
                raw_content = _normalize_message_content(message)
                if "pseudo tool-call text" in raw_content:
                    saw_pseudo_stop = True
                if "Action-required run stopped" in raw_content:
                    saw_action_stop = True
                summary, _ = ToolResult.split_tool_output(raw_content)
                parsed = parse_tool_result(raw_content)
                if parsed is not None:
                    print(parsed.to_pretty_text())
                elif summary:
                    print(summary)
                else:
                    print(raw_content)

            # if node_name == "brain" and last_usage:
            #     print(
            #         f"  [tokens] prompt={last_usage.get('prompt_tokens', '?')}  "
            #         f"completion={last_usage.get('completion_tokens', '?')}  "
            #         f"total={last_usage.get('total_tokens', '?')}"
            #     )

    if latest_step_count >= MAX_REASONING_STEPS:
        print(
            f"\n[system]\nMax reasoning steps reached ({MAX_REASONING_STEPS}). "
            "Stopping to avoid unbounded loops."
        )

    if saw_pseudo_stop:
        print(
            "\n[system]\nThe model returned pseudo tool syntax, so the run was halted without executing those actions."
        )

    if saw_action_stop:
        print(
            "\n[system]\nRun ended without a final executable tool call. See the last [brain] message for the stop reason."
        )

    return final_messages, latest_summary
