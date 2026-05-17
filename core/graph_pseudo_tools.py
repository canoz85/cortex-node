import ast
import json
import re
from uuid import uuid4

from langchain_core.messages import AIMessage

from core.graph_constants import PSEUDO_JSON_TOOL_CALL_PATTERN, PSEUDO_TOOL_CALL_PATTERN
from core.graph_messages import is_effectively_empty_response, normalize_message_content


def looks_like_pseudo_tool_text(content: str) -> bool:
    text = content or ""
    return bool(PSEUDO_TOOL_CALL_PATTERN.search(text) or PSEUDO_JSON_TOOL_CALL_PATTERN.search(text))


def is_pseudo_tool_response(message: AIMessage) -> bool:
    if getattr(message, "tool_calls", None):
        return False
    return looks_like_pseudo_tool_text(str(getattr(message, "content", "")))


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
    """Best-effort parse for pseudo tool text shaped like JSON."""
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


def recover_pseudo_tool_response(message: AIMessage, allowed_tool_names: set[str]) -> AIMessage:
    content = normalize_message_content(message)
    recovered_tool_call = _extract_json_pseudo_tool_call(content, allowed_tool_names)
    if recovered_tool_call is None:
        recovered_tool_call = _extract_function_pseudo_tool_call(content, allowed_tool_names)
    if recovered_tool_call is not None:
        return AIMessage(content="Recovered pseudo tool-call text into executable tool call.", tool_calls=[recovered_tool_call])
    return message


def finalize_action_response(response: AIMessage, allowed_tool_names: set[str]) -> AIMessage:
    if getattr(response, "tool_calls", None):
        return response

    if is_pseudo_tool_response(response):
        recovered = recover_pseudo_tool_response(response, allowed_tool_names)
        if getattr(recovered, "tool_calls", None):
            return recovered
        raw_preview = normalize_message_content(response).strip()
        raw_preview = re.sub(r"\s+", " ", raw_preview)
        raw_preview = raw_preview[:240]
        return AIMessage(
            content=(
                "Action-required run stopped because the model returned pseudo tool syntax that could not be recovered into an executable tool call. "
                f"Pseudo text preview: {raw_preview}"
            )
        )

    if is_effectively_empty_response(response):
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
