import ast
import json
import re
from uuid import uuid4

from langchain_core.messages import AIMessage

from core.graph_constants import PSEUDO_JSON_TOOL_CALL_PATTERN, PSEUDO_TOOL_CALL_PATTERN
from core.graph_messages import is_effectively_empty_response, normalize_message_content

def is_generic_json_tool_response(message: AIMessage) -> bool:
    if getattr(message, "tool_calls", None):
        return False

    content = normalize_message_content(message).strip()
    if not content:
        return False

    # Check leading fenced block first; fallback to whole content.
    envelope_text, _ = _extract_leading_fenced_json_and_trailing(content)
    candidate = envelope_text if envelope_text else content

    # If content is mixed (json + prose), isolate first balanced object.
    balanced = _extract_balanced_object(candidate)
    if balanced:
        candidate = balanced.strip()

    # Shape check only: this is a routing guard, not strict validation.
    lowered = candidate.lower()
    has_name = '"name"' in lowered
    has_args = ('"arguments"' in lowered) or ('"args"' in lowered)
    return has_name and has_args

def looks_like_pseudo_tool_text(content: str) -> bool:
    text = content or ""
    return bool(PSEUDO_TOOL_CALL_PATTERN.search(text) or PSEUDO_JSON_TOOL_CALL_PATTERN.search(text))

def is_pseudo_tool_response(message: AIMessage) -> bool:
    if getattr(message, "tool_calls", None):
        return False
    
    return looks_like_pseudo_tool_text(normalize_message_content(message))

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

def _extract_balanced_object(text: str, start_index: int = 0) -> str | None:
    if not text:
        return None

    try:
        open_index = text.index("{", start_index)
    except ValueError:
        return None

    depth = 0
    in_string = False
    quote_char = ""
    escaped = False

    for index in range(open_index, len(text)):
        char = text[index]

        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if in_string:
            if char == quote_char:
                in_string = False
                quote_char = ""
            continue

        if char in {'"', "'"}:
            in_string = True
            quote_char = char
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_index : index + 1]

    return None

def _as_recovered_tool_call(parsed: object, allowed_tool_names: set[str]) -> dict | None:
    candidate = parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
        tool_calls = parsed.get("tool_calls") or []
        if tool_calls and isinstance(tool_calls[0], dict):
            first_call = tool_calls[0]
            if isinstance(first_call.get("function"), dict):
                function_payload = first_call.get("function") or {}
                candidate = {
                    "name": function_payload.get("name"),
                    "arguments": function_payload.get("arguments", {}),
                }
            else:
                candidate = first_call

    if not isinstance(candidate, dict):
        return None

    name = candidate.get("name")
    if not isinstance(name, str) or name not in allowed_tool_names:
        return None

    arguments = candidate.get("arguments", candidate.get("args", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            return None
    if not isinstance(arguments, dict):
        return None

    return {
        "name": name,
        "args": arguments,
        "id": f"pseudo-{uuid4()}",
        "type": "tool_call",
    }

def _extract_json_field_slice(text: str, field_name: str) -> str | None:
    field_match = re.search(rf'["\']{re.escape(field_name)}["\']\s*:\s*', text, flags=re.IGNORECASE)
    if not field_match:
        return None
    return text[field_match.end() :]

def _scan_relaxed_quoted_value(text: str, quote_char: str) -> tuple[str, int]:
    result: list[str] = []
    escaped = False

    for index, char in enumerate(text):
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\":
            result.append(char)
            escaped = True
            continue
        if char == quote_char:
            trailer = text[index + 1 :]
            # Accept as a closing quote only at likely value boundaries.
            if re.match(r"\s*(?:,\s*[\"\'][a-zA-Z_][a-zA-Z0-9_]*[\"\']\s*:|[}\]])", trailer):
                return "".join(result), index + 1
            result.append(char)
            continue
        result.append(char)

    # No reliable closing quote found; salvage the rest as a partial value.
    return "".join(result), len(text)

def _extract_relaxed_string_field(text: str, field_name: str) -> str | None:
    tail = _extract_json_field_slice(text, field_name)
    if tail is None:
        return None
    stripped = tail.lstrip()
    if not stripped or stripped[0] not in {'"', "'"}:
        return None
    quote_char = stripped[0]
    raw_value, _ = _scan_relaxed_quoted_value(stripped[1:], quote_char)
    try:
        return json.loads(f'"{raw_value}"')
    except Exception:
        return (
            raw_value.replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace('\\"', '"')
            .replace("\\'", "'")
        )

def _extract_relaxed_bool_field(text: str, field_name: str) -> bool | None:
    tail = _extract_json_field_slice(text, field_name)
    if tail is None:
        return None
    lowered = tail.lstrip().lower()
    if lowered.startswith("true"):
        return True
    if lowered.startswith("false"):
        return False
    return None

def _extract_relaxed_json_tool_call(content: str, allowed_tool_names: set[str]) -> dict | None:
    if not content:
        return None

    name = _extract_relaxed_string_field(content, "name")
    if not name or name not in allowed_tool_names:
        return None

    arguments_tail = _extract_json_field_slice(content, "arguments")
    arguments_text = arguments_tail if arguments_tail is not None else content

    if name == "write_file":
        path = _extract_relaxed_string_field(arguments_text, "path")
        content_value = _extract_relaxed_string_field(arguments_text, "content")
        if not path or content_value is None:
            return None
        overwrite = _extract_relaxed_bool_field(arguments_text, "overwrite")
        args: dict[str, object] = {"path": path, "content": content_value}
        if overwrite is not None:
            args["overwrite"] = overwrite
        return {
            "name": name,
            "args": args,
            "id": f"pseudo-{uuid4()}",
            "type": "tool_call",
        }

    path = _extract_relaxed_string_field(arguments_text, "path")
    if path:
        return {
            "name": name,
            "args": {"path": path},
            "id": f"pseudo-{uuid4()}",
            "type": "tool_call",
        }

    return {
        "name": name,
        "args": {},
        "id": f"pseudo-{uuid4()}",
        "type": "tool_call",
    }

def _extract_json_pseudo_tool_call(content: str, allowed_tool_names: set[str]) -> dict | None:
    """Best-effort parse for pseudo tool text shaped like JSON."""
    if not content:
        return None

    candidates: list[str] = []
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    fenced_match = re.search(r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        fenced_candidate = _extract_balanced_object(fenced_match.group(1))
        if fenced_candidate:
            candidates.append(fenced_candidate.strip())

    inline_name_index = content.find('"name"')
    if inline_name_index >= 0:
        inline_candidate = _extract_balanced_object(content, max(0, inline_name_index - 1))
        if inline_candidate:
            candidates.append(inline_candidate.strip())

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

        recovered = _as_recovered_tool_call(parsed, allowed_tool_names)
        if recovered is not None:
            return recovered

    relaxed_recovered = _extract_relaxed_json_tool_call(content, allowed_tool_names)
    if relaxed_recovered is not None:
        return relaxed_recovered

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
        return AIMessage(content="", tool_calls=[recovered_tool_call])
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


# ****************************
#todo: not valid for coder model, start debugging here..

def _normalize_existing_tool_calls(message: AIMessage, allowed_tool_names: set[str]) -> AIMessage | None:
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return None

    normalized: list[dict] = []
    for raw in tool_calls:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not isinstance(name, str) or name not in allowed_tool_names:
            continue
        args = raw.get("args", {})
        if not isinstance(args, dict):
            continue
        normalized.append(
            {
                "name": name,
                "args": args,
                "id": str(raw.get("id") or f"pseudo-{uuid4()}"),
                "type": "tool_call",
            }
        )

    if not normalized:
        return None
    return AIMessage(content="", tool_calls=normalized)

def _extract_leading_fenced_json_and_trailing(content: str) -> tuple[str | None, str]:
    if not content:
        return (None, "")

    # Correct triple-backtick fence.
    match = re.match(
        r'^\s*```(?:json|JSON)?\s*\n?(.*?)\s*```\s*(.*)$',
        content,
        flags=re.DOTALL,
    )
    if not match:
        return (None, "")
    return (match.group(1).strip(), (match.group(2) or "").strip())


def _extract_trailing_content_when_invalid_tool_envelope(
    content: str,
    allowed_tool_names: set[str],
) -> str:
    envelope_text, trailing = _extract_leading_fenced_json_and_trailing(content)
    if not envelope_text or not trailing.strip():
        return ""

    # Reuse existing recovery: if it is a valid recoverable tool call, do not treat as invalid.
    recovered = _extract_json_pseudo_tool_call(envelope_text, allowed_tool_names)
    if recovered is not None:
        return ""

    # Heuristic: envelope-ish but unrecoverable -> salvage trailing prose.
    # This catches {"name": null, "arguments": null} and similar.
    lower_env = envelope_text.lower()
    has_tool_shape = ('"name"' in lower_env) and (('"arguments"' in lower_env) or ('"args"' in lower_env))
    if has_tool_shape:
        return trailing.strip()

    return ""

def recover_action_response(
    message: AIMessage,
    allowed_tool_names: set[str],
) -> AIMessage | None:
    direct = _normalize_existing_tool_calls(message, allowed_tool_names)
    if direct is not None:
        return direct

    recovered = recover_pseudo_tool_response(message, allowed_tool_names)
    if getattr(recovered, "tool_calls", None):
        return recovered

    content = normalize_message_content(message)
    trailing = _extract_trailing_content_when_invalid_tool_envelope(content, allowed_tool_names)
    if trailing:
        return AIMessage(content=trailing)

    return None