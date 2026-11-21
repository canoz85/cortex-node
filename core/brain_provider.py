"""LangChain/Ollama implementation of the framework-neutral Brain provider port."""

import json
import logging
import re
from copy import copy
from functools import partial
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError
from pydantic.v1 import ValidationError as LegacyValidationError

from core.brain import BrainMessage
from core.brain_normalization import (
    normalize_brain_output, normalize_brain_usage, validate_native_call,
)
from core.protocol.enums import BrainOutcomeKind
from core.protocol.models import BrainInput, BrainOutcome
from core.tool_call_guard import DuplicateSuccessfulCall, reject_duplicate_successful_calls

from core.debug import log_llm_exchange
from core.logging.live_status import add_response_usage, begin_provider_invocation, set_brain_step_context
from tools.registry import ToolRegistry, get_tool_argument_schema
from core.brain_batch_policy import (
    MAX_BATCH_CALLS, batch_call_limit, is_oversized_tool_batch, validate_tool_batch,
)


logger = logging.getLogger(__name__)

_MAX_ARGUMENT_ISSUES = 8
_MAX_ARGUMENT_FEEDBACK_CHARS = 4000
_MAX_NATIVE_CORRECTIONS = 2
_MAX_TEXTUAL_LIFECYCLE_CHARS = 16_384


def _native_argument_feedback(raw, authorized_tools, *, argument_schema_for):
    """Diagnose executable arguments only, without creating runtime requests.

    Malformed/unauthorized calls and invalid lifecycle actions retain their
    existing strict normalization path. Never recover invalid_tool_calls.
    """
    calls = getattr(raw, "tool_calls", None)
    if (not isinstance(calls, (list, tuple)) or not calls
            or getattr(raw, "invalid_tool_calls", None)
            or not _are_valid_native_calls(calls, authorized_tools, validate_arguments=False)):
        return None
    feedback = {"code": "invalid_native_tool_arguments", "issues": [], "issues_truncated": False}
    for index, call in enumerate(calls):
        if call["name"] not in authorized_tools:
            continue
        schema = argument_schema_for(call["name"])
        if schema is None or isinstance(schema, dict):
            continue  # No new validator or fallback for unavailable executable schemas.
        validate = getattr(schema, "model_validate", None)
        if not callable(validate):
            validate = getattr(schema, "parse_obj", None)
        if not callable(validate):
            continue
        try:
            validate(call["args"])
        except (ValidationError, LegacyValidationError) as exc:
            errors = (exc.errors(include_input=False, include_url=False, include_context=False)
                      if isinstance(exc, ValidationError) else exc.errors())
        except (ValueError, TypeError):
            # Equivalent validators may not expose structured diagnostics.
            errors = [{"loc": (), "type": "schema_validation_failed",
                       "msg": "Arguments must match the executable schema."}]
        else:
            continue
        for error in errors:
            issue = {
                "tool": call["name"][:96], "call_index": index,
                "field": [part if isinstance(part, int) else str(part)[:32]
                          for part in error.get("loc", ())[:8]],
                "type": str(error.get("type", "schema_validation_failed"))[:64],
                "reason": ("Arguments must match the executable schema."
                           if error.get("type") in {"value_error", "assertion_error"}
                           else " ".join(str(error.get("msg", "Invalid argument.")).split())[:160]),
            }
            candidate = {**feedback, "issues": [*feedback["issues"], issue]}
            if (len(feedback["issues"]) == _MAX_ARGUMENT_ISSUES
                    or len(json.dumps(candidate, ensure_ascii=True)) > _MAX_ARGUMENT_FEEDBACK_CHARS):
                feedback["issues_truncated"] = True
                return feedback
            feedback["issues"].append(issue)
    return feedback if feedback["issues"] else None

def _ensure_native_call(
    *,
    llm,
    provider_messages,
    raw,
    authorized_tools: dict[str, object],
    argument_schema_for: Callable[[str], type | None] | None = None,
    brain_input: BrainInput | None = None,
    show_raw_llm: bool = False,
):
    if argument_schema_for is None:
        argument_schema_for = partial(
            get_tool_argument_schema, registry=ToolRegistry.from_tools(authorized_tools.values()),
        )
    # The initial response is already available; at most two further invocations.
    for attempt in range(1, _MAX_NATIVE_CORRECTIONS + 2):
        raw = _canonical_brain_response(raw, authorized_tools)
        calls = getattr(raw, "tool_calls", None)
        missing_call = not calls and not getattr(raw, "invalid_tool_calls", None)
        argument_feedback = _native_argument_feedback(
            raw, authorized_tools, argument_schema_for=argument_schema_for,
        )
        valid_batch = _is_valid_native_batch(raw, authorized_tools)
        disallowed_batch = valid_batch and not _is_permitted_native_batch(
            raw, authorized_tools, argument_schema_for=argument_schema_for,
        )
        duplicate_feedback = None
        if (brain_input is not None and calls and not disallowed_batch
                and not getattr(raw, "content", None) and not getattr(raw, "invalid_tool_calls", None)
                and _are_valid_native_calls(calls, authorized_tools, allow_unknown_schema=True)):
            try:
                reject_duplicate_successful_calls(
                    [(call["name"], call["args"]) for call in calls if call["name"] in authorized_tools],
                    brain_input, argument_schema_for=argument_schema_for,
                )
            except DuplicateSuccessfulCall as exc:
                duplicate_feedback = str(exc)
        correction_reason = (
            "invalid_native_tool_arguments" if argument_feedback is not None else
            "duplicate_successful_call" if duplicate_feedback is not None else
            "unsupported_native_batch" if disallowed_batch else
            "missing_native_call" if missing_call else None
        )
        if correction_reason is None or attempt > _MAX_NATIVE_CORRECTIONS:
            return raw, attempt, correction_reason
        bare_action = getattr(raw, "content", None)
        if missing_call and isinstance(bare_action, str) and bare_action in _LIFECYCLE_PARAMETERS:
            parameters = _LIFECYCLE_PARAMETERS[bare_action]
            required = parameters["required"]
            descriptions = "\n".join(
                f"- {field}: {parameters['properties'][field]['description']}" for field in required
            )
            correction_reason = "bare_lifecycle_repair"
            instruction = (
                f"You returned the bare lifecycle name {bare_action}. "
                f"Missing required arguments: {', '.join(required)}.\n{descriptions}\n"
                f"Invoke {bare_action} natively exactly once with those arguments. "
                "Leave assistant content empty. Do not reconsider the task or choose another action."
            )
            rejected_response = [AIMessage(content=bare_action)]
        elif argument_feedback is not None:
            instruction = (
                "The previous native action was rejected; no tool was executed.\n"
                + json.dumps(argument_feedback, ensure_ascii=True, separators=(",", ":"))
                + "\nRepair the arguments using the currently bound schemas, or choose another bound native action. "
                "Return a permitted native action with empty content. "
                "The entire response must validate before any member can execute."
            )
            rejected_response = [raw] if isinstance(raw, AIMessage) else []
        elif duplicate_feedback is not None:
            instruction = duplicate_feedback + " Return a currently bound native action with empty content."
            rejected_response = [raw] if isinstance(raw, AIMessage) else []
        elif disallowed_batch and _is_oversized_native_batch(
            raw, authorized_tools, argument_schema_for=argument_schema_for,
        ):
            limits = ", ".join(
                f"{name}: {batch_call_limit(name, argument_schema_for=argument_schema_for)}"
                for name in dict.fromkeys(call["name"] for call in calls)
            )
            instruction = (
                f"The previous response contained {len(calls)} calls in an otherwise valid "
                "independent tool batch, but a call limit was exceeded. "
                f"Return one native action containing at most {MAX_BATCH_CALLS} of those useful calls. "
                f"Per-tool call limits: {limits}. Do not add new calls. Leave content empty."
            )
            rejected_response = [raw] if isinstance(raw, AIMessage) else []
        elif disallowed_batch:
            instruction = (
                "The previous response contained multiple native calls and was not accepted. "
                "Choose exactly one next action from the useful candidates for this turn. "
                "Return exactly one native call using a currently bound tool. "
                "Do not batch or parallelize calls. "
                "Leave content empty."
            )
            rejected_response = [raw] if isinstance(raw, AIMessage) else []
        else:
            intended = _recognizable_intended_action(raw, authorized_tools)
            if intended is not None:
                instruction = (
                    f"Serialization repair only for the previous intended action {intended}. "
                    "Repair only the previous response into native-call format. Preserve the same "
                    "intended action and arguments exactly. Do not reconsider the task or choose "
                    "another action. Do not write the action as assistant text. Do not output "
                    "textual function-call syntax such as action(...). Invoke the already-bound intended "
                    "action natively so it is returned as a native tool call. Leave assistant "
                    "content empty. Do not continue execution."
                )
            else:
                instruction = (
                    "The previous response contained no native call and was not accepted. "
                    "Return exactly one native action: either one authorized executable tool call, "
                    "or one lifecycle action: brain_step_completed, brain_step_failed, or "
                    "brain_replan_requested. Supply exactly the arguments required by its bound schema. "
                    "Leave assistant content empty."
                )
            rejected_response = [AIMessage(content=raw.content)] if isinstance(raw, AIMessage) else []
        _log_native_call_attempt(raw, attempt, correction_triggered=True,
                                 correction_reason=correction_reason, accepted=False)
        provider_messages = [
            *provider_messages,
            *rejected_response,
            SystemMessage(content=instruction),
        ]
        begin_provider_invocation(worker="brain")
        raw = llm.invoke(provider_messages)
        add_response_usage(raw, worker="brain")
        log_llm_exchange(
            worker="brain", operation="step", messages=provider_messages, response=raw,
            execution_id=brain_input.identity.execution_id if brain_input is not None else None,
            enabled=show_raw_llm,
        )
        if correction_reason == "bare_lifecycle_repair":
            # Inspect the native reply before any textual or incidental-content recovery.
            calls = getattr(raw, "tool_calls", None)
            if calls and (not isinstance(calls, (list, tuple)) or len(calls) != 1
                          or not isinstance(calls[0], dict) or calls[0].get("name") != bare_action):
                correction_reason = "lifecycle_repair_action_mismatch"
            return raw, attempt + 1, correction_reason


def _is_valid_native_batch(raw, authorized_tools: dict[str, object]) -> bool:
    calls = getattr(raw, "tool_calls", None)
    if (
        not isinstance(calls, (list, tuple)) or len(calls) <= 1
        or getattr(raw, "invalid_tool_calls", None)
        or getattr(raw, "content", None)
    ):
        return False
    if not _are_valid_native_calls(calls, authorized_tools):
        return False
    try:
        for call in calls:
            schema = get_tool_argument_schema(call["name"])
            if schema is not None:
                schema.model_validate(call["args"])
    except (ValueError, TypeError, KeyError):
        return False
    return True


def _is_permitted_native_batch(
    raw, authorized_tools: dict[str, object],
    *, argument_schema_for: Callable[[str], type | None] | None = None,
) -> bool:
    """Execution eligibility is stricter than valid correction candidates."""
    if not _is_valid_native_batch(raw, authorized_tools):
        return False
    if argument_schema_for is None:
        argument_schema_for = partial(
            get_tool_argument_schema, registry=ToolRegistry.from_tools(authorized_tools.values()),
        )
    try:
        validate_tool_batch(
            [(call["name"], call["args"]) for call in raw.tool_calls], authorized_tools,
            argument_schema_for=argument_schema_for,
        )
    except (ValueError, TypeError, KeyError):
        return False
    return True


def _is_oversized_native_batch(
    raw, authorized_tools: dict[str, object],
    *, argument_schema_for: Callable[[str], type | None] | None = None,
) -> bool:
    if not _is_valid_native_batch(raw, authorized_tools):
        return False
    if argument_schema_for is None:
        argument_schema_for = partial(
            get_tool_argument_schema, registry=ToolRegistry.from_tools(authorized_tools.values()),
        )
    return is_oversized_tool_batch(
        [(call["name"], call["args"]) for call in raw.tool_calls], authorized_tools,
        argument_schema_for=argument_schema_for,
    )


def _are_valid_native_calls(
    calls, authorized_tools: dict[str, object], *, allow_unknown_schema=False,
    validate_arguments=True,
) -> bool:
    try:
        allowed_tools = set(authorized_tools)
        registry = ToolRegistry.from_tools(authorized_tools.values()) if validate_arguments else None
        for call in calls:
            name, arguments = validate_native_call(call, allowed_tools)
            if call.get("type", "tool_call") != "tool_call":
                return False
            if "id" in call and (not isinstance(call["id"], str) or not call["id"].strip()):
                return False
            json.dumps(arguments, allow_nan=False)
            if name in authorized_tools and validate_arguments:
                schema = get_tool_argument_schema(name, registry=registry)
                if isinstance(schema, dict):
                    # Do not correct a batch whose argument validity is unknown.
                    if not allow_unknown_schema:
                        return False
                    continue
                if schema is not None:
                    if callable(getattr(schema, "model_validate", None)):
                        schema.model_validate(arguments)
                    else:
                        schema.parse_obj(arguments)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return False
    return True


def _recover_textual_call(raw, authorized_tools: dict[str, object]):
    """Adapt one exact textual bound action; leave validation to the native path."""
    content = getattr(raw, "content", None)
    if (getattr(raw, "tool_calls", None) or getattr(raw, "invalid_tool_calls", None)
            or not isinstance(content, str) or not content):
        return raw
    colon_lifecycle = _parse_colon_lifecycle_call(content)
    if colon_lifecycle is not None:
        name, arguments = colon_lifecycle
        return _make_textual_call(raw, name, arguments)
    lifecycle = _parse_textual_lifecycle_call(content)
    if lifecycle is not None:
        name, arguments = lifecycle
        return _make_textual_call(raw, name, arguments)
    lifecycle = _parse_unquoted_key_lifecycle_call(content)
    if lifecycle is not None:
        name, arguments = lifecycle
        return _make_textual_call(raw, name, arguments)
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*\})", content, re.DOTALL)
    if match is None:
        return raw
    name, encoded = match.groups()
    bound_names = set(authorized_tools) | {
        schema["function"]["name"] for schema in LIFECYCLE_ACTION_SCHEMAS
    }
    if name not in bound_names:
        return raw

    def reject_constant(value):
        raise ValueError("Non-JSON constant")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    try:
        arguments = json.loads(encoded, parse_constant=reject_constant, object_pairs_hook=unique_object)
        # Reject overflow to non-finite floats as well as NaN/Infinity tokens.
        json.dumps(arguments, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        return raw
    return _make_textual_call(raw, name, arguments)


def _parse_colon_lifecycle_call(content: str) -> tuple[str, dict] | None:
    """Recover exactly one plain colon-style lifecycle outcome."""
    if len(content) > _MAX_TEXTUAL_LIFECYCLE_CHARS:
        return None
    match = re.fullmatch(
        r"\s*(brain_step_completed|brain_step_failed|brain_replan_requested)\s*:\s*(.*?)\s*",
        content, re.DOTALL,
    )
    if match is None:
        return None
    name, message = match.groups()
    message = message.strip()
    if not message:
        return None
    if re.search(
        r"(?:^|\n)\s*brain_(?:step_completed|step_failed|replan_requested)\s*:",
        message,
    ):
        return None
    if name == "brain_replan_requested":
        arguments = {"reason": message, "constraints": []}
    else:
        arguments = {"message": message}
    return name, arguments


def _recognizable_intended_action(raw, authorized_tools: dict[str, object]) -> str | None:
    """Identify a leading action label without interpreting prose as an action."""
    if getattr(raw, "invalid_tool_calls", None):
        return None
    content = getattr(raw, "content", None)
    if not isinstance(content, str):
        return None
    names = set(authorized_tools) | {
        schema["function"]["name"] for schema in LIFECYCLE_ACTION_SCHEMAS
    }
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?=[:({{\s])", content)
    if match is None or match.group(1) not in names:
        return None
    return match.group(1)


def _make_textual_call(raw, name: str, arguments: dict):
    canonical = copy(raw)
    canonical.content = ""
    canonical.tool_calls = [{"name": name, "args": arguments,
                             "id": "textual-recovery", "type": "tool_call"}]
    logger.debug("Brain textual-call recovery: recovered_from_text=True tool_call_name=%s", name)
    return canonical


def _parse_unquoted_key_lifecycle_call(content: str) -> tuple[str, dict] | None:
    """Parse one lifecycle action with bare keys and JSON values.

    Single bare identifiers are also accepted as string values. Schema and
    lifecycle argument validation remain the native normalization path's job.
    """
    text = content.strip()
    if len(text) > _MAX_TEXTUAL_LIFECYCLE_CHARS:
        return None
    head = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\{", text)
    names = {schema["function"]["name"] for schema in LIFECYCLE_ACTION_SCHEMAS}
    if head is None or head.group(1) not in names:
        return None

    def reject_constant(_value):
        raise ValueError("Non-JSON constant")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    decoder = json.JSONDecoder(parse_constant=reject_constant, object_pairs_hook=unique_object)
    index = head.end()
    arguments = {}

    def skip_space(position):
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    try:
        index = skip_space(index)
        if index < len(text) and text[index] == "}":
            index += 1
        else:
            while True:
                key = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[index:])
                if key is None or key.group() in arguments:
                    return None
                name = key.group()
                index = skip_space(index + key.end())
                if index >= len(text) or text[index] != ":":
                    return None
                index = skip_space(index + 1)
                try:
                    value, index = decoder.raw_decode(text, index)
                except json.JSONDecodeError:
                    bare = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[index:])
                    if bare is None or bare.group() in {"NaN", "Infinity"}:
                        return None
                    value = bare.group()
                    index += bare.end()
                arguments[name] = value
                index = skip_space(index)
                if index < len(text) and text[index] == "}":
                    index += 1
                    break
                if index >= len(text) or text[index] != ",":
                    return None
                index = skip_space(index + 1)
        if skip_space(index) != len(text):
            return None
        json.dumps(arguments, allow_nan=False)
        return head.group(1), arguments
    except (ValueError, TypeError, RecursionError):
        return None


def _parse_textual_lifecycle_call(content: str) -> tuple[str, dict] | None:
    """Parse one strict Python-style or Gemma-style lifecycle serialization."""
    text = content.strip()
    if len(text) > _MAX_TEXTUAL_LIFECYCLE_CHARS:
        return None
    head = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*([({])", text)
    if head is None:
        return None
    name = head.group(1)
    opening = head.group(2)
    schema = _LIFECYCLE_PARAMETERS.get(name)
    if schema is None:
        return None

    if opening == "{":
        return _parse_gemma_lifecycle_arguments(text, head.end(), name, schema)

    index = head.end()
    arguments = {}

    def skip_space(position):
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    def parse_string(position):
        if position >= len(text) or text[position] not in "\"'":
            raise ValueError("Expected a quoted string literal")
        quote = text[position]
        index = position + 1
        value = []
        escapes = {"\\": "\\", "n": "\n", "r": "\r", "t": "\t", "'": "'", '"': '"'}
        while index < len(text):
            char = text[index]
            if char == quote:
                return "".join(value), index + 1
            if char == "\\":
                index += 1
                if index >= len(text) or text[index] not in escapes:
                    raise ValueError("Invalid string escape")
                value.append(escapes[text[index]])
            else:
                value.append(char)
            index += 1
        raise ValueError("Unclosed string literal")

    try:
        index = skip_space(index)
        if index < len(text) and text[index] == ")":
            index += 1
        else:
            while True:
                match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", text[index:])
                if match is None:
                    return None
                argument_name = match.group(1)
                index += match.end()
                if argument_name not in schema["properties"] or argument_name in arguments:
                    return None
                index = skip_space(index)
                if index >= len(text) or text[index] != "=":
                    return None
                index = skip_space(index + 1)

                if schema["properties"][argument_name]["type"] == "array":
                    if index >= len(text) or text[index] != "[":
                        return None
                    index = skip_space(index + 1)
                    values = []
                    if index < len(text) and text[index] == "]":
                        index += 1
                    else:
                        while True:
                            value, index = parse_string(index)
                            values.append(value)
                            if len(values) > 64:
                                return None
                            index = skip_space(index)
                            if index < len(text) and text[index] == "]":
                                index += 1
                                break
                            if index >= len(text) or text[index] != ",":
                                return None
                            index = skip_space(index + 1)
                    arguments[argument_name] = values
                else:
                    value, index = parse_string(index)
                    arguments[argument_name] = value

                index = skip_space(index)
                if index < len(text) and text[index] == ")":
                    index += 1
                    break
                if index >= len(text) or text[index] != ",":
                    return None
                index = skip_space(index + 1)

        index = skip_space(index)
        if index != len(text):
            return None
        if set(arguments) != set(schema["required"]):
            return None
        return name, arguments
    except (ValueError, TypeError, RecursionError):
        return None


def _parse_gemma_lifecycle_arguments(text: str, index: int, name: str, schema: dict):
    """Parse Gemma fields as delimited strings and schema-declared string lists."""
    delimiter = '<|"|>'
    arguments = {}

    def skip_space(position):
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    def parse_gemma_string(position):
        if not text.startswith(delimiter, position):
            raise ValueError("Expected Gemma string delimiter")
        start = position + len(delimiter)
        end = text.find(delimiter, start)
        if end < 0:
            raise ValueError("Unclosed Gemma string delimiter")
        return text[start:end], end + len(delimiter)

    try:
        index = skip_space(index)
        if index < len(text) and text[index] == "}":
            index += 1
        else:
            while True:
                match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", text[index:])
                if match is None:
                    return None
                argument_name = match.group(1)
                index += match.end()
                if argument_name not in schema["properties"] or argument_name in arguments:
                    return None
                index = skip_space(index)
                if index >= len(text) or text[index] != ":":
                    return None
                index = skip_space(index + 1)

                if schema["properties"][argument_name]["type"] == "array":
                    if index >= len(text) or text[index] != "[":
                        return None
                    index = skip_space(index + 1)
                    values = []
                    if index < len(text) and text[index] == "]":
                        index += 1
                    else:
                        while True:
                            value, index = parse_gemma_string(index)
                            values.append(value)
                            if len(values) > 64:
                                return None
                            index = skip_space(index)
                            if index < len(text) and text[index] == "]":
                                index += 1
                                break
                            if index >= len(text) or text[index] != ",":
                                return None
                            index = skip_space(index + 1)
                    arguments[argument_name] = values
                else:
                    value, index = parse_gemma_string(index)
                    arguments[argument_name] = value

                index = skip_space(index)
                if index < len(text) and text[index] == "}":
                    index += 1
                    break
                if index >= len(text) or text[index] != ",":
                    return None
                index = skip_space(index + 1)

        index = skip_space(index)
        if index != len(text) or set(arguments) != set(schema["required"]):
            return None
        return name, arguments
    except (ValueError, TypeError, RecursionError):
        return None


def _canonical_brain_response(raw, authorized_tools: dict[str, object]):
    """Adapt strict textual actions or discard incidental text beside native calls.

    Preserve the raw log. Candidates still pass the existing argument validation,
    normalization, and execution safety policy.
    """
    if not getattr(raw, "tool_calls", None):
        return _recover_textual_call(raw, authorized_tools)
    calls = getattr(raw, "tool_calls", None)
    if (
        not getattr(raw, "content", None)
        or not isinstance(calls, (list, tuple)) or not calls
        or getattr(raw, "invalid_tool_calls", None)
        or not _are_valid_native_calls(
            calls, authorized_tools, validate_arguments=len(calls) == 1,
        )
    ):
        return raw
    canonical = copy(raw)
    canonical.content = ""
    return canonical


def _log_native_call_attempt(
    raw, attempt: int, *, correction_triggered: bool = False,
    correction_exhausted: bool = False,
    correction_reason: str | None = None,
    accepted: bool = False,
) -> None:
    calls = getattr(raw, "tool_calls", None)
    names = [
        call["name"] for call in calls
        if isinstance(call, dict) and isinstance(call.get("name"), str)
    ] if isinstance(calls, (list, tuple)) else []
    logger.debug(
        "Brain native-call compliance: attempt=%s call_count=%s "
        "tool_call_names=%s correction_triggered=%s correction_exhausted=%s correction_reason=%s accepted=%s",
        attempt, len(calls) if isinstance(calls, (list, tuple)) else 0, names,
        correction_triggered, correction_exhausted, correction_reason, accepted,
    )

LIFECYCLE_ACTION_SCHEMAS = (
    {
        "type": "function",
        "function": {
            "name": "brain_step_completed",
            "description": "Report that the active execution step is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "minLength": 1,
                        "description": "The evidence-grounded semantic result: include the requested finding, summary, interpretation, comparison or calculation, not merely a completion notice.",
                    },
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "brain_step_failed",
            "description": "Report that the active step cannot be achieved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "minLength": 1, "description": "Evidence-grounded reason the objective is unachievable."},
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "brain_replan_requested",
            "description": "Request a revised plan from Controller.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "minLength": 1, "description": "Evidence-grounded reason the current strategy or plan must change."},
                    "constraints": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Constraints the replacement plan must respect.",
                    },
                },
                "required": ["reason", "constraints"],
                "additionalProperties": False,
            },
        },
    },
)

_LIFECYCLE_PARAMETERS = {
    schema["function"]["name"]: schema["function"]["parameters"] for schema in LIFECYCLE_ACTION_SCHEMAS
}


def _resolve_authorized_tools(
    brain_input: BrainInput,
    registry: ToolRegistry,
) -> tuple[set[str], list]:
    authorized_names = set(
        brain_input.active_plan.available_tools or ()
        if brain_input.active_plan is not None
        else ()
    )

    authorized_tools = registry.resolve(authorized_names & registry.names)

    return authorized_names, authorized_tools


def _to_provider_messages(
    messages: tuple[BrainMessage, ...],
) -> list:
    provider_messages = []

    for message in messages:
        if message.role == "human":
            provider_messages.append(HumanMessage(content=message.content))
        elif message.role == "system":
            provider_messages.append(SystemMessage(content=message.content))
        else:
            raise ValueError(f"Unsupported Brain message role: {message.role}")

    return provider_messages


def _batch_guidance(authorized_tools, *, argument_schema_for: Callable[[str], type | None]) -> str:
    lines = []
    for name in authorized_tools:
        try:
            limit = batch_call_limit(name, argument_schema_for=argument_schema_for)
        except (ValueError, TypeError, KeyError):
            continue
        lines.append(f"- {name}: up to {limit}")
    if lines:
        lines.insert(0, f"Independent multiple calls allowed (up to {MAX_BATCH_CALLS} total; tools may be mixed):")
    lines.append("All other tools and lifecycle actions require one call." if lines
                 else "All tools and lifecycle actions require one call.")
    return "\n".join(lines)


class LangChainBrainProvider:
    def __init__(
        self, *, brain_llm, executable_tools,
        show_raw_llm: bool = False,
    ):
        self.brain_llm = brain_llm
        self.tool_registry = ToolRegistry.from_tools(executable_tools)
        self.executable_tools = self.tool_registry.by_name
        self.show_raw_llm = show_raw_llm

    def generate(
        self, brain_input: BrainInput, messages: tuple[BrainMessage, ...],
    ) -> BrainOutcome:

        set_brain_step_context(
            execution_id=brain_input.identity.execution_id, plan=brain_input.active_plan,
            step_id=brain_input.active_step.step_id if brain_input.active_step is not None else None,
        )
        authorized_tool_names, authorized_tools = _resolve_authorized_tools(
            brain_input, self.tool_registry
        )
        authorized_by_name = {tool.name: tool for tool in authorized_tools}
        argument_schema_for = partial(get_tool_argument_schema, registry=self.tool_registry)

        provider_messages = _to_provider_messages(messages)
        try:
            # Keep the service's native contract immediately before the step input.
            provider_messages.insert(-2, SystemMessage(content=_batch_guidance(
                authorized_by_name, argument_schema_for=argument_schema_for,
            )))
            llm = self.brain_llm.bind_tools([*authorized_tools, *LIFECYCLE_ACTION_SCHEMAS])
            begin_provider_invocation(worker="brain")
            raw = llm.invoke(provider_messages)
            add_response_usage(raw, worker="brain")
            log_llm_exchange(
                worker="brain", operation="step", messages=provider_messages,
                response=raw, execution_id=brain_input.identity.execution_id,
                enabled=self.show_raw_llm,
            )
            canonical, attempt, correction_reason = _ensure_native_call(
                llm=llm, provider_messages=provider_messages,
                raw=raw,
                authorized_tools=authorized_by_name,
                argument_schema_for=argument_schema_for,
                brain_input=brain_input,
                show_raw_llm=self.show_raw_llm,
            )
        except Exception as exc:
            # Provider failures are values at the service boundary.
            # Exception retries remain a Controller decision.
            return BrainOutcome(
                outcome=BrainOutcomeKind.PROVIDER_FAILURE,
                step_id=brain_input.active_step.step_id if brain_input.active_step else None,
                error_code=type(exc).__name__,
                message=f"Brain provider failed ({type(exc).__name__}).",
            )

        if correction_reason == "lifecycle_repair_action_mismatch":
            outcome = BrainOutcome(
                outcome=BrainOutcomeKind.INVALID_OUTPUT,
                step_id=brain_input.active_step.step_id if brain_input.active_step else None,
                error_code=correction_reason,
                message="Lifecycle repair must invoke the originally named action exactly once.",
            )
        elif _native_argument_feedback(canonical, authorized_by_name, argument_schema_for=argument_schema_for):
            # Reject invalid arguments after the bounded correction loop before
            # normalization can create ToolRequest identities.
            outcome = BrainOutcome(
                outcome=BrainOutcomeKind.INVALID_OUTPUT,
                step_id=brain_input.active_step.step_id if brain_input.active_step else None,
                error_code="invalid_native_tool_arguments",
                message="Brain returned invalid native tool arguments.",
            )
        else:
            outcome = normalize_brain_output(
                canonical, brain_input, set(self.executable_tools) & authorized_tool_names,
                argument_schema_for=argument_schema_for,
            )
        # A proposal must also satisfy the bound tool schemas. Normalization's
        # batch policy cannot override a provider schema rejection.
        calls = getattr(canonical, "tool_calls", None)
        if (outcome.outcome == BrainOutcomeKind.TOOL_REQUESTED
                and not _are_valid_native_calls(
                    calls, authorized_by_name,
                    allow_unknown_schema=outcome.tool_requests is None,
                )):
            outcome = BrainOutcome(
                outcome=BrainOutcomeKind.INVALID_OUTPUT, step_id=outcome.step_id,
                error_code="invalid_native_tool_arguments",
                message="Brain returned invalid native tool arguments.",
            )
        repair_finished = correction_reason in {"bare_lifecycle_repair", "lifecycle_repair_action_mismatch"}
        _log_native_call_attempt(
            canonical, attempt, correction_exhausted=(
                outcome.outcome == BrainOutcomeKind.INVALID_OUTPUT
                and (attempt > _MAX_NATIVE_CORRECTIONS or repair_finished)
            ),
            correction_reason=correction_reason,
            accepted=outcome.outcome != BrainOutcomeKind.INVALID_OUTPUT,
        )
        return outcome.model_copy(update={"usage": normalize_brain_usage(canonical)})
