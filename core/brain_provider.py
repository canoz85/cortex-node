"""LangChain/Ollama implementation of the framework-neutral Brain provider port."""

import json
import logging
from copy import copy
from functools import partial
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.brain import BrainMessage
from core.brain_normalization import (
    normalize_brain_output, normalize_brain_usage, validate_native_call,
)
from core.protocol.enums import BrainOutcomeKind
from core.protocol.models import BrainInput, BrainOutcome

from core.debug import log_llm_exchange
from core.logging.live_status import add_response_usage, begin_provider_invocation, set_brain_step_context
from tools.registry import ToolRegistry, get_tool_argument_schema
from core.brain_batch_policy import (
    MAX_BATCH_CALLS, batch_call_limit, is_oversized_read_only_batch, validate_read_only_batch,
)


logger = logging.getLogger(__name__)

def _ensure_native_call(
    *,
    llm,
    provider_messages,
    raw,
    authorized_tools: dict[str, object],
    argument_schema_for: Callable[[str], type | None] | None = None,
):
    if argument_schema_for is None:
        argument_schema_for = partial(
            get_tool_argument_schema, registry=ToolRegistry.from_tools(authorized_tools.values()),
        )
    calls = getattr(raw, "tool_calls", None)
    missing_call = not calls and not getattr(raw, "invalid_tool_calls", None)
    valid_batch = _is_valid_native_batch(raw, authorized_tools)
    disallowed_batch = valid_batch and not _is_permitted_native_batch(
        raw, authorized_tools, argument_schema_for=argument_schema_for,
    )
    correction_triggered = missing_call or disallowed_batch
    _log_native_call_attempt(raw, 1, correction_triggered=correction_triggered)
    if not correction_triggered:
        return raw, provider_messages

    if disallowed_batch and _is_oversized_native_batch(
        raw, authorized_tools, argument_schema_for=argument_schema_for,
    ):
        limits = ", ".join(
            f"{name}: {batch_call_limit(name, argument_schema_for=argument_schema_for)}"
            for name in dict.fromkeys(call["name"] for call in calls)
        )
        instruction = (
            f"The previous response contained {len(calls)} calls in an otherwise valid "
            "independent read-only batch, but a call limit was exceeded. "
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
        instruction = (
            "The previous response contained no native call and was not accepted. "
            "Return exactly one native tool call using a currently bound tool, "
            "with its required arguments and empty content."
        )
        rejected_response = [AIMessage(content=raw.content)] if isinstance(raw, AIMessage) else []
    corrected_messages = [
        *provider_messages,
        *rejected_response,
        SystemMessage(content=instruction),
    ]
    begin_provider_invocation(worker="brain")
    corrected = llm.invoke(corrected_messages)
    add_response_usage(corrected, worker="brain")

    return corrected, corrected_messages


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
        validate_read_only_batch(
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
    return is_oversized_read_only_batch(
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
            name, arguments, _ = validate_native_call(call, allowed_tools)
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


def _canonical_brain_response(raw, authorized_tools: dict[str, object]):
    """Discard incidental text beside valid native candidates; preserve the raw log.

    This validates candidates, not batch eligibility. Classification still checks
    the complete group against the existing execution safety policy.
    """
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
) -> None:
    calls = getattr(raw, "tool_calls", None)
    names = [
        call["name"] for call in calls
        if isinstance(call, dict) and isinstance(call.get("name"), str)
    ] if isinstance(calls, (list, tuple)) else []
    logger.debug(
        "Brain native-call compliance: attempt=%s call_count=%s "
        "tool_call_names=%s correction_triggered=%s correction_exhausted=%s",
        attempt, len(calls) if isinstance(calls, (list, tuple)) else 0, names,
        correction_triggered, correction_exhausted,
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
                    "exact_collection_source_record_index": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Optional nonnegative index of the displayed eligible evidence record in current_attempts. "
                            "Provide together with exact_collection_data_path only when that path resolves directly "
                            "to a structured array/list already present in tool result data. "
                            "Do not use for stdout, rendered text, prose, or a list inferred/extracted from text. "
                            "If the semantic result is expressed in the completion message but no structured collection exists, "
                            "omit all exact_collection arguments. Controller binds members from the original tool data."
                        ),
                    },
                    "exact_collection_data_path": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                        "description": (
                            "Path of string keys or integer indexes resolving directly to an existing structured array/list; "
                            "[] selects the root. Required together with exact_collection_source_record_index."
                        ),
                    },
                    "exact_collection_label": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Optional nonempty collection label; use only with both exact_collection reference arguments.",
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
            raw, retry_messages = _ensure_native_call(
                llm=llm, provider_messages=provider_messages,
                raw=_canonical_brain_response(
                    raw, authorized_by_name,
                ),
                authorized_tools=authorized_by_name,
                argument_schema_for=argument_schema_for,
            )
            if retry_messages is not provider_messages:
                log_llm_exchange(
                    worker="brain", operation="step", messages=retry_messages,
                    response=raw, execution_id=brain_input.identity.execution_id,
                    enabled=self.show_raw_llm,
                )
            canonical = _canonical_brain_response(
                raw, authorized_by_name,
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
        if retry_messages is not provider_messages:
            _log_native_call_attempt(
                raw, 2, correction_exhausted=outcome.outcome == BrainOutcomeKind.INVALID_OUTPUT,
            )
        return outcome.model_copy(update={"usage": normalize_brain_usage(raw)})
