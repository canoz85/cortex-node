"""LangChain/Ollama implementation of the framework-neutral Brain provider port."""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.brain import BrainMessage
from core.brain_normalization import (
    normalize_brain_output, normalize_brain_usage, validate_native_call,
)
from core.protocol.enums import BrainOutcomeKind
from core.protocol.models import BrainInput, BrainOutcome

from core.debug import log_llm_exchange
from core.logging.live_status import add_response_usage, begin_provider_invocation


logger = logging.getLogger(__name__)

def _ensure_native_call(
    *,
    llm,
    provider_messages,
    raw,
    authorized_tools: dict[str, object],
):
    calls = getattr(raw, "tool_calls", None)
    missing_call = not calls and not getattr(raw, "invalid_tool_calls", None)
    valid_batch = _is_valid_native_batch(raw, authorized_tools)
    correction_triggered = missing_call or valid_batch
    _log_native_call_attempt(raw, 1, correction_triggered=correction_triggered)
    if not correction_triggered:
        return raw, provider_messages

    if valid_batch:
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
    try:
        allowed_tools = set(authorized_tools)
        for call in calls:
            name, arguments, _ = validate_native_call(call, allowed_tools)
            if call.get("type", "tool_call") != "tool_call":
                return False
            if "id" in call and (not isinstance(call["id"], str) or not call["id"].strip()):
                return False
            json.dumps(arguments, allow_nan=False)
            if name in authorized_tools:
                tool = authorized_tools[name]
                schema = getattr(tool, "args_schema", None)
                if isinstance(schema, dict):
                    # Do not correct a batch whose argument validity is unknown.
                    return False
                if schema is None and callable(getattr(tool, "get_input_schema", None)):
                    schema = tool.get_input_schema()
                if schema is not None:
                    if callable(getattr(schema, "model_validate", None)):
                        schema.model_validate(arguments)
                    else:
                        schema.parse_obj(arguments)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return False
    return True


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
    executable_tools: dict[str, object],
) -> tuple[set[str], list]:
    authorized_names = set(
        brain_input.active_plan.available_tools or ()
        if brain_input.active_plan is not None
        else ()
    )

    authorized_tools = [
        tool
        for name, tool in executable_tools.items()
        if name in authorized_names
    ]

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


class LangChainBrainProvider:
    def __init__(
        self, *, brain_llm, executable_tools,
        show_raw_llm: bool = False,
    ):
        self.brain_llm = brain_llm
        self.executable_tools = {
            tool.name: tool for tool in executable_tools
            if isinstance(getattr(tool, "name", None), str) and tool.name
        }
        self.show_raw_llm = show_raw_llm

    def generate(
        self, brain_input: BrainInput, messages: tuple[BrainMessage, ...],
    ) -> BrainOutcome:

        authorized_tool_names, authorized_tools = _resolve_authorized_tools(
            brain_input, self.executable_tools
        )

        provider_messages = _to_provider_messages(messages)
        try:
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
                llm=llm, provider_messages=provider_messages, raw=raw,
                authorized_tools={tool.name: tool for tool in authorized_tools},
            )
            if retry_messages is not provider_messages:
                log_llm_exchange(
                    worker="brain", operation="step", messages=retry_messages,
                    response=raw, execution_id=brain_input.identity.execution_id,
                    enabled=self.show_raw_llm,
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
            raw, brain_input, set(self.executable_tools) & authorized_tool_names,
        )
        if retry_messages is not provider_messages:
            _log_native_call_attempt(
                raw, 2, correction_exhausted=outcome.outcome == BrainOutcomeKind.INVALID_OUTPUT,
            )
        return outcome.model_copy(update={"usage": normalize_brain_usage(raw)})
