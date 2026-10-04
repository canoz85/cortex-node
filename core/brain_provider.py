"""LangChain/Ollama implementation of the framework-neutral Brain provider port."""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.brain import BrainMessage
from core.brain_normalization import normalize_brain_output, normalize_brain_usage
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
):
    _log_native_call_attempt(raw, 1)

    if getattr(raw, "tool_calls", None) or getattr(raw, "invalid_tool_calls", None):
        return raw, provider_messages

    corrected_messages = [
        *provider_messages,
        *([AIMessage(content=raw.content)] if isinstance(raw, AIMessage) else []),
        SystemMessage(content=(
            "The previous response contained no native call and was not accepted. "
            "Return exactly one native tool call using a currently bound tool, "
            "with its required arguments and empty content."
        )),
    ]
    begin_provider_invocation(worker="brain")
    corrected = llm.invoke(corrected_messages)
    add_response_usage(corrected, worker="brain")

    _log_native_call_attempt(corrected, 2)
    return corrected, corrected_messages

def _log_native_call_attempt(raw, attempt: int) -> None:
    calls = getattr(raw, "tool_calls", None)
    names = [
        call["name"] for call in calls
        if isinstance(call, dict) and isinstance(call.get("name"), str)
    ] if isinstance(calls, (list, tuple)) else []
    logger.debug(
        "Brain native-call compliance: attempt=%s native_tool_calls_present=%s "
        "tool_call_names=%s retry_triggered=%s retry_exhausted=%s",
        attempt, bool(calls), names,
        attempt == 1 and not calls and not getattr(raw, "invalid_tool_calls", None),
        attempt == 2 and not calls,
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
                    "exact_collection": {
                        "type": "object",
                        "description": (
                            "Reference an exact collection in one current_attempts record's structured evidence. "
                            "Required fields: source_record_index (nonnegative integer index in current_attempts) "
                            "and data_path (array of string keys or integer indexes; [] selects the root). "
                            "Optional label is a nonempty string. "
                            "Do not copy members; Controller binds them from the original tool data."
                        ),
                        "properties": {
                            "source_record_index": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Index in the displayed current_attempts array.",
                            },
                            "data_path": {
                                "type": "array",
                                "items": {"anyOf": [
                                    {"type": "string"},
                                    {"type": "integer"},
                                ]},
                            },
                            "label": {"type": "string", "minLength": 1},
                        },
                        "required": ["source_record_index", "data_path"],
                        "additionalProperties": False,
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
        return outcome.model_copy(update={"usage": normalize_brain_usage(raw)})
