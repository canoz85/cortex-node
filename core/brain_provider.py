"""LangChain/Ollama implementation of the framework-neutral Brain provider port."""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

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

    if getattr(raw, "tool_calls", None):
        return raw, provider_messages

    corrected_messages = [
        *provider_messages,
        *([AIMessage(content=raw.content)] if isinstance(raw, AIMessage) else []),
        SystemMessage(content=(
            "The previous response was invalid because it did not contain a native tool call. "
            "Do not write function-call syntax as text. Return exactly one native tool call. "
            "Use one of the currently bound executable or lifecycle tools. "
            "Lifecycle actions returned in content are text, not native calls. "
            "If reporting a lifecycle outcome, invoke brain_step_completed, "
            "brain_step_failed, or brain_replan_requested through the native tool channel "
            "with its required arguments and leave content empty. "
            "Keep the same active-step decision; correct only the response protocol."
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
        attempt == 1 and not calls, attempt == 2 and not calls,
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
                        "description": (
                            "The semantic result produced by completing the active step, "
                            "grounded in the available evidence. Include the actual finding, "
                            "summary, interpretation, comparison, calculation, or other "
                            "requested result when the step produces one; do not merely state "
                            "that the step was completed."
                        ),
                    },
                    "exact_collection": {
                        "type": "object",
                        "description": (
                            "When the accepted result is an exact list, table, path set, "
                            "identifier set, or other collection already present in one "
                            "tool result's structured data, reference it here. Do not copy "
                            "the members into this object."
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
                            "label": {"type": "string"},
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
            "description": (
                "Report genuine impossibility: the active step and overall request cannot "
                "reasonably be achieved, including by a reasonable revised plan. Controller "
                "may retry the same step and ultimately terminate the execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Concise step-local failure reason."},
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
            "description": (
                "Request Controller-authorized replanning because the accepted strategy or plan "
                "must change, while the overall objective may still be achievable. Repeated "
                "identical tool failures are not required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Concise reason replanning is required."},
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


def native_brain_tools(executable_tools) -> list:
    """Return executable tools plus non-executable provider response actions."""
    return [*executable_tools, *LIFECYCLE_ACTION_SCHEMAS]


def text_tool_definitions(tools) -> str:
    """Expose tool schemas to models that cannot receive native tool bindings."""
    return json.dumps(
        [convert_to_openai_tool(tool)["function"] for tool in tools],
        ensure_ascii=True,
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

def _limit_visible_tools(messages, authorized_tools, *, native: bool):
    """Replace the construction-time tool catalog with this plan's authorization."""
    limited = []
    for message in messages:
        content = message.content
        if isinstance(message, SystemMessage) and content.startswith("AVAILABLE TOOLS:\n"):
            visible = (
                "\n".join(f"- {tool.name}" for tool in authorized_tools)
                if native else text_tool_definitions(authorized_tools)
            )
            _, separator, environment = content.partition("\nENVIRONMENT:\n")
            content = f"AVAILABLE TOOLS:\n{visible}\n"
            if separator:
                content += separator + environment
            message = SystemMessage(content=content)
        limited.append(message)
    return limited

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
        show_raw_llm: bool = False, supports_native_tool_calls: bool = True,
    ):
        self.brain_llm = brain_llm
        self.executable_tools = {
            tool.name: tool for tool in executable_tools
            if isinstance(getattr(tool, "name", None), str) and tool.name
        }
        self.show_raw_llm = show_raw_llm
        self.supports_native_tool_calls = supports_native_tool_calls

    def generate(
        self, brain_input: BrainInput, messages: tuple[BrainMessage, ...], *, tools_enabled: bool,
    ) -> BrainOutcome:

        authorized_tool_names, authorized_tools = _resolve_authorized_tools(
            brain_input, self.executable_tools
        )
        native_tools_enabled = tools_enabled and self.supports_native_tool_calls
        
        provider_messages = _to_provider_messages(messages)
        try:
            if tools_enabled:
                provider_messages = _limit_visible_tools(
                    provider_messages, authorized_tools,
                    native=self.supports_native_tool_calls,
                )
            llm = (
                self.brain_llm.bind_tools(native_brain_tools(authorized_tools))
                if native_tools_enabled else self.brain_llm
            )
            begin_provider_invocation(worker="brain")
            raw = llm.invoke(provider_messages)
            add_response_usage(raw, worker="brain")
            log_llm_exchange(
                worker="brain", operation="step", messages=provider_messages,
                response=raw, execution_id=brain_input.identity.execution_id,
                enabled=self.show_raw_llm,
            )
            if native_tools_enabled:
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
            # Provider/structured-output errors are values at the service boundary.
            # Exception retries remain a Controller decision.
            return BrainOutcome(
                outcome=BrainOutcomeKind.PROVIDER_FAILURE,
                step_id=brain_input.active_step.step_id if brain_input.active_step else None,
                error_code=type(exc).__name__,
                message=f"Brain provider failed ({type(exc).__name__}).",
            )

        outcome = normalize_brain_output(
            raw, brain_input, set(self.executable_tools) & authorized_tool_names,
            allow_text_tool_calls=not self.supports_native_tool_calls,
        )
        return outcome.model_copy(update={"usage": normalize_brain_usage(raw)})
