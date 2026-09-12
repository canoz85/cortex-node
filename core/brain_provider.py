"""LangChain/Ollama implementation of the framework-neutral Brain provider port."""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from core.brain import BrainMessage
from core.brain_normalization import normalize_brain_output, normalize_brain_usage
from core.protocol.enums import BrainOutcomeKind
from core.protocol.models import BrainInput, BrainOutcome


logger = logging.getLogger(__name__)


def _log_native_call_attempt(raw, attempt: int) -> None:
    calls = getattr(raw, "tool_calls", None)
    names = [
        call["name"] for call in calls
        if isinstance(call, dict) and isinstance(call.get("name"), str)
    ] if isinstance(calls, (list, tuple)) else []
    logger.info(
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
                    "message": {"type": "string", "description": "Concise step-local completion summary."},
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
            "description": "Report that the active execution step failed.",
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
            "description": "Request replanning because the active step or plan cannot proceed safely.",
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


class LangChainBrainProvider:
    def __init__(
        self, *, brain_llm, tool_brain_llm, tools_set: set[str],
        show_raw_llm: bool = False, supports_native_tool_calls: bool = True,
    ):
        self.brain_llm = brain_llm
        self.tool_brain_llm = tool_brain_llm
        self.tools_set = set(tools_set)
        self.show_raw_llm = show_raw_llm
        self.supports_native_tool_calls = supports_native_tool_calls

    def generate(
        self, brain_input: BrainInput, messages: tuple[BrainMessage, ...], *, tools_enabled: bool,
    ) -> BrainOutcome:
        provider_messages = [
            HumanMessage(content=message.content) if message.role == "human"
            else SystemMessage(content=message.content)
            for message in messages
        ]
        llm = self.tool_brain_llm if tools_enabled else self.brain_llm
        try:
            raw = llm.invoke(provider_messages)
            if tools_enabled and self.supports_native_tool_calls:
                _log_native_call_attempt(raw, 1)
            if tools_enabled and self.supports_native_tool_calls and not getattr(raw, "tool_calls", None):
                raw = llm.invoke([
                    *provider_messages,
                    # Show the rejected response so this is a protocol correction,
                    # not a fresh task invocation. Never interpret its content.
                    *([AIMessage(content=raw.content)] if isinstance(raw, AIMessage) else []),
                    SystemMessage(content=(
                        "The previous response was invalid because it did not contain a native tool call. "
                        "Do not write function-call syntax as text. Return exactly one native tool call. "
                        "Use one of the currently bound executable or lifecycle tools."
                        " Lifecycle actions returned in content are text, not native calls. "
                        "If reporting a lifecycle outcome, invoke brain_step_completed, "
                        "brain_step_failed, or brain_replan_requested through the native tool channel "
                        "with its required arguments and leave content empty. "
                        "Keep the same active-step decision; correct only the response protocol."
                    )),
                ])
                _log_native_call_attempt(raw, 2)
        except Exception as exc:
            # Provider/structured-output errors are values at the service boundary.
            # Exception retries remain a Controller decision.
            return BrainOutcome(
                outcome=BrainOutcomeKind.PROVIDER_FAILURE,
                step_id=brain_input.active_step.step_id if brain_input.active_step else None,
                error_code=type(exc).__name__,
                message=f"Brain provider failed ({type(exc).__name__}).",
            )
        if self.show_raw_llm:
            for message in messages:
                print(f"[raw-llm][{message.role}]\n{message.content}")
            print(f"[raw-llm][response]\n{raw}")
        outcome = normalize_brain_output(
            raw, brain_input, self.tools_set,
            allow_text_tool_calls=not self.supports_native_tool_calls,
        )
        return outcome.model_copy(update={"usage": normalize_brain_usage(raw)})
