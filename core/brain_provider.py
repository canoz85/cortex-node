"""LangChain/Ollama implementation of the framework-neutral Brain provider port."""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from core.brain import BrainMessage
from core.brain_normalization import normalize_brain_output, normalize_brain_usage
from core.protocol.enums import BrainOutcomeKind
from core.protocol.models import BrainInput, BrainOutcome

from core.debug import save_raw_llm


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

def _summarize_response(raw) -> dict:
    response_metadata = getattr(raw, "response_metadata", {}) or {}
    usage_metadata = getattr(raw, "usage_metadata", {}) or {}

    return {
        "content": getattr(raw, "content", ""),
        "tool_calls": getattr(raw, "tool_calls", []) or [],
        "model": response_metadata.get("model")
            or response_metadata.get("model_name"),
        "done_reason": response_metadata.get("done_reason"),
        "usage": {
            "input_tokens": usage_metadata.get("input_tokens"),
            "output_tokens": usage_metadata.get("output_tokens"),
            "total_tokens": usage_metadata.get("total_tokens"),
        },
    }

def _log_brain_exchange(
    *,
    messages: tuple[BrainMessage, ...],
    raw,
    execution_id: str,
    show_raw_llm: bool,
) -> None:
    for message in messages:
        save_raw_llm(
            "brain",
            f"message:{message.role}",
            message.content,
            execution_id=execution_id,
        )

    save_raw_llm(
        "brain",
        "response",
        _summarize_response(raw),
        execution_id=execution_id,
    )

    if not show_raw_llm:
        return

    for message in messages:
        print(f"[raw-llm][{message.role}]\n{message.content}")

    print(f"[raw-llm][response]\n{raw}")


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

        native_tools_enabled = tools_enabled and self.supports_native_tool_calls
        
        provider_messages = [
            HumanMessage(content=message.content) if message.role == "human"
            else SystemMessage(content=message.content)
            for message in messages
        ]
        llm = self.tool_brain_llm if tools_enabled else self.brain_llm
        try:
            raw = llm.invoke(provider_messages)
            if native_tools_enabled:
                _log_native_call_attempt(raw, 1)
            if native_tools_enabled and not getattr(raw, "tool_calls", None):
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

        _log_brain_exchange(
            messages=messages,
            raw=raw,
            execution_id=brain_input.identity.execution_id,
            show_raw_llm=self.show_raw_llm,
        )

        outcome = normalize_brain_output(
            raw, brain_input, self.tools_set,
            allow_text_tool_calls=not self.supports_native_tool_calls,
        )
        return outcome.model_copy(update={"usage": normalize_brain_usage(raw)})
