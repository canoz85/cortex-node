"""LangChain/Ollama implementation of the framework-neutral Brain provider port."""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from core.brain import BrainMessage
from core.brain_normalization import normalize_brain_output, normalize_brain_usage
from core.protocol.enums import BrainOutcomeKind
from core.protocol.models import BrainInput, BrainOutcome


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
        snapshots = [message.evidence_snapshot for message in messages if message.evidence_snapshot is not None]
        if len(snapshots) > 1:
            raise ValueError("ambiguous_evidence_snapshot")
        evidence_snapshot = snapshots[0] if snapshots else None
        provider_messages = [
            HumanMessage(content=message.content) if message.role == "human"
            else SystemMessage(content=message.content)
            for message in messages
        ]
        llm = self.tool_brain_llm if tools_enabled else self.brain_llm
        try:
            raw = llm.invoke(provider_messages)
        except Exception as exc:
            # Provider/structured-output errors are values at the service boundary.
            # Retrying is a Controller decision, never another model/parser loop.
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
            evidence_snapshot=evidence_snapshot,
            allow_text_tool_calls=not self.supports_native_tool_calls,
        )
        return outcome.model_copy(update={"usage": normalize_brain_usage(raw)})
