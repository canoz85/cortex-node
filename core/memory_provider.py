"""Optional LangChain transport for narrow structured memory proposals."""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from core.conversation_memory_updater import MemoryProposal, MemoryUpdateRequest
from core.debug import log_llm_exchange


_SYSTEM_PROMPT = """Extract only durable memory supported by the supplied evidence.
Return a MemoryProposal object with a facts array. Return an empty facts array when unsure.
Do not store greetings, one-off tasks, temporary results, execution mechanics, errors,
speculation, or arbitrary final-answer prose as durable facts.
Apply this durability test before proposing any fact: would this fact still be useful
and expected to remain valid in a future conversation after the immediate execution
context is gone? If not, do not propose it. Current file or directory listings and
counts, git or branch status, test pass/fail results, command output, process state,
and temporary runtime, network, or device state are transient snapshots, not durable
facts. Stable project configuration, architecture decisions, selected models or
providers, persistent tooling choices, and durable workflow conventions may qualify.
For a user fact, use source_handle human_current. Set text to a concise, standalone
normalized durable fact (for example,
"The user prefers concise answers"), not the verbatim quote or the whole conversational
utterance. Omit greetings, questions, and incidental wording from text.
For a project fact explicitly stated in user_request, use source_handle human_current;
this records user-stated project knowledge, not independently verified knowledge. For a
project fact supported by accepted execution evidence, use source_handle
accepted:<evidence_id> from accepted_completions. Set text to a concise, standalone
normalized durable fact. Set evidence_quote to an exact supporting span of the selected
accepted completion. The normalized text may differ from the evidence quote. The shared
proposal schema still requires evidence_quote for human_current, but Cortex ignores that
field and deterministically binds human_current to the complete current user_request.
Never treat the user request or accepted_answer as verified project evidence.
Never invent source handles, identifiers, categories, or execution outcomes.
Use short stable scope_key values; reuse an existing scope_key for a correction.
Current explicit user corrections take precedence over older existing_facts.
Do not propose question changes. Do not rewrite the memory store."""


class LangChainMemoryProposalProvider:
    def __init__(self, llm, *, show_raw_llm: bool = False):
        self.llm = llm
        self.show_raw_llm = show_raw_llm

    def generate(self, request: MemoryUpdateRequest) -> MemoryProposal:
        try:
            structured = self.llm.with_structured_output(
                MemoryProposal, method="json_schema", include_raw=True,
            )
        except TypeError:
            # Compatibility for provider/test doubles without include_raw.
            structured = self.llm.with_structured_output(
                MemoryProposal, method="json_schema",
            )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(request.model_dump(mode="json"), ensure_ascii=False)),
        ]
        exchange = structured.invoke(messages)
        if isinstance(exchange, dict) and "raw" in exchange and "parsed" in exchange:
            raw = exchange.get("raw")
            result = exchange.get("parsed")
            parsing_error = exchange.get("parsing_error")
            if parsing_error is not None:
                raise parsing_error
        else:
            raw = exchange
            result = exchange
        log_llm_exchange(
            worker="memory", operation="update", messages=messages,
            response=raw, execution_id=request.execution_id,
            enabled=self.show_raw_llm,
        )
        if isinstance(result, MemoryProposal):
            return result
        return MemoryProposal.model_validate(result)
