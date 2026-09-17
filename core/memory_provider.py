"""Optional LangChain transport for narrow structured memory proposals."""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from core.conversation_memory_updater import MemoryProposal, MemoryUpdateRequest


_SYSTEM_PROMPT = """Extract only durable memory supported by the supplied evidence.
Return a MemoryProposal object with a facts array. Return an empty facts array when unsure.
Do not store greetings, one-off tasks, temporary results, execution mechanics, errors,
speculation, or arbitrary final-answer prose as durable facts.
For a user fact, use source_handle human_current and copy an exact evidence_quote from
user_request. Set text to a concise, standalone normalized durable fact (for example,
"The user prefers concise answers"), not the verbatim quote or the whole conversational
utterance. Omit greetings, questions, and incidental wording from text. The quote
must be an exact supporting span of an explicit user statement, not a guess.
For a project fact, use source_handle accepted:<evidence_id> from accepted_completions.
Its text and evidence_quote must be the same exact span of that accepted summary.
Never treat the user request or accepted_answer as verified project evidence.
Never invent source handles, identifiers, categories, or execution outcomes.
Use short stable scope_key values; reuse an existing scope_key for a correction.
Current explicit user corrections take precedence over older existing_facts.
Do not propose question changes. Do not rewrite the memory store."""


class LangChainMemoryProposalProvider:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, request: MemoryUpdateRequest) -> MemoryProposal:
        structured = self.llm.with_structured_output(MemoryProposal, method="json_schema")
        result = structured.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(request.model_dump(mode="json"), ensure_ascii=False)),
        ])
        if isinstance(result, MemoryProposal):
            return result
        return MemoryProposal.model_validate(result)
