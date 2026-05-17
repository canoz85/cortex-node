from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from core.graph_constants import MAX_SUMMARY_CHARS
from core.rag import WorkspaceRAG


def retrieval_message(rag_service: WorkspaceRAG, query: str, top_k: int) -> list[SystemMessage]:
    context = rag_service.format_context(query=query, top_k=top_k)
    if not context:
        return []
    return [SystemMessage(content=context)]


def _clip_summary(text: str, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def rolling_summary_message(summary: str) -> list[SystemMessage]:
    compact = (summary or "").strip()
    if not compact:
        return []
    return [
        SystemMessage(
            content=(
                "Rolling summary from earlier turns (authoritative context):\n"
                f"{compact}"
            )
        )
    ]


def update_rolling_summary(
    planner_llm: ChatOllama,
    existing_summary: str,
    recent_history: list,
) -> str:
    if not recent_history and not existing_summary:
        return ""

    summarization_prompt = (
        "Update the rolling conversation summary for a coding agent. "
        "Keep only durable facts and unresolved items. "
        "Output plain text with these headings exactly: Goal, Constraints, Decisions, Done, Next, Open Questions, Facts. "
        "Use short bullet-like lines, no markdown code blocks, no verbosity. "
        f"Hard limit: {MAX_SUMMARY_CHARS} characters."
    )

    summary_messages = [
        SystemMessage(content=summarization_prompt),
        SystemMessage(content=f"Existing summary:\n{existing_summary or '(none)'}"),
        *recent_history,
    ]
    response = planner_llm.invoke(summary_messages)
    text = str(getattr(response, "content", "") or "").strip()
    if not text:
        return _clip_summary(existing_summary or "")
    return _clip_summary(text)
