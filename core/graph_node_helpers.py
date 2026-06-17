from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from core.graph_summarize import rolling_summary_message
from core.graph_filegen_policy import last_tool_stderr
from core.graph_messages import normalize_message_content
from core.graph_pseudo_tools import looks_like_pseudo_tool_text
from core.models import TokenUsage
from tools.info_ops import update_token_usage


def response_with_usage(state: dict, response: AIMessage) -> dict:
    """Return a standard graph node payload with token usage updates."""
    meta = getattr(response, "response_metadata", {}) or {}
    usage = TokenUsage.from_response_metadata(meta)
    update_token_usage(usage.model_dump())
    return {
        "messages": [response],
        "steps": state.get("steps", 0) + 1,
        "token_usage": usage,
        "tool_text_retry_used": False,
    }


def direct_discussion_response(
    planner_llm: ChatOllama,
    system_prompt: str,
    retrieval_messages: list[SystemMessage],
    rolling_summary: str,
    recent_history: list,
) -> AIMessage:
    def _finalize_llm_text(source: AIMessage, content: str) -> AIMessage:
        metadata = getattr(source, "response_metadata", None)
        if isinstance(metadata, dict) and metadata:
            return AIMessage(content=content, response_metadata=metadata)
        return AIMessage(content=content)

    messages = [
        SystemMessage(content=system_prompt),
        *retrieval_messages,
        *rolling_summary_message(rolling_summary),
        *recent_history,
        SystemMessage(
            content=(
                "This turn is discussion-only. Answer directly in concise prose. "
                "Do not call tools, do not propose tool syntax, and do not create or modify files."
            )
        ),
    ]
    response = planner_llm.invoke(messages)
    content = normalize_message_content(response).strip()
    if getattr(response, "tool_calls", None) or looks_like_pseudo_tool_text(content) or not content:
        fallback = planner_llm.invoke(
            [
                *messages,
                SystemMessage(
                    content=(
                        "Your previous reply was not a direct discussion answer. "
                        "Reply with plain prose only, no code blocks and no tool-like syntax."
                    )
                ),
            ]
        )
        fallback_content = normalize_message_content(fallback).strip()
        if fallback_content and not looks_like_pseudo_tool_text(fallback_content):
            return _finalize_llm_text(fallback, fallback_content)
        return AIMessage(
            content=(
                "Describe the error message, the JSON input, and the code path that fails, and I will help isolate the parsing bug directly."
            )
        )
    return _finalize_llm_text(response, content)


def detect_missing_dependency(tool_output_raw: str) -> str | None:
    """Extract package name from ModuleNotFoundError or ImportError in stderr. Returns package name or None."""
    if not tool_output_raw:
        return None
    stderr = last_tool_stderr(tool_output_raw)
    if not stderr:
        return None
    stderr_lower = stderr.lower()
    if "modulenotfounderror" in stderr_lower or "importerror" in stderr_lower:
        if "no module named" in stderr_lower:
            start = stderr.find("'") + 1
            end = stderr.find("'", start)
            if start > 0 and end > start:
                return stderr[start:end]
        return "unknown_module"
    return None


def planner_execution_brief(route: str, plan_text: str) -> str:
    """Return a concise planner brief for the brain LLM on action routes."""
    normalized_plan = str(plan_text or "").strip()
    if not normalized_plan:
        return ""

    if len(normalized_plan) > 1500:
        normalized_plan = f"{normalized_plan[:1500]}..."

    return (
        "Planner execution brief below. Use it as guidance, but produce executable next steps now.\n"
        f"Route: {route}\n"
        "Plan:\n"
        f"{normalized_plan}"
    )