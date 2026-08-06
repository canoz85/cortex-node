from langchain_core.messages import AIMessage

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
