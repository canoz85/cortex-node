from core.graph_constants import (
    ACTION_INTENT_PATTERN,
    AGENT_INFO_INTENT_PATTERN,
    CASUAL_CHAT_PATTERN,
    CODE_DISCUSSION_PATTERN,
    CODING_DISCUSSION_QUESTION_PATTERN,
    CURRENT_TIME_INTENT_PATTERN,
    FILE_GENERATION_PATTERN,
    TOKEN_USAGE_INTENT_PATTERN,
)


def requires_action(user_text: str) -> bool:
    text = (user_text or "").strip()
    if not text:
        return False
    return bool(ACTION_INTENT_PATTERN.search(text))


def preferred_info_tool(user_text: str) -> str | None:
    text = (user_text or "").strip()
    if not text:
        return None
    if TOKEN_USAGE_INTENT_PATTERN.search(text):
        return "token_usage"
    if CURRENT_TIME_INTENT_PATTERN.search(text):
        return "current_time"
    if AGENT_INFO_INTENT_PATTERN.search(text):
        return "agent_info"
    return None


def _is_casual_chat(user_text: str) -> bool:
    """Return True for social/identity chat that should skip planning/tool routing."""
    text = (user_text or "").strip()
    if not text:
        return False
    if preferred_info_tool(text) or requires_action(text):
        return False
    if CODE_DISCUSSION_PATTERN.search(text):
        return False
    if CASUAL_CHAT_PATTERN.search(text):
        return True

    word_count = len(text.split())
    if word_count <= 6 and text.endswith("?"):
        return True
    return False


def is_file_generation_request(user_text: str) -> bool:
    text = (user_text or "").strip()
    if not text or not requires_action(text):
        return False
    return bool(FILE_GENERATION_PATTERN.search(text))


def planner_route(user_text: str) -> str:
    text = (user_text or "").strip()
    if not text:
        return "conversation"
    if preferred_info_tool(text):
        return "info"
    if _is_casual_chat(text):
        return "casual"
    if CODE_DISCUSSION_PATTERN.search(text) and CODING_DISCUSSION_QUESTION_PATTERN.search(text):
        return "coding_discussion"
    if is_file_generation_request(text):
        return "action:file_generation"
    if requires_action(text):
        return "action"
    if CODE_DISCUSSION_PATTERN.search(text):
        return "coding_discussion"
    return "conversation"
