from langchain_core.messages import AIMessage, HumanMessage


def latest_user_message(history: list) -> str:
    for message in reversed(history):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def current_turn_messages(history: list) -> list:
    """Return messages from the latest user turn onward."""
    if not history:
        return []

    for index in range(len(history) - 1, -1, -1):
        if isinstance(history[index], HumanMessage):
            return history[index:]
    return history


def recent_messages(history: list, limit: int) -> list:
    if limit <= 0:
        return []
    return history[-limit:]


def normalize_message_content(message: AIMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def is_effectively_empty_response(message: AIMessage) -> bool:
    if getattr(message, "tool_calls", None):
        return False

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        return len(content) == 0
    return not bool(content)
