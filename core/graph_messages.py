from langchain_core.messages import HumanMessage


def latest_human_message_str(history: list) -> str:
    
    latest_message = latest_human_message(history)
    if latest_message is not None:
        raw_text = str(latest_message.content)
        return raw_text.strip()
    
    return ""

def latest_human_message(history: list) -> HumanMessage | None:
    for message in reversed(history):
        if isinstance(message, HumanMessage):
            return message
    return None

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


def normalize_message_content(message: object) -> str:
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
                    continue
                # Providers may emit text blocks under alternative keys.
                for fallback_key in ("content", "value"):
                    fallback_value = item.get(fallback_key)
                    if fallback_value:
                        parts.append(str(fallback_value))
                        break
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def is_effectively_empty_response(message: object) -> bool:
    if getattr(message, "tool_calls", None):
        return False

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        return len(content) == 0
    return not bool(content)
