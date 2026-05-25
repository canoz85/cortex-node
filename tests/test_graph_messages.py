from langchain_core.messages import AIMessage

from core.graph_messages import normalize_message_content


def test_normalize_message_content_reads_text_blocks():
    message = AIMessage(content=[{"type": "text", "text": "alpha"}, {"type": "text", "text": "beta"}])

    assert normalize_message_content(message) == "alpha\nbeta"


def test_normalize_message_content_reads_content_fallback_key():
    message = AIMessage(content=[{"type": "text", "content": "alpha"}, {"type": "text", "content": "beta"}])

    assert normalize_message_content(message) == "alpha\nbeta"


def test_normalize_message_content_reads_value_fallback_key():
    message = AIMessage(content=[{"type": "text", "value": "alpha"}, {"type": "text", "value": "beta"}])

    assert normalize_message_content(message) == "alpha\nbeta"
