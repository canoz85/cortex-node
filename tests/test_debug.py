"""Shared raw LLM exchange logging contract."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.debug import _message_value, log_llm_exchange, sanitize_raw_value


def test_plain_messages_keep_only_role_and_content():
    assert _message_value(HumanMessage(content="hello")) == {
        "role": "human",
        "content": "hello",
    }
    assert _message_value(SystemMessage(content="instructions")) == {
        "role": "system",
        "content": "instructions",
    }


def test_ai_message_preserves_structured_tool_calls():
    tool_calls = [{"name": "inspect", "args": {"path": "status"}, "id": "call-1"}]
    message = AIMessage(content="", tool_calls=tool_calls)

    value = _message_value(message)

    assert value["tool_calls"] == message.tool_calls
    assert isinstance(value["tool_calls"], list)
    assert isinstance(value["tool_calls"][0]["args"], dict)


def test_ai_message_preserves_structured_additional_kwargs():
    additional_kwargs = {
        "function_call": {"name": "inspect", "arguments": '{"path":"status"}'},
    }

    value = _message_value(
        AIMessage(content="", additional_kwargs=additional_kwargs)
    )

    assert value["additional_kwargs"] == additional_kwargs


def test_mapping_preserves_provider_fields_without_mutation():
    message = {
        "type": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "provider_field": {"cache_control": {"type": "ephemeral"}},
    }
    original = dict(message)

    value = _message_value(message)

    assert value == {**message, "role": "assistant"}
    assert message == original
    assert "role" not in message


def test_json_looking_message_content_remains_a_string():
    content = '{"tool":"inspect","args":{"path":"status"}}'

    value = _message_value(HumanMessage(content=content))

    assert value["content"] == content
    assert isinstance(value["content"], str)


def test_message_extras_are_recursively_sanitized():
    message = AIMessage(
        content="ok",
        tool_calls=[{
            "name": "inspect",
            "args": {"api_key": "hidden-key"},
            "id": "call-1",
        }],
        additional_kwargs={"provider": {"authorization": "Bearer hidden-token"}},
    )

    value = sanitize_raw_value(_message_value(message))

    assert value["tool_calls"][0]["args"]["api_key"] == "[REDACTED]"
    assert value["additional_kwargs"]["provider"]["authorization"] == "[REDACTED]"


def test_exchange_is_one_structured_sanitized_record(monkeypatch, capsys):
    path = Path(".tmp/raw-exchange-test.jsonl")
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        path.unlink()
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", str(path))
    messages = [
        {"role": "system", "content": {"credentials": "hidden", "nested": ["safe"]}},
        HumanMessage(content="Bearer hidden-token https://user:pass@example.com"),
    ]
    response = AIMessage(
        content="ok",
        tool_calls=[{"name": "inspect", "args": {"api_key": "hidden-key"}, "id": "1"}],
        response_metadata={"model": "demo", "done_reason": "stop"},
        usage_metadata={"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
    )

    log_llm_exchange(
        worker="brain", operation="step", messages=messages,
        response=response, execution_id="exec-1", enabled=True,
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == {"execution_id", "worker", "operation", "messages", "response", "usage"}
    assert record["messages"][0]["content"] == {
        "credentials": "[REDACTED]", "nested": ["safe"],
    }
    assert record["response"]["tool_calls"][0]["args"]["api_key"] == "[REDACTED]"
    serialized = json.dumps(record)
    for secret in ("hidden", "hidden-token", "hidden-key", "user:pass"):
        assert secret not in serialized
    output = capsys.readouterr().out
    assert "[raw-llm:brain:step]" in output
    assert json.loads(output.split("\n", 1)[1]) == record


def test_file_logging_is_independent_from_console_flag(monkeypatch, capsys):
    path = Path(".tmp/raw-exchange-no-console.jsonl")
    if path.exists():
        path.unlink()
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", str(path))

    log_llm_exchange(
        worker="finalizer", operation="render",
        messages=[HumanMessage(content="hello")],
        response=AIMessage(content="hi"), enabled=False,
    )

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert capsys.readouterr().out == ""
