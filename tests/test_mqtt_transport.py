import json
from types import SimpleNamespace

import pytest

from transports.mqtt import MQTTTransport, PayloadError, parse_command


class FakeClient:
    def __init__(self):
        self.published = []
        self.subscribed = []

    def publish(self, topic, payload, qos):
        self.published.append((topic, json.loads(payload), qos))

    def subscribe(self, topic, qos):
        self.subscribed.append((topic, qos))


def message(payload):
    return SimpleNamespace(payload=payload)


def response(client):
    topic, payload, qos = client.published[-1]
    assert topic == "cortex/v1/responses"
    assert qos == 1
    return payload


def test_valid_command_parsing():
    command = parse_command(
        b'{"request_id":"abc-123","message":"Check git status"}'
    )
    assert command.request_id == "abc-123"
    assert command.message == "Check git status"


@pytest.mark.parametrize("payload", [b"not-json", b"[]", b'{"request_id":"x"}'])
def test_malformed_payload_handling(payload):
    client = FakeClient()
    transport = MQTTTransport(client, lambda _: "unused")
    transport._on_message(client, None, message(payload))
    result = response(client)
    assert result["status"] == "failed"
    assert result["answer"] is None
    assert result["error"]


def test_request_id_correlation_for_payload_failure():
    client = FakeClient()
    transport = MQTTTransport(client, lambda _: "unused")
    transport._on_message(
        client, None, message(b'{"request_id":"abc-123","message":null}')
    )
    assert response(client)["request_id"] == "abc-123"


def test_successful_run_prompt_result_publication():
    client = FakeClient()
    seen = []
    transport = MQTTTransport(client, lambda prompt: seen.append(prompt) or "clean tree")
    transport._on_message(
        client,
        None,
        message(b'{"request_id":"abc-123","message":"Check git status"}'),
    )
    assert seen == ["Check git status"]
    assert response(client) == {
        "request_id": "abc-123",
        "status": "completed",
        "answer": "clean tree",
        "error": None,
    }


def test_failed_execution_result_publication():
    client = FakeClient()

    def fail(_prompt):
        raise RuntimeError("execution failed")

    transport = MQTTTransport(client, fail)
    transport._on_message(
        client,
        None,
        message(b'{"request_id":"abc-123","message":"do work"}'),
    )
    assert response(client) == {
        "request_id": "abc-123",
        "status": "failed",
        "answer": None,
        "error": "execution failed",
    }
