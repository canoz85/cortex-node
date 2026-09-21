"""Minimal MQTT transport adapter for CortexNode prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable


COMMAND_TOPIC = "cortex/v1/commands"
RESPONSE_TOPIC = "cortex/v1/responses"


@dataclass(frozen=True)
class MQTTCommand:
    request_id: str
    message: str


class PayloadError(ValueError):
    def __init__(self, message: str, *, request_id: str | None = None):
        super().__init__(message)
        self.request_id = request_id


def parse_command(payload: bytes | str) -> MQTTCommand:
    """Parse and validate one command without coupling it to Cortex internals."""
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    except UnicodeDecodeError as exc:
        raise PayloadError("payload must be valid UTF-8") from exc

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise PayloadError("payload must be valid JSON") from exc

    if not isinstance(data, dict):
        raise PayloadError("payload must be a JSON object")

    request_id = data.get("request_id")
    correlation_id = request_id if isinstance(request_id, str) else None
    if not isinstance(request_id, str) or not request_id.strip():
        raise PayloadError("request_id must be a non-empty string")

    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        raise PayloadError(
            "message must be a non-empty string", request_id=correlation_id
        )

    return MQTTCommand(request_id=request_id, message=message)


class MQTTTransport:
    """Translate MQTT messages to calls against an injected prompt runner."""

    def __init__(
        self,
        client,
        run_prompt: Callable[[str], str],
        *,
        command_topic: str = COMMAND_TOPIC,
        response_topic: str = RESPONSE_TOPIC,
    ) -> None:
        self.client = client
        self.run_prompt = run_prompt
        self.command_topic = command_topic
        self.response_topic = response_topic
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _publish(self, payload: dict) -> None:
        self.client.publish(
            self.response_topic,
            json.dumps(payload, ensure_ascii=False),
            qos=1,
        )

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code != 0:
            raise ConnectionError(f"MQTT connection rejected: {reason_code}")
        client.subscribe(self.command_topic, qos=1)

    def _on_message(self, client, userdata, mqtt_message):
        request_id = None
        try:
            command = parse_command(mqtt_message.payload)
            request_id = command.request_id
            answer = self.run_prompt(command.message)
            self._publish({
                "request_id": request_id,
                "status": "completed",
                "answer": answer,
                "error": None,
            })
        except PayloadError as exc:
            self._publish({
                "request_id": exc.request_id,
                "status": "failed",
                "answer": None,
                "error": str(exc),
            })
        except Exception as exc:
            self._publish({
                "request_id": request_id,
                "status": "failed",
                "answer": None,
                "error": str(exc) or type(exc).__name__,
            })

    def run(self, host: str, port: int, keepalive: int = 60) -> None:
        self.client.connect(host, port, keepalive)
        self.client.loop_forever()


def create_transport(run_prompt: Callable[[str], str]) -> MQTTTransport:
    """Create the production adapter, importing the optional dependency lazily."""
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError(
            "MQTT mode requires paho-mqtt; install dependencies from requirements.txt"
        ) from exc

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    return MQTTTransport(client, run_prompt)
