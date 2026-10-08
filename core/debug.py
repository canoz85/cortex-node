"""Shared raw LLM exchange diagnostics."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path


_SENSITIVE_KEY = re.compile(
    r"(?i)^(?:api[_-]?key|password|passwd|secrets?|access[_-]?token|"
    r"refresh[_-]?token|authorization|credentials?)$"
)


def _sanitize_text(value: str) -> str:
    value = re.sub(r'(?i)\bBearer\s+[^\s"\']+', 'Bearer [REDACTED]', value)
    value = re.sub(
        r'(?i)(["\']?(?:api[_-]?key|password|passwd|secrets?|access[_-]?token|'
        r'refresh[_-]?token|authorization|credentials?)["\']?\s*[:=]\s*)'
        r'(?:(?:"[^"\n]*")|(?:\'[^\'\n]*\')|[^\s,;}]+)',
        r'\1[REDACTED]',
        value,
    )
    value = re.sub(r'(?i)(https?://)[^\s/@]+:[^\s/@]+@', r'\1[REDACTED]@', value)
    return re.sub(
        r'-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----',
        '[REDACTED]', value, flags=re.DOTALL,
    )


def sanitize_raw_value(value, *, key: str | None = None):
    """Recursively redact secrets without flattening structured values."""
    if key is not None and _SENSITIVE_KEY.fullmatch(key):
        return "[REDACTED]"
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, Mapping):
        return {
            str(item_key): sanitize_raw_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_raw_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return sanitize_raw_value(value.model_dump(mode="json"))
    return value


def _message_value(message) -> dict:
    if isinstance(message, Mapping):
        result = dict(message)
        if "role" not in result:
            result["role"] = result.get("type", "unknown")
        return result

    result = {
        "role": str(
            getattr(message, "role", None)
            or getattr(message, "type", "unknown")
        ),
        "content": getattr(message, "content", None),
    }
    for name in ("tool_calls", "invalid_tool_calls", "additional_kwargs"):
        value = getattr(message, name, None)
        if value:
            result[name] = value
    return result


def _response_value(response) -> tuple[dict, dict]:
    get = response.get if isinstance(response, Mapping) else (
        lambda name, default=None: getattr(response, name, default)
    )
    metadata = get("response_metadata", {}) or {}
    usage = get("usage_metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    if not isinstance(usage, Mapping):
        usage = {}

    response_value = {
        "content": get("content"),
        "tool_calls": get("tool_calls", []) or [],
        "model": metadata.get("model") or metadata.get("model_name"),
        "done_reason": metadata.get("done_reason"),
    }
    if get("invalid_tool_calls"):
        response_value["invalid_tool_calls"] = get("invalid_tool_calls")

    # if metadata:
    #     metadata_value = dict(metadata)

    #     for key in ("model", "model_name", 
    #                 "done_reason", "done",
    #                 "prompt_eval_count", "eval_count"):
    #         metadata_value.pop(key, None)

    #     if metadata_value:
    #         response_value["metadata"] = metadata_value

    usage_value = {
        "input_tokens": usage.get("input_tokens", metadata.get("prompt_eval_count", 0)),
        "output_tokens": usage.get("output_tokens", metadata.get("eval_count", 0)),
        "total_tokens": usage.get("total_tokens", 0),
    }
    if not usage_value["total_tokens"]:
        usage_value["total_tokens"] = (
            (usage_value["input_tokens"] or 0)
            + (usage_value["output_tokens"] or 0)
        )
    return response_value, usage_value


def log_llm_exchange(
    *, worker: str, operation: str, messages, response,
    execution_id: str | None = None, enabled: bool = False,
    invocation: Mapping | None = None,
) -> None:
    """Write one sanitized JSONL record for one completed LLM invocation."""
    try:
        response_value, usage = _response_value(response)
        record = {
            "execution_id": execution_id,
            "worker": worker,
            "operation": operation,
            "messages": [_message_value(message) for message in messages],
            "response": response_value,
            "usage": usage,
        }
        if invocation is not None:
            record["invocation"] = dict(invocation)
        record = sanitize_raw_value(record)

        file_path = os.getenv("CORTEX_RAW_LLM_FILE") or "logs/raw_llm.jsonl"
        if file_path:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        if enabled:
            print(
                f"[raw-llm:{worker}:{operation}]\n"
                + json.dumps(record, ensure_ascii=False, indent=2, default=str)
            )
    except Exception:
        # Diagnostics must never change an invocation's result.
        pass
