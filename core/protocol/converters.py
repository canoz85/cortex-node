"""Conversion utilities for legacy values to typed representations."""

from typing import Any


def _to_int(value: Any, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


def _to_optional_int(value: Any, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None and parsed < minimum:
        return None
    return parsed


def _to_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default