"""Deterministic presentation of Controller-bound exact collections."""

from __future__ import annotations

import json

from core.protocol.models import FinalizationRequest


def render_exact_collections(request: FinalizationRequest) -> str | None:
    """Return exact collection output, or None for the model-backed prose path."""
    collections = [
        result.completion_evidence.exact_collection
        for result in request.accepted_step_results
        if result.completion_evidence.exact_collection is not None
    ]
    if not collections:
        return None

    blocks: list[str] = []
    for collection in collections:
        if collection.items is None:
            raise ValueError("exact collection was not Controller-bound")
        lines = [f"{collection.label}:"]
        lines.extend(f"- {_format_member(item)}" for item in collection.items)
        if not collection.items:
            lines.append("- (none)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_member(item) -> str:
    if isinstance(item, str):
        return item
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"))
