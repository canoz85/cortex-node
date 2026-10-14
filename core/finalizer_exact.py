"""Deterministic presentation of Controller-bound exact collections."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
import re

from core.protocol.models import FinalizationRequest


def render_exact_collections(request: FinalizationRequest) -> str | None:
    """Return exact collection output, or None for the model-backed prose path."""
    results = [
        result
        for result in request.accepted_step_results
        if result.completion_evidence.exact_collection is not None
    ]
    if not results:
        return None

    blocks: list[str] = []
    for result in results:
        collection = result.completion_evidence.exact_collection
        if collection.items is None:
            raise ValueError("exact collection was not Controller-bound")
        lines = [f"{collection.label}:"]
        if _group_by_file_type(request, result.completion_evidence.step_id) and all(
            isinstance(item, str) for item in collection.items
        ):
            # Stable groups and stable occurrence order within each group. Values
            # remain untouched; duplicates are appended rather than deduplicated.
            groups: dict[str, list[str]] = {}
            for item in collection.items:
                extension = PurePosixPath(item).suffix
                groups.setdefault(extension or "No extension", []).append(item)
            for group, members in groups.items():
                lines.append(f"{group}:")
                lines.extend(f"- {member}" for member in members)
        else:
            lines.extend(f"- {_format_member(item)}" for item in collection.items)
        if not collection.items:
            lines.append("- (none)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _group_by_file_type(request: FinalizationRequest, step_id: str) -> bool:
    """Small presentation accommodation, never used to select exact membership."""
    steps = request.accepted_plan.steps if request.accepted_plan is not None else ()
    step = next((step for step in steps if step.step_id == step_id), None)
    semantics = f"{step.title}\n{step.description}" if step is not None else ""
    return re.search(
        r"\bgroup(?:ed|ing)?\b[^.\n]*?\bby\s+(?:file\s+)?(?:types?|extensions?)\b",
        semantics + "\n" + request.context.user_request, re.IGNORECASE,
    ) is not None


def render_exact_collection_answer(request: FinalizationRequest) -> str | None:
    """Compose exact blocks with all other accepted step conclusions.

    A collection-bearing step keeps its existing deterministic presentation.
    Other steps retain their accepted semantic text, in acceptance order. No
    model is needed to regenerate either the collections or these conclusions.
    """
    exact = render_exact_collections(request)
    if exact is None:
        return None

    semantics = (
        result.semantic_content
        for result in request.accepted_step_results
        if result.completion_evidence.exact_collection is None
    )
    return "\n\n".join((exact, *semantics))


def _format_member(item) -> str:
    if isinstance(item, str):
        return item
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"))
