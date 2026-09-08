"""Immutable, prompt-local references to domain tool execution evidence."""

import hashlib
import json
from dataclasses import dataclass

from core.protocol.models import BrainInput


def evidence_scope(brain_input: BrainInput) -> str:
    """Scope identity only; never rebuild evidence from post-response history."""
    return json.dumps({
        "identity": brain_input.identity.model_dump(mode="json"),
        "cursor": brain_input.cursor.model_dump(mode="json"),
        "retry": brain_input.retry.model_dump(mode="json"),
        "step": brain_input.active_step.model_dump(mode="json") if brain_input.active_step else None,
    }, sort_keys=True)


@dataclass(frozen=True)
class EvidenceSnapshot:
    scope: str
    bindings: tuple[tuple[str, str], ...]

    @classmethod
    def capture(
        cls, brain_input: BrainInput, request_ids: tuple[str, ...], prompt_content: str,
    ) -> "EvidenceSnapshot":
        # Same prompt/input snapshot has stable refs. Changes to the prompt,
        # history, execution, step, retry or cursor give a different namespace.
        fingerprint = hashlib.sha256(json.dumps({
            "input": brain_input.model_dump(mode="json"),
            "prompt": prompt_content,
        }, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:16]
        return cls(
            scope=evidence_scope(brain_input),
            bindings=tuple(
                (f"e{index}-{fingerprint}", request_id)
                for index, request_id in enumerate(request_ids, start=1)
            ),
        )
