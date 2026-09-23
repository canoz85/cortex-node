from typing import Any
import uuid

from core.artifacts import ToolArtifact
from core.models import ToolResult
from core.protocol.models import ArtifactRecord

def compute_repeat_fail_count(
    *,
    previous: ToolResult | None,
    previous_repeat_count: int,
    current: ToolResult,
) -> int:
    if current.is_async_job and not current.async_terminal:
        return 0

    if (
        not current.success
        and current.signature
        and previous is not None
        and previous.signature == current.signature
        and previous.success is False
    ):
        return previous_repeat_count + 1

    if not current.success and current.signature:
        return 1

    return 0

def build_artifact_records(
    artifacts: tuple[ToolArtifact, ...],
    *,
    step_id: str,
) -> tuple[ArtifactRecord, ...]:
    return tuple(
        ArtifactRecord(
            artifact_id=f"art-{uuid.uuid4()}",
            step_id=step_id,
            path=artifact.path,
            action=artifact.action,
        )
        for artifact in artifacts
    )


def parse_tool_result(raw: Any) -> ToolResult | None:
    """Return a ToolResult if the raw value contains a valid structured payload."""
    return ToolResult.try_parse(raw)


def unwrap_tool_output(raw: Any) -> dict[str, Any] | list[Any] | str | None:
    """Unwrap summary-plus-JSON tool output into Python values."""
    return ToolResult.unwrap_tool_output(raw)