import json
from typing import Any
import uuid
from dataclasses import dataclass

from core.artifacts import ToolArtifact
from core.graph_response_formatters import format_tool_result_response
from core.models import ToolOutputEnvelope
from core.protocol.enums import AsyncJobStatus
from core.protocol.models import ArtifactRecord, ContentIntegrity, PaginationMetadata, ToolRequest, ToolResult


@dataclass(frozen=True, slots=True)
class NormalizedToolPayload:
    success: bool
    message: str
    data: object | None
    rendered_output: str
    error_code: str | None
    integrity: ContentIntegrity
    pagination: PaginationMetadata | None
    is_async_job: bool
    async_job_id: str | None
    async_job_status: AsyncJobStatus | None
    async_terminal: bool
    async_observed_at_utc: str | None
    artifacts: tuple[ToolArtifact, ...] = ()

def _build_tool_signature(request: ToolRequest) -> str:
    return f"{request.tool_name}:{json.dumps(request.arguments, sort_keys=True)}"

def _structured_result_data(unwrapped: dict, tool_name: str):
    """Preserve known file-result fields without inventing absent evidence.

    Existing data envelopes take precedence, including explicit null values.
    This is capture only: downstream consumers must validate evidence semantics.
    """
    if "data" in unwrapped:
        return unwrapped["data"]
    fields = {
        "list_files": ("path", "entries", "is_file"),
        "read_file": (
            "path", "content", "total_chars", "offset", "read_chars", "is_truncated",
        ),
    }.get(tool_name, ())
    data = {field: unwrapped[field] for field in fields if field in unwrapped}
    return data or None

def _extract_integrity_and_pagination(
    raw_content: str,
    unwrapped: object | None,
) -> tuple[ContentIntegrity, PaginationMetadata | None]:
    is_truncated = False
    total_items = 0
    returned_items = 0
    offset = 0
    limit = None
    has_pagination = False
    has_more = False

    if isinstance(unwrapped, dict):
        if "is_truncated" in unwrapped:
            is_truncated = bool(unwrapped["is_truncated"])
        elif "content_truncated" in unwrapped:
            is_truncated = bool(unwrapped["content_truncated"])

        if "offset" in unwrapped and isinstance(unwrapped["offset"], int):
            offset = unwrapped["offset"]
            has_pagination = True

        if "limit" in unwrapped and isinstance(unwrapped["limit"], int):
            limit = unwrapped["limit"]
            has_pagination = True

        if "total_chars" in unwrapped and isinstance(unwrapped["total_chars"], int):
            total_items = unwrapped["total_chars"]
            has_pagination = True
        elif "total_items" in unwrapped and isinstance(unwrapped["total_items"], int):
            total_items = unwrapped["total_items"]
            has_pagination = True

        if "read_chars" in unwrapped and isinstance(unwrapped["read_chars"], int):
            returned_items = unwrapped["read_chars"]
            has_pagination = True
        elif "returned_items" in unwrapped and isinstance(unwrapped["returned_items"], int):
            returned_items = unwrapped["returned_items"]
            has_pagination = True

        if "has_more" in unwrapped:
            has_more = bool(unwrapped["has_more"])
            has_pagination = True
        else:
            has_more = is_truncated or (total_items > 0 and (offset + returned_items) < total_items)

    if not is_truncated:
        if "[TRUNCATED]" in raw_content or "...[truncated]" in raw_content:
            is_truncated = True

    integrity = ContentIntegrity(
        is_truncated=is_truncated,
        original_bytes=total_items if total_items > 0 else len(raw_content),
        captured_bytes=returned_items if returned_items > 0 else len(raw_content),
        stdout_truncated=is_truncated,
        stderr_truncated=False,
    )

    pagination = None
    if has_pagination or is_truncated:
        pagination = PaginationMetadata(
            has_more=is_truncated or has_more,
            total_items=total_items,
            returned_items=returned_items,
            offset=offset,
            limit=limit,
        )

    return integrity, pagination

def _normalize_transport_payload(raw_content: str, *, tool_name: str = "") -> NormalizedToolPayload:
    parsed = parse_tool_output(raw_content)
    unwrapped = unwrap_tool_output(raw_content)
    integrity, pagination = _extract_integrity_and_pagination(raw_content, unwrapped)

    success = (
        parsed.success
        if parsed is not None
        else bool(isinstance(unwrapped, dict) and unwrapped.get("success") is True)
    )

    if isinstance(unwrapped, dict):
        rendered_output = format_tool_result_response(unwrapped).strip() or str(unwrapped)
        return NormalizedToolPayload(
            success=success,
            message=str(unwrapped.get("message", "")),
            data=_structured_result_data(unwrapped, tool_name),
            rendered_output=rendered_output,
            error_code=unwrapped.get("error_code"),
            integrity=integrity,
            pagination=pagination,
            is_async_job=bool(unwrapped.get("is_async_job", False)),
            async_job_id=unwrapped.get("async_job_id"),
            async_job_status=unwrapped.get("async_job_status"),
            async_terminal=bool(unwrapped.get("async_terminal", False)),
            async_observed_at_utc=unwrapped.get("async_observed_at_utc"),
            artifacts=unwrapped.get("artifacts", ()),
        )

    if isinstance(unwrapped, list):
        text = str(unwrapped)
        return NormalizedToolPayload(
            success=success,
            message=text,
            data=unwrapped,
            rendered_output=text,
            error_code=None,
            integrity=integrity,
            pagination=pagination,
            is_async_job=False,
            async_job_id=None,
            async_job_status=None,
            async_terminal=False,
            async_observed_at_utc=None,
        )

    if isinstance(unwrapped, str):
        return NormalizedToolPayload(
            success=success,
            message=unwrapped,
            data=None,
            rendered_output=unwrapped,
            error_code=None,
            integrity=integrity,
            pagination=pagination,
            is_async_job=False,
            async_job_id=None,
            async_job_status=None,
            async_terminal=False,
            async_observed_at_utc=None,
        )

    return NormalizedToolPayload(
        success=success,
        message=raw_content,
        data=None,
        rendered_output=str(raw_content or ""),
        error_code=None,
        integrity=integrity,
        pagination=pagination,
        is_async_job=False,
        async_job_id=None,
        async_job_status=None,
        async_terminal=False,
        async_observed_at_utc=None,
    )

def normalize_tool_output(*, raw_content: str, request: ToolRequest) -> ToolResult:
    """Normalize an existing serialized tool envelope without graph semantics."""

    payload = _normalize_transport_payload(raw_content, tool_name=request.tool_name)
    signature = _build_tool_signature(request)

    return ToolResult(
        request_id=request.request_id,
        signature=signature,
        success=payload.success,
        message=payload.message,
        data=payload.data,
        rendered_output=payload.rendered_output,
        error_code=payload.error_code,
        integrity=payload.integrity,
        pagination=payload.pagination,
        is_async_job=payload.is_async_job,
        async_job_id=payload.async_job_id,
        async_job_status=payload.async_job_status,
        async_terminal=payload.async_terminal,
        async_observed_at_utc=payload.async_observed_at_utc,
        artifacts=payload.artifacts
    )


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


def parse_tool_output(raw: Any) -> ToolOutputEnvelope | None:
    """Return a ToolOutputEnvelope if the raw value contains a valid structured payload."""
    return ToolOutputEnvelope.try_parse(raw)


def unwrap_tool_output(raw: Any) -> dict[str, Any] | list[Any] | str | None:
    """Unwrap summary-plus-JSON tool output into Python values."""
    return ToolOutputEnvelope.unwrap_tool_output(raw)