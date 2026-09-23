
import uuid
from dataclasses import dataclass

from langchain_core.messages import ToolMessage

from core.graph_messages import tool_message_content
from core.graph_node_helpers import build_tool_signature
from core.artifacts import ToolArtifact
from core.protocol.enums import AsyncJobStatus
from core.protocol.models import ArtifactRecord, ContentIntegrity, PaginationMetadata, ToolExecutionRecord, ToolRequest, ToolResult
from core.graph_response_formatters import format_tool_result_response
from core.state import AgentState
from core.tool_output import parse_tool_result, unwrap_tool_output, build_artifact_records, compute_repeat_fail_count


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


def _normalize_transport_payload(raw_content: str, *, tool_name: str = "") -> NormalizedToolPayload:
    parsed = parse_tool_result(raw_content)
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

def _build_tool_execution_record(
    *,
    execution_state,
    request: ToolRequest,
    result: ToolResult,
) -> ToolExecutionRecord:
    protocol = execution_state.protocol_visible
    active_step = protocol.active_step
    active_plan = protocol.active_plan
    step_id = active_step.step_id 

    return ToolExecutionRecord(
        execution_id=protocol.identity.execution_id,
        plan_id=active_plan.plan_id if active_plan is not None else None,
        plan_revision=active_plan.revision if active_plan is not None else None,
        step_id=step_id,
        tool_name=request.tool_name,
        arguments=request.arguments,
        result=result,
        artifacts=build_artifact_records(
            result.artifacts,
            step_id=step_id,
        ),
    )

def normalize_tool_output(*, raw_content: str, request: ToolRequest) -> ToolResult:
    """Normalize an existing serialized tool envelope without graph semantics."""

    payload = _normalize_transport_payload(raw_content, tool_name=request.tool_name)
    signature = build_tool_signature(request)

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

def create_capture_tool_output_node():
    def capture_tool_output_node(state: AgentState):

        execution_state = state["execution_state"]
        decision = state.get("controller_decision")
        history = state.get("messages", [])

        active_step = execution_state.protocol_visible.active_step
        if active_step is None:
            raise RuntimeError(
                "Capture executed without an active step."
            )

        if not history:
            return {}

        last_message = history[-1]
        if not isinstance(last_message, ToolMessage):
            return {}

        if decision is None or decision.pending_tool_request is None:
            raise RuntimeError(
                "Capture executed without pending_tool_request."
            )

        raw_content = tool_message_content(last_message)

        tool_result = normalize_tool_output(
            raw_content=raw_content,
            request=decision.pending_tool_request,
        )

        working = execution_state.working
        active_step = execution_state.protocol_visible.active_step

        repeat_fail_count = compute_repeat_fail_count(
            previous=working.last_tool_result,
            previous_repeat_count=working.repeat_fail_count,
            current=tool_result,
        )

        tool_execution_record = _build_tool_execution_record(
            execution_state=execution_state,
            request=decision.pending_tool_request,
            result=tool_result,
        )

        updated_history = (
            *working.tool_execution_history,
            tool_execution_record,
        )

        working = working.model_copy(
            update={
                "last_tool_result": tool_result,
                "tool_execution_history": updated_history,
                "repeat_fail_count": repeat_fail_count,
            }
        )

        updated_execution_state = execution_state.model_copy(
            update={
                "working": working,
            }
        )


        return {
            "execution_state": updated_execution_state,
        }

    return capture_tool_output_node
