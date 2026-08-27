
import uuid
from dataclasses import dataclass

from langchain_core.messages import ToolMessage

from core.graph_node_helpers import build_tool_signature
from core.protocol.models import ArtifactRecord, ContentIntegrity, PaginationMetadata, ToolExecutionRecord, ToolRequest, ToolResult
from core.graph_messages import normalize_message_content, tool_message_content
from core.graph_response_formatters import format_tool_result_response
from core.state import AgentState
from core.tool_output import parse_tool_result, unwrap_tool_output


@dataclass(frozen=True, slots=True)
class NormalizedToolPayload:
    success: bool
    message: str
    data: object | None
    rendered_output: str
    error_code: str | None
    integrity: ContentIntegrity
    pagination: PaginationMetadata | None


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
            has_more=is_truncated or (total_items > 0 and (offset + returned_items) < total_items),
            total_items=total_items,
            returned_items=returned_items,
            offset=offset,
            limit=limit,
        )

    return integrity, pagination


def _extract_artifact_records(
    request: ToolRequest,
    unwrapped: object | None,
    step_id: str,
) -> tuple[ArtifactRecord, ...]:
    if not isinstance(unwrapped, dict):
        return ()

    path = unwrapped.get("path")
    if not isinstance(path, str) or not path.strip():
        path = request.arguments.get("path") if isinstance(request.arguments, dict) else None

    if not isinstance(path, str) or not path.strip():
        return ()

    tool_name = request.tool_name.lower()
    action = "modified"
    if "write" in tool_name or "create" in tool_name or "make" in tool_name:
        action = "created"
    elif "delete" in tool_name or "remove" in tool_name:
        action = "deleted"

    return (
        ArtifactRecord(
            artifact_id=f"art-{uuid.uuid4()}",
            step_id=step_id,
            path=path.strip(),
            action=action,
        ),
    )


def _normalize_transport_payload(raw_content: str) -> NormalizedToolPayload:
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
            data=unwrapped.get("data"),
            rendered_output=rendered_output,
            error_code=unwrapped.get("error_code"),
            integrity=integrity,
            pagination=pagination,
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
        )

    return NormalizedToolPayload(
        success=success,
        message=raw_content,
        data=None,
        rendered_output=str(raw_content or ""),
        error_code=None,
        integrity=integrity,
        pagination=pagination,
    )


def _build_tool_result(
    *,
    raw_content: str,
    request: ToolRequest,
) -> ToolResult:
    payload = _normalize_transport_payload(raw_content)

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
    )


def _compute_repeat_fail_count(
    *,
    previous: ToolResult | None,
    previous_repeat_count: int,
    current: ToolResult,
) -> int:
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


def create_capture_tool_output_node():
    def capture_tool_output_node(state: AgentState):

        execution_state = state["execution_state"]
        decision = state.get("controller_decision")
        history = state.get("messages", [])

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

        tool_result = _build_tool_result(
            raw_content=raw_content,
            request=decision.pending_tool_request,
        )

        working = execution_state.working
        active_step = execution_state.protocol_visible.active_step

        repeat_fail_count = _compute_repeat_fail_count(
            previous=working.last_tool_result,
            previous_repeat_count=working.repeat_fail_count,
            current=tool_result,
        )

        tool_execution_record = ToolExecutionRecord(
            step_id=active_step.step_id if active_step is not None else "",
            tool_name=decision.pending_tool_request.tool_name,
            arguments=decision.pending_tool_request.arguments,
            result=tool_result,
            artifacts=_extract_artifact_records(
                request=decision.pending_tool_request,
                unwrapped=unwrap_tool_output(raw_content),
                step_id=active_step.step_id if active_step is not None else "",
            ),
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