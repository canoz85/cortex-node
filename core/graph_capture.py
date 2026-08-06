
import uuid
from dataclasses import dataclass

from langchain_core.messages import ToolMessage

from core.protocol.bridge import build_execution_state, with_cursor, WorkerRole
from core.protocol.models import ToolResult
from core.graph_messages import normalize_message_content
from core.graph_response_formatters import format_tool_result_response
from core.graph_tool_events import extract_tool_signature
from core.state import AgentState
from core.tool_output import parse_tool_result, unwrap_tool_output


@dataclass(frozen=True, slots=True)
class NormalizedToolPayload:
    success: bool
    message: str
    data: object | None
    rendered_output: str
    error_code: str | None


def _normalize_transport_payload(raw_content: str) -> NormalizedToolPayload:
    parsed = parse_tool_result(raw_content)
    unwrapped = unwrap_tool_output(raw_content)

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
        )

    if isinstance(unwrapped, list):
        text = str(unwrapped)
        return NormalizedToolPayload(
            success=success,
            message=text,
            data=unwrapped,
            rendered_output=text,
            error_code=None,
        )

    if isinstance(unwrapped, str):
        return NormalizedToolPayload(
            success=success,
            message=unwrapped,
            data=None,
            rendered_output=unwrapped,
            error_code=None,
        )

    return NormalizedToolPayload(
        success=success,
        message=raw_content,
        data=None,
        rendered_output=str(raw_content or ""),
        error_code=None,
    )


def _build_tool_result(
    *,
    raw_content: str,
    tool_call_id: str | None,
    signature: str,
) -> ToolResult:
    payload = _normalize_transport_payload(raw_content)

    # TODO(protocol):
    # request_id should originate from ToolRequest.
    # Remove UUID fallback after protocol migration is complete.
    request_id = tool_call_id or signature or uuid.uuid4().hex

    return ToolResult(
        request_id=request_id,
        signature=signature,
        success=payload.success,
        message=payload.message,
        data=payload.data,
        rendered_output=payload.rendered_output,
        error_code=payload.error_code,
    )


def _compute_repeat_fail_count(
    *,
    previous: ToolResult | None,
    current: ToolResult,
    previous_repeat_count: int,
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
        history = state.get("messages", [])
        if not history:
            return {"repeat_fail_count": 0}

        last_message = history[-1]
        if not isinstance(last_message, ToolMessage):
            return {}

        tool_call_id = getattr(last_message, "tool_call_id", None)
        signature = extract_tool_signature(history[:-1], tool_call_id)
        raw_content = normalize_message_content(last_message)

        tool_result = _build_tool_result(
            raw_content=raw_content,
            tool_call_id=tool_call_id,
            signature=signature,
        )

        repeat_fail_count = _compute_repeat_fail_count(
            previous=state.get("last_tool_result"),
            current=tool_result,
            previous_repeat_count=state.get("repeat_fail_count", 0),
        )

        # updated_execution_state = with_cursor(
        #     build_execution_state(state),
        #     current_worker=WorkerRole.TOOL_RUNTIME,
        # )

        return {
            "last_tool_result": tool_result,
            # "execution_state": updated_execution_state,
            "brain_result": None,
            "repeat_fail_count": repeat_fail_count,
        }

    return capture_tool_output_node