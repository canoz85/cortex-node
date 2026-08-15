
import uuid
from dataclasses import dataclass

from langchain_core.messages import ToolMessage

from core.graph_node_helpers import build_tool_signature
from core.protocol.models import ToolExecutionRecord, ToolRequest, ToolResult
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
            return {"repeat_fail_count": 0}

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