import uuid
from langchain_core.messages import ToolMessage

from core.protocol.bridge import build_execution_state, tool_result_to_legacy, with_cursor, WorkerRole
from core.protocol.models import ToolResult

from core.graph_messages import normalize_message_content
from core.graph_response_formatters import format_tool_result_response
from core.graph_tool_events import extract_tool_signature
from core.state import AgentState
from core.tool_output import parse_tool_result, unwrap_tool_output


def create_capture_tool_output_node():
    def _render_tool_output(unwrapped: object, raw_content: str) -> str:
        if isinstance(unwrapped, dict):
            rendered = format_tool_result_response(unwrapped).strip()
            return rendered or str(unwrapped)
        if isinstance(unwrapped, list):
            return str(unwrapped)
        if isinstance(unwrapped, str):
            return unwrapped
        return str(raw_content or "")
    
    def _build_tool_result(
        *,
        request_id: str,
        success: bool,
        message: str,
        data,
        error_code: str | None = None,
    ) -> ToolResult:
        return ToolResult(
            request_id=request_id,
            success=success,
            message=message,
            data=data,
            error_code=error_code,
        )

    def capture_tool_output_node(state: AgentState):
        history = state.get("messages", [])
        if not history:
            return {
                "last_tool_output": "",
                "last_tool_rendered": "",
                "last_tool_signature": "",
                "last_tool_success": None,
                "repeat_fail_count": 0,
            }

        last_message = history[-1]

        # Ignore non-tool messages.
        if not isinstance(last_message, ToolMessage):
            return {}
        
        raw_content = normalize_message_content(last_message)
        parsed = parse_tool_result(raw_content)
        unwrapped = unwrap_tool_output(raw_content)
        
        success = parsed.success if parsed is not None else bool(isinstance(unwrapped, dict) and unwrapped.get("success") is True)
        current_signature = extract_tool_signature(history[:-1], getattr(last_message, "tool_call_id", None))

        previous_signature = state.get("last_tool_signature", "")
        previous_success = state.get("last_tool_success", None)
        previous_repeat_count = state.get("repeat_fail_count", 0)
        
        if not success and current_signature and previous_signature == current_signature and previous_success is False:
            repeat_fail_count = previous_repeat_count + 1
        elif not success and current_signature:
            repeat_fail_count = 1
        else:
            repeat_fail_count = 0

        rendered_output = _render_tool_output(unwrapped, raw_content)

        if isinstance(unwrapped, dict):
            data = unwrapped.get("data")
            message = str(unwrapped.get("message", ""))
            error_code = unwrapped.get("error_code")
        elif isinstance(unwrapped, list):
            data = unwrapped
            message = str(unwrapped)
            error_code = None
        elif isinstance(unwrapped, str):
            data = None
            message = unwrapped
            error_code = None
        else:
            data = None
            message = raw_content
            error_code = None

        tool_result = _build_tool_result(
            request_id=(
                getattr(last_message, "tool_call_id", None)
                or current_signature
                or uuid.uuid4().hex
            ),
            success=success,
            message=message,
            data=data,
            error_code=error_code,
        )
        
        execution_state = build_execution_state(state)
        updated_execution_state = with_cursor(
            execution_state,
            current_worker=WorkerRole.TOOL_RUNTIME, 
        )

        return {
            "execution_state": updated_execution_state,
            "last_tool_result": tool_result,    # protocol
            "brain_result": None,               # Brain request has now been consumed.

            # Legacy compatibility
            "last_tool_output": tool_result.model_dump(),
            **tool_result_to_legacy(tool_result),

            # Diagnostics
            "last_tool_rendered": rendered_output,
            "last_tool_signature": current_signature,
            "last_tool_success": tool_result.success,
            "repeat_fail_count": repeat_fail_count,
        }

    return capture_tool_output_node