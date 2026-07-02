from langchain_core.messages import ToolMessage

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

        # Guard: Only process state changes if the last node was an actual tool execution
        if not isinstance(last_message, ToolMessage):
            return {}  # Safe: Returns empty dict so existing state values remain pristine

        if isinstance(last_message, ToolMessage):
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
                return {
                    "last_tool_output": unwrapped,
                    "last_tool_rendered": rendered_output,
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            if isinstance(unwrapped, list):
                return {
                    "last_tool_output": {"message": str(unwrapped), "data": unwrapped, "success": success},
                    "last_tool_rendered": rendered_output,
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            if isinstance(unwrapped, str):
                return {
                    "last_tool_output": {"message": unwrapped, "data": None, "success": success},
                    "last_tool_rendered": rendered_output,
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            return {
                "last_tool_output": raw_content,
                "last_tool_rendered": rendered_output,
                "last_tool_signature": current_signature,
                "last_tool_success": success,
                "repeat_fail_count": repeat_fail_count,
            }
        return {
            "last_tool_output": state.get("last_tool_output", ""),
            "last_tool_rendered": state.get("last_tool_rendered", ""),
            "last_tool_signature": state.get("last_tool_signature", ""),
            "last_tool_success": state.get("last_tool_success", None),
            "repeat_fail_count": state.get("repeat_fail_count", 0),
        }

    return capture_tool_output_node