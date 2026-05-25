from langchain_core.messages import ToolMessage

from core.graph_messages import normalize_message_content
from core.graph_tool_events import extract_tool_signature
from core.state import AgentState
from core.tool_output import parse_tool_result, unwrap_tool_output


def create_capture_tool_output_node():
    def capture_tool_output_node(state: AgentState):
        history = state.get("messages", [])
        if not history:
            return {
                "last_tool_output": "",
                "last_tool_signature": "",
                "last_tool_success": True,
                "repeat_fail_count": 0,
            }

        last_message = history[-1]
        if isinstance(last_message, ToolMessage):
            raw_content = normalize_message_content(last_message)
            parsed = parse_tool_result(raw_content)
            unwrapped = unwrap_tool_output(raw_content)
            success = parsed.success if parsed is not None else bool(isinstance(unwrapped, dict) and unwrapped.get("success") is True)
            current_signature = extract_tool_signature(history[:-1], getattr(last_message, "tool_call_id", None))

            previous_signature = state.get("last_tool_signature", "")
            previous_success = state.get("last_tool_success", True)
            previous_repeat_count = state.get("repeat_fail_count", 0)
            if not success and current_signature and previous_signature == current_signature and not previous_success:
                repeat_fail_count = previous_repeat_count + 1
            elif not success and current_signature:
                repeat_fail_count = 1
            else:
                repeat_fail_count = 0

            if isinstance(unwrapped, dict):
                return {
                    "last_tool_output": unwrapped,
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            if isinstance(unwrapped, list):
                return {
                    "last_tool_output": {"message": str(unwrapped), "data": unwrapped, "success": success},
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            if isinstance(unwrapped, str):
                return {
                    "last_tool_output": {"message": unwrapped, "data": None, "success": success},
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            return {
                "last_tool_output": raw_content,
                "last_tool_signature": current_signature,
                "last_tool_success": success,
                "repeat_fail_count": repeat_fail_count,
            }
        return {
            "last_tool_output": state.get("last_tool_output", ""),
            "last_tool_signature": state.get("last_tool_signature", ""),
            "last_tool_success": state.get("last_tool_success", True),
            "repeat_fail_count": state.get("repeat_fail_count", 0),
        }

    return capture_tool_output_node