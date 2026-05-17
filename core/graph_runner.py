import json

from langchain_core.messages import AIMessage, HumanMessage

from core.graph_constants import ANSI_GREEN, ANSI_ITALIC, ANSI_RED, ANSI_RESET, MAX_REASONING_STEPS
from core.graph_messages import normalize_message_content
from core.graph_response_formatters import format_tool_call_preview
from core.models import ToolResult
from core.state import AgentState
from core.tool_output import parse_tool_result


def _print_raw_llm_response(raw_text: str) -> None:
    text = (raw_text or "").strip()
    if not text:
        text = "<empty>"
    print(f"{ANSI_RED}{ANSI_ITALIC}[raw-llm]{ANSI_RESET}")
    print(f"{ANSI_RED}{ANSI_ITALIC}{text}{ANSI_RESET}")


def _raw_ai_message_payload(message: AIMessage) -> str:
    payload: dict[str, object] = {
        "content": getattr(message, "content", ""),
    }
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return json.dumps(payload, ensure_ascii=True, indent=2, default=str)


def _is_llm_generated_message(message: AIMessage) -> bool:
    meta = getattr(message, "response_metadata", None)
    if isinstance(meta, dict) and bool(meta):
        return True
    # Ollama tool-call responses often have empty response_metadata but
    # always have empty content (all data is in tool_calls).
    # Our manually-constructed deterministic messages always have non-empty content.
    tool_calls = getattr(message, "tool_calls", None) or []
    content = (getattr(message, "content", "") or "").strip()
    return bool(tool_calls) and not content


def run_prompt(
    app,
    prompt: str,
    history: list | None = None,
    rolling_summary: str = "",
    show_raw_llm: bool = False,
) -> tuple[list, str]:
    """Run a single prompt and return updated history with rolling summary."""
    prior_messages = history or []
    initial_state: AgentState = {
        "messages": [*prior_messages, HumanMessage(content=prompt)],
        "steps": 0,
        "plan": "",
        "planner_plan_source": "",
        "planner_route": "",
        "rolling_summary": rolling_summary,
        "last_tool_output": "",
        "last_tool_signature": "",
        "last_tool_success": True,
        "repeat_fail_count": 0,
        "tool_text_retry_used": False,
    }

    final_messages = list(initial_state["messages"])
    latest_step_count = 0
    saw_pseudo_stop = False
    latest_summary = ""
    saw_action_stop = False
    events = app.stream(initial_state)
    for event in events:
        for node_name, value in event.items():
            if isinstance(value, dict):
                latest_step_count = int(value.get("steps", latest_step_count) or 0)
                if "rolling_summary" in value:
                    latest_summary = str(value.get("rolling_summary") or "")

                if node_name == "planner" and value.get("plan"):
                    current_route = str(value.get("planner_route") or "")
                    header = "[planner]"
                    if current_route:
                        header = f"[planner:{current_route}]"
                    print(f"\n{ANSI_GREEN}{header}{ANSI_RESET}")
                    print(str(value.get("plan")))
                    plan_source = str(value.get("planner_plan_source") or "")
                    if show_raw_llm and plan_source == "llm":
                        _print_raw_llm_response(str(value.get("plan")))

            messages = value.get("messages") if isinstance(value, dict) else None
            if not messages:
                continue

            message = messages[-1]
            final_messages.append(message)
            print(f"\n[{node_name}]")
            if show_raw_llm and isinstance(message, AIMessage) and _is_llm_generated_message(message):
                _print_raw_llm_response(_raw_ai_message_payload(message))
            if getattr(message, "tool_calls", None):
                print(format_tool_call_preview(message))
            else:
                raw_content = normalize_message_content(message)
                if "pseudo tool-call text" in raw_content:
                    saw_pseudo_stop = True
                if "Action-required run stopped" in raw_content:
                    saw_action_stop = True
                summary, _ = ToolResult.split_tool_output(raw_content)
                parsed = parse_tool_result(raw_content)
                if parsed is not None:
                    print(parsed.to_pretty_text())
                elif summary:
                    print(summary)
                else:
                    print(raw_content)

    if latest_step_count >= MAX_REASONING_STEPS:
        print(
            f"\n[system]\nMax reasoning steps reached ({MAX_REASONING_STEPS}). "
            "Stopping to avoid unbounded loops."
        )

    if saw_pseudo_stop:
        print(
            "\n[system]\nThe model returned pseudo tool syntax, so the run was halted without executing those actions."
        )

    if saw_action_stop:
        print(
            "\n[system]\nRun ended without a final executable tool call. See the last [brain] message for the stop reason."
        )

    return final_messages, latest_summary
