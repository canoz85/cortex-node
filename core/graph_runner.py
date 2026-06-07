import json
import logging
import uuid
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage

from core.graph_constants import ANSI_BLUE, ANSI_GREEN, ANSI_ITALIC, ANSI_RED, ANSI_RESET, MAX_REASONING_STEPS
from core.graph_messages import normalize_message_content
from core.graph_response_formatters import format_preferred_tool_response, format_tool_call_preview
from core.logging_utils import get_logger, log_event
from core.models import ToolResult
from core.state import AgentState
from core.tool_output import parse_tool_result, unwrap_tool_output


logger = get_logger(__name__)


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


def _handle_non_tool_call_message(
    *,
    raw_content: str,
    logger_obj,
    run_id: str,
    node_name: str,
) -> bool:
    """Render one non-tool-call message and log tool-result events when present.

    Returns True if a structured tool result was captured, otherwise False.
    """
    unwrapped = unwrap_tool_output(raw_content)
    if isinstance(unwrapped, dict) and ("success" in unwrapped or "message" in unwrapped):
        print(format_preferred_tool_response(unwrapped))
        log_event(
            logger_obj,
            logging.INFO,
            "Tool result captured",
            event_name="tool_result",
            run_id=run_id,
            node=node_name,
            success=bool(unwrapped.get("success", False)),
            tool_message=str(unwrapped.get("message", "")),
        )
        return True

    parsed = parse_tool_result(raw_content)
    if parsed is not None:
        print(parsed.to_pretty_text())
        log_event(
            logger_obj,
            logging.INFO,
            "Tool result captured",
            event_name="tool_result",
            run_id=run_id,
            node=node_name,
            success=parsed.success,
            tool_message=parsed.message,
        )
        return True

    summary, _ = ToolResult.split_tool_output(raw_content)
    if summary:
        print(summary)
    else:
        print(raw_content)
    return False


def run_prompt(
    app,
    prompt: str,
    history: list | None = None,
    rolling_summary: str = "",
    show_raw_llm: bool = False,
    show_summary: bool = False,
) -> tuple[list, str]:
    """Run a single prompt and return updated history with rolling summary."""
    prior_messages = history or []
    run_id = uuid.uuid4().hex[:12]
    started_at = perf_counter()
    log_event(
        logger,
        logging.INFO,
        "Prompt received",
        event_name="prompt_received",
        run_id=run_id,
        prompt_chars=len(prompt or ""),
        history_messages=len(prior_messages),
    )

    initial_state: AgentState = {
        "messages": [*prior_messages, HumanMessage(content=prompt)],
        "steps": 0,
        "plan": "",
        "planner_route": "",
        "rolling_summary": rolling_summary,
        "last_tool_output": "",
        "last_tool_rendered": "",
        "last_tool_signature": "",
        "last_tool_success": True,
        "repeat_fail_count": 0,
        "tool_text_retry_used": False,
        "run_id": run_id,
    }

    final_messages = list(initial_state["messages"])
    latest_step_count = 0
    saw_pseudo_stop = False
    latest_summary = ""
    saw_action_stop = False
    node_updates = 0
    tool_call_messages = 0
    tool_result_messages = 0
    tool_call_count = 0
    events = app.stream(initial_state)
    for event in events:
        for node_name, value in event.items():
            if isinstance(value, dict):
                node_updates += 1
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
                    if show_raw_llm:
                        _print_raw_llm_response(str(value.get("plan")))

                log_event(
                    logger,
                    logging.INFO,
                    "Graph node update",
                    event_name="graph_node_update",
                    run_id=run_id,
                    node=node_name,
                    steps=latest_step_count,
                )

            messages = value.get("messages") if isinstance(value, dict) else None
            if not messages:
                continue

            message = messages[-1]
            final_messages.append(message)
            print(f"\n[{node_name}]")
            if show_raw_llm and isinstance(message, AIMessage) and _is_llm_generated_message(message):
                _print_raw_llm_response(_raw_ai_message_payload(message))
            if getattr(message, "tool_calls", None):
                tool_call_messages += 1
                print(format_tool_call_preview(message))
                tool_calls = getattr(message, "tool_calls", None) or []
                tool_call_count += len(tool_calls)
                for call in tool_calls:
                    log_event(
                        logger,
                        logging.INFO,
                        "Tool call proposed",
                        event_name="tool_call_proposed",
                        run_id=run_id,
                        node=node_name,
                        tool_name=call.get("name"),
                    )
            else:
                raw_content = normalize_message_content(message)
                captured = _handle_non_tool_call_message(
                    raw_content=raw_content,
                    logger_obj=logger,
                    run_id=run_id,
                    node_name=node_name,
                )
                if captured:
                    tool_result_messages += 1

    if latest_step_count >= MAX_REASONING_STEPS:
        print(
            f"\n{ANSI_BLUE}[system]{ANSI_RESET}\n"
            f"{ANSI_BLUE}Max reasoning steps reached ({MAX_REASONING_STEPS}). "
            f"Stopping to avoid unbounded loops.{ANSI_RESET}"
        )

    if show_summary and latest_summary.strip():
        print(f"\n{ANSI_BLUE}[summary]{ANSI_RESET}")
        print(f"{ANSI_BLUE}{latest_summary.strip()}{ANSI_RESET}")

    stop_reason = "max_steps" if latest_step_count >= MAX_REASONING_STEPS else "pseudo_tool" if saw_pseudo_stop else "action_stop" if saw_action_stop else "completed"
    duration_ms = round((perf_counter() - started_at) * 1000.0, 3)
    log_event(
        logger,
        logging.INFO,
        "Prompt completed",
        event_name="prompt_completed",
        run_id=run_id,
        steps=latest_step_count,
        node_updates=node_updates,
        tool_call_messages=tool_call_messages,
        tool_call_count=tool_call_count,
        tool_result_messages=tool_result_messages,
        duration_ms=duration_ms,
        stop_reason=stop_reason,
        max_steps_reached=latest_step_count >= MAX_REASONING_STEPS,
        stopped_on_pseudo_call=saw_pseudo_stop,
        stopped_on_action_stop=saw_action_stop,
    )

    return final_messages, latest_summary
