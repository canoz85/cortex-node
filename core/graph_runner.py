import json
import logging
import uuid
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from core.error_codes import TOOL_UNSTRUCTURED_RESULT
from core.graph_constants import ANSI_BLUE, ANSI_CYAN, ANSI_GREEN, ANSI_ITALIC, ANSI_LIGHT_BLUE, ANSI_RED, ANSI_RESET, MAX_REASONING_STEPS
from core.graph_messages import normalize_message_content
from core.graph_response_formatters import format_tool_call_preview
from core.logging_utils import get_logger, log_event
from core.models import ToolResult
from core.protocol.bridge import legacy_state_to_execution_state
from core.runtime.accessors import get_execution_state
from core.state import AgentState
from core.tool_output import parse_tool_result, unwrap_tool_output


logger = get_logger(__name__)


@dataclass
class EventFrame:
    seq: int
    node: str
    step: int
    route: str
    has_messages: bool
    message_kind: str  # planner_plan | tool_call | tool_result | ai_text | none
    message: Any | None
    message_text: str
    tool_calls: list[dict[str, Any]]
    rolling_summary: str
    has_summary_update: bool


@dataclass
class RunMetrics:
    node_updates: int = 0
    tool_call_messages: int = 0
    tool_call_count: int = 0
    tool_result_messages: int = 0
    latest_step_count: int = 0
    latest_summary: str = ""
    terminal_node: str = ""
    terminal_message_kind: str = "none"
    planner_route: str = ""
    error_counts: dict[str, int] = field(default_factory=dict)


def _print_raw_llm_response(raw_text: str) -> None:
    text = (raw_text or "").strip() or "<empty>"
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
    tool_calls = getattr(message, "tool_calls", None) or []
    content = (getattr(message, "content", "") or "").strip()
    return bool(tool_calls) and not content

def _pretty_summary_text(raw_summary: str) -> str:
    def _compact_value(v: Any) -> str:
        if isinstance(v, dict):
            return ", ".join(
                f'"{k}": {json.dumps(val, ensure_ascii=True)}'
                for k, val in v.items()
            )

        if isinstance(v, list):
            if not v:
                return "[]"

            chunks: list[str] = []
            for item in v:
                if isinstance(item, dict):
                    chunks.append(_compact_value(item))
                else:
                    chunks.append(json.dumps(item, ensure_ascii=True))
                    
            return "\n  - " + "\n  - ".join(chunks)
        
        return json.dumps(v, ensure_ascii=True)

    text = (raw_summary or "").strip()
    if not text:
        return ""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text

    if not isinstance(payload, dict):
        return json.dumps(payload, ensure_ascii=True)

    lines: list[str] = []
    for key, value in payload.items():
        lines.append(f"{key}: {_compact_value(value)}")

    return "\n".join(lines)

def _render_non_tool_text(*, raw_content: str) -> bool:
    """
    Render non-tool-call content.
    Returns True if content is a structured tool result.
    """
    unwrapped = unwrap_tool_output(raw_content)
    if isinstance(unwrapped, dict) and ("success" in unwrapped or "message" in unwrapped):
        # Use existing formatter logic if needed:
        # print(format_tool_result_response(unwrapped))
        summary, _ = ToolResult.split_tool_output(raw_content)
        print(summary or str(unwrapped))
        return True

    parsed = parse_tool_result(raw_content)
    if parsed is not None:
        print(parsed.to_pretty_text())
        return True

    summary, _ = ToolResult.split_tool_output(raw_content)
    print(summary or raw_content)
    return False


def _build_event_frame(
    *,
    seq: int,
    node_name: str,
    value: dict[str, Any],
    latest_step_count: int,
    latest_summary: str,
) -> EventFrame:
    step = int(value.get("steps", latest_step_count) or 0)
    route = str(value.get("planner_route") or "")

    has_summary_update = "rolling_summary" in value
    summary_now = str(value.get("rolling_summary") or "")
    rolling_summary = summary_now if has_summary_update else latest_summary

    messages = value.get("messages")
    if not messages:
        kind = "planner_plan" if node_name == "planner" and bool(value.get("plan")) else "none"
        text = str(value.get("plan") or "") if kind == "planner_plan" else ""
        return EventFrame(
            seq=seq,
            node=node_name,
            step=step,
            route=route,
            has_messages=False,
            message_kind=kind,
            message=None,
            message_text=text,
            tool_calls=[],
            rolling_summary=rolling_summary,
            has_summary_update=has_summary_update,
        )

    message = messages[-1]
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    if tool_calls:
        kind = "tool_call"
        text = ""
    else:
        text = normalize_message_content(message)
        unwrapped = unwrap_tool_output(text)
        parsed = parse_tool_result(text)
        kind = "tool_result" if (parsed is not None or isinstance(unwrapped, dict)) else "ai_text"

    return EventFrame(
        seq=seq,
        node=node_name,
        step=step,
        route=route,
        has_messages=True,
        message_kind=kind,
        message=message,
        message_text=text,
        tool_calls=tool_calls,
        rolling_summary=rolling_summary,
        has_summary_update=has_summary_update,
    )


def _render_frame(*, frame: EventFrame) -> None:

    if frame.message_kind == "none":
        return
    
    color = ANSI_GREEN if frame.node == "planner" else ANSI_CYAN

    if frame.message_kind == "planner_plan":
        header = "[planner]"
        if frame.route:
            header = f"[planner:{frame.route}]"
        print(f"\n{color}{header}{ANSI_RESET}")
        print(frame.message_text)
        return

    print(f"\n{color}[{frame.node}]{ANSI_RESET}")

    if frame.message_kind == "tool_call":
        print(format_tool_call_preview(frame.message))
        return

    if frame.message_kind == "tool_result":
        _render_non_tool_text(raw_content=frame.message_text)
        return
    
    color = ANSI_LIGHT_BLUE

    # ai_text
    print(f"{color}{frame.message_text}{ANSI_RESET}")


def _log_tool_calls(*, run_id: str, node_name: str, tool_calls: list[dict[str, Any]]) -> None:
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


def _update_metrics_from_frame(*, frame: EventFrame, metrics: RunMetrics) -> None:
    metrics.node_updates += 1
    metrics.latest_step_count = frame.step
    metrics.latest_summary = frame.rolling_summary
    metrics.terminal_node = frame.node
    metrics.terminal_message_kind = frame.message_kind
    if frame.route:
        metrics.planner_route = frame.route

    if frame.message_kind == "tool_call":
        metrics.tool_call_messages += 1
        metrics.tool_call_count += len(frame.tool_calls)

    if frame.message_kind == "tool_result":
        metrics.tool_result_messages += 1


def _log_frame(*, frame: EventFrame, run_id: str, prev_node: str) -> None:
    log_event(
        logger,
        logging.INFO,
        "Graph node update",
        event_name="graph_node_update",
        run_id=run_id,
        event_seq=frame.seq,
        node=frame.node,
        prev_node=prev_node,
        transition=f"{prev_node}->{frame.node}" if prev_node else frame.node,
        steps=frame.step,
        message_kind=frame.message_kind,
        tool_call_count=len(frame.tool_calls),
        has_summary_update=frame.has_summary_update,
        planner_route=frame.route or None,
    )


def _derive_stop_reason(*, metrics: RunMetrics, last_text: str) -> tuple[str, bool, bool]:
    if metrics.latest_step_count >= MAX_REASONING_STEPS:
        return "max_steps", False, False

    text = (last_text or "").lower()
    pseudo_hit = (
        "pseudo tool-call text instead of executable tool calls" in text
        or "pseudo tool syntax" in text
    )
    action_stop_hit = "action-required run stopped" in text

    if pseudo_hit:
        return "pseudo_tool", True, False
    if action_stop_hit:
        return "action_stop", False, True
    return "completed", False, False


def _extract_error_code(*, parsed: ToolResult | None, unwrapped: Any) -> str | None:
    if parsed is not None and isinstance(parsed.error_code, str) and parsed.error_code.strip():
        return parsed.error_code.strip()

    if isinstance(unwrapped, dict):
        maybe_error = unwrapped.get("error_code")
        if isinstance(maybe_error, str) and maybe_error.strip():
            return maybe_error.strip()

        if unwrapped.get("success") is False:
            return TOOL_UNSTRUCTURED_RESULT

    return None

def _collect_tool_result_event(
    *,
    raw_content: str,
    metrics: RunMetrics,
) -> tuple[bool, str | None, str]:
    parsed = parse_tool_result(raw_content)
    unwrapped = unwrap_tool_output(raw_content)

    success = (
        parsed.success
        if parsed is not None
        else bool(isinstance(unwrapped, dict) and unwrapped.get("success") is True)
    )
    error_code = _extract_error_code(parsed=parsed, unwrapped=unwrapped)
    if error_code:
        metrics.error_counts[error_code] = metrics.error_counts.get(error_code, 0) + 1

    message_text = (
        parsed.message
        if parsed is not None
        else str(unwrapped.get("message", "")) if isinstance(unwrapped, dict) else ""
    )
    return success, error_code, message_text

def run_prompt(
    app,
    prompt: str,
    history: list | None = None,
    rolling_summary: str = "",
    show_summary: bool = False,
) -> tuple[list, str]:
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
        "retrieval_messages": [],
        "last_tool_output": "",
        "last_tool_rendered": "",
        "last_tool_signature": "",
        "last_tool_success": True,
        "repeat_fail_count": 0,
        "tool_text_retry_used": False,
        "run_id": run_id,
    }

    # Migration boundary: legacy runtime state and protocol ExecutionState coexist here.
    # The protocol state is read-only and mirrors the same legacy inputs without
    # changing execution order, routing, or worker behavior.
    execution_state = legacy_state_to_execution_state(initial_state)
    initial_state["execution_state"] = execution_state
    _ = get_execution_state(initial_state)

    final_messages = list(initial_state["messages"])
    metrics = RunMetrics()
    seq = 0
    prev_node = ""
    last_text = ""

    events = app.stream(initial_state)

    for event in events:
        for node_name, value in event.items():
            if not isinstance(value, dict):
                continue

            seq += 1
            frame = _build_event_frame(
                seq=seq,
                node_name=node_name,
                value=value,
                latest_step_count=metrics.latest_step_count,
                latest_summary=metrics.latest_summary,
            )

            _render_frame(frame=frame)
            _update_metrics_from_frame(frame=frame, metrics=metrics)
            _log_frame(frame=frame, run_id=run_id, prev_node=prev_node)

            if frame.has_messages and frame.message is not None:
                final_messages.append(frame.message)

                if frame.message_kind == "tool_call":
                    _log_tool_calls(
                        run_id=run_id,
                        node_name=frame.node,
                        tool_calls=frame.tool_calls,
                    )
                else:
                    last_text = frame.message_text

                    if frame.message_kind == "tool_result":
                        # Optional per-tool-result event if you want to preserve this signal.
                        success, error_code, message_text = _collect_tool_result_event(
                            raw_content=frame.message_text,
                            metrics=metrics,
                        )
                        log_event(
                            logger,
                            logging.INFO,
                            "Tool result captured",
                            event_name="tool_result",
                            run_id=run_id,
                            node=frame.node,
                            success=success,
                            error_code=error_code,
                            tool_message=message_text,
                        )

            prev_node = frame.node

    if metrics.latest_step_count >= MAX_REASONING_STEPS:
        print(
            f"\n{ANSI_BLUE}[system]{ANSI_RESET}\n"
            f"{ANSI_BLUE}Max reasoning steps reached ({MAX_REASONING_STEPS}). "
            f"Stopping to avoid unbounded loops.{ANSI_RESET}"
        )

    if show_summary and metrics.latest_summary.strip():
        print(f"\n{ANSI_BLUE}[summary]{ANSI_RESET}")
        pretty = _pretty_summary_text(metrics.latest_summary)
        print(f"{ANSI_BLUE}{pretty}{ANSI_RESET}")

    stop_reason, saw_pseudo_stop, saw_action_stop = _derive_stop_reason(metrics=metrics, last_text=last_text)
    duration_ms = round((perf_counter() - started_at) * 1000.0, 3)

    log_event(
        logger,
        logging.INFO,
        "Prompt completed",
        event_name="prompt_completed",
        run_id=run_id,
        steps=metrics.latest_step_count,
        node_updates=metrics.node_updates,
        tool_call_messages=metrics.tool_call_messages,
        tool_call_count=metrics.tool_call_count,
        tool_result_messages=metrics.tool_result_messages,
        duration_ms=duration_ms,
        stop_reason=stop_reason,
        max_steps_reached=metrics.latest_step_count >= MAX_REASONING_STEPS,
        stopped_on_pseudo_call=saw_pseudo_stop,
        stopped_on_action_stop=saw_action_stop,
        terminal_node=metrics.terminal_node,
        terminal_message_kind=metrics.terminal_message_kind,
        planner_route=metrics.planner_route or None,
        error_counts=metrics.error_counts or None,
    )

    return final_messages, metrics.latest_summary