import json
import logging
import uuid
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from langchain_core.messages import HumanMessage

from core.graph_constants import ANSI_BLUE, ANSI_CYAN, ANSI_GREEN, ANSI_ITALIC, ANSI_LIGHT_BLUE, ANSI_RED, ANSI_RESET, MAX_REASONING_STEPS
from core.logging.node_update import extract_node_update
from core.logging.renderer import render_node_update
from core.logging_utils import get_logger, log_event
from core.protocol.bridge import legacy_state_to_execution_state
from core.protocol.enums import ControllerDecisionType
from core.protocol.models import AsyncJobPolicy, ControllerDecision
from core.runtime.accessors import get_execution_state
from core.state import AgentState

logger = get_logger(__name__)


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





def run_prompt(
    app,
    prompt: str,
    history: list | None = None,
    rolling_summary: str = "",
    show_summary: bool = False,
    run_id: str | None = None,
    async_job_policy: AsyncJobPolicy | None = None,
) -> tuple[list, str]:
    prior_messages = history or []
    run_id = run_id or uuid.uuid4().hex[:12]
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
        "tool_text_retry_used": False,
        "run_id": run_id,
    }
    if async_job_policy is not None:
        initial_state["async_job_policy"] = async_job_policy

    # Migration boundary: legacy runtime state and protocol ExecutionState coexist here.
    # The protocol state is read-only and mirrors the same legacy inputs without
    # changing execution order, routing, or worker behavior.
    execution_state = legacy_state_to_execution_state(initial_state)
    initial_state["execution_state"] = execution_state
    _ = get_execution_state(initial_state)

    final_messages = list(initial_state["messages"])
    metrics = RunMetrics()
    from_node = ""

    async_runtime = getattr(app, "async_runtime", None)
    graph_config = {
        "configurable": {
            "thread_id": run_id,
        }
    }
    events = (
        app.stream(initial_state, config=graph_config)
        if async_runtime is not None
        else app.stream(initial_state)
    )

    while True:
        latest_controller_decision: ControllerDecision | None = None

        for event in events:
            for node_name, value in event.items():
                if not isinstance(value, dict):
                    continue

                decision = value.get("controller_decision")
                if isinstance(decision, ControllerDecision):
                    latest_controller_decision = decision

                node_messages = value.get("messages")
                if node_messages:
                    final_messages.extend(node_messages)

                node_update = extract_node_update(
                    from_node=from_node,
                    to_node=node_name,
                    value=value,
                )

                if node_update is None:
                    continue

                if node_update.transition:
                    log_event(
                        logger,
                        logging.INFO,
                        "Graph node update",
                        node=node_update.to_node,
                        transition=node_update.transition,
                    )

                render_node_update(node_update)

                from_node = node_name

        if (
            async_runtime is None
            or latest_controller_decision is None
            or latest_controller_decision.decision_type
            != ControllerDecisionType.AWAIT_ASYNC_JOB
        ):
            break

        events = async_runtime.poll_and_resume(
            config=graph_config,
            decision=latest_controller_decision,
        )

    # if metrics.latest_step_count >= MAX_REASONING_STEPS:
    #     print(
    #         f"\n{ANSI_BLUE}[system]{ANSI_RESET}\n"
    #         f"{ANSI_BLUE}Max reasoning steps reached ({MAX_REASONING_STEPS}). "
    #         f"Stopping to avoid unbounded loops.{ANSI_RESET}"
    #     )

    if show_summary and metrics.latest_summary.strip():
        print(f"\n{ANSI_BLUE}[summary]{ANSI_RESET}")
        pretty = _pretty_summary_text(metrics.latest_summary)
        print(f"{ANSI_BLUE}{pretty}{ANSI_RESET}")

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
        max_steps_reached=metrics.latest_step_count >= MAX_REASONING_STEPS,
        terminal_node=metrics.terminal_node,
        terminal_message_kind=metrics.terminal_message_kind,
        planner_route=metrics.planner_route or None,
        error_counts=metrics.error_counts or None,
    )

    return final_messages, metrics.latest_summary
