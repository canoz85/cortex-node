import json
import logging
import uuid
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from core.application_session import bounded_recent_conversation
from core.graph_constants import ANSI_BLUE, ANSI_RESET, MAX_REASONING_STEPS
from core.graph_messages import (
    ACCEPTED_FINALIZER_PROVENANCE, CONVERSATION_PROVENANCE_KEY,
    conversational_messages,
)
from core.logging.node_update import extract_node_update
from core.logging.renderer import render_node_update
from core.logging.live_status import LiveStatus
from core.logging_utils import get_logger, log_event
from core.memory.terminal import AcceptedCompletion, CompletedTurnEvidence
from core.memory import TurnStatus
from core.protocol.controller import CortexController
from core.protocol.enums import ControllerDecisionType, ExecutionStatus
from core.protocol.models import (
    AsyncJobPolicy, ControllerDecision, ExecutionState, FinalizationResult,
    PlannerMemoryContext,
)
from core.runtime.async_wake import AsyncExecutionWake
from core.state import AgentState

logger = get_logger(__name__)

def _accepted_finalizer_message(result: FinalizationResult) -> AIMessage:
    return AIMessage(
        content=result.final_answer,
        additional_kwargs={CONVERSATION_PROVENANCE_KEY: ACCEPTED_FINALIZER_PROVENANCE},
    )


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


@dataclass(frozen=True)
class PendingClarification:
    """Application-boundary handle for one Controller-paused execution."""

    execution_state: ExecutionState

    @property
    def run_id(self) -> str:
        return self.execution_state.protocol_visible.identity.execution_id

    @property
    def prompt(self) -> str:
        return self.execution_state.protocol_visible.planning_clarification.prompt

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
    completed_turn_evidence: list[CompletedTurnEvidence] | None = None,
    turn_index: int = 1,
    planner_memory_context: PlannerMemoryContext | None = None,
    verbose: bool = False,
    live_status: LiveStatus | None = None,
    pending_clarification: PendingClarification | None = None,
    clarification_sink: list[PendingClarification] | None = None,
) -> tuple[list, str]:
    prior_messages = list(bounded_recent_conversation(conversational_messages(history or [])))
    if pending_clarification is not None:
        if run_id is not None and run_id != pending_clarification.run_id:
            raise ValueError("clarification resume run_id does not match paused execution")
        run_id = pending_clarification.run_id
    else:
        run_id = run_id or uuid.uuid4().hex[:12]
    started_at = perf_counter()
    owns_live_status = live_status is None
    live_status = live_status or LiveStatus()
    live_status.start("planner")

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
        "user_input": prompt,
        "retrieval_messages": [],
        "run_id": run_id,
    }
    if async_job_policy is not None:
        initial_state["async_job_policy"] = async_job_policy
    if planner_memory_context is not None:
        initial_state["planner_memory_context"] = planner_memory_context

    execution_state = (
        pending_clarification.execution_state
        if pending_clarification is not None
        else CortexController.start_execution(run_id, async_job_policy)
    )
    initial_state["execution_state"] = execution_state

    conversation_history = list(initial_state["messages"])
    metrics = RunMetrics()
    from_node = ""
    terminal_decision: ControllerDecision | None = None
    terminal_state: ExecutionState | None = None
    terminal_finalization: FinalizationResult | None = None
    latest_execution_state = execution_state
    latest_controller_decision: ControllerDecision | None = None

    async_runtime = getattr(app, "async_runtime", None)
    if async_runtime is not None and pending_clarification is not None:
        # Session history is a replacement projection. Checkpoint add_messages
        # must not append recreated copies of the previous user turns.
        initial_state["messages"] = [
            RemoveMessage(id=REMOVE_ALL_MESSAGES), *conversation_history,
        ]
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

    try:
        while True:
            latest_controller_decision = None

            for event in events:
                for node_name, value in event.items():
                    if not isinstance(value, dict):
                        continue

                    decision = value.get("controller_decision")
                    if isinstance(decision, ControllerDecision):
                        latest_controller_decision = decision
                        event_state = value.get("execution_state")
                        if isinstance(event_state, ExecutionState):
                            latest_execution_state = event_state
                        if decision.terminal and isinstance(event_state, ExecutionState):
                            terminal_decision = decision
                            terminal_state = event_state

                    finalization_result = value.get("finalization_result")
                    if isinstance(finalization_result, FinalizationResult):
                        conversation_history.append(_accepted_finalizer_message(finalization_result))
                        terminal_finalization = finalization_result

                    node_update = extract_node_update(
                        from_node=from_node,
                        to_node=node_name,
                        value=value,
                    )

                    if node_update is None:
                        continue

                    metrics.node_updates += 1

                    render_node_update(node_update, verbose=verbose)

                    from_node = node_name

            if (
                async_runtime is None
                or latest_controller_decision is None
                or latest_controller_decision.decision_type
                != ControllerDecisionType.AWAIT_ASYNC_JOB
            ):
                break

            events = async_runtime.wake_and_resume(
                config=graph_config,
                wake=AsyncExecutionWake(
                    execution_id=execution_state.protocol_visible.identity.execution_id,
                    async_job_id=latest_controller_decision.async_job_id,
                ),
            )
    except Exception:
        live_status.stop()
        raise
    finally:
        if owns_live_status:
            live_status.stop()

    if show_summary and metrics.latest_summary.strip():
        print(f"\n{ANSI_BLUE}[summary]{ANSI_RESET}")
        pretty = _pretty_summary_text(metrics.latest_summary)
        print(f"{ANSI_BLUE}{pretty}{ANSI_RESET}")

    if clarification_sink is not None:
        clarification_sink.clear()
        marker = latest_execution_state.protocol_visible.planning_clarification
        if (
            latest_controller_decision is not None
            and latest_controller_decision.decision_type == ControllerDecisionType.PAUSE
            and marker is not None
        ):
            clarification_sink.append(PendingClarification(
                execution_state=latest_execution_state,
            ))

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

    if completed_turn_evidence is not None and terminal_decision is not None and terminal_state is not None:
        try:
            protocol = terminal_state.protocol_visible
            completed_turn_evidence.append(CompletedTurnEvidence(
                turn_id=run_id,
                turn_index=turn_index,
                user_request=protocol.original_user_request or prompt,
                execution_id=protocol.identity.execution_id,
                status=TurnStatus(protocol.status.value),
                direct_response=(
                    protocol.status == ExecutionStatus.COMPLETED
                    and protocol.active_plan is None
                    and not protocol.completion_provenance
                ),
                accepted_completions=tuple(
                    AcceptedCompletion(
                        step_id=item.step_id,
                        summary=item.summary,
                        evidence_id=item.evidence_id,
                        plan_id=item.plan_id,
                        plan_revision=item.plan_revision,
                    )
                    for item in protocol.completion_provenance
                ),
                terminal_reason=terminal_decision.failure_reason or terminal_decision.reason,
                accepted_answer=(
                    terminal_finalization.final_answer
                    if terminal_finalization is not None
                    and terminal_finalization.execution_summary.execution_id == protocol.identity.execution_id
                    else None
                ),
            ))
        except Exception as exc:
            logger.warning("Terminal memory evidence unavailable: %s", exc)

    return list(bounded_recent_conversation(conversation_history)), metrics.latest_summary
