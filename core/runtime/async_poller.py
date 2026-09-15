"""
Generic Async Job Polling Runtime Adapter for CortexNode.

This module provides background polling capabilities for any asynchronous 
provider workflow. It ensures state isolation by bypassing LLM inference 
during wait cycles and interacting strictly via execution evidence capture.
"""

import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.protocol.enums import ControllerDecisionType
from core.protocol.models import (
    ControllerDecision,
    ExecutionState,
    ToolRequest,
)
from core.protocol.bridge import build_controller_input
from core.graph_async_resume import LangGraphAsyncResumeAdapter
from core.runtime.gpu_resources import GpuResourceCoordinator, RuntimeGpuObserver
from core.runtime.async_wake import AsyncExecutionWake
from core.runtime.portable_orchestration import PortableExecutionRuntime
from core.runtime.tool_result_integration import (
    SerializedToolRuntimePort,
    integrate_tool_result,
)


@dataclass(frozen=True, slots=True)
class AsyncToolRoute:
    """Map provider submission/evidence tools to one status observation tool."""

    provider: str
    evidence_tool_names: frozenset[str]
    status_tool_name: str
    status_tool_arg_key: str


DEFAULT_ASYNC_TOOL_ROUTES: tuple[AsyncToolRoute, ...] = (
    AsyncToolRoute(
        provider="comfyui",
        evidence_tool_names=frozenset({
            "run_comfy_workflow",
            "get_comfy_history",
        }),
        status_tool_name="get_comfy_history",
        status_tool_arg_key="prompt_id",
    ),
)


class CheckpointedGraphApp:
    """Compatibility wrapper that supplies checkpoint thread IDs automatically."""

    def __init__(self, compiled_graph: Any, async_runtime: "LocalAsyncPollingRuntime"):
        self._compiled_graph = compiled_graph
        self.async_runtime = async_runtime

    @staticmethod
    def _default_config(input_state: Any) -> dict[str, Any]:
        thread_id = ""
        if isinstance(input_state, Mapping):
            run_id = input_state.get("run_id")
            if isinstance(run_id, str):
                thread_id = run_id

            if not thread_id:
                execution_state = input_state.get("execution_state")
                if isinstance(execution_state, ExecutionState):
                    thread_id = execution_state.protocol_visible.identity.execution_id

        return {
            "configurable": {
                "thread_id": thread_id or f"cortex-{uuid.uuid4().hex}",
            }
        }

    def stream(
        self,
        input_state: Any,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        resolved_config = config or self._default_config(input_state)
        return self._compiled_graph.stream(
            input_state,
            config=resolved_config,
            **kwargs,
        )

    def invoke(
        self,
        input_state: Any,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        resolved_config = config or self._default_config(input_state)
        return self._compiled_graph.invoke(
            input_state,
            config=resolved_config,
            **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._compiled_graph, name)


class LocalAsyncPollingRuntime:
    """Run one provider status observation, then resume from the tools boundary."""

    def __init__(
        self,
        *,
        compiled_graph: Any,
        tools: Iterable[Any],
        routes: Iterable[AsyncToolRoute] = DEFAULT_ASYNC_TOOL_ROUTES,
        sleep: Callable[[float], None] = time.sleep,
        now_utc: Callable[[], datetime] | None = None,
        resource_observer: RuntimeGpuObserver | None = None,
        resource_coordinator: GpuResourceCoordinator | None = None,
        portable_runtime: PortableExecutionRuntime | None = None,
        resume_adapter: LangGraphAsyncResumeAdapter | None = None,
    ):
        self._compiled_graph = compiled_graph
        self._tools = tuple(tools)
        self._routes = tuple(routes)
        self._sleep = sleep
        self._now_utc = now_utc or (lambda: datetime.now(timezone.utc))
        self._resource_observer = resource_observer
        self._resource_coordinator = resource_coordinator
        self._portable_runtime = portable_runtime or getattr(
            compiled_graph, "_cortex_portable_runtime", None
        )
        self._resume_adapter = resume_adapter or getattr(
            compiled_graph, "_cortex_async_resume_adapter", None
        )
        self._cancel_events: dict[str, threading.Event] = {}
        self._cancel_events_lock = threading.Lock()

    def request_cancel(self, thread_id: str) -> None:
        """Request local cancellation and interrupt an active poll wait."""
        if not thread_id:
            raise ValueError("Cancellation requires a checkpoint thread_id.")
        self._get_cancel_event(thread_id).set()

    def wake_and_resume(
        self,
        *,
        config: dict[str, Any],
        wake: AsyncExecutionWake,
    ):
        """Enter through a semantic wake, then bridge to legacy graph polling."""
        thread_id = self._get_thread_id(config)
        if self._resume_adapter is None:
            raise RuntimeError("Async wake requires a graph resume adapter.")
        state = self._resume_adapter.load(config)

        execution_state = state.get("execution_state")
        if not isinstance(execution_state, ExecutionState):
            raise RuntimeError("Checkpoint does not contain ExecutionState.")
        PortableExecutionRuntime.begin_async_wake(execution_state, wake)

        decision = state.get("controller_decision")
        if not isinstance(decision, ControllerDecision):
            return iter(())
        if decision.decision_type != ControllerDecisionType.AWAIT_ASYNC_JOB:
            return iter(())

        return self._poll_and_resume_legacy(
            config=config,
            decision=decision,
            state=state,
            wake=wake,
            thread_id=thread_id,
            execution_state=execution_state,
        )

    def _poll_and_resume_legacy(
        self,
        *,
        config: dict[str, Any],
        decision: ControllerDecision,
        state: Mapping[str, Any],
        wake: AsyncExecutionWake,
        thread_id: str,
        execution_state: ExecutionState,
    ):
        """Stage 6A bridge preserving the existing poll and graph-resume path."""
        if decision.decision_type != ControllerDecisionType.AWAIT_ASYNC_JOB:
            raise ValueError("async wake bridge requires AWAIT_ASYNC_JOB.")
        if decision.async_job_id is None:
            raise ValueError("AWAIT_ASYNC_JOB requires async_job_id.")

        if self._get_cancel_event(thread_id).is_set():
            return self._resume_local_cancellation(
                config=config,
                execution_state=execution_state,
            )

        terminal_result = next(
            (
                record.result
                for record in reversed(execution_state.working.tool_execution_history)
                if record.result.is_async_job
                and record.result.async_job_id == decision.async_job_id
                and record.result.async_terminal
            ),
            None,
        )
        route = self._resolve_route(
            execution_state=execution_state,
            async_job_id=decision.async_job_id,
        )

        if terminal_result is None and self._wait_until(
            decision.resume_after_utc, thread_id=thread_id
        ):
            return self._resume_local_cancellation(
                config=config,
                execution_state=execution_state,
            )

        if self._portable_runtime is None:
            raise RuntimeError("Async wake requires PortableExecutionRuntime.")
        authorized_turn = self._portable_runtime.authorize_async_wake(
            execution_state,
            build_controller_input(state),
            wake,
            status_tool_name=route.status_tool_name,
            status_argument_key=route.status_tool_arg_key,
        )
        poll_decision = authorized_turn.driver_turn.decision
        updated_execution_state = authorized_turn.driver_turn.execution_state

        if poll_decision.decision_type != ControllerDecisionType.DISPATCH_TOOL_RUNTIME:
            if terminal_result is None:
                return iter(())
            if (
                self._resource_coordinator is not None
                and route.provider == "comfyui"
            ):
                self._resource_coordinator.prepare_for_llm()
            restored_execution_state = execution_state.model_copy(
                update={
                    "working": execution_state.working.model_copy(
                        update={"last_tool_result": terminal_result}
                    )
                }
            )
            return self._resume_adapter.resume(
                config=config,
                state=state,
                update={"execution_state": restored_execution_state},
            )

        request = poll_decision.pending_tool_request
        if request is None:
            raise RuntimeError("Controller poll authorization omitted ToolRequest.")

        def observe(_request: ToolRequest):
            if self._resource_observer is None:
                return nullcontext()
            return self._resource_observer.observe_operation(
                component="async_provider",
                operation=_request.tool_name,
                fields={
                    "provider": route.provider,
                    "async_job_id": decision.async_job_id,
                    "thread_id": thread_id,
                },
            )

        result = self._portable_runtime.dispatch_authorized_async_poll(
            authorized_turn,
            SerializedToolRuntimePort(self._tools, observe=observe),
        )
        if result is None:
            raise RuntimeError("Authorized async poll returned no ToolResult.")
        integrated_execution_state = integrate_tool_result(
            updated_execution_state,
            poll_decision,
            result,
        )
        return self._resume_adapter.resume(
            config=config,
            state=state,
            update={
                "controller_decision": poll_decision,
                "execution_state": integrated_execution_state,
            },
        )

    def _wait_until(
        self,
        resume_after_utc: datetime | None,
        *,
        thread_id: str,
    ) -> bool:
        if resume_after_utc is None:
            return self._get_cancel_event(thread_id).is_set()

        target = resume_after_utc
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        now = self._now_utc()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        delay_seconds = (target.astimezone(timezone.utc) - now.astimezone(timezone.utc)).total_seconds()
        if delay_seconds > 0:
            cancel_event = self._get_cancel_event(thread_id)
            if self._sleep is time.sleep:
                return cancel_event.wait(delay_seconds)
            self._sleep(delay_seconds)
        return self._get_cancel_event(thread_id).is_set()

    def _resume_local_cancellation(
        self,
        *,
        config: dict[str, Any],
        execution_state: ExecutionState,
    ):
        cancelled_state = execution_state.model_copy(
            update={
                "working": execution_state.working.model_copy(
                    update={"cancel_requested": True}
                )
            }
        )
        state = self._resume_adapter.load(config)
        return self._resume_adapter.resume(
            config=config,
            state=state,
            update={"execution_state": cancelled_state},
        )

    def _get_cancel_event(self, thread_id: str) -> threading.Event:
        with self._cancel_events_lock:
            event = self._cancel_events.get(thread_id)
            if event is None:
                event = threading.Event()
                self._cancel_events[thread_id] = event
            return event

    @staticmethod
    def _get_thread_id(config: Mapping[str, Any]) -> str:
        configurable = config.get("configurable")
        if not isinstance(configurable, Mapping):
            raise ValueError("Async polling requires configurable.thread_id.")
        thread_id = configurable.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("Async polling requires configurable.thread_id.")
        return thread_id

    def _resolve_route(
        self,
        *,
        execution_state: ExecutionState,
        async_job_id: str,
    ) -> AsyncToolRoute:
        for record in reversed(execution_state.working.tool_execution_history):
            result = record.result
            if not result.is_async_job or result.async_job_id != async_job_id:
                continue

            for route in self._routes:
                if record.tool_name in route.evidence_tool_names:
                    return route

        raise RuntimeError(
            f"No async status route is registered for job {async_job_id}."
        )
