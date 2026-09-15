"""Framework-neutral preparation and execution of one application turn."""

from __future__ import annotations

from dataclasses import dataclass

from core.completion import CompletionService
from core.planner_revision import RevisionRejection, reconcile_revision
from core.protocol.completion_identity import accepted_step
from core.protocol.enums import ControllerDecisionType, PlanningOperation
from core.protocol.models import ControllerInput, ExecutionState, FinalizationResult
from core.runtime.execution_driver import ExecutionDriver, ExecutionDriverTurn
from core.runtime.async_wake import AsyncExecutionWake, AsyncWakeIntent
from core.runtime.worker_ports import ToolRuntimePort


@dataclass(frozen=True, slots=True)
class PortableRuntimeTurn:
    """Portable result plus completion state committed around the transition."""

    driver_turn: ExecutionDriverTurn
    controller_input: ControllerInput
    terminal_dispatch_error: Exception | None = None


class PortableExecutionRuntime:
    """Own normal-path preparation, reconciliation, and driver invocation."""

    def __init__(self, *, driver: ExecutionDriver, completion_service: CompletionService):
        self._driver = driver
        self._completion_service = completion_service

    @staticmethod
    def begin_async_wake(
        execution_state: ExecutionState,
        wake: AsyncExecutionWake,
    ) -> AsyncExecutionWake:
        """Accept semantic wake correlation without treating it as authorization.

        Stage 6A establishes this framework-neutral entrypoint only. The legacy
        polling adapter still recovers the checkpointed wait decision and owns
        poll construction, execution, and graph resume until later Stage 6 slices.
        """

        if not isinstance(execution_state, ExecutionState):
            raise TypeError("execution_state must be an ExecutionState")
        if not isinstance(wake, AsyncExecutionWake):
            raise TypeError("wake must be an AsyncExecutionWake")
        if wake.intent != AsyncWakeIntent.POLL_DUE:
            raise ValueError(f"Unsupported async wake intent: {wake.intent!r}")
        if wake.execution_id != execution_state.protocol_visible.identity.execution_id:
            raise ValueError("Async wake execution identity mismatch")
        return wake

    def authorize_async_wake(
        self,
        execution_state: ExecutionState,
        controller_input: ControllerInput,
        wake: AsyncExecutionWake,
        *,
        status_tool_name: str,
        status_argument_key: str,
    ) -> PortableRuntimeTurn:
        """Ask Controller to authorize a correlated async status observation."""
        self.begin_async_wake(execution_state, wake)
        wake_input = controller_input.model_copy(
            update={
                "planner_result": None,
                "brain_result": None,
                "tool_result": None,
                "async_wake_job_id": wake.async_job_id,
                "async_poll_tool_name": status_tool_name,
                "async_poll_argument_key": status_argument_key,
            }
        )
        return self.turn(execution_state, wake_input, dispatch_worker=False)

    def dispatch_authorized_async_poll(
        self,
        authorized_turn: PortableRuntimeTurn,
        tool_runtime: ToolRuntimePort,
    ):
        """Execute one Controller-authorized poll through ExecutionDriver."""
        turn = authorized_turn.driver_turn
        return self._driver.dispatch_authorized(
            turn.execution_state,
            turn.decision,
            authorized_turn.controller_input,
            tool_runtime=tool_runtime,
        )

    def turn(
        self,
        execution_state: ExecutionState,
        controller_input: ControllerInput,
        *,
        dispatch_worker: bool = True,
    ) -> PortableRuntimeTurn:
        prepared, bindings, assessment, frozen = self._prepare(
            execution_state, controller_input
        )
        # The portable runtime owns dispatch sequencing.  Asking the driver for
        # the applied transition first lets terminal dispatch consume that exact
        # authorization without recomputing or applying it again.
        driver_turn = self._driver.transition(execution_state, prepared)
        decision = driver_turn.decision
        transitioned = driver_turn.execution_state
        previous_protocol = execution_state.protocol_visible

        if decision.accepted_plan is None:
            bindings = previous_protocol.accepted_requirements
        next_protocol = transitioned.protocol_visible
        if (
            next_protocol.active_step != previous_protocol.active_step
            or next_protocol.active_plan != previous_protocol.active_plan
        ):
            assessment, frozen = self._completion_service.evaluate(
                prepared.identity,
                next_protocol.active_plan,
                next_protocol.active_step,
                prepared.tool_execution_history,
                frozen,
                previous=assessment,
                bindings=bindings,
            )

        transitioned = transitioned.model_copy(update={
            "protocol_visible": next_protocol.model_copy(update={
                "resolved_coverages": frozen,
                "accepted_requirements": bindings,
            }),
            "working": transitioned.working.model_copy(update={
                "coverage_assessment": assessment,
            }),
        })
        worker_result = None
        terminal_dispatch_error = None
        dispatchable = decision.terminal or decision.decision_type in {
            ControllerDecisionType.DISPATCH_PLANNER,
            ControllerDecisionType.DISPATCH_BRAIN,
            ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
            ControllerDecisionType.PAUSE,
        }
        if dispatch_worker and dispatchable or decision.terminal:
            try:
                worker_result = self._driver.dispatch_authorized(
                    transitioned, decision, prepared
                )
            except Exception as exc:
                if not decision.terminal:
                    raise
                terminal_dispatch_error = exc
        if decision.terminal and worker_result is not None and not isinstance(
            worker_result, FinalizationResult
        ):
            terminal_dispatch_error = TypeError(
                "portable driver did not return FinalizationResult"
            )
            worker_result = None

        return PortableRuntimeTurn(
            driver_turn=ExecutionDriverTurn(
                execution_state=transitioned,
                decision=decision,
                worker_result=worker_result,
            ),
            controller_input=prepared,
            terminal_dispatch_error=terminal_dispatch_error,
        )

    def _prepare(
        self, execution_state: ExecutionState, controller_input: ControllerInput
    ):
        protocol = execution_state.protocol_visible
        if controller_input.active_step is not None:
            accepted_step(
                controller_input.active_plan,
                controller_input.active_step,
                controller_input.cursor,
            )
        assessment, frozen = self._completion_service.evaluate(
            controller_input.identity,
            controller_input.active_plan,
            controller_input.active_step,
            controller_input.tool_execution_history,
            protocol.resolved_coverages,
            previous=execution_state.working.coverage_assessment,
            bindings=protocol.accepted_requirements,
        )
        bindings = protocol.accepted_requirements
        validation_id = validation_error = None
        planner_result = controller_input.planner_result
        if planner_result is not None and planner_result.proposed_plan is not None:
            request = controller_input.planning_request
            if request is not None and request.operation == PlanningOperation.REVISE:
                try:
                    reconciled = reconcile_revision(
                        request, controller_input.active_plan, planner_result.proposed_plan
                    )
                    planner_result = planner_result.model_copy(
                        update={"proposed_plan": reconciled}
                    )
                except RevisionRejection:
                    pass
            validation_id, validation_error, bindings = self._completion_service.bind_plan(
                controller_input.identity, planner_result.proposed_plan, bindings
            )
        prepared = controller_input.model_copy(update={
            "planner_result": planner_result,
            "coverage_assessment": assessment,
            "accepted_requirements": bindings,
            "completion_validation_id": validation_id,
            "completion_validation_error": validation_error,
        })
        return prepared, bindings, assessment, frozen


__all__ = ["PortableExecutionRuntime", "PortableRuntimeTurn"]
