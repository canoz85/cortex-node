"""Framework-neutral, Controller-authorized one-turn execution driver."""

from __future__ import annotations

from dataclasses import dataclass

from core.protocol.enums import (
    BrainOutcome,
    ControllerDecisionType,
    ExecutionStatus,
    WorkerRole,
)
from core.protocol.models import (
    AcceptedStepResult,
    BrainInput,
    BrainResult,
    ControllerDecision,
    ControllerInput,
    ExecutionState,
    FinalizationRequest,
    FinalizationResult,
    PlannerResult,
    ToolResult,
)
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.worker_ports import (
    BrainPort,
    FinalizerPort,
    PlannerPort,
    ToolRuntimePort,
)


WorkerResult = PlannerResult | BrainResult | ToolResult | FinalizationResult


class WorkerDispatchError(RuntimeError):
    """A Controller decision cannot be safely dispatched by this driver."""


@dataclass(frozen=True, slots=True)
class ExecutionDriverTurn:
    """One Controller transition and its authorized worker result, if any."""

    execution_state: ExecutionState
    decision: ControllerDecision
    worker_result: WorkerResult | None = None


class ExecutionDriver:
    """Apply one Controller turn, then invoke exactly its authorized worker."""

    def __init__(
        self,
        *,
        coordinator: ControllerCoordinator,
        planner: PlannerPort,
        brain: BrainPort,
        tool_runtime: ToolRuntimePort,
        finalizer: FinalizerPort,
    ) -> None:
        self._coordinator = coordinator
        self._planner = planner
        self._brain = brain
        self._tool_runtime = tool_runtime
        self._finalizer = finalizer

    def turn(
        self,
        execution_state: ExecutionState,
        controller_input: ControllerInput,
        *,
        dispatch_worker: bool = True,
    ) -> ExecutionDriverTurn:
        """Perform one decision/application turn and optionally dispatch its worker."""

        authorized = self.transition(execution_state, controller_input)
        decision = authorized.decision
        updated_state = authorized.execution_state
        worker_result = (
            self._dispatch(updated_state, decision, controller_input)
            if dispatch_worker
            else None
        )
        return ExecutionDriverTurn(
            execution_state=updated_state,
            decision=decision,
            worker_result=worker_result,
        )

    def transition(
        self,
        execution_state: ExecutionState,
        controller_input: ControllerInput,
    ) -> ExecutionDriverTurn:
        """Produce and apply one authorization without dispatching it."""

        value = self._coordinator.transition(execution_state, controller_input)
        return ExecutionDriverTurn(
            execution_state=value.execution_state,
            decision=value.decision,
        )

    def dispatch_authorized(
        self,
        execution_state: ExecutionState,
        decision: ControllerDecision,
        controller_input: ControllerInput,
        *,
        tool_runtime: ToolRuntimePort | None = None,
    ) -> WorkerResult | None:
        """Dispatch a previously applied authorization without recomputing it."""

        return self._dispatch(
            execution_state,
            decision,
            controller_input,
            tool_runtime=tool_runtime,
        )

    def _dispatch(
        self,
        execution_state: ExecutionState,
        decision: ControllerDecision,
        controller_input: ControllerInput,
        *,
        tool_runtime: ToolRuntimePort | None = None,
    ) -> WorkerResult | None:
        if decision.terminal:
            return self._dispatch_finalizer(
                execution_state,
                decision,
                controller_input,
            )

        match decision.decision_type:
            case ControllerDecisionType.DISPATCH_PLANNER:
                self._require_worker(decision, WorkerRole.PLANNER)
                request = decision.planning_request
                if request is None:
                    raise WorkerDispatchError(
                        "DISPATCH_PLANNER requires a PlanningRequest"
                    )
                run_authorized = getattr(self._planner, "run_authorized", None)
                result = (
                    run_authorized(request, execution_state, decision)
                    if callable(run_authorized)
                    else self._planner.run(request)
                )
                if not isinstance(result, PlannerResult):
                    raise WorkerDispatchError("Planner returned an invalid result type")
                if result.request_id != request.request_id:
                    raise WorkerDispatchError("Planner result request identity mismatch")
                return result

            case ControllerDecisionType.DISPATCH_BRAIN:
                self._require_worker(decision, WorkerRole.BRAIN)
                protocol = execution_state.protocol_visible
                brain_input = BrainInput(
                    identity=protocol.identity,
                    cursor=protocol.cursor,
                    context=controller_input.context.model_copy(
                        update={"role": WorkerRole.BRAIN}
                    ),
                    active_plan=protocol.active_plan,
                    active_step=protocol.active_step,
                    last_tool_result=(
                        controller_input.tool_result
                        if controller_input.tool_result is not None
                        else execution_state.working.last_tool_result
                    ),
                    tool_execution_history=controller_input.tool_execution_history,
                    retry=protocol.retry,
                    direct_response=decision.direct_response,
                    coverage_assessment=controller_input.coverage_assessment,
                )
                run_authorized = getattr(self._brain, "run_authorized", None)
                result = (
                    run_authorized(brain_input, execution_state, decision)
                    if callable(run_authorized)
                    else self._brain.run(brain_input)
                )
                if not isinstance(result, BrainResult):
                    raise WorkerDispatchError("Brain returned an invalid result type")
                return result

            case ControllerDecisionType.DISPATCH_TOOL_RUNTIME:
                self._require_worker(decision, WorkerRole.TOOL_RUNTIME)
                request = decision.pending_tool_request
                if request is None:
                    raise WorkerDispatchError(
                        "DISPATCH_TOOL_RUNTIME requires a ToolRequest"
                    )
                if execution_state.protocol_visible.pending_tool_request != request:
                    raise WorkerDispatchError(
                        "Tool request does not match the authorized execution state"
                    )
                selected_tool_runtime = tool_runtime or self._tool_runtime
                execute_authorized = getattr(
                    selected_tool_runtime, "execute_authorized", None
                )
                result = (
                    execute_authorized(request, execution_state, decision)
                    if callable(execute_authorized)
                    else selected_tool_runtime.execute(request)
                )
                if not isinstance(result, ToolResult):
                    raise WorkerDispatchError("Tool runtime returned an invalid result type")
                if result.request_id != request.request_id:
                    raise WorkerDispatchError("Tool result request identity mismatch")
                return result

            case ControllerDecisionType.PAUSE:
                if decision.next_worker not in (None, WorkerRole.CONTROLLER):
                    raise WorkerDispatchError("PAUSE contains a mismatched worker")
                return None

            case _:
                raise WorkerDispatchError(
                    f"Unsupported Controller decision: {decision.decision_type.value}"
                )

    def _dispatch_finalizer(
        self,
        execution_state: ExecutionState,
        decision: ControllerDecision,
        controller_input: ControllerInput,
    ) -> FinalizationResult:
        allowed_workers = {
            ControllerDecisionType.DISPATCH_SUMMARY: (WorkerRole.SUMMARY,),
            ControllerDecisionType.TERMINATE: (None, WorkerRole.CONTROLLER),
            ControllerDecisionType.CANCEL: (WorkerRole.CONTROLLER,),
        }
        expected_workers = allowed_workers.get(decision.decision_type)
        if expected_workers is None or decision.next_worker not in expected_workers:
            raise WorkerDispatchError(
                "Terminal decision does not authorize Finalizer execution"
            )

        protocol = execution_state.protocol_visible
        if protocol.status == ExecutionStatus.NON_TERMINAL:
            raise WorkerDispatchError("Finalizer requires terminal ExecutionState")

        direct_response = bool(
            protocol.accepted_direct_response is not None
            or (protocol.status == ExecutionStatus.COMPLETED
            and protocol.active_plan is None
            and controller_input.brain_result is not None
            and controller_input.brain_result.outcome == BrainOutcome.FINAL_ANSWER)
        )
        request = FinalizationRequest(
            identity=protocol.identity,
            status=protocol.status,
            context=controller_input.context,
            accepted_plan=protocol.active_plan,
            tool_execution_history=controller_input.tool_execution_history,
            completed_step_ids=protocol.completed_step_ids,
            accepted_step_results=tuple(
                AcceptedStepResult(completion_evidence=evidence)
                for evidence in protocol.completion_provenance
            ),
            accepted_direct_response=protocol.accepted_direct_response,
            terminal_reason=decision.failure_reason or decision.reason,
            direct_response=direct_response,
            cancellation_source=protocol.cancellation_source,
        )
        result = self._finalizer.finalize(request)
        if not isinstance(result, FinalizationResult):
            raise WorkerDispatchError("Finalizer returned an invalid result type")
        return result

    @staticmethod
    def _require_worker(
        decision: ControllerDecision,
        expected: WorkerRole,
    ) -> None:
        if decision.next_worker != expected:
            raise WorkerDispatchError(
                f"{decision.decision_type.value} does not authorize {expected.value}"
            )


__all__ = [
    "ExecutionDriver",
    "ExecutionDriverTurn",
    "WorkerDispatchError",
    "WorkerResult",
]
