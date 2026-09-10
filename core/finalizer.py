"""Framework-neutral deterministic execution finalization."""

from core.protocol.enums import StepStatus
from core.protocol.models import (
    ExecutionSummary,
    FinalizationRequest,
    FinalizationResult,
)


class ExecutionSummaryBuilder:
    """Build an authoritative summary from accepted terminal domain facts only."""

    def build(self, request: FinalizationRequest) -> ExecutionSummary:
        plan = request.accepted_plan
        failed_step_ids = (
            tuple(step.step_id for step in plan.steps if step.status == StepStatus.FAILED)
            if plan is not None
            else ()
        )

        lines = [
            f"Execution {request.identity.execution_id}: {request.status.value}.",
        ]
        if request.direct_response:
            lines.append("Mode: direct response.")
        if plan is not None:
            lines.append(f"Plan: {plan.plan_id} revision {plan.revision}.")
        lines.append(
            "Completed steps: "
            + (", ".join(request.completed_step_ids) if request.completed_step_ids else "none")
            + "."
        )
        lines.append(
            "Failed steps: "
            + (", ".join(failed_step_ids) if failed_step_ids else "none")
            + "."
        )
        if request.terminal_reason:
            lines.append(f"Terminal reason: {request.terminal_reason}")
        if request.cancellation_source is not None:
            lines.append(f"Cancellation source: {request.cancellation_source.value}.")

        return ExecutionSummary(
            execution_id=request.identity.execution_id,
            status=request.status,
            summary_text="\n".join(lines),
            completed_step_ids=request.completed_step_ids,
            failed_step_ids=failed_step_ids,
        )


class Finalizer:
    """Stage 3A application service; it is not connected to live routing."""

    def __init__(self, summary_builder: ExecutionSummaryBuilder | None = None):
        self._summary_builder = summary_builder or ExecutionSummaryBuilder()

    def finalize(self, request: FinalizationRequest) -> FinalizationResult:
        return FinalizationResult(
            execution_summary=self._summary_builder.build(request),
        )
