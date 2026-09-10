"""Framework-neutral execution finalization application service."""

from typing import Protocol

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


class FinalAnswerRenderer(Protocol):
    """Framework-neutral port for rendering the user-facing final answer."""

    def render(
        self,
        request: FinalizationRequest,
        summary: ExecutionSummary,
    ) -> str:
        ...


class SummaryFinalAnswerRenderer:
    """Deterministic renderer used when no model-backed adapter is configured."""

    def render(
        self,
        request: FinalizationRequest,
        summary: ExecutionSummary,
    ) -> str:
        return summary.summary_text


class Finalizer:
    """Produce the authoritative summary and answer after terminal authorization."""

    _RENDER_FAILURE_ANSWER = "Execution finished, but the final answer could not be rendered."

    def __init__(
        self,
        summary_builder: ExecutionSummaryBuilder | None = None,
        answer_renderer: FinalAnswerRenderer | None = None,
    ):
        self._summary_builder = summary_builder or ExecutionSummaryBuilder()
        self._answer_renderer = answer_renderer or SummaryFinalAnswerRenderer()

    def finalize(self, request: FinalizationRequest) -> FinalizationResult:
        summary = self._summary_builder.build(request)
        try:
            final_answer = self._answer_renderer.render(request, summary).strip()
            if not final_answer:
                raise ValueError("Final answer renderer returned empty content")
        except Exception as exc:
            return FinalizationResult(
                execution_summary=summary,
                final_answer=self._RENDER_FAILURE_ANSWER,
                final_answer_error=f"{type(exc).__name__}: {exc}",
            )
        return FinalizationResult(
            execution_summary=summary,
            final_answer=final_answer,
        )
