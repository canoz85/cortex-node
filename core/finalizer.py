"""Framework-neutral execution finalization application service."""

from typing import Protocol

from core.finalizer_debug import log_finalizer
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
        show_raw_llm: bool = False,
    ):
        self._summary_builder = summary_builder or ExecutionSummaryBuilder()
        self._answer_renderer = answer_renderer or SummaryFinalAnswerRenderer()
        self.show_raw_llm = show_raw_llm

    def finalize(self, request: FinalizationRequest) -> FinalizationResult:
        summary = self._summary_builder.build(request)
        if self.show_raw_llm:
            log_finalizer("request", {
                "identity": request.identity.model_dump(mode="json"),
                "status": request.status.value,
                "accepted_plan": request.accepted_plan.model_dump(mode="json") if request.accepted_plan else None,
                "completed_step_ids": summary.completed_step_ids,
                "failed_step_ids": summary.failed_step_ids,
                "terminal_reason": request.terminal_reason,
                "tool_execution_history": [record.model_dump(mode="json") for record in request.tool_execution_history],
            })
        try:
            final_answer = self._answer_renderer.render(request, summary).strip()
            if not final_answer:
                raise ValueError("Final answer renderer returned empty content")
        except Exception as exc:
            log_finalizer("error", {"stage": "render", "type": type(exc).__name__, "message": str(exc)},
                          enabled=self.show_raw_llm)
            result = FinalizationResult(
                execution_summary=summary,
                final_answer=self._RENDER_FAILURE_ANSWER,
                final_answer_error=f"{type(exc).__name__}: {exc}",
            )
        else:
            result = FinalizationResult(execution_summary=summary, final_answer=final_answer)
        if self.show_raw_llm:
            log_finalizer("normalized", result.model_dump(mode="json"))
        return result
