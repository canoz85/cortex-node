"""Framework-neutral execution finalization application service."""

from typing import Protocol

from core.finalizer_exact import render_exact_collection_answer, render_exact_collections
from core.protocol.enums import ExecutionStatus, StepStatus
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
        if request.accepted_direct_response is not None:
            return request.accepted_direct_response.content
        exact = render_exact_collection_answer(request)
        if exact is not None:
            return exact
        return summary.summary_text


def render_terminal_failure(
    request: FinalizationRequest,
    summary: ExecutionSummary,
) -> str:
    """Render non-success terminal facts without generative additions."""
    lines = [
        "Execution failed."
        if request.status == ExecutionStatus.FAILED
        else "Execution cancelled."
    ]

    if request.accepted_step_results:
        lines.extend(("", "Completed:"))
        for result in request.accepted_step_results:
            evidence = result.completion_evidence
            if evidence.exact_collection is None:
                lines.append(f"- {evidence.step_id}: {result.semantic_content}")
        exact = render_exact_collections(request)
        if exact is not None:
            lines.extend(("", exact))
    elif summary.completed_step_ids:
        lines.extend(("", "Completed:"))
        lines.extend(f"- {step_id}" for step_id in summary.completed_step_ids)

    if summary.failed_step_ids:
        lines.extend(("", "Failed:"))
        lines.extend(f"- {step_id}" for step_id in summary.failed_step_ids)

    if request.terminal_reason:
        lines.extend(("", "Reason:", f"- {request.terminal_reason}"))

    return "\n".join(lines)


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
        if hasattr(self._answer_renderer, "show_raw_llm"):
            self._answer_renderer.show_raw_llm = show_raw_llm

    def finalize(self, request: FinalizationRequest) -> FinalizationResult:
        summary = self._summary_builder.build(request)
        if request.status != ExecutionStatus.COMPLETED:
            return FinalizationResult(
                execution_summary=summary,
                final_answer=render_terminal_failure(request, summary),
            )
        try:
            final_answer = self._answer_renderer.render(request, summary).strip()
            if not final_answer:
                raise ValueError("Final answer renderer returned empty content")
        except Exception as exc:
            result = FinalizationResult(
                execution_summary=summary,
                final_answer=self._RENDER_FAILURE_ANSWER,
                final_answer_error=f"{type(exc).__name__}: {exc}",
            )
        else:
            result = FinalizationResult(execution_summary=summary, final_answer=final_answer)
        return result
