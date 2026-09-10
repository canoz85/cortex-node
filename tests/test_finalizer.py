import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from core.finalizer import Finalizer
from core.finalizer_provider import LangChainFinalAnswerRenderer
from core.protocol.enums import ExecutionStatus, StepStatus, WorkerRole
from core.protocol.models import (
    ExecutionContext,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionStep,
    FinalizationRequest,
    FinalizationResult,
)


IDENTITY = ExecutionIdentity(execution_id="stage-3a", protocol_version="1.0")
CONTEXT = ExecutionContext(user_request="Complete the request", role=WorkerRole.CONTROLLER)


def _plan(*, failed: bool = False) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan-1",
        revision=2,
        objective="Perform accepted work",
        steps=(
            ExecutionStep(step_id="step-1", title="First", status=StepStatus.COMPLETED),
            ExecutionStep(
                step_id="step-2",
                title="Second",
                status=StepStatus.FAILED if failed else StepStatus.COMPLETED,
                depends_on_step_ids=("step-1",),
            ),
        ),
    )


def test_completed_execution_produces_deterministic_summary_without_final_answer():
    request = FinalizationRequest(
        identity=IDENTITY,
        status=ExecutionStatus.COMPLETED,
        context=CONTEXT,
        accepted_plan=_plan(),
        completed_step_ids=("step-1", "step-2"),
        terminal_reason="All accepted steps completed.",
    )

    first = Finalizer().finalize(request)
    second = Finalizer().finalize(request)

    assert first == second
    assert isinstance(first, FinalizationResult)
    assert first.final_answer == first.execution_summary.summary_text
    assert first.execution_summary.model_dump() == {
        "execution_id": "stage-3a",
        "status": ExecutionStatus.COMPLETED,
        "summary_text": (
            "Execution stage-3a: completed.\n"
            "Plan: plan-1 revision 2.\n"
            "Completed steps: step-1, step-2.\n"
            "Failed steps: none.\n"
            "Terminal reason: All accepted steps completed."
        ),
        "completed_step_ids": ("step-1", "step-2"),
        "failed_step_ids": (),
    }


def test_failed_execution_reports_only_accepted_failed_step_facts():
    result = Finalizer().finalize(FinalizationRequest(
        identity=IDENTITY,
        status=ExecutionStatus.FAILED,
        context=CONTEXT,
        accepted_plan=_plan(failed=True),
        completed_step_ids=("step-1",),
        terminal_reason="Retry budget exhausted.",
    ))

    assert result.execution_summary.status == ExecutionStatus.FAILED
    assert result.execution_summary.completed_step_ids == ("step-1",)
    assert result.execution_summary.failed_step_ids == ("step-2",)
    assert "Retry budget exhausted." in result.execution_summary.summary_text


def test_direct_response_terminal_execution_has_no_plan_or_step_outcomes():
    result = Finalizer().finalize(FinalizationRequest(
        identity=IDENTITY,
        status=ExecutionStatus.COMPLETED,
        context=CONTEXT,
        direct_response=True,
        terminal_reason="Direct response accepted.",
    ))

    summary = result.execution_summary
    assert "Mode: direct response." in summary.summary_text
    assert "Plan:" not in summary.summary_text
    assert summary.completed_step_ids == ()
    assert summary.failed_step_ids == ()


def test_finalizer_does_not_mutate_request_or_nested_plan():
    request = FinalizationRequest(
        identity=IDENTITY,
        status=ExecutionStatus.COMPLETED,
        context=CONTEXT,
        accepted_plan=_plan(),
        completed_step_ids=("step-1", "step-2"),
    )
    before = request.model_dump(mode="json")

    Finalizer().finalize(request)

    assert request.model_dump(mode="json") == before


def test_finalization_rejects_non_terminal_state():
    with pytest.raises(ValueError, match="terminal execution status"):
        FinalizationRequest(identity=IDENTITY, status=ExecutionStatus.NON_TERMINAL, context=CONTEXT)


def test_answer_renderer_owns_final_answer_and_failure_has_no_brain_fallback():
    class Renderer:
        def __init__(self, fails=False):
            self.fails = fails

        def render(self, request, summary):
            if self.fails:
                raise RuntimeError("provider unavailable")
            return f"Finalized: {request.context.user_request} ({summary.status.value})"

    request = FinalizationRequest(
        identity=IDENTITY,
        status=ExecutionStatus.COMPLETED,
        context=CONTEXT,
        direct_response=True,
    )
    success = Finalizer(answer_renderer=Renderer()).finalize(request)
    failure = Finalizer(answer_renderer=Renderer(fails=True)).finalize(request)

    assert success.final_answer == "Finalized: Complete the request (completed)"
    assert success.final_answer_error is None
    assert failure.final_answer == "Execution finished, but the final answer could not be rendered."
    assert failure.final_answer_error == "RuntimeError: provider unavailable"


def test_langchain_renderer_is_an_outward_adapter_over_framework_neutral_facts():
    class Model:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content="Rendered by Finalizer")

    model = Model()
    request = FinalizationRequest(
        identity=IDENTITY,
        status=ExecutionStatus.COMPLETED,
        context=CONTEXT,
        accepted_plan=_plan(),
        completed_step_ids=("step-1", "step-2"),
    )
    result = Finalizer(answer_renderer=LangChainFinalAnswerRenderer(
        llm=model,
    )).finalize(request)

    assert result.final_answer == "Rendered by Finalizer"
    assert result.final_answer_error is None
    assert len(model.calls) == 1
    rendered = "\n".join(message.content for message in model.calls[0])
    assert "Complete the request" in rendered
    assert '"execution_id": "stage-3a"' in rendered
    assert "BRAIN OUTCOME CONTRACT" not in rendered


def test_finalizer_contract_runs_with_graph_provider_and_memory_imports_blocked():
    script = r'''
import sys
class BlockOutwardLayers:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("langgraph", "langchain", "ollama", "core.graph_summarize")):
            raise AssertionError("outward import: " + fullname)
sys.meta_path.insert(0, BlockOutwardLayers())
from core.finalizer import Finalizer
from core.protocol.enums import ExecutionStatus
from core.protocol.models import ExecutionContext, ExecutionIdentity, FinalizationRequest
request = FinalizationRequest(
    identity=ExecutionIdentity(execution_id="plain", protocol_version="1"),
    status=ExecutionStatus.COMPLETED,
    context=ExecutionContext(user_request="hi"),
    direct_response=True,
)
result = Finalizer().finalize(request)
assert result.execution_summary.execution_id == "plain"
assert result.final_answer == result.execution_summary.summary_text
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
