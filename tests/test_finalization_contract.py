import pytest
from pydantic import ValidationError

from core.protocol.enums import ExecutionStatus
from core.protocol.models import (
    AcceptedStepResult,
    ExecutionContext,
    ExecutionIdentity,
    FinalizationRequest,
    StepCompletionEvidence,
)


IDENTITY = ExecutionIdentity(execution_id="finalization-contract", protocol_version="1.0")
CONTEXT = ExecutionContext(user_request="Explain the accepted results")


def bound_evidence(summary: str, *, step_id: str, revision: int) -> StepCompletionEvidence:
    return StepCompletionEvidence(
        execution_id=IDENTITY.execution_id,
        plan_id="plan",
        plan_revision=revision,
        step_id=step_id,
        summary=summary,
        tool_request_ids=(f"request-{step_id}",),
        evidence_id=f"evidence-{step_id}",
    )


def test_accepted_step_result_is_immutable_and_derives_semantic_content():
    evidence = bound_evidence("Accepted conclusion", step_id="s1", revision=2)
    result = AcceptedStepResult(completion_evidence=evidence)

    assert result.semantic_content == "Accepted conclusion"
    assert result.completion_evidence is evidence
    assert result.model_dump()["semantic_content"] == "Accepted conclusion"
    with pytest.raises(ValidationError, match="frozen"):
        result.completion_evidence = bound_evidence(
            "Replacement", step_id="s1", revision=2
        )


def test_accepted_step_result_retains_controller_bound_evidence_identity():
    evidence = bound_evidence("Accepted conclusion", step_id="s1", revision=7)

    result = AcceptedStepResult(completion_evidence=evidence)

    assert result.completion_evidence.execution_id == IDENTITY.execution_id
    assert result.completion_evidence.plan_id == "plan"
    assert result.completion_evidence.plan_revision == 7
    assert result.completion_evidence.step_id == "s1"
    assert result.completion_evidence.evidence_id == "evidence-s1"


def test_finalization_request_preserves_multiple_accepted_result_order():
    first = AcceptedStepResult(
        completion_evidence=bound_evidence("First", step_id="s1", revision=2)
    )
    second = AcceptedStepResult(
        completion_evidence=bound_evidence("Second", step_id="s2", revision=3)
    )

    request = FinalizationRequest(
        identity=IDENTITY,
        status=ExecutionStatus.COMPLETED,
        context=CONTEXT,
        accepted_step_results=(first, second),
    )

    assert request.accepted_step_results == (first, second)
    assert tuple(
        result.semantic_content for result in request.accepted_step_results
    ) == ("First", "Second")


def test_accepted_step_result_rejects_unbound_brain_evidence():
    with pytest.raises(ValidationError, match="Controller-bound"):
        AcceptedStepResult(
            completion_evidence=StepCompletionEvidence(
                step_id="s1",
                summary="Unaccepted proposal",
            )
        )
