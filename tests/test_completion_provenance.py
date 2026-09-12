"""Runtime-owned completion provenance regressions."""

from core.graph_state_machine import apply_controller_decision_to_state
from core.protocol.completion_identity import completion_provenance_records, evidence_identity
from core.protocol.controller import CortexController
from core.protocol.enums import BrainOutcomeKind, ExecutionPhase, StepStatus
from core.protocol.models import (
    BrainOutcome, ControllerInput, ExecutionContext, ExecutionCursor,
    ExecutionIdentity, ExecutionPlan, ExecutionState, ExecutionStep,
    ProtocolVisibleState, RetryMetadata, StepCompletionEvidence,
    ToolExecutionRecord, ToolResult, WorkingState,
)


def fixture():
    identity = ExecutionIdentity(execution_id="exec", protocol_version="1")
    step = ExecutionStep(step_id="s1", title="Inspect", status=StepStatus.ACTIVE)
    plan = ExecutionPlan(plan_id="plan", revision=2, objective="Inspect", steps=(step,))
    return identity, plan, step


def record(request_id, *, execution="exec", plan="plan", revision=2, step="s1", success=True):
    return ToolExecutionRecord(
        execution_id=execution, plan_id=plan, plan_revision=revision,
        step_id=step, tool_name="read_file",
        result=ToolResult(request_id=request_id, success=success, message="observed"),
    )


def controller_input(records):
    identity, plan, step = fixture()
    return ControllerInput(
        identity=identity,
        cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING, step_id="s1", plan_revision=2),
        context=ExecutionContext(user_request="Inspect"), active_plan=plan, active_step=step,
        retry=RetryMetadata(max_retries=1), tool_execution_history=tuple(records),
        brain_result=BrainOutcome(
            outcome=BrainOutcomeKind.STEP_COMPLETED, step_id="s1", message="done",
            completion_evidence=StepCompletionEvidence(step_id="s1", summary="done"),
        ),
    )


def test_exact_scope_filter_accumulates_retries_and_preserves_history_order():
    identity, plan, step = fixture()
    records = (
        record("first", success=False), record("foreign-execution", execution="other"),
        record("foreign-plan", plan="other"), record("foreign-revision", revision=1),
        record("foreign-step", step="s2"), record("retry-success"),
    )
    eligible = completion_provenance_records(identity, plan, step, records)
    assert tuple(item.result.request_id for item in eligible) == ("first", "retry-success")
    assert eligible[0].result.success is False


def test_controller_binds_all_records_not_only_brain_projection_window():
    records = tuple(record(f"r{i}") for i in range(30))
    decision = CortexController(24).decide(controller_input(records))
    provenance = decision.completion_evidence
    assert provenance.tool_request_ids == tuple(f"r{i}" for i in range(30))
    assert (provenance.execution_id, provenance.plan_id, provenance.plan_revision, provenance.step_id) == (
        "exec", "plan", 2, "s1",
    )
    assert provenance.evidence_id == evidence_identity(records)


def test_reasoning_only_completion_binds_empty_stable_provenance():
    decision = CortexController(24).decide(controller_input(()))
    assert decision.completion_evidence.tool_request_ids == ()
    assert decision.completion_evidence.evidence_id == evidence_identity(())


def test_accepted_provenance_is_checkpoint_stable_and_not_recomputed():
    decision = CortexController(24).decide(controller_input((record("captured"),)))
    identity, plan, step = fixture()
    state = ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=identity,
            cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING, step_id="s1", plan_revision=2),
            active_plan=plan, active_step=step,
        ),
        working=WorkingState(tool_execution_history=(record("captured"),)),
    )
    accepted = apply_controller_decision_to_state(state, decision)
    restored = ExecutionState.model_validate_json(accepted.model_dump_json())
    later = restored.model_copy(update={
        "working": restored.working.model_copy(update={
            "tool_execution_history": (*restored.working.tool_execution_history, record("later")),
        }),
    })
    assert later.protocol_visible.completion_provenance == accepted.protocol_visible.completion_provenance
    assert later.protocol_visible.completion_provenance[0].tool_request_ids == ("captured",)
