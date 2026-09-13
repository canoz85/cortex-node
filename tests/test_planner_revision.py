"""P4 deterministic revision acceptance and reconciliation regressions."""

from datetime import datetime, timezone

import pytest

from core.runtime.controller_transition import apply_controller_decision_to_state
from core.planner_revision import RevisionRejection, reconcile_revision
from core.protocol.controller import CortexController
from core.protocol.completion_identity import eligible_records
from core.protocol.enums import (
    BrainOutcome, ControllerDecisionType, ExecutionPhase, PlanningOperation,
    ReplanTrigger, StepStatus,
)
from core.protocol.models import (
    BrainResult, ControllerInput, ExecutionContext, ExecutionCursor, ExecutionIdentity,
    ExecutionPlan, ExecutionState, ExecutionStep, PlannerResult, PlanningCapabilities,
    PlanningRequest, ProtocolVisibleState, ReplanRequest, StepCompletionEvidence,
    ToolExecutionRecord, ToolResult,
)


IDENTITY = ExecutionIdentity(execution_id="p4", protocol_version="1")
DONE = ExecutionStep(step_id="done", title="Collect facts", description="facts",
                     primary_tool="read_file", status=StepStatus.COMPLETED)
FAILED = ExecutionStep(step_id="failed", title="Old approach", description="blocked",
                       primary_tool="read_file", status=StepStatus.FAILED, attempt=2,
                       depends_on_step_ids=("done",))
BASE = ExecutionPlan(plan_id="plan", revision=3, objective="Finish", steps=(DONE, FAILED))


def request(**updates):
    value = PlanningRequest(
        request_id="revision", episode_id="revision-episode", identity=IDENTITY, operation=PlanningOperation.REVISE,
        context=ExecutionContext(user_request="Finish"), capabilities=PlanningCapabilities(),
        sequence=2, created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        base_plan=BASE, base_plan_id="plan", base_revision=3,
        completed_step_ids=("done",), completed_steps=(DONE,), interrupted_step=FAILED,
        trigger=ReplanTrigger.BRAIN_REQUESTED, reason="old approach blocked",
    )
    return value.model_copy(update=updates)


def proposal(*steps, plan_id="plan", revision=99):
    return ExecutionPlan(plan_id=plan_id, revision=revision, objective="Finish", steps=steps)


def replacement(step_id="new", title="New approach"):
    return ExecutionStep(step_id=step_id, title=title, description="alternative",
                         depends_on_step_ids=("done",))


def test_valid_meaningful_revision_is_controller_versioned_and_completed_work_preserved():
    accepted = reconcile_revision(request(), BASE, proposal(replacement()))
    assert (accepted.plan_id, accepted.revision) == ("plan", 4)
    assert accepted.steps == (DONE, replacement())
    assert accepted.steps[0].status == StepStatus.COMPLETED


def test_wrong_plan_id_and_stale_base_are_rejected():
    with pytest.raises(RevisionRejection, match="plan_id"):
        reconcile_revision(request(), BASE, proposal(replacement(), plan_id="other"))
    newer = BASE.model_copy(update={"revision": 4})
    with pytest.raises(RevisionRejection, match="base_revision"):
        reconcile_revision(request(), newer, proposal(replacement()))


def test_failed_step_can_be_replaced_but_completed_definition_cannot_change():
    assert reconcile_revision(request(), BASE, proposal(replacement())).steps[-1].step_id == "new"
    changed_done = DONE.model_copy(update={"title": "Redo facts", "status": StepStatus.PENDING})
    with pytest.raises(RevisionRejection, match="completed_step_changed"):
        reconcile_revision(request(), BASE, proposal(changed_done, replacement()))


def test_pending_copy_of_completed_step_is_reconciled_not_rerun():
    pending_done = DONE.model_copy(update={"status": StepStatus.PENDING, "attempt": 0})
    accepted = reconcile_revision(request(), BASE, proposal(pending_done, replacement()))
    assert accepted.steps[0] == DONE
    assert accepted.steps[0].status == StepStatus.COMPLETED


def test_structurally_identical_remaining_plan_is_ineffective():
    same = FAILED.model_copy(update={"status": StepStatus.PENDING, "attempt": 0})
    with pytest.raises(RevisionRejection, match="ineffective_revision"):
        reconcile_revision(request(), BASE, proposal(same))


def test_rejection_decision_is_atomic_and_provenance_survives_acceptance_round_trip():
    provenance = StepCompletionEvidence(
        step_id="done", summary="facts captured", execution_id="p4", plan_id="plan",
        plan_revision=3, evidence_id="stable",
    )
    protocol = ProtocolVisibleState(
        identity=IDENTITY,
        cursor=ExecutionCursor(phase=ExecutionPhase.REPLANNING, plan_revision=3),
        active_plan=BASE, completed_step_ids=("done",), completion_provenance=(provenance,),
        planning_request=request(), planning_sequence=2,
    )
    state = ExecutionState(protocol_visible=protocol)
    ctrl = CortexController(20)
    rejected = ctrl.decide(ControllerInput(
        identity=IDENTITY, cursor=protocol.cursor,
        context=ExecutionContext(user_request="Finish"), active_plan=BASE,
        planner_result=PlannerResult(outcome="execution_plan", request_id="revision", proposed_plan=proposal(
            FAILED.model_copy(update={"status": StepStatus.PENDING, "attempt": 0}))),
        planning_request=request(), planning_sequence=2, completed_step_ids=("done",),
    ))
    after_rejection = apply_controller_decision_to_state(state, rejected)
    assert rejected.decision_type == ControllerDecisionType.PAUSE
    assert after_rejection.protocol_visible.active_plan == BASE
    assert after_rejection.protocol_visible.completion_provenance == (provenance,)

    accepted = ctrl.decide(ControllerInput(
        identity=IDENTITY, cursor=protocol.cursor,
        context=ExecutionContext(user_request="Finish"), active_plan=BASE,
        planner_result=PlannerResult(outcome="execution_plan", request_id="revision", proposed_plan=proposal(replacement())),
        planning_request=request(), planning_sequence=2, completed_step_ids=("done",),
    ))
    restored = ExecutionState.model_validate_json(
        apply_controller_decision_to_state(state, accepted).model_dump_json())
    assert restored.protocol_visible.active_plan.revision == 4
    assert restored.protocol_visible.completed_step_ids == ("done",)
    assert restored.protocol_visible.completion_provenance == (provenance,)


def test_create_path_is_not_revision_reconciled():
    plan = ExecutionPlan(plan_id="fresh", revision=1, steps=(replacement(),))
    ctrl = CortexController(20)
    initial = ControllerInput(identity=IDENTITY, cursor=ExecutionCursor(),
                              context=ExecutionContext(user_request="Finish"))
    dispatch = ctrl.decide(initial)
    request = dispatch.planning_request
    decision = ctrl.decide(initial.model_copy(update={
        "cursor": dispatch.cursor, "planning_request": request,
        "planning_sequence": request.sequence,
        "planner_result": PlannerResult(outcome="execution_plan",
            request_id=request.request_id, proposed_plan=plan),
    }))
    assert decision.accepted_plan == plan


def test_only_structurally_carried_completed_evidence_crosses_revision_boundary():
    revised = reconcile_revision(request(), BASE, proposal(replacement()))
    records = (
        ToolExecutionRecord(execution_id="p4", plan_id="plan", plan_revision=3,
                            step_id="done", tool_name="read_file",
                            result=ToolResult(request_id="done-evidence", success=True, message="done")),
        ToolExecutionRecord(execution_id="p4", plan_id="plan", plan_revision=3,
                            step_id="failed", tool_name="read_file",
                            result=ToolResult(request_id="failed-evidence", success=False, message="failed")),
        ToolExecutionRecord(execution_id="p4", plan_id="plan", plan_revision=4,
                            step_id="new", tool_name="read_file",
                            result=ToolResult(request_id="new-evidence", success=True, message="new")),
    )
    assert tuple(r.result.request_id for r in eligible_records(IDENTITY, revised, records)) == (
        "done-evidence", "new-evidence",
    )
