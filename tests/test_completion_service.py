"""Domain-neutral completion lifecycle, exercised only with fake providers."""

import pytest

from core.completion import CompletionService, Resolution
from core.graph_controller import create_controller_node
from core.protocol.controller import CortexController
from core.protocol.models import (
    CoverageRequirement, CoverageAssessment, ExecutionStep, ExecutionPlan,
    ExecutionIdentity, ExecutionCursor, ExecutionContext, ControllerInput,
    BrainOutcome, PlannerResult, ToolExecutionRecord, ToolResult, ExecutionState,
    ProtocolVisibleState, WorkingState, StepCompletionEvidence,
)
from core.protocol.enums import BrainOutcomeKind, PlannerOutcome, StepStatus, ExecutionPhase


class FakeProvider:
    contract_version = "1"

    def __init__(self):
        self.items = ("a", "b")
        self.satisfied = ()
        self.resolve_calls = 0
        self.assess_calls = 0
        self.explode = False
        self.stale = False

    def validate(self, specification):
        if specification.get("invalid"):
            raise ValueError("invalid")
        with pytest.raises(TypeError):
            specification["changed"] = True

    def resolve(self, specification, evidence):
        self.resolve_calls += 1
        for record in evidence.records:
            with pytest.raises(TypeError):
                record["result"]["data"]["values"] = ()
        return Resolution(self.items, reason="waiting" if self.items is None else "")

    def assess(self, specification, resolved, evidence):
        self.assess_calls += 1
        if self.explode:
            raise RuntimeError("internal")
        missing = tuple(i for i in resolved.required_item_ids if i not in self.satisfied)
        return CoverageAssessment(
            scope_id=resolved.scope_id,
            evidence_id="old" if self.stale else evidence.evidence_id,
            status="missing" if missing else "satisfied",
            satisfied_item_ids=self.satisfied, missing_item_ids=missing,
        )


def context(requirement=True, revision=1):
    step = ExecutionStep(step_id="s", title="Work", status=StepStatus.ACTIVE,
        completion_requirement=CoverageRequirement(provider_id="fake") if requirement else None)
    ctx = ControllerInput(
        identity=ExecutionIdentity(execution_id="e", protocol_version="1"),
        cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING, step_id="s", plan_revision=revision),
        context=ExecutionContext(user_request="Work"),
        active_plan=ExecutionPlan(plan_id="p", revision=revision, steps=(step,)), active_step=step,
        brain_result=BrainOutcome(outcome=BrainOutcomeKind.STEP_COMPLETED, step_id="s",
            completion_evidence=StepCompletionEvidence(step_id="s", summary="Done", tool_request_ids=())),
    )
    bindings = CompletionService({"fake": FakeProvider()}).bind_plan(ctx.identity, ctx.active_plan)[2]
    return ctx.model_copy(update={"accepted_requirements": bindings})



def record(n):
    return ToolExecutionRecord(execution_id="e", plan_id="p", plan_revision=1, step_id="s", tool_name="fake_operation",
        result=ToolResult(request_id=str(n), success=True, message="ok", data={"values": [n]}))


def evaluate(service, ctx, frozen=(), previous=None):
    return service.evaluate(ctx.identity, ctx.active_plan, ctx.active_step,
                            ctx.tool_execution_history, frozen, previous, ctx.accepted_requirements)


def test_validation_and_unknown_provider_block_plan_acceptance():
    service = CompletionService({"fake": FakeProvider()})
    ctx = context()
    receipt, error = service.validate_plan(ctx.active_plan)
    assert receipt and error is None
    planner = ctx.model_copy(update={"brain_result": None, "planner_result": PlannerResult(
        outcome=PlannerOutcome.EXECUTION_PLAN, proposed_plan=ctx.active_plan.model_copy(update={
            "steps": (ctx.active_step.model_copy(update={"status": StepStatus.PENDING}),)}))})
    plan = planner.planner_result.proposed_plan
    receipt, error = CompletionService().validate_plan(plan)
    decision = CortexController(20).decide(planner.model_copy(update={
        "completion_validation_id": receipt, "completion_validation_error": error}))
    assert decision.accepted_plan is None
    assert decision.reason == "unknown_completion_provider"
    invalid = ctx.active_step.model_copy(update={"completion_requirement": CoverageRequirement(
        provider_id="fake", specification={"invalid": True})})
    assert service.validate_plan(ctx.active_plan.model_copy(update={"steps": (invalid,)}))[1] == "invalid_completion_specification"
    receipt, error = service.validate_plan(plan)
    assert CortexController(20).decide(planner.model_copy(update={
        "completion_validation_id": receipt, "completion_validation_error": error})).accepted_plan == plan


def test_unresolved_retries_on_new_evidence_and_empty_is_resolved():
    provider = FakeProvider()
    provider.items = None
    service = CompletionService({"fake": provider})
    ctx = context()
    first, frozen = evaluate(service, ctx)
    assert first.status == "unresolved" and frozen == ()
    evaluate(service, ctx, previous=first)
    assert provider.resolve_calls == 1
    ctx = ctx.model_copy(update={"tool_execution_history": (record(1),)})
    second, frozen = evaluate(service, ctx, previous=first)
    assert second.status == "unresolved" and provider.resolve_calls == 2
    provider.items = ()
    ctx = ctx.model_copy(update={"tool_execution_history": (record(1), record(2))})
    assessment, frozen = evaluate(service, ctx, previous=second)
    assert assessment.status == "satisfied" and frozen[0].required_item_ids == ()


def test_membership_freezes_and_assessment_is_fresh():
    provider = FakeProvider()
    service = CompletionService({"fake": provider})
    ctx = context()
    first, frozen = evaluate(service, ctx)
    provider.items = ("a",)
    provider.satisfied = ("a",)
    ctx = ctx.model_copy(update={"tool_execution_history": (record(1),)})
    second, again = evaluate(service, ctx, frozen)
    assert again == frozen and frozen[0].required_item_ids == ("a", "b")
    assert provider.resolve_calls == 1 and provider.assess_calls == 2
    assert second.missing_item_ids == ("b",) and first.evidence_id != second.evidence_id
    assert CortexController(20).decide(ctx.model_copy(update={"coverage_assessment": second})).completed_step_id is None
    provider.satisfied = ("a", "b")
    complete, _ = evaluate(service, ctx, frozen)
    assert CortexController(20).decide(ctx.model_copy(update={"coverage_assessment": complete})).completed_step_id == "s"
    newer = ctx.model_copy(update={"tool_execution_history": (record(1), record(2)), "coverage_assessment": complete})
    assert CortexController(20).decide(newer).completed_step_id is None


@pytest.mark.parametrize("mode", ["exception", "stale", "unavailable"])
def test_provider_failures_preserve_frozen_state_and_block(mode):
    provider = FakeProvider()
    service = CompletionService({"fake": provider})
    ctx = context()
    _, frozen = evaluate(service, ctx)
    before = frozen[0].model_dump_json()
    if mode == "exception":
        provider.explode = True
    elif mode == "stale":
        provider.stale = True
    else:
        service = CompletionService()
    assessment, after = evaluate(service, ctx, frozen)
    assert assessment.status in {"error", "stale"}
    assert after == frozen and after[0].model_dump_json() == before
    decision = CortexController(20).decide(ctx.model_copy(update={"coverage_assessment": assessment}))
    assert decision.next_step_id == "s" and decision.failed_step_id is None
    assert not decision.requires_replan and decision.completed_step_id is None


def test_new_revision_resolves_again_and_citations_are_unchanged():
    provider = FakeProvider()
    service = CompletionService({"fake": provider})
    ctx = context()
    _, frozen = evaluate(service, ctx)
    provider.items = ("c",)
    assessment, frozen = evaluate(service, context(revision=2), frozen)
    assert len(frozen) == 2 and frozen[0].scope_id != frozen[1].scope_id
    assert frozen[1].required_item_ids == ("c",) and provider.resolve_calls == 2
    assert ctx.brain_result.completion_evidence.tool_request_ids == ()
    assert set(StepCompletionEvidence.model_fields) == {"step_id", "summary", "tool_request_ids"}


def test_semantic_step_behavior_unchanged():
    ctx = context(False)
    assert evaluate(CompletionService(), ctx) == (None, ())
    assert CortexController(20).decide(ctx).completed_step_id == "s"


def test_graph_commits_resolution_and_checkpoint_roundtrip():
    provider = FakeProvider()
    ctx = context()
    state = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=ctx.identity, cursor=ctx.cursor, active_plan=ctx.active_plan, active_step=ctx.active_step, accepted_requirements=ctx.accepted_requirements),
        working=WorkingState())
    node = create_controller_node(completion_service=CompletionService({"fake": provider}))
    result = node({"execution_state": state, "brain_result": ctx.brain_result})
    restored = ExecutionState.model_validate_json(result["execution_state"].model_dump_json())
    assert restored.protocol_visible.resolved_coverages[0].required_item_ids == ("a", "b")
    assert restored.working.coverage_assessment.status == "missing"
    provider.items = ()
    result = node({"execution_state": restored, "brain_result": ctx.brain_result})
    assert result["execution_state"].protocol_visible.resolved_coverages == restored.protocol_visible.resolved_coverages
    assert provider.resolve_calls == 1


def test_resolution_exception_does_not_freeze_and_can_recover():
    class Broken(FakeProvider):
        def resolve(self, specification, evidence):
            raise RuntimeError("broken")
    ctx = context()
    assessment, frozen = evaluate(CompletionService({"fake": Broken()}), ctx)
    assert assessment.status == "error" and frozen == ()
    assessment, frozen = evaluate(CompletionService({"fake": FakeProvider()}), ctx, frozen, assessment)
    assert assessment.status == "missing" and len(frozen) == 1


def test_inconsistent_assessment_cannot_claim_coverage():
    class Inconsistent(FakeProvider):
        def assess(self, specification, resolved, evidence):
            return CoverageAssessment(scope_id=resolved.scope_id, evidence_id=evidence.evidence_id,
                                      status="satisfied", satisfied_item_ids=("a",))
    ctx = context()
    assessment, frozen = evaluate(CompletionService({"fake": Inconsistent()}), ctx)
    assert assessment.status == "error" and frozen[0].required_item_ids == ("a", "b")


def test_provider_version_change_blocks_existing_resolution():
    provider = FakeProvider()
    service = CompletionService({"fake": provider})
    ctx = context()
    _, frozen = evaluate(service, ctx)
    provider.contract_version = "2"
    assessment, after = evaluate(service, ctx, frozen)
    assert assessment.reason == "completion_provider_version_changed"
    assert after == frozen and provider.resolve_calls == 1


def test_missing_assessment_blocks_bounded_completion():
    decision = CortexController(20).decide(context())
    assert decision.completed_step_id is None and decision.next_step_id == "s"


def test_nested_specification_and_record_snapshots_are_immutable():
    class Nested(FakeProvider):
        def validate(self, specification):
            with pytest.raises(TypeError):
                specification["nested"]["items"][0]["value"] = 2
    provider = Nested()
    ctx = context()
    spec = {"nested": {"items": [{"value": 1}]}}
    step = ctx.active_step.model_copy(update={"completion_requirement": CoverageRequirement(
        provider_id="fake", specification=spec)})
    ctx = ctx.model_copy(update={"active_step": step,
        "active_plan": ctx.active_plan.model_copy(update={"steps": (step,)}),
        "tool_execution_history": (record(1),)})
    service = CompletionService({"fake": provider})
    ctx = ctx.model_copy(update={"accepted_requirements": service.bind_plan(ctx.identity, ctx.active_plan)[2]})
    assessment, _ = evaluate(service, ctx)
    assert assessment.status == "missing"
    assert spec["nested"]["items"][0]["value"] == 1


def test_acceptance_binding_survives_recovery_before_resolution():
    provider = FakeProvider()
    provider.items = None
    ctx = context()
    pending = ctx.active_step.model_copy(update={"status": StepStatus.PENDING})
    plan = ctx.active_plan.model_copy(update={"steps": (pending,)})
    state = ExecutionState(protocol_visible=ProtocolVisibleState(identity=ctx.identity,
        cursor=ExecutionCursor(phase=ExecutionPhase.PLANNING)))
    node = create_controller_node(completion_service=CompletionService({"fake": provider}))
    result = node({"execution_state": state, "planner_result": PlannerResult(
        outcome=PlannerOutcome.EXECUTION_PLAN, proposed_plan=plan)})
    restored = ExecutionState.model_validate_json(result["execution_state"].model_dump_json())
    assert not restored.protocol_visible.resolved_coverages
    binding = restored.protocol_visible.accepted_requirements[0]
    assert binding.provider_version == "1"
    provider.contract_version = "2"
    result = node({"execution_state": restored, "brain_result": ctx.brain_result})
    assert result["execution_state"].working.coverage_assessment.reason == "completion_provider_version_changed"
    assert result["execution_state"].protocol_visible.accepted_requirements == (binding,)
    assert result["controller_decision"].completed_step_id is None


def test_nested_mutation_cannot_rebind_same_revision():
    ctx = context()
    requirement = CoverageRequirement(provider_id="fake", specification={"nested": {"x": 1}})
    step = ctx.active_step.model_copy(update={"completion_requirement": requirement})
    plan = ctx.active_plan.model_copy(update={"steps": (step,)})
    service = CompletionService({"fake": FakeProvider()})
    _, _, bindings = service.bind_plan(ctx.identity, plan)
    before = bindings[0].specification_json
    requirement.specification["nested"]["x"] = 2
    ctx = ctx.model_copy(update={"active_plan": plan, "active_step": step, "accepted_requirements": bindings})
    assessment, frozen = evaluate(service, ctx)
    assert assessment.reason == "accepted_requirement_changed" and frozen == ()
    assert bindings[0].specification_json == before
    assert service.bind_plan(ctx.identity, plan, bindings)[1] == "accepted_requirement_changed"


def test_revision_evidence_filter_and_retry_accumulation():
    class FromEvidence(FakeProvider):
        def resolve(self, specification, evidence):
            return Resolution(tuple(r["result"]["request_id"] for r in evidence.records))
    service = CompletionService({"fake": FromEvidence()})
    old = record(1)
    current = record(2).model_copy(update={"plan_revision": 2})
    foreign = record(3).model_copy(update={"execution_id": "other", "plan_revision": 2})
    ctx = context(revision=2).model_copy(update={"tool_execution_history": (old, current, foreign)})
    assessment, frozen = evaluate(service, ctx)
    assert frozen[0].required_item_ids == ("2",)
    assert frozen[0].evidence_id == assessment.evidence_id
    retry = ctx.model_copy(update={"retry": ctx.retry.model_copy(update={"retry_count": 1}),
        "tool_execution_history": (old, current, record(4).model_copy(update={"plan_revision": 2}))})
    second, again = evaluate(service, retry, frozen)
    assert again == frozen and second.evidence_id != assessment.evidence_id


def test_final_answer_cannot_bypass_coverage():
    ctx = context()
    ctx = ctx.model_copy(update={"brain_result": BrainOutcome(
        outcome=BrainOutcomeKind.FINAL_ANSWER_READY, final_answer="Done")})
    assert CortexController(20).decide(ctx).completed_step_id is None
    provider = FakeProvider()
    provider.satisfied = ("a", "b")
    assessment, _ = evaluate(CompletionService({"fake": provider}), ctx)
    assert CortexController(20).decide(ctx.model_copy(update={"coverage_assessment": assessment})).completed_step_id == "s"
    semantic = context(False).model_copy(update={"brain_result": ctx.brain_result})
    assert CortexController(20).decide(semantic).completed_step_id == "s"


def test_service_and_controller_reject_active_step_disagreement():
    ctx = context()
    ctx = ctx.model_copy(update={"active_step": ctx.active_step.model_copy(update={"title": "different"})})
    with pytest.raises(ValueError, match="disagrees"):
        evaluate(CompletionService({"fake": FakeProvider()}), ctx)
    with pytest.raises(ValueError, match="disagrees"):
        CortexController(20).decide(ctx)
    ctx = context().model_copy(update={"cursor": context().cursor.model_copy(update={"plan_revision": 2})})
    with pytest.raises(ValueError, match="revision"):
        CortexController(20).decide(ctx)
