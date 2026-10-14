"""Semantic completion after mutation keeps provenance under Controller authority."""

import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from core.brain import BrainService
from core.brain_provider import LangChainBrainProvider
from core.graph_constants import MUTATING_TOOLS
from core.planner import PlannerService
from core.protocol.completion_identity import evidence_identity
from core.protocol.controller import CortexController
from core.protocol.enums import BrainOutcomeKind as Kind, ControllerDecisionType as Decision, StepStatus
from core.protocol.models import (
    BrainInput, BrainOutcome, ExactCollection, ExecutionContext, ExecutionState,
    PlanningCapabilities, StepCompletionEvidence,
)
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver
from core.runtime.tool_result_integration import SerializedToolRuntimePort, integrate_tool_result
from test_brain_outcomes import FakeModel, SequenceModel, native_action
from test_execution_driver import FinalizerService, controller_input, initial_state
from test_planner_service import FakeProvider, FakeRouter
from tools.discovery_ops import get_discovery_tools
from tools.file_ops import get_file_tools


def input_for(state, result=None):
    return controller_input(state, result).model_copy(update={
        "context": ExecutionContext(user_request="In x.txt replace night with day."),
    })


@pytest.fixture
def patched_completion(tmp_path):
    target = tmp_path / "x.txt"
    before = b"The stars shine bright in the velvet night.\r\n"
    target.write_bytes(before)
    tools = [*get_file_tools(str(tmp_path)), *get_discovery_tools(str(tmp_path))]
    calls = []

    @contextmanager
    def observe(request):
        calls.append(request.tool_name)
        yield

    summary = "Replaced the single occurrence of night with day in x.txt."
    model = SequenceModel(
        native_action("find_files", {"path": "", "pattern": "x.txt", "recursive": True}),
        native_action("read_file", {"path": "x.txt"}),
        native_action("apply_patch", {"path": "x.txt", "old_text": "night", "new_text": "day"}),
        native_action("brain_step_completed", {"message": summary}),
    )
    proposal = {
        "result": "PLAN_PROPOSED", "objective": "Replace night with day", "message": "",
        "steps": [{"step_id": "patch", "title": "Patch x.txt",
                   "description": "Replace the single occurrence of night with day in x.txt.",
                   "deliverable_kind": "semantic", "primary_tool": "apply_patch", "dependencies": []}],
    }
    planner = PlannerService(provider=FakeProvider(proposal), router=FakeRouter("action"),
                             mutating_tools=MUTATING_TOOLS)
    brain = BrainService(provider=LangChainBrainProvider(brain_llm=model, executable_tools=tools),
                         agent_system_prompt="Execute the active step.")
    controller = CortexController(24, planning_capabilities=PlanningCapabilities(
        available_tools=tuple(tool.name for tool in tools),
    ))
    driver = ExecutionDriver(
        coordinator=ControllerCoordinator(controller), planner=planner, brain=brain,
        tool_runtime=SerializedToolRuntimePort(tools, observe=observe), finalizer=FinalizerService(),
    )
    state = initial_state()
    planned = driver.turn(state, input_for(state))
    turn = driver.turn(planned.execution_state, input_for(planned.execution_state, planned.worker_result))
    for name in ("find_files", "read_file", "apply_patch"):
        assert turn.worker_result.tool_request.tool_name == name
        executed = driver.turn(turn.execution_state, input_for(turn.execution_state, turn.worker_result))
        assert executed.decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME
        assert executed.worker_result.success
        captured = integrate_tool_result(executed.execution_state, executed.decision, executed.worker_result)
        turn = driver.turn(captured, input_for(captured, executed.worker_result))

    assert turn.worker_result.kind == Kind.STEP_COMPLETED
    records = turn.execution_state.working.tool_execution_history
    receipt = records[2].result
    assert receipt.data["replacements"] == 1
    assert receipt.data["before_sha256"] == hashlib.sha256(before).hexdigest()
    assert receipt.data["after_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert "artifacts" not in receipt.data
    evidence_message = next(m for m in model.calls[-1] if m.content.startswith("Execution evidence v1:"))
    displayed = json.loads(evidence_message.content.split("\n", 1)[1])["current_attempts"][2]
    assert displayed["record_index"] == 2
    assert displayed["artifacts"] == [{"path": "x.txt", "action": "modified"}]
    assert "artifacts" not in displayed["evidence"]
    return SimpleNamespace(driver=driver, controller=controller, state=turn.execution_state,
                           completion=turn.worker_result, calls=calls, target=target, model=model,
                           tools=tools, summary=summary)


def test_semantic_patch_completion_binds_all_provenance_without_reexecuting_mutation(patched_completion):
    case = patched_completion
    checkpoint = case.state.model_dump_json()
    after_patch = case.target.read_bytes()
    records = case.state.working.tool_execution_history
    accepted = case.driver.turn(case.state, input_for(case.state, case.completion))
    assert accepted.decision.completed_step_id == "patch"
    assert accepted.decision.accepted_plan.steps[0].status == StepStatus.COMPLETED
    evidence = accepted.decision.completion_evidence
    protocol = case.state.protocol_visible
    assert (evidence.execution_id, evidence.plan_id, evidence.plan_revision, evidence.step_id) == (
        protocol.identity.execution_id, protocol.active_plan.plan_id, protocol.active_plan.revision, "patch",
    )
    assert evidence.summary == case.summary
    assert evidence.tool_request_ids == tuple(record.result.request_id for record in records)
    assert evidence.evidence_id == evidence_identity(records)
    assert evidence.exact_collection is None
    assert accepted.execution_state.protocol_visible.completion_provenance == (evidence,)
    assert accepted.worker_result.kind == Kind.FINAL_ANSWER_READY

    restored = ExecutionState.model_validate_json(checkpoint)
    replayed = case.driver.turn(restored, input_for(restored, case.completion))
    assert replayed.decision.completion_evidence == evidence
    assert replayed.execution_state.protocol_visible.completion_provenance == (evidence,)
    assert case.calls == ["find_files", "read_file", "apply_patch"]
    assert len(case.model.calls) == 4
    assert case.target.read_bytes() == after_patch == b"The stars shine bright in the velvet day.\r\n"


def test_live_patch_collection_locators_are_rejected_at_native_boundary(patched_completion):
    case = patched_completion
    value = input_for(case.state)
    model = FakeModel(native_action("brain_step_completed", {
        "message": case.summary, "exact_collection_source_record_index": 2,
        "exact_collection_data_path": ["artifacts"],
    }))
    provider = LangChainBrainProvider(brain_llm=model, executable_tools=case.tools)
    outcome = provider.generate(BrainInput(
        identity=value.identity, cursor=value.cursor, context=value.context,
        active_plan=value.active_plan, active_step=value.active_step,
        tool_execution_history=value.tool_execution_history,
    ), ())
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.error_code == "unexpected_envelope_fields"
    assert outcome.completion_evidence is None
    assert len(model.calls) == 1
    decision = case.controller.decide(input_for(case.state, outcome))
    assert decision.completed_step_id is decision.completion_evidence is None
    assert decision.retry.last_error_code == "unexpected_envelope_fields"
    assert case.calls == ["find_files", "read_file", "apply_patch"]
    assert case.target.read_bytes() == b"The stars shine bright in the velvet day.\r\n"


def test_internal_patch_collection_cannot_traverse_artifact_metadata(patched_completion):
    case = patched_completion
    internal = BrainOutcome(
        outcome=Kind.STEP_COMPLETED, step_id="patch",
        completion_evidence=StepCompletionEvidence(
            step_id="patch", summary=case.summary,
            exact_collection=ExactCollection(source_record_index=2, data_path=("artifacts",)),
        ),
    )
    decision = case.controller.decide(input_for(case.state, internal))
    assert decision.completed_step_id is decision.completion_evidence is None
    assert decision.retry.last_error_code == "invalid_exact_collection"
    assert decision.failure_reason == "Brain completion cannot supply exact collection provenance"
    assert case.calls == ["find_files", "read_file", "apply_patch"]
    assert case.target.read_bytes() == b"The stars shine bright in the velvet day.\r\n"
