"""Planner semantics gate generic Controller binding, rendering and replay."""

from collections import Counter
from contextlib import contextmanager
import json

import pytest
from langchain_core.messages import AIMessage

from core.brain import BrainService
from core.brain_evidence_policy import MAX_CURRENT_ATTEMPT_RECORDS
from core.brain_provider import LangChainBrainProvider
from core.finalizer import Finalizer
from core.finalizer_provider import LangChainFinalAnswerRenderer
from core.planner import PlannerService
from core.planner_normalization import normalize_planner_proposal
from core.planner_validation import PlanValidationError, validate_execution_plan
from core.protocol.completion_identity import evidence_identity
from core.protocol.controller import CortexController
from core.protocol.enums import BrainOutcomeKind, ExecutionStatus, PlannerOutcome
from core.protocol.models import (
    AcceptedStepResult, ArtifactRecord, BrainOutcome, ContentIntegrity, ExecutionContext,
    ExecutionState, FinalizationRequest, PaginationMetadata, PlanningCapabilities,
    ToolExecutionRecord, ToolRequest, ToolResult,
)
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver
from core.runtime.tool_result_integration import SerializedToolRuntimePort, integrate_tool_result
from test_brain_outcomes import SequenceModel, brain_input, controller_input, native_action
from test_execution_driver import controller_input as state_input, initial_state
from test_finalizer_rendering import Model
from test_planner_service import FakeProvider, FakeRouter, planner_input
from tools.discovery_ops import get_discovery_tools
from tools.file_ops import get_file_tools
from tools.registry import ExactCollectionOutput, ToolDefinition, get_tool_definition, planning_capability_projection


def proposal(kind="exact_enumeration", description="List the files in the workspace root.", tool="list_files"):
    return {"result": "PLAN_PROPOSED", "objective": description, "message": "", "steps": [{
        "step_id": "s1", "title": description, "description": description,
        "primary_tool": tool, "deliverable_kind": kind, "dependencies": [],
    }]}


@pytest.mark.parametrize("kind", ["semantic", "exact_enumeration"])
def test_normalization_and_validation_preserve_deliverable_semantics(kind):
    request = planner_input()
    result = normalize_planner_proposal(proposal(kind), request, route="info")
    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    plan = validate_execution_plan(result.proposed_plan, request, route="info")
    assert plan.steps[0].deliverable_kind == kind
    forged = plan.model_copy(update={"steps": (plan.steps[0].model_copy(update={"deliverable_kind": "all"}),)})
    with pytest.raises(PlanValidationError, match="invalid_execution_plan_model"):
        validate_execution_plan(forged, request, route="info")


@pytest.mark.parametrize("kind", [None, "exact", True])
def test_normalizer_rejects_invalid_deliverable_kind(kind):
    result = normalize_planner_proposal(proposal(kind), planner_input(), route="info")
    assert result.outcome == PlannerOutcome.FAILED
    assert result.proposed_plan is None


def test_normalizer_rejects_missing_kind_and_legacy_checkpoint_defaults_to_semantic():
    payload = proposal()
    del payload["steps"][0]["deliverable_kind"]
    assert normalize_planner_proposal(payload, planner_input(), route="info").outcome == PlannerOutcome.FAILED
    from core.protocol.models import ExecutionStep
    assert ExecutionStep.model_validate({"step_id": "s", "title": "Legacy step"}).deliverable_kind == "semantic"


def test_registry_declares_only_trusted_output_locations_and_hides_mechanics_from_planner():
    expected = {"list_files": "entries", "find_files": "files",
                "git_changed_files": "changed_files", "search_text": "matches"}
    for name, field in expected.items():
        assert get_tool_definition(name).exact_collection_outputs == (ExactCollectionOutput((field,)),)
    assert ToolDefinition("future", "workspace").exact_collection_outputs == ()
    assert get_tool_definition("read_file").exact_collection_outputs == ()
    cards = planning_capability_projection(expected)
    assert all("exact_collection_outputs" not in card and "data_path" not in card for card in cards)


@pytest.mark.parametrize("path", [(), (0,), ("",), ("entries", False), ["entries"]])
def test_registry_rejects_invalid_trusted_paths(path):
    with pytest.raises(ValueError, match="nonempty field path"):
        ExactCollectionOutput(path)


def record(context, name="list_files", data=None, request_id="list", **result_fields):
    return ToolExecutionRecord(
        execution_id=context.identity.execution_id, plan_id=context.active_plan.plan_id,
        plan_revision=context.active_plan.revision, step_id=context.active_step.step_id, tool_name=name,
        result=ToolResult(request_id=request_id, success=result_fields.pop("success", True),
                          message="Evidence captured", data=data, **result_fields),
    )


def complete(context, records):
    outcome = BrainOutcome(outcome=BrainOutcomeKind.STEP_COMPLETED, step_id=context.active_step.step_id,
                           message="A summary that omits every filename.")
    return CortexController(24).decide(controller_input(context, outcome).model_copy(
        update={"tool_execution_history": tuple(records)}))


def rendered(context, decision, *, provider=False):
    model = Model(AssertionError("Exact rendering must not ask a model to rewrite members"))
    request = FinalizationRequest(
        identity=context.identity, status=ExecutionStatus.COMPLETED, context=context.context,
        accepted_plan=decision.accepted_plan, completed_step_ids=(context.active_step.step_id,),
        accepted_step_results=(AcceptedStepResult(completion_evidence=decision.completion_evidence),),
    )
    finalizer = Finalizer(answer_renderer=LangChainFinalAnswerRenderer(llm=model)) if provider else Finalizer()
    result = finalizer.finalize(request)
    assert result.final_answer_error is None
    assert model.calls == []
    return result.final_answer


@pytest.mark.parametrize("provider", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_exact_members_order_case_unicode_and_duplicate_occurrences_survive(provider, grouped):
    context = brain_input(deliverable_kind="exact_enumeration")
    request = "List the files in the workspace root" + (" and group them by file type." if grouped else ".")
    context = context.model_copy(update={"context": ExecutionContext(user_request=request)})
    members = ["Z.py", "image.PNG", "README", "Z.py", "résumé.txt", "a.py"]
    source = record(context, data={"entries": members})
    decision = complete(context, [source])
    evidence = decision.completion_evidence
    assert evidence.exact_collection.items == tuple(members)
    assert evidence.exact_collection.source_request_id == source.result.request_id
    assert evidence.evidence_id == evidence_identity((source,))
    answer = rendered(context, decision, provider=provider)
    lines = [line[2:] for line in answer.splitlines() if line.startswith("- ")]
    assert Counter(lines) == Counter(members)
    assert len(lines) == len(members)
    if grouped:
        assert lines == ["Z.py", "Z.py", "a.py", "image.PNG", "README", "résumé.txt"]
        assert ".py:\n" in answer and ".PNG:\n" in answer and "No extension:\n" in answer
    else:
        assert lines == members


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("same_members", [False, True])
def test_ambiguous_collections_fail_closed_even_if_their_members_match(reverse, same_members):
    context = brain_input(deliverable_kind="exact_enumeration")
    records = [record(context, data={"entries": ["a.py"]}),
               record(context, "find_files", {"files": ["a.py" if same_members else "b.py"]}, "find")]
    decision = complete(context, reversed(records) if reverse else records)
    assert decision.terminal and decision.execution_status == ExecutionStatus.FAILED
    assert decision.reason == "exact_collection_unresolved"
    assert "found 2" in decision.failure_reason
    assert decision.completion_evidence is None and decision.completed_step_id is None
    assert decision.pending_tool_request is None and decision.retry is None


def test_one_trusted_collection_among_supporting_arrays_and_foreign_records_binds():
    context = brain_input(deliverable_kind="exact_enumeration")
    source = record(context, "find_files", {"files": ["src/a.py", "src/a.py"]}, "find")
    foreign = [source.model_copy(update=scope) for scope in (
        {"execution_id": "foreign"}, {"plan_id": "foreign"}, {"plan_revision": 2},
        {"plan_revision": 0}, {"step_id": "other"},
    )]
    supporting = [record(context, "read_file", {"entries": ["not-authoritative"], "content": "code"}, "read"),
                  record(context, "unknown_tool", ["arbitrary-array"], "unknown"),
                  record(context, data={"entries": ["failed"]}, request_id="failed", success=False)]
    records = [*foreign, *supporting, source]
    decision = complete(context, records)
    evidence = decision.completion_evidence
    assert evidence.tool_request_ids == ("read", "unknown", "failed", "find")
    assert evidence.evidence_id == evidence_identity((*supporting, source))
    assert evidence.exact_collection.source_record_index == 3
    assert evidence.exact_collection.source_request_id == "find"
    assert evidence.exact_collection.items == ("src/a.py", "src/a.py")


def test_full_step_history_can_bind_a_collection_outside_brain_evidence_window():
    context = brain_input(deliverable_kind="exact_enumeration")
    source = record(context, data={"entries": ["first.py"]})
    records = [source, *(record(context, "read_file", {"entries": ["support"]}, f"read-{i}")
                         for i in range(MAX_CURRENT_ATTEMPT_RECORDS + 1))]
    decision = complete(context, records)
    assert decision.completion_evidence.exact_collection.source_record_index == 0
    assert decision.completion_evidence.exact_collection.items == ("first.py",)
    assert len(decision.completion_evidence.tool_request_ids) == len(records)


@pytest.mark.parametrize("data,fields", [
    ({"entries": "scalar"}, {}), ({"other": ["arbitrary"]}, {}), (None, {}),
    ({"entries": ["partial"]}, {"integrity": ContentIntegrity(is_truncated=True)}),
    ({"entries": ["partial"]}, {"pagination": PaginationMetadata(has_more=True, total_items=2, returned_items=1)}),
    ({"entries": ["last-page"]}, {"pagination": PaginationMetadata(offset=1, total_items=2, returned_items=1)}),
    ({"entries": ["inconsistent"]}, {"pagination": PaginationMetadata(total_items=2, returned_items=1)}),
])
def test_missing_invalid_or_partial_collections_are_unresolved(data, fields):
    context = brain_input(deliverable_kind="exact_enumeration")
    decision = complete(context, [record(context, data=data, **fields)])
    assert decision.reason == "exact_collection_unresolved" and decision.terminal
    assert decision.completion_evidence is None and "found 0" in decision.failure_reason


def test_empty_collection_binds_and_artifact_siblings_cannot_supply_missing_data():
    context = brain_input(deliverable_kind="exact_enumeration")
    artifact = ArtifactRecord(artifact_id="a", path="decoy.txt", action="created", step_id="s1")
    source = record(context, data={"entries": []}).model_copy(update={"artifacts": (artifact,)})
    decision = complete(context, [source])
    assert decision.completion_evidence.exact_collection.items == ()
    assert rendered(context, decision) == "Results:\n- (none)"
    source = source.model_copy(update={"result": source.result.model_copy(update={"data": None})})
    assert complete(context, [source]).reason == "exact_collection_unresolved"


@pytest.mark.parametrize("second_output", [False, True])
def test_metadata_supports_nested_paths_and_multiple_outputs_without_tool_specific_code(monkeypatch, second_output):
    import tools.registry as registry
    outputs = (ExactCollectionOutput(("catalog", "members")),)
    if second_output:
        outputs += (ExactCollectionOutput(("other",)),)
    definition = ToolDefinition("future_catalog", "general", exact_collection_outputs=outputs)
    monkeypatch.setattr(registry, "TOOL_DEFINITIONS", (*registry.TOOL_DEFINITIONS, definition))
    context = brain_input(deliverable_kind="exact_enumeration")
    source = record(context, "future_catalog", {"catalog": {"members": ["ID:AbC", "ID:AbC"]},
                                                "other": ["ignored"]})
    decision = complete(context, [source])
    if second_output:
        assert decision.reason == "exact_collection_unresolved"
        assert "found 2" in decision.failure_reason
    else:
        exact = decision.completion_evidence.exact_collection
        assert exact.items == ("ID:AbC", "ID:AbC") and exact.data_path == ("catalog", "members")


def make_driver(tmp_path, request, kind, actions, summary, *, final_model=None):
    tools = [*get_file_tools(str(tmp_path)), *get_discovery_tools(str(tmp_path))]
    calls = []

    @contextmanager
    def observe(tool_request):
        calls.append(tool_request.tool_name)
        yield

    model = SequenceModel(*(native_action(name, args) for name, args in actions),
                          native_action("brain_step_completed", {"message": summary}))
    planner = PlannerService(provider=FakeProvider(proposal(kind, request, actions[0][0])),
                             router=FakeRouter("action"), mutating_tools={"write_file"})
    brain = BrainService(provider=LangChainBrainProvider(brain_llm=model, executable_tools=tools),
                         agent_system_prompt="Execute the active step.")
    runtime = SerializedToolRuntimePort(tools, observe=observe)
    driver = ExecutionDriver(
        coordinator=ControllerCoordinator(CortexController(24, planning_capabilities=PlanningCapabilities(
            available_tools=tuple(tool.name for tool in tools)))), planner=planner, brain=brain,
        tool_runtime=runtime, finalizer=Finalizer(answer_renderer=LangChainFinalAnswerRenderer(llm=final_model))
        if final_model is not None else Finalizer(),
    )

    def input_for(state, result=None):
        return state_input(state, result).model_copy(update={"context": ExecutionContext(user_request=request)})

    state = initial_state()
    planned = driver.turn(state, input_for(state))
    turn = driver.turn(planned.execution_state, input_for(planned.execution_state, planned.worker_result))
    for name, _ in actions:
        assert turn.worker_result.tool_request.tool_name == name
        executed = driver.turn(turn.execution_state, input_for(turn.execution_state, turn.worker_result))
        assert executed.worker_result.success
        captured = integrate_tool_result(executed.execution_state, executed.decision, executed.worker_result)
        turn = driver.turn(captured, input_for(captured, executed.worker_result))
    assert turn.worker_result.kind == BrainOutcomeKind.STEP_COMPLETED
    accepted = driver.turn(turn.execution_state, input_for(turn.execution_state, turn.worker_result))
    return driver, accepted, input_for, calls, model, runtime


@pytest.mark.parametrize("grouped", [False, True])
def test_root_enumeration_native_completion_and_checkpoint_replay_do_not_reselect_or_execute(tmp_path, monkeypatch, grouped):
    members = ["a.py", "z.py", "README.md", "photo.png"]
    for name in members:
        (tmp_path / name).write_text("fixture", encoding="utf-8")
    request = "List the files in the workspace root" + (" and group them by file type." if grouped else ".")
    driver, accepted, input_for, calls, model, runtime = make_driver(
        tmp_path, request, "exact_enumeration", [("list_files", {"path": "."})],
        "Python scripts: 2 files. Text/markdown: 1 file. Image files: 1 file.")
    evidence = accepted.decision.completion_evidence
    assert Counter(evidence.exact_collection.items) == Counter(members)
    assert accepted.execution_state.protocol_visible.completion_provenance == (evidence,)
    restored = ExecutionState.model_validate_json(accepted.execution_state.model_dump_json())

    def unexpected(*args, **kwargs):
        raise AssertionError("Accepted exact results must not be reselected or tools re-executed")

    monkeypatch.setattr(CortexController, "_resolve_exact_collection", unexpected)
    monkeypatch.setattr(runtime, "execute", unexpected)
    terminal = driver.turn(restored, input_for(restored, accepted.worker_result))
    assert terminal.worker_result.final_answer_error is None
    answer = terminal.worker_result.final_answer
    assert Counter(line[2:] for line in answer.splitlines() if line.startswith("- ")) == Counter(members)
    assert "2 files" not in answer
    assert terminal.execution_state.protocol_visible.completion_provenance == (evidence,)
    assert calls == ["list_files"] and len(model.calls) == 2
    if grouped:
        assert ".py:\n" in answer and ".md:\n" in answer and ".png:\n" in answer


def test_snake_discovery_is_semantic_and_does_not_force_candidate_list_into_final_answer(tmp_path):
    for name in ("snake_ui.py", "snake_legacy.py", "snake_test.py"):
        (tmp_path / name).write_text("def snake_ui(): pass", encoding="utf-8")
    finding = "The Snake UI is implemented by snake_ui() in snake_ui.py."
    final_model = Model(AIMessage(content=finding))
    driver, accepted, input_for, calls, model, _ = make_driver(
        tmp_path, "Find the Snake UI implementation.", "semantic", [
            ("find_files", {"pattern": "snake*"}), ("read_file", {"path": "snake_ui.py"}),
        ], finding, final_model=final_model)
    assert accepted.decision.completion_evidence.exact_collection is None
    terminal = driver.turn(accepted.execution_state, input_for(accepted.execution_state, accepted.worker_result))
    assert terminal.worker_result.final_answer == finding
    facts = json.loads(final_model.calls[0][-1].content.split("\n", 1)[1])
    assert "tool_execution_history" not in facts
    assert "snake_legacy.py" not in terminal.worker_result.final_answer
    assert calls == ["find_files", "read_file"] and len(model.calls) == 3


def test_failed_binding_after_successful_mutation_never_retries_tools(tmp_path):
    driver, failed, _, calls, model, _ = make_driver(
        tmp_path, "Create receipt.txt and list the files.", "exact_enumeration", [
            ("write_file", {"path": "receipt.txt", "content": "once"}),
            ("list_files", {"path": "."}), ("find_files", {"pattern": "*"}),
        ], "Created and enumerated.")
    assert failed.decision.terminal and failed.decision.reason == "exact_collection_unresolved"
    assert failed.decision.execution_status == ExecutionStatus.FAILED
    assert failed.decision.retry is None and failed.decision.pending_tool_request is None
    assert calls == ["write_file", "list_files", "find_files"] and len(model.calls) == 4
    assert (tmp_path / "receipt.txt").read_text(encoding="utf-8") == "once"
    assert failed.worker_result.final_answer.startswith("Execution failed.")


@pytest.mark.parametrize("tool_name,arguments,field", [
    ("find_files", {"pattern": "*.py"}, "files"),
    ("search_text", {"query": "needle"}, "matches"),
])
def test_normalized_discovery_outputs_bind_and_structured_matches_render_without_loss(tmp_path, tool_name, arguments, field):
    (tmp_path / "a.py").write_text("needle α\nneedle β\n", encoding="utf-8")
    runtime = SerializedToolRuntimePort(get_discovery_tools(str(tmp_path)))
    result = runtime.execute(ToolRequest(request_id="discovery", tool_name=tool_name, arguments=arguments))
    context = brain_input(deliverable_kind="exact_enumeration")
    source = record(context, tool_name).model_copy(update={"result": result})
    decision = complete(context, [source])
    assert decision.completion_evidence.exact_collection.items == tuple(result.data[field])
    answer = rendered(context, decision, provider=True)
    expected = [item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                for item in result.data[field]]
    assert answer.splitlines()[1:] == [f"- {item}" for item in expected]
