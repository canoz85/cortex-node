"""Source capabilities through existing Planner, Brain, Controller and replay."""

from datetime import datetime
from functools import partial
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from core.brain import BrainService
from core.brain_normalization import normalize_brain_output
from core.brain_provider import LangChainBrainProvider
from core.graph_constants import MUTATING_TOOLS
from core.graph_messages import ACCEPTED_FINALIZER_PROVENANCE, CONVERSATION_PROVENANCE_KEY
from core.graph_runner import run_prompt
from core.models import SourceUrlRequest
from core.planner import PlannerService
from core.planner_routing import LangChainPlannerRouter
from core.protocol.bridge import build_brain_input, build_controller_input
from core.protocol.controller import CortexController
from core.protocol.enums import BrainOutcomeKind as Kind, ControllerDecisionType as Decision, ExecutionStatus, PlannerOutcome
from core.protocol.models import ExecutionState, PlanningCapabilities, PlannerMemoryContext, PlannerMemoryFact, ProtocolVisibleState
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver, WorkerDispatchError
from core.runtime.tool_result_integration import SerializedToolRuntimePort, integrate_tool_result
from test_brain_outcomes import SequenceModel, brain_input, native_action
from test_planner_routing import RouterLLM
from test_planner_runtime_flow import StructuredModel, production_app
from test_planner_service import FakeProvider, planner_input
from test_source_ops import URL, BEFORE, AFTER, baseline_path, page
from tools import info_ops, source_ops as source, web_ops as web
from core.source_snapshots import SourceSnapshotStore
from tools.registry import (
    ToolRegistry, build_tool_registry, default_enabled_tools, get_tool_argument_schema,
    get_tool_definition, planning_capability_projection,
)


SOURCE_NAMES = ("web_fetch", "source_snapshot", "source_diff", "source_status")


@pytest.fixture
def case(tmp_path, monkeypatch):
    store = SourceSnapshotStore(tmp_path / "internal")
    fetch = Mock(return_value=page())
    monkeypatch.setattr(web, "fetch_web", fetch)
    monkeypatch.setattr(web, "create_connection", Mock(side_effect=AssertionError("Unexpected network")))
    tools = [*source.get_source_tools(str(tmp_path / "workspace"), store=store), *web.get_web_tools()]
    registry = ToolRegistry.from_tools(tools)
    lookup = partial(get_tool_argument_schema, registry=registry)
    context = brain_input()
    context = context.model_copy(update={"active_plan": context.active_plan.model_copy(update={"available_tools": SOURCE_NAMES})})
    state = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=context.identity, cursor=context.cursor, active_plan=context.active_plan,
        active_step=context.active_step, retry=context.retry, original_user_request=f"Inspect {URL}",
    ))
    controller = CortexController(24, argument_schema_for=lookup)
    runtime = SerializedToolRuntimePort(tools)
    driver = ExecutionDriver(coordinator=ControllerCoordinator(controller), planner=Mock(), brain=Mock(),
                             tool_runtime=runtime, finalizer=Mock())
    return SimpleNamespace(store=store, fetch=fetch, tools=tools, registry=registry, lookup=lookup,
                           context=context, state=state, controller=controller, runtime=runtime, driver=driver)


def test_production_registry_exposes_exactly_three_minimal_source_tools(tmp_path, monkeypatch):
    monkeypatch.setattr("core.source_snapshots.default_source_store_root", lambda: tmp_path / "internal")
    registry = build_tool_registry(str(tmp_path / "workspace"), str(tmp_path / "knowledge"), SimpleNamespace(), "test")
    assert {name for name in registry.names if name.startswith("source_")} == {"source_snapshot", "source_diff", "source_status"}
    assert {"source_snapshot", "source_diff", "source_status"} <= {tool.name for tool in default_enabled_tools(registry)}
    assert get_tool_definition("source_snapshot").mutating is True and "source_snapshot" in MUTATING_TOOLS
    assert get_tool_definition("source_diff").mutating is False and "source_diff" not in MUTATING_TOOLS
    assert get_tool_definition("source_status").mutating is False and "source_status" not in MUTATING_TOOLS
    for name in ("source_snapshot", "source_diff", "source_status"):
        definition = get_tool_definition(name)
        assert definition.args_schema is get_tool_argument_schema(name, registry=registry) is SourceUrlRequest
        assert not definition.exact_collection_outputs and definition.max_batch_calls == 1
        schema = convert_to_openai_tool(registry.by_name[name])["function"]["parameters"]
        assert schema["required"] == ["url"] and set(schema["properties"]) == {"url"}
        assert schema["properties"]["url"]["type"] == "string"
    assert SourceUrlRequest.model_json_schema()["additionalProperties"] is False
    assert not (tmp_path / "internal").exists()


def test_cards_distinguish_reading_saving_and_comparing_without_pagination_or_exact_collection():
    cards = {card["name"]: card for card in planning_capability_projection(SOURCE_NAMES)}
    assert cards["source_snapshot"]["mutating"] and not cards["source_diff"]["mutating"]
    assert "start tracking" in cards["source_snapshot"]["use_when"]
    assert "read/summarize" in cards["source_snapshot"]["avoid_when"]
    assert "saved baseline" in cards["source_diff"]["use_when"]
    assert "replace the baseline" in cards["source_diff"]["avoid_when"]
    assert "only needs its current content" in cards["web_fetch"]["use_when"]
    assert "source baseline" in cards["web_fetch"]["avoid_when"]
    status = cards["source_status"]
    assert status["mutating"] is False and "saved baseline itself" in status["use_when"]
    assert "authoritative over conversation history or Planner memory" in " ".join(status["limits"])
    assert "Does not fetch current web content" in " ".join(status["limits"])
    assert all("pagination" not in card and "exact_collection" not in card and "async_kind" not in card
               for card in cards.values())


@pytest.mark.parametrize("user_text,route,name", [
    (f"Track {URL} from now on.", "action", "source_snapshot"),
    (f"Save the current version of {URL} as my baseline.", "action", "source_snapshot"),
    (f"Has {URL} changed since I last saved it?", "info", "source_diff"),
    (f"What changed on {URL} since my baseline?", "info", "source_diff"),
    (f"Read {URL}.", "info", "web_fetch"),
    (f"When was the baseline for {URL} saved?", "info", "source_status"),
    (f"What hash did we save for {URL}?", "info", "source_status"),
    (f"How many characters are in the saved baseline for {URL}?", "info", "source_status"),
])
def test_scripted_route_and_capability_selection_preserve_source_distinctions(user_text, route, name):
    router = LangChainPlannerRouter(router_llm=RouterLLM(route=route))
    provider = FakeProvider({"result": "PLAN_PROPOSED", "objective": user_text, "steps": [{
        "step_id": "source", "title": user_text, "description": user_text, "primary_tool": name,
        "deliverable_kind": "semantic", "dependencies": [],
    }], "message": ""})
    value = planner_input()
    value = value.model_copy(update={
        "context": value.context.model_copy(update={"user_request": user_text}),
        "capabilities": PlanningCapabilities(available_tools=SOURCE_NAMES),
    })
    outcome = PlannerService(provider=provider, router=router, mutating_tools=MUTATING_TOOLS).run(value)
    assert outcome.outcome == PlannerOutcome.EXECUTION_PLAN and outcome.planner_route == route
    assert outcome.proposed_plan.steps[0].primary_tool == name
    assert outcome.proposed_plan.steps[0].deliverable_kind == "semantic"
    prompt = provider.messages[0][0].content
    expected = SOURCE_NAMES if route == "action" else ("web_fetch", "source_diff", "source_status")
    assert set(outcome.proposed_plan.available_tools) == set(expected)
    for tool_name in expected:
        assert f'"name":"{tool_name}"' in prompt
    if route == "info":
        assert '"name":"source_snapshot"' not in prompt


def test_info_route_rejects_mutating_snapshot_proposal_from_scripted_provider():
    provider = FakeProvider({"result": "PLAN_PROPOSED", "objective": "Read source", "steps": [{
        "step_id": "source", "title": "Read source", "description": f"Read {URL}",
        "primary_tool": "source_snapshot", "deliverable_kind": "semantic", "dependencies": [],
    }], "message": ""})
    value = planner_input().model_copy(update={"capabilities": PlanningCapabilities(available_tools=SOURCE_NAMES)})
    outcome = PlannerService(provider=provider, router=LangChainPlannerRouter(router_llm=RouterLLM(route="info")),
                             mutating_tools=MUTATING_TOOLS).run(value)
    assert outcome.outcome != PlannerOutcome.EXECUTION_PLAN
    assert '"name":"source_snapshot"' not in provider.messages[0][0].content


@pytest.mark.parametrize("name,route", [
    ("source_snapshot", "action"), ("source_diff", "info"), ("web_fetch", "info"), ("source_status", "info"),
])
def test_native_source_calls_complete_through_existing_production_controller(name, route, case):
    if name in {"source_diff", "source_status"}:
        assert source.snapshot_source(URL, case.store).success
        case.fetch.reset_mock()
    if name == "source_diff":
        case.fetch.return_value = page("Changed evidence.", at=AFTER)
    if name == "source_status":
        case.fetch.side_effect = AssertionError("Status must not fetch")
    before = baseline_path(case.store).read_bytes() if name in {"source_diff", "source_status"} else None
    replies = iter([native_action(name, {"url": URL}),
                    native_action("brain_step_completed", {"message": "Explained the returned source evidence."})])

    class Model(StructuredModel):
        def __init__(self):
            super().__init__([{"result": "PLAN_PROPOSED", "objective": "Use the source", "steps": [{
                "step_id": "source", "title": "Use the source", "description": f"Use {URL}",
                "primary_tool": name, "deliverable_kind": "semantic", "dependencies": [],
            }]}], route=route)
            self.brain_calls = []

        def invoke(self, messages):
            if any(message.content.startswith("Active step:") for message in messages):
                self.brain_calls.append(messages)
                return next(replies)
            return AIMessage(content="Source evidence explained.")

    model = Model()
    app = production_app(model, case.tools)
    run_id = f"source-production-{name}"
    history, _ = run_prompt(app, f"Use {URL}", run_id=run_id)
    state = app.get_state({"configurable": {"thread_id": run_id}}).values["execution_state"]
    assert state.protocol_visible.status == ExecutionStatus.COMPLETED
    record, = state.working.tool_execution_history
    assert record.tool_name == name and record.result.success
    assert case.fetch.call_count == (0 if name == "source_status" else 1)
    assert record.result.pagination is None and not record.result.is_async_job
    evidence = next(message.content for message in model.brain_calls[1]
                    if message.content.startswith("Execution evidence v1:"))
    data = json.loads(evidence.split("\n", 1)[1])["current_attempts"][0]["evidence"]
    if name == "source_snapshot":
        assert data["baseline_created"] is True and "text" not in data
        assert not record.artifacts  # Internal state is not a workspace artifact.
    elif name == "source_diff":
        assert data["changed"] is True and "+Changed evidence." in data["diff"]
        assert baseline_path(case.store).read_bytes() == before
    elif name == "source_status":
        assert data["baseline_exists"] is True and data["retrieved_at"] == page().data["retrieved_at"]
        assert data["canonical_url"] == URL and "text" not in data and not record.artifacts
        assert not record.result.integrity.is_truncated
        assert baseline_path(case.store).read_bytes() == before
    else:
        assert data["text"] == "Original text." and not case.store.root.exists()
    assert history[-1].content == "Source evidence explained."


@pytest.mark.parametrize("name", ["source_snapshot", "source_diff"])
def test_successful_source_calls_remain_guarded_after_checkpoint_and_authorization_replay(name, case):
    if name == "source_diff":
        assert source.snapshot_source(URL, case.store).success
        case.fetch.reset_mock()
    raw = native_action(name, {"url": URL})
    outcome = BrainService(provider=LangChainBrainProvider(brain_llm=SequenceModel(raw), executable_tools=case.tools),
                           agent_system_prompt="Use the source tool.").run(case.context)
    assert outcome.kind == Kind.TOOL_REQUESTED
    value = build_controller_input({"execution_state": case.state, "brain_result": outcome})
    turn = case.driver.turn(case.state, value)
    captured = integrate_tool_result(turn.execution_state, turn.decision, turn.worker_result)
    assert turn.worker_result.success and case.fetch.call_count == 1
    before = baseline_path(case.store).read_bytes()
    restored = ExecutionState.model_validate_json(captured.model_dump_json())
    proposal = outcome.model_copy(update={"tool_request": outcome.tool_request.model_copy(update={"request_id": "new-id"})})
    repeated = build_controller_input({"execution_state": restored, "brain_result": proposal}).model_copy(update={"tool_result": None})
    rejected = case.driver.turn(restored, repeated, dispatch_worker=False)
    assert rejected.decision.retry.last_error_code == "duplicate_successful_call"
    assert rejected.decision.pending_tool_request is None and case.fetch.call_count == 1
    with pytest.raises(WorkerDispatchError, match="duplicate_successful_call"):
        case.driver.dispatch_authorized(restored, turn.decision, value)
    provider = LangChainBrainProvider(brain_llm=SequenceModel(raw, native_action("brain_step_completed", {
        "message": "Reused the successful source evidence.",
    })), executable_tools=case.tools)
    corrected = provider.generate(build_brain_input({"execution_state": restored}), ())
    assert corrected.kind == Kind.STEP_COMPLETED and len(provider.brain_llm.calls) == 2
    assert case.fetch.call_count == 1 and baseline_path(case.store).read_bytes() == before
    assert len(restored.working.tool_execution_history) == 1


def test_diff_truncation_integrity_is_not_fake_pagination(case):
    case.fetch.return_value = page("old " * 10000)
    assert source.snapshot_source(URL, case.store).success
    case.fetch.return_value = page("new " * 10000)
    proposal = normalize_brain_output(native_action("source_diff", {"url": URL}), case.context, set(SOURCE_NAMES))
    result = case.runtime.execute(proposal.tool_request)
    assert result.success and result.data["changed"] and result.data["diff_truncated"]
    assert result.integrity.is_truncated and result.pagination is None
    assert result.integrity.original_bytes == result.data["total_diff_bytes"]
    assert result.integrity.captured_bytes == len(result.data["diff"].encode())
    assert len(result.data["diff"]) <= 8000


def test_incomplete_comparison_is_normalized_failure_without_unchanged_evidence(case):
    assert source.snapshot_source(URL, case.store).success
    case.fetch.return_value = page(truncated=True, total_chars=200001)
    proposal = normalize_brain_output(native_action("source_diff", {"url": URL}), case.context, set(SOURCE_NAMES))
    value = build_controller_input({"execution_state": case.state, "brain_result": proposal})
    turn = case.driver.turn(case.state, value)
    captured = integrate_tool_result(turn.execution_state, turn.decision, turn.worker_result)
    assert not turn.worker_result.success and turn.worker_result.error_code == "SOURCE_DIFF_INCOMPLETE"
    assert "changed" not in turn.worker_result.data
    continuation = case.driver.turn(captured, build_controller_input({"execution_state": captured}), dispatch_worker=False)
    assert continuation.decision.decision_type == Decision.DISPATCH_PLANNER
    assert continuation.decision.completed_step_id is None


def test_source_diff_in_new_execution_still_compares_original_baseline(case):
    assert source.snapshot_source(URL, case.store).success
    case.fetch.return_value = page("Changed evidence.", at=AFTER)
    before = baseline_path(case.store).read_bytes()
    results = []
    for execution_id in ("first-comparison", "next-comparison"):
        context = case.context.model_copy(update={"identity": case.context.identity.model_copy(update={"execution_id": execution_id})})
        outcome = LangChainBrainProvider(brain_llm=SequenceModel(native_action("source_diff", {"url": URL})),
                                         executable_tools=case.tools).generate(context, ())
        result = case.runtime.execute(outcome.tool_request)
        assert result.success and result.data["changed"]
        results.append(result.data)
    assert results[0] == results[1] and baseline_path(case.store).read_bytes() == before


def test_baseline_age_uses_fresh_status_evidence_over_stale_history_and_planner_memory(case, monkeypatch):
    assert source.snapshot_source(URL, case.store).success
    case.fetch.return_value = page("Replacement baseline.", at=AFTER)
    assert source.snapshot_source(URL, SourceSnapshotStore(case.store.root)).success
    original = baseline_path(case.store).read_bytes()
    case.fetch.reset_mock()
    case.fetch.side_effect = AssertionError("Reading saved metadata must not fetch")
    clock = Mock()
    clock.fromisoformat.side_effect = datetime.fromisoformat
    clock.now.return_value.astimezone.return_value = datetime.fromisoformat("2026-10-17T12:10:00+03:00")
    monkeypatch.setattr(info_ops, "datetime", clock)
    current_time = next(tool for tool in info_ops.get_info_tools("test", "workspace") if tool.name == "current_time")

    class Model(StructuredModel):
        def __init__(self):
            super().__init__([{"result": "PLAN_PROPOSED", "objective": "Read baseline and calculate its age", "steps": [
                {"step_id": "saved", "title": "Read saved baseline metadata", "description": f"Read persistent metadata for {URL}",
                 "primary_tool": "source_status", "deliverable_kind": "semantic", "dependencies": []},
                {"step_id": "age", "title": "Calculate baseline age", "description": "Pass saved retrieved_at to current_time as reference_time",
                 "primary_tool": "current_time", "deliverable_kind": "semantic", "dependencies": ["saved"]},
            ]}], route="info")
            self.brain_calls = 0

        def invoke(self, messages):
            if any(message.content.startswith("Active step:") for message in messages):
                self.brain_calls += 1
                if self.brain_calls == 1:
                    return native_action("source_status", {"url": URL})
                if self.brain_calls in {2, 4}:
                    evidence = next(message.content for message in messages
                                    if message.content.startswith("Execution evidence v1:"))
                    data = json.loads(evidence.split("\n", 1)[1])["current_attempts"][0]["evidence"]
                    if self.brain_calls == 2:
                        self.saved_metadata = data
                    else:
                        self.elapsed = data["elapsed_seconds"]
                    return native_action("brain_step_completed", {"message": "Read the authoritative runtime evidence."})
                assert self.brain_calls == 3
                return native_action("current_time", {"reference_time": self.saved_metadata["retrieved_at"]})
            return AIMessage(content=f"The saved baseline is {self.elapsed} seconds old.")

    model = Model()
    app = production_app(model, [*case.tools, current_time])
    previous = [HumanMessage(content=f"Save {URL} as my baseline."), AIMessage(
        content=f"Baseline saved at {BEFORE}.",
        additional_kwargs={CONVERSATION_PROVENANCE_KEY: ACCEPTED_FINALIZER_PROVENANCE},
    )]
    memory = PlannerMemoryContext(project_facts=(PlannerMemoryFact(
        category="source", text=f"The baseline for {URL} was saved at {BEFORE}.",
        authority="tool_supported", source_turn_index=1,
    ),))
    run_id = "source-status-fresh-baseline-age"
    history, _ = run_prompt(app, "son kaydedilen site için baseline saati ile şimdiki zaman farkı ne kadar",
                            history=previous, planner_memory_context=memory, run_id=run_id)
    state = app.get_state({"configurable": {"thread_id": run_id}}).values["execution_state"]
    assert state.protocol_visible.status == ExecutionStatus.COMPLETED
    status, elapsed = state.working.tool_execution_history
    assert status.tool_name == "source_status" and status.result.data["retrieved_at"] == AFTER
    assert elapsed.tool_name == "current_time" and elapsed.arguments == {"reference_time": AFTER}
    assert elapsed.result.data["elapsed_seconds"] == 600.0 and model.elapsed == 600.0
    assert history[-1].content == "The saved baseline is 600.0 seconds old."
    assert any(message.content.startswith("PLANNER MEMORY CONTEXT") and BEFORE in message.content
               for _schema, messages in model.exchanges for message in messages)
    assert baseline_path(case.store).read_bytes() == original
    case.fetch.assert_not_called()
