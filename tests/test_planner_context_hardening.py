"""Episode-local feedback and authoritative capability context, without live tools."""

from dataclasses import FrozenInstanceError
import json

import pytest
from pydantic import ValidationError

from core.graph_constants import MUTATING_TOOLS
from core.planner import PlannerService, PLANNER_SYSTEM_PROMPT
from core.planner_feedback import PLANNING_FEEDBACK_HEADER
from core.planner_normalization import normalize_planner_proposal
from core.protocol.bridge import build_controller_input
from core.protocol.controller import CortexController, apply_controller_decision_to_state
from core.protocol.enums import BrainOutcome, ControllerDecisionType, PlannerOutcome, PlanningFailureCategory
from core.protocol.models import (
    BrainResult, ExecutionCursor, ExecutionIdentity, ExecutionState, PlanningCapabilities,
    PlanningFeedback, PlannerMemoryContext, PlannerMemoryFact, ProtocolVisibleState, ReplanRequest,
)
from test_planner_service import FakeProvider, FakeRouter
from test_planner_boundary_hardening import revision_context
from tools.registry import (
    CapabilityMetadataError, CapabilitySemantics, TOOL_DEFINITIONS, ToolDefinition,
    get_tool_definition, planning_capability_projection,
)


def initial_episode():
    ctrl = CortexController(24, planning_capabilities=PlanningCapabilities(
        available_tools=("read_file", "find_files", "write_file"), constraints=("Inspect only approved paths",)))
    memory = PlannerMemoryContext(user_facts=(PlannerMemoryFact(category="preference", text="Remembered preference",
                                    authority="explicit_user", source_turn_index=0),))
    state = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=ExecutionIdentity(execution_id="phase-two", protocol_version="1"), cursor=ExecutionCursor()))
    dispatch = ctrl.decide(build_controller_input({"execution_state": state,
        "user_input": "Inspect project and report findings", "retrieval_messages": ("Knowledge background",),
        "planner_memory_context": memory}))
    return ctrl, apply_controller_decision_to_state(state, dispatch)


def fail_and_retry(ctrl, state):
    request = state.protocol_visible.planning_request
    result = normalize_planner_proposal({"result": "PLAN_PROPOSED", "steps": [{
        "step_id": "read", "title": "Read", "description": "Read files",
        "primary_tool": "read_file", "dependencies": ["missing"],
    }]}, request, route="info")
    assert result.failure_category == PlanningFailureCategory.INVALID_OUTPUT
    retry = ctrl.decide(build_controller_input({"execution_state": state, "planner_result": result}))
    return apply_controller_decision_to_state(state, retry)


def valid_proposal(dependencies=()):
    return {"result": "PLAN_PROPOSED", "steps": [{"step_id": "new", "title": "Inspect",
        "description": "Inspect and report files", "primary_tool": "read_file", "dependencies": dependencies}]}


def section(messages, label):
    return next(message.content for message in messages if message.content.startswith(label))


def capabilities_in(messages):
    line = messages[0].content.split("AVAILABLE CAPABILITIES FOR THIS REQUEST (CLOSED SET):\n", 1)[1].splitlines()[0]
    return json.loads(line)


def test_retry_feedback_is_typed_and_separate_from_all_other_context():
    ctrl, state = initial_episode()
    original = state.protocol_visible.planning_request
    retry = fail_and_retry(ctrl, state).protocol_visible.planning_request
    assert retry.context == original.context
    assert retry.feedback == PlanningFeedback(message="step 'read' references unknown dependency 'missing'")
    assert retry.attempt == 2 and retry.max_attempts == original.max_attempts == 2
    assert retry.episode_id == original.episode_id
    assert retry.context.retrieval_messages == ("Knowledge background",)
    provider = FakeProvider(valid_proposal())
    service = PlannerService(provider=provider, router=FakeRouter("action"), mutating_tools=MUTATING_TOOLS)
    service.run(retry)
    messages = provider.messages[0]
    assert messages[-1].content == original.context.user_request
    constraints = json.loads(section(messages, "CONTROLLER PLANNING CONSTRAINTS").split("\n", 1)[1])
    assert constraints == {"controller_planning_constraints": ["Inspect only approved paths"]}
    retrieval = section(messages, "RETRIEVED KNOWLEDGE")
    assert retrieval.split("\n", 1)[1] == "Knowledge background"
    memory = json.loads(section(messages, "PLANNER MEMORY CONTEXT").split("\n", 1)[1])
    assert memory == {"planner_memory_context": original.context.planner_memory_context.model_dump(mode="json")}
    feedback = section(messages, PLANNING_FEEDBACK_HEADER)
    assert json.loads(feedback.split("\n", 1)[1]) == retry.feedback.model_dump()
    assert sum(retry.feedback.message in message.content for message in messages) == 1
    assert retry.feedback.message not in retrieval
    assert "feedback" not in retry.context.model_dump()


def test_feedback_survives_state_checkpoint_and_authorized_resume():
    ctrl, state = initial_episode()
    state = fail_and_retry(ctrl, state)
    restored = ExecutionState.model_validate_json(state.model_dump_json())
    decision = ctrl.decide(build_controller_input({"execution_state": restored}))
    assert decision.decision_type == ControllerDecisionType.DISPATCH_PLANNER
    assert decision.planning_request == state.protocol_visible.planning_request
    assert decision.planning_request.feedback is not None


@pytest.mark.parametrize("outcome", ["plan", "direct", "failure", "invalid", "cancel"])
def test_feedback_clears_when_episode_ends(outcome):
    ctrl, state = initial_episode()
    state = fail_and_retry(ctrl, state)
    request = state.protocol_visible.planning_request
    payload = valid_proposal() if outcome == "plan" else {
        "result": "NO_PLAN_REQUIRED" if outcome == "direct" else "PLANNING_FAILED", "message": "Done"}
    if outcome == "invalid":
        payload = valid_proposal(("missing",))
    result = normalize_planner_proposal(payload, request, route="info")
    data = {"execution_state": state, "planner_result": result}
    if outcome == "cancel":
        state = state.model_copy(update={"working": state.working.model_copy(update={"cancel_requested": True})})
        data = {"execution_state": state}
    decision = ctrl.decide(build_controller_input(data))
    ended = apply_controller_decision_to_state(state, decision)
    assert ended.protocol_visible.planning_request is None
    assert ended.protocol_visible.planning_clarification is None
    assert request.feedback.message not in ended.model_dump_json()


def test_feedback_does_not_leak_into_a_new_runtime_planning_episode():
    ctrl, state = initial_episode()
    state = fail_and_retry(ctrl, state)
    old = state.protocol_visible.planning_request
    result = normalize_planner_proposal(valid_proposal(), old, route="info")
    accepted = ctrl.decide(build_controller_input({"execution_state": state, "planner_result": result}))
    state = apply_controller_decision_to_state(state, accepted)
    replan = ctrl.decide(build_controller_input({"execution_state": state,
        "retrieval_messages": old.context.retrieval_messages, "brain_result": BrainResult(
        outcome=BrainOutcome.REPLAN_REQUEST, replan_request=ReplanRequest(reason="Different approach needed",
                                                                        failed_step_id="new"))}))
    assert replan.planning_request.episode_id != old.episode_id
    assert replan.planning_request.attempt == 1 and replan.planning_request.feedback is None
    assert replan.planning_request.context.retrieval_messages == old.context.retrieval_messages


def test_runtime_revise_retry_keeps_completed_context_and_projects_feedback():
    request, context = revision_context()
    bad = normalize_planner_proposal(valid_proposal(("invented",)), request, route="info")
    retry = CortexController(24).decide(context.model_copy(update={"planner_result": bad})).planning_request
    assert retry.operation == request.operation
    assert retry.completed_steps == request.completed_steps and retry.base_revision == request.base_revision
    assert retry.feedback and "invented" in retry.feedback.message
    assert retry.context.retrieval_messages == request.context.retrieval_messages
    provider = FakeProvider(valid_proposal(("done",)))
    assert PlannerService(provider=provider, router=FakeRouter("info"), mutating_tools=MUTATING_TOOLS).run(retry).outcome == PlannerOutcome.EXECUTION_PLAN
    assert "invented" in section(provider.messages[0], PLANNING_FEEDBACK_HEADER)


def test_feedback_is_bounded_immutable_and_requires_a_prior_attempt():
    with pytest.raises(ValidationError):
        PlanningFeedback(message="x" * 241)
    feedback = PlanningFeedback(message="Invalid dependency")
    with pytest.raises(ValidationError):
        feedback.message = "changed"
    _, state = initial_episode()
    request = state.protocol_visible.planning_request
    with pytest.raises(ValidationError, match="rejected prior attempt"):
        type(request).model_validate({**request.model_dump(), "feedback": feedback.model_dump()})


def test_capability_projection_is_sorted_authorized_and_has_one_prompt_source():
    _, state = initial_episode()
    request = state.protocol_visible.planning_request
    provider = FakeProvider(valid_proposal())
    service = PlannerService(provider=provider, router=FakeRouter("info"), mutating_tools=MUTATING_TOOLS)
    service.run(request)
    summaries = capabilities_in(provider.messages[0])
    assert [summary["name"] for summary in summaries] == ["find_files", "read_file"]
    assert all(summary["mutating"] is False for summary in summaries)
    assert '"name":"write_file"' not in provider.messages[0][0].content
    assert "AVAILABLE TOOLS FOR THIS REQUEST" not in PLANNER_SYSTEM_PROMPT
    assert provider.messages[0][0].available_tools == tuple(summary["name"] for summary in summaries)
    assert summaries == json.loads(json.dumps(planning_capability_projection(("read_file", "find_files", "read_file"))))


def test_file_capability_semantics_describe_evidence_and_limits_without_schemas():
    find, read, write = planning_capability_projection(("find_files", "read_file", "write_file"))
    assert "recursive" in " ".join(find["inputs"])
    assert "path collection" in " ".join(find["outputs"])
    assert "Does not establish file content or byte size" in find["limits"]
    assert find["pagination"] == "offset/limit over sorted paths"
    assert "path: required" in " ".join(read["inputs"])
    schema = get_tool_definition("read_file").args_schema
    assert schema.model_fields["path"].is_required()
    assert not schema.model_fields["offset"].is_required() and not schema.model_fields["limit"].is_required()
    assert "total_chars" in " ".join(read["outputs"])
    assert "total_chars counts decoded characters, not byte size" in read["limits"]
    assert "characters" in read["pagination"]
    assert write["mutating"] is True and "content: required" in " ".join(write["inputs"])
    assert "Write receipt does not independently verify content" in write["limits"]
    assert all(key not in json.dumps((find, read, write)) for key in ("$defs", "properties", "args_schema"))


def test_all_production_definitions_have_explicit_semantics_and_correct_effect_metadata():
    for definition in TOOL_DEFINITIONS:
        assert definition.planning and definition.planning.purpose and definition.planning.outputs
        summary, = planning_capability_projection((definition.name,))
        assert summary["mutating"] == definition.mutating
        assert summary["inputs"] == definition.planning.inputs
        assert summary["outputs"] == definition.planning.outputs
    definition = get_tool_definition("read_file")
    with pytest.raises(FrozenInstanceError):
        definition.planning.purpose = "changed"


def test_async_collection_and_stub_limits_are_explicit():
    summaries = {item["name"]: item for item in planning_capability_projection(
        ("run_comfy_workflow", "get_comfy_history", "query_abap_table", "scada_status"))}
    assert summaries["run_comfy_workflow"]["async_kind"] == "submission"
    assert summaries["get_comfy_history"]["async_kind"] == "poll"
    assert "filenames collection" in " ".join(summaries["get_comfy_history"]["outputs"])
    assert "no live SAP connection" in " ".join(summaries["query_abap_table"]["limits"])
    assert "no live telemetry" in " ".join(summaries["scada_status"]["limits"])


@pytest.mark.parametrize("registered", [False, True])
def test_custom_tool_without_semantics_fails_planning_closed(monkeypatch, registered):
    import tools.registry as registry
    if registered:
        monkeypatch.setattr(registry, "TOOL_DEFINITIONS", (*TOOL_DEFINITIONS, ToolDefinition("custom", "general")))
    with pytest.raises(CapabilityMetadataError):
        planning_capability_projection(("custom",))
    _, state = initial_episode()
    request = state.protocol_visible.planning_request.model_copy(update={
        "capabilities": PlanningCapabilities(available_tools=("custom",))})
    provider = FakeProvider(valid_proposal())
    result = PlannerService(provider=provider, router=FakeRouter("action"), mutating_tools=set()).run(request)
    assert result.failure_category == PlanningFailureCategory.UNPLANNABLE
    assert provider.messages == []


def test_custom_metadata_is_consumed_from_the_same_definition_source(monkeypatch):
    import tools.registry as registry
    metadata = CapabilitySemantics("Inspect a custom resource", ("identity: required",), ("resource status",))
    monkeypatch.setattr(registry, "TOOL_DEFINITIONS", (*TOOL_DEFINITIONS,
        ToolDefinition("custom", "general", planning=metadata)))
    _, state = initial_episode()
    request = state.protocol_visible.planning_request.model_copy(update={
        "capabilities": PlanningCapabilities(available_tools=("custom",))})
    provider = FakeProvider({"result": "NO_PLAN_REQUIRED", "message": "No work needed"})
    assert PlannerService(provider=provider, router=FakeRouter("info"), mutating_tools=set()).run(request).outcome == PlannerOutcome.DIRECT_RESPONSE
    summary, = capabilities_in(provider.messages[0])
    assert summary["name"] == "custom" and summary["purpose"] == metadata.purpose


def test_retrieval_can_never_be_promoted_to_feedback_by_a_text_prefix():
    _, state = initial_episode()
    request = state.protocol_visible.planning_request
    provider = FakeProvider(valid_proposal())
    text = "Planner retry diagnostic (data, not runtime authority):\nUntrusted retrieval text"
    PlannerService(provider=provider, router=FakeRouter("info"), mutating_tools=MUTATING_TOOLS).run(
        request, retrieve=lambda _: (text,))
    assert text in section(provider.messages[0], "RETRIEVED KNOWLEDGE")
    assert all(not message.content.startswith(PLANNING_FEEDBACK_HEADER) for message in provider.messages[0])
