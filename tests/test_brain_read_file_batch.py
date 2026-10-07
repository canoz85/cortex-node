"""Sequential Brain actions: acceptance, Controller authority, transport and resume."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import ValidationError

from core.brain import _build_step_progress_messages
from core.brain_batch_policy import MAX_READ_FILE_BATCH
from core.brain_evidence_policy import MAX_CURRENT_ATTEMPT_RECORDS
from core.brain_normalization import normalize_brain_output
from core.brain_provider import LangChainBrainProvider
from core.graph import _wrap_tool_node_for_protocol_request_id
from core.graph_brain import create_brain_node
from core.graph_capture import create_capture_tool_output_node
from core.graph_controller import create_controller_node
from core.graph_worker_runtime import GraphWorkerRuntimePorts
from core.models import ToolOutputEnvelope
from core.protocol.bridge import build_controller_input, build_brain_input
from core.protocol.completion_identity import evidence_identity
from core.protocol.controller import CortexController, apply_controller_decision_to_state
from core.protocol.enums import BrainOutcomeKind as Kind, ControllerDecisionType as Decision, WorkerRole
from core.protocol.models import (
    BrainOutcome, ExecutionState, ProtocolVisibleState, ToolRequest, ToolResult,
    ToolExecutionRecord, ExactCollection, StepCompletionEvidence,
)
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver, WorkerDispatchError
from core.runtime.tool_result_integration import integrate_tool_result, SerializedToolRuntimePort
from tools.file_ops import get_file_tools
from test_brain_outcomes import brain_input, SequenceModel, native_action


def native_batch(count=2, *, names=None, args=None):
    return AIMessage(content="", tool_calls=[{
        "name": names[i] if names else "read_file",
        "args": args[i] if args else {"path": f"file-{i}.txt", "limit": 1},
        "id": f"provider-{i}",
    } for i in range(count)])


def provider_output(*responses, context=None):
    model = SequenceModel(*responses)
    tools = get_file_tools(str(Path.cwd()))
    provider = LangChainBrainProvider(brain_llm=model, executable_tools=tools)
    return provider.generate(context or brain_input(), ()), model


def initial_state():
    context = brain_input()
    return ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=context.identity, active_plan=context.active_plan, active_step=context.active_step,
        cursor=context.cursor.model_copy(update={"current_worker": WorkerRole.BRAIN,
                                                 "step_attempt": context.active_step.attempt}),
    ))


def input_for(state, proposal=None):
    return build_controller_input({"execution_state": state, "brain_result": proposal})


def proposal(count=2):
    return normalize_brain_output(native_batch(count), brain_input(), {"read_file"})


def result_for(request, success=True):
    return ToolResult(request_id=request.request_id, success=success,
                      signature=f"read_file:{request.arguments['path']}",
                      error_code=None if success else "FILE_NOT_FOUND",
                      message="Read" if success else "Missing",
                      data={"entries": [request.arguments["path"]]})


@pytest.mark.parametrize("count", [2, MAX_READ_FILE_BATCH])
def test_permitted_batch_uses_one_provider_call_and_distinct_ordinary_requests(count):
    output, model = provider_output(native_batch(count))
    assert output.kind == Kind.TOOL_REQUESTED
    assert output.tool_request is None
    assert len(output.tool_requests) == count
    assert len({request.request_id for request in output.tool_requests}) == count
    assert all(type(request) is ToolRequest for request in output.tool_requests)
    assert [r.arguments["path"] for r in output.tool_requests] == [f"file-{i}.txt" for i in range(count)]
    assert len(model.calls) == 1


@pytest.mark.parametrize("content", ["", "Incidental description"])
def test_singleton_canonicalization_unchanged(content):
    raw = native_action("read_file", {"path": "a"})
    raw.content = content
    output, model = provider_output(raw)
    assert output.tool_request.arguments == {"path": "a"}
    assert output.tool_requests is None and len(model.calls) == 1
    assert raw.content == content


@pytest.mark.parametrize("raw", [
    native_batch(MAX_READ_FILE_BATCH + 1),
    native_batch(args=[{"path": "a"}, {"path": "./a", "offset": 0, "limit": 10000}]),
    native_batch(names=["read_file", "list_files"], args=[{"path": "a"}, {}]),
    native_batch(names=["write_file"] * 2, args=[{"path": "a", "content": "x"}, {"path": "b", "content": "y"}]),
    native_batch(names=["read_file", "brain_step_completed"], args=[{"path": "a"}, {"message": "Done"}]),
    native_batch(names=["describe_image"] * 2),
])
def test_valid_disallowed_groups_never_execute_and_use_at_most_one_correction(raw):
    # describe_image is deliberately unbound here, hence strict invalid output.
    output, model = provider_output(raw, native_action("read_file", {"path": "chosen"}))
    if raw.tool_calls[0]["name"] == "describe_image":
        assert output.kind == Kind.INVALID_OUTPUT and len(model.calls) == 1
    else:
        assert output.tool_request.arguments == {"path": "chosen"}
        assert output.tool_requests is None and len(model.calls) == 2


@pytest.mark.parametrize("args", [{}, {"path": []}, {"path": "a", "offset": -1},
                                   {"path": "a", "unexpected": True}, {"path": "a", "limit": 0}])
def test_invalid_argument_group_is_strict_without_correction(args):
    raw = native_batch(args=[{"path": "good"}, args])
    output, model = provider_output(raw)
    assert output.kind == Kind.INVALID_OUTPUT and len(model.calls) == 1


def test_payload_exclusivity_and_wrong_kind_are_enforced():
    requests = proposal().tool_requests
    with pytest.raises(ValidationError, match="exclusive"):
        BrainOutcome(outcome=Kind.TOOL_REQUESTED, tool_request=requests[0], tool_requests=requests)
    with pytest.raises(ValidationError, match="kind"):
        BrainOutcome(outcome=Kind.STEP_FAILED, tool_requests=requests)


@pytest.mark.parametrize("count", [2, MAX_READ_FILE_BATCH])
@pytest.mark.parametrize("failed", [None, 0])
def test_driver_orders_members_reenters_controller_and_returns_all_evidence_once(count, failed):
    state = initial_state()
    action = proposal(count)
    brain = Mock(spec=["run"])
    brain.run.return_value = BrainOutcome(outcome=Kind.STEP_COMPLETED, step_id="s1", message="Inspected")
    observed = []

    class Tools:
        def execute_authorized(self, request, authorized_state, decision):
            assert authorized_state.protocol_visible.pending_tool_request == request
            assert decision.pending_tool_request == request
            assert request not in authorized_state.protocol_visible.tool_request_continuation.remaining
            observed.append(request)
            brain.run.assert_not_called()
            return result_for(request, len(observed) - 1 != failed)

    controller = CortexController(24)
    trace = Mock(wraps=controller)
    driver = ExecutionDriver(coordinator=ControllerCoordinator(trace), planner=Mock(), brain=brain,
                             tool_runtime=Tools(), finalizer=Mock())
    for index in range(count):
        turn = driver.turn(state, input_for(state, action if index == 0 else None))
        assert turn.decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME
        assert turn.execution_state.protocol_visible.cursor.controller_iteration == 1
        state = integrate_tool_result(turn.execution_state, turn.decision, turn.worker_result)
    assert observed == list(action.tool_requests)
    records = state.working.tool_execution_history
    assert all(type(record) is ToolExecutionRecord for record in records)
    assert [r.result.request_id for r in records] == [r.request_id for r in observed]
    assert [r.result.success for r in records] == [i != failed for i in range(count)]
    turn = driver.turn(state, input_for(state))
    assert turn.decision.decision_type == Decision.DISPATCH_BRAIN
    assert turn.execution_state.protocol_visible.pending_tool_request is None
    assert turn.execution_state.protocol_visible.tool_request_continuation is None
    brain.run.assert_called_once()
    assert brain.run.call_args.args[0].tool_execution_history == records
    assert trace.decide.call_count == count + 1
    completion = controller.decide(input_for(turn.execution_state, turn.worker_result))
    assert completion.completion_evidence.tool_request_ids == tuple(r.request_id for r in observed)
    assert completion.completion_evidence.evidence_id == evidence_identity(records)


def accepted_first(count=3):
    state = initial_state()
    decision = CortexController(24).decide(input_for(state, proposal(count)))
    return apply_controller_decision_to_state(state, decision), decision


@pytest.mark.parametrize("captured", [False, True])
def test_checkpoint_resume_after_a_and_with_b_authorized_or_captured(captured):
    controller = CortexController(24)
    state, decision = accepted_first()
    first = decision.pending_tool_request
    state = integrate_tool_result(state, decision, result_for(first))
    decision = controller.decide(input_for(state))
    state = apply_controller_decision_to_state(state, decision)
    second = decision.pending_tool_request
    if captured:
        state = integrate_tool_result(state, decision, result_for(second))
    serializer = JsonPlusSerializer()
    restored = serializer.loads_typed(serializer.dumps_typed(state))
    assert isinstance(restored, ExecutionState)
    assert ExecutionState.model_validate_json(state.model_dump_json()) == restored
    resumed = controller.decide(input_for(restored))
    if not captured:
        assert resumed.pending_tool_request == second
        restored = apply_controller_decision_to_state(restored, resumed)
        restored = integrate_tool_result(restored, resumed, result_for(second))
        resumed = controller.decide(input_for(restored))
    third = resumed.pending_tool_request
    assert third.arguments["path"] == "file-2.txt"
    restored = apply_controller_decision_to_state(restored, resumed)
    restored = integrate_tool_result(restored, resumed, result_for(third))
    assert [r.result.request_id for r in restored.working.tool_execution_history] == [
        first.request_id, second.request_id, third.request_id,
    ]
    assert controller.decide(input_for(restored)).decision_type == Decision.DISPATCH_BRAIN


@pytest.mark.parametrize("interruption", ["cancel", "max_steps", "identity", "replan", "async", "scope"])
def test_controller_interruption_discards_remainder(interruption):
    state, decision = accepted_first()
    request = decision.pending_tool_request
    result = result_for(request, interruption != "replan")
    state = integrate_tool_result(state, decision, result)
    value = input_for(state)
    controller = CortexController(24)
    if interruption == "cancel":
        value = value.model_copy(update={"cancel_requested": True})
    elif interruption == "max_steps":
        controller = CortexController(1)
    elif interruption == "identity":
        value = value.model_copy(update={"tool_result": result.model_copy(update={"request_id": "alien"})})
    elif interruption == "replan":
        value = value.model_copy(update={"retry": value.retry.model_copy(update={"max_retries": 1})})
    elif interruption == "async":
        from core.protocol.enums import AsyncJobStatus
        value = value.model_copy(update={"tool_result": ToolResult(
            request_id=request.request_id, success=True, message="Unexpected async result", is_async_job=True,
            async_job_id="unexpected-job", async_job_status=AsyncJobStatus.RUNNING,
        )})
    else:
        value = value.model_copy(update={"tool_request_continuation":
            value.tool_request_continuation.model_copy(update={"plan_revision": 2})})
    interrupted = controller.decide(value)
    assert interrupted.decision_type != Decision.DISPATCH_TOOL_RUNTIME
    updated = apply_controller_decision_to_state(state, interrupted)
    assert updated.protocol_visible.tool_request_continuation is None
    assert updated.protocol_visible.pending_tool_request is None
    assert len(updated.working.tool_execution_history) == 1


@pytest.mark.parametrize("invalid", ["duplicate", "oversized", "heterogeneous", "mutating", "args", "unauthorized", "ids"])
def test_controller_revalidates_whole_injected_proposal_before_any_authorization(invalid):
    action = proposal()
    requests = list(action.tool_requests)
    state = initial_state()
    if invalid == "duplicate":
        requests[1] = requests[1].model_copy(update={"arguments": {"path": "file-0.txt", "limit": 1, "offset": 0}})
    elif invalid == "oversized":
        requests = [ToolRequest(request_id=str(i), tool_name="read_file", arguments={"path": str(i)})
                    for i in range(MAX_READ_FILE_BATCH + 1)]
    elif invalid == "heterogeneous":
        requests[1] = requests[1].model_copy(update={"tool_name": "list_files"})
    elif invalid == "mutating":
        requests = [r.model_copy(update={"tool_name": "write_file"}) for r in requests]
    elif invalid == "args":
        requests[1] = requests[1].model_copy(update={"arguments": {"path": "x", "limit": -1}})
    elif invalid == "unauthorized":
        plan = state.protocol_visible.active_plan.model_copy(update={"available_tools": ("list_files",)})
        state = state.model_copy(update={"protocol_visible": state.protocol_visible.model_copy(update={"active_plan": plan})})
    else:
        requests[1] = requests[1].model_copy(update={"request_id": requests[0].request_id})
    action = action.model_copy(update={"tool_requests": tuple(requests)})
    decision = CortexController(24).decide(input_for(state, action))
    assert decision.terminal
    updated = apply_controller_decision_to_state(state, decision)
    assert updated.protocol_visible.pending_tool_request is None
    assert updated.protocol_visible.tool_request_continuation is None


def test_maximum_batch_window_order_and_record_local_exact_collection_with_full_provenance():
    state, decision = accepted_first(MAX_READ_FILE_BATCH)
    prefix = tuple(ToolExecutionRecord(execution_id=state.protocol_visible.identity.execution_id,
        plan_id="p1", plan_revision=1, step_id="s1", tool_name="read_file",
        result=ToolResult(request_id=f"prior-{i}", success=True, message="Earlier read", data={"entries": [f"old-{i}"]}))
        for i in range(10))
    state = state.model_copy(update={"working": state.working.model_copy(update={"tool_execution_history": prefix})})
    controller = CortexController(24)
    for i in range(MAX_READ_FILE_BATCH):
        state = integrate_tool_result(state, decision, result_for(decision.pending_tool_request, i != 0))
        decision = controller.decide(input_for(state))
        state = apply_controller_decision_to_state(state, decision)
    context = build_brain_input({"execution_state": state})
    payload = json.loads(_build_step_progress_messages(brain_input=context)[0].content.split("\n", 1)[1])
    current = payload["current_attempts"]
    assert [r["record_index"] for r in current] == list(range(MAX_CURRENT_ATTEMPT_RECORDS))
    assert [r["args"]["path"] for r in current[-MAX_READ_FILE_BATCH:]] == [
        f"file-{i}.txt" for i in range(MAX_READ_FILE_BATCH)]
    assert [r["success"] for r in current[-MAX_READ_FILE_BATCH:]] == [
        False, *([True] * (MAX_READ_FILE_BATCH - 1))]
    completion = BrainOutcome(outcome=Kind.STEP_COMPLETED, step_id="s1",
        completion_evidence=StepCompletionEvidence(step_id="s1", summary="Inspected",
            exact_collection=ExactCollection(source_record_index=MAX_CURRENT_ATTEMPT_RECORDS - 1,
                                            data_path=("entries",))))
    decision = controller.decide(input_for(state, completion))
    records = state.working.tool_execution_history
    assert decision.completion_evidence.tool_request_ids == tuple(r.result.request_id for r in records)
    assert len(decision.completion_evidence.tool_request_ids) == len(prefix) + MAX_READ_FILE_BATCH
    assert decision.completion_evidence.exact_collection.items == (f"file-{MAX_READ_FILE_BATCH - 1}.txt",)
    assert decision.completion_evidence.exact_collection.source_request_id == records[-1].result.request_id


@pytest.mark.parametrize("mode", ["portable_direct", "portable_injected", "legacy_deferred"])
def test_graph_transport_authorizes_one_member_and_captures_each_record(mode):
    graph = {"execution_state": initial_state(), "brain_result": proposal(3), "messages": []}
    seen = []

    def tool_node(state):
        calls = state["messages"][-1].tool_calls
        assert len(calls) == 1
        request = state["execution_state"].protocol_visible.pending_tool_request
        assert calls[0]["id"] == request.request_id
        seen.append(request)
        return {"messages": [ToolMessage(tool_call_id=request.request_id,
            content=ToolOutputEnvelope(success=len(seen) != 1, message="observed",
                                       data={"path": request.arguments["path"]}).to_tool_output())]}

    wrapped = _wrap_tool_node_for_protocol_request_id(tool_node)
    capture = create_capture_tool_output_node()
    brain = Mock(return_value={"brain_result": BrainOutcome(outcome=Kind.STEP_COMPLETED, step_id="s1")})
    def direct_execute(request):
        seen.append(request)
        return result_for(request, len(seen) != 1)

    ports = GraphWorkerRuntimePorts(tool_runtime=SimpleNamespace(execute=direct_execute)
                                    if mode == "portable_direct" else None)
    ports.bind_nodes(planner=Mock(), brain=brain, tool=wrapped, capture=capture)
    controller = create_controller_node(controller=CortexController(24),
        worker_ports=ports if mode != "legacy_deferred" else None)
    for _ in range(3):
        update = controller(graph)
        graph = {**graph, **update, "messages": [*graph["messages"], *update.get("messages", [])]}
        if mode == "legacy_deferred":
            update = wrapped(graph)
            graph = {**graph, **update, "messages": [*graph["messages"], *update["messages"]]}
            graph.update(capture(graph))
        brain.assert_not_called()
    update = controller(graph)
    assert update["controller_decision"].decision_type == Decision.DISPATCH_BRAIN
    if mode != "legacy_deferred":
        brain.assert_called_once()
    assert [r.arguments["path"] for r in seen] == [f"file-{i}.txt" for i in range(3)]
    assert [r.result.success for r in update["execution_state"].working.tool_execution_history] == [False, True, True]


def test_runtime_exception_and_result_identity_failure_stop_before_next_member():
    for failure in [RuntimeError("runtime failed"), result_for(ToolRequest(request_id="alien", tool_name="read_file", arguments={"path": "x"}))]:
        tools = Mock(spec=["execute"])
        if isinstance(failure, Exception):
            tools.execute.side_effect = failure
        else:
            tools.execute.return_value = failure
        driver = ExecutionDriver(coordinator=ControllerCoordinator(CortexController(24)),
            planner=Mock(), brain=Mock(), tool_runtime=tools, finalizer=Mock())
        with pytest.raises(RuntimeError):
            driver.turn(initial_state(), input_for(initial_state(), proposal()))
        tools.execute.assert_called_once()


def test_actual_read_file_batch_preserves_missing_file_and_later_success(tmp_path):
    (tmp_path / "file-1.txt").write_text("hello", encoding="utf-8")
    tools = SerializedToolRuntimePort(get_file_tools(str(tmp_path)))
    state, decision = accepted_first(2)
    for _ in range(2):
        result = tools.execute(decision.pending_tool_request)
        state = integrate_tool_result(state, decision, result)
        decision = CortexController(24).decide(input_for(state))
        state = apply_controller_decision_to_state(state, decision)
    assert [r.result.success for r in state.working.tool_execution_history] == [False, True]
    assert decision.decision_type == Decision.DISPATCH_BRAIN


@pytest.mark.parametrize("name", ["describe_image", "run_comfy_workflow", "git_status"])
def test_other_bound_read_only_or_async_tools_are_not_batch_eligible(name):
    model = SequenceModel(native_batch(names=[name, name], args=[{}, {}]),
                          native_action("brain_step_failed", {"message": "Cannot proceed"}))
    context = brain_input()
    context = context.model_copy(update={"active_plan": context.active_plan.model_copy(
        update={"available_tools": (name,)})})
    provider = LangChainBrainProvider(brain_llm=model,
        executable_tools=[SimpleNamespace(name=name, args_schema=None)])
    output = provider.generate(context, ())
    assert output.kind == Kind.STEP_FAILED and len(model.calls) == 2
    assert output.tool_requests is None


def test_correction_can_return_permitted_batch_but_invalid_singleton_arguments_remain_strict():
    duplicate = native_batch(args=[{"path": "same"}, {"path": "same"}])
    output, model = provider_output(duplicate, native_batch())
    assert len(output.tool_requests) == 2 and len(model.calls) == 2
    output, model = provider_output(duplicate, native_action("read_file", {}))
    assert output.kind == Kind.INVALID_OUTPUT and len(model.calls) == 2


def test_batch_result_does_not_grant_execution_authority_to_legacy_toolnode():
    state = initial_state()
    service = SimpleNamespace(run=lambda _: proposal())
    decision = CortexController(24)._dispatch_brain(state.protocol_visible.cursor, reason="test")
    state = apply_controller_decision_to_state(state, decision)
    update = create_brain_node(brain_llm=None, executable_tools=[], agent_system_prompt="",
                              show_raw_llm=False, brain_service=service)({
        "execution_state": state, "controller_decision": decision,
    })
    assert update["brain_result"].tool_requests == proposal().tool_requests
    assert update["messages"][0].tool_calls == []
    invoked = Mock()
    with pytest.raises(WorkerDispatchError):
        _wrap_tool_node_for_protocol_request_id(invoked)({
            "execution_state": state, "controller_decision": decision, "messages": update["messages"],
        })
    invoked.assert_not_called()


def test_checkpoint_after_a_capture_before_b_authorization_does_not_reexecute_a():
    state, decision = accepted_first()
    state = integrate_tool_result(state, decision, result_for(decision.pending_tool_request))
    restored = ExecutionState.model_validate_json(state.model_dump_json())
    resumed = CortexController(24).decide(input_for(restored))
    assert resumed.pending_tool_request.arguments["path"] == "file-1.txt"
    assert len(resumed.tool_request_continuation.remaining) == 1
    assert restored.working.tool_execution_history[0].result.request_id != resumed.pending_tool_request.request_id


def test_coordinator_rejects_untracked_queue_and_stale_scope_before_dispatch():
    state, _ = accepted_first()
    coordinator = ControllerCoordinator(CortexController(24))
    for changes in [
        {"tool_request_continuation": None},
        {"pending_tool_request": None},
        {"active_plan": state.protocol_visible.active_plan.model_copy(update={"revision": 2})},
    ]:
        with pytest.raises(ValueError, match="does not match ExecutionState"):
            coordinator.transition(state, input_for(state).model_copy(update=changes))


def test_completion_provenance_filters_foreign_execution_plan_and_revision():
    state, decision = accepted_first(2)
    controller = CortexController(24)
    for _ in range(2):
        state = integrate_tool_result(state, decision, result_for(decision.pending_tool_request))
        decision = controller.decide(input_for(state))
        state = apply_controller_decision_to_state(state, decision)
    records = state.working.tool_execution_history
    foreign = tuple(records[0].model_copy(update={**scope,
        "result": records[0].result.model_copy(update={"request_id": f"foreign-{i}"})})
        for i, scope in enumerate([{"execution_id": "other"}, {"plan_id": "other"}, {"plan_revision": 2}]))
    state = state.model_copy(update={"working": state.working.model_copy(
        update={"tool_execution_history": (*foreign, *records)})})
    context = build_brain_input({"execution_state": state})
    payload = json.loads(_build_step_progress_messages(brain_input=context)[0].content.split("\n", 1)[1])
    assert [r["record_index"] for r in payload["current_attempts"]] == [0, 1]
    assert [r["args"]["path"] for r in payload["current_attempts"]] == ["file-0.txt", "file-1.txt"]
    completion = controller.decide(input_for(state, BrainOutcome(
        outcome=Kind.STEP_COMPLETED, step_id="s1", message="Done",
        completion_evidence=StepCompletionEvidence(step_id="s1", summary="Done",
            exact_collection=ExactCollection(source_record_index=1, data_path=("entries",))))))
    assert completion.completion_evidence.tool_request_ids == tuple(r.result.request_id for r in records)
    assert completion.completion_evidence.evidence_id == evidence_identity(records)
    assert completion.completion_evidence.exact_collection.source_request_id == records[1].result.request_id


@pytest.mark.parametrize("mode", ["portable", "deferred"])
def test_graph_checkpoint_restart_resumes_b_without_skipping_or_reexecuting_a(mode):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph, END
    from core.state import AgentState

    seen = []
    action = proposal(3)
    brain = Mock(return_value={"brain_result": BrainOutcome(outcome=Kind.STEP_COMPLETED, step_id="s1")})

    def tool(state):
        request = state["execution_state"].protocol_visible.pending_tool_request
        assert len(state["messages"][-1].tool_calls) == 1
        seen.append(request.request_id)
        return {"messages": [ToolMessage(tool_call_id=request.request_id,
            content=ToolOutputEnvelope(success=True, message="Read", data={}).to_tool_output())]}

    def compile_graph(saver):
        wrapped = _wrap_tool_node_for_protocol_request_id(tool)
        capture = create_capture_tool_output_node()
        ports = GraphWorkerRuntimePorts()
        ports.bind_nodes(planner=Mock(), brain=brain, tool=wrapped, capture=capture)
        controller = create_controller_node(controller=CortexController(24),
            worker_ports=ports if mode == "portable" else None)
        graph = StateGraph(AgentState)
        graph.add_node("controller", controller)
        graph.set_entry_point("controller")
        if mode == "portable":
            graph.add_conditional_edges("controller", lambda s:
                END if s["controller_decision"].decision_type == Decision.DISPATCH_BRAIN else "controller")
            return graph.compile(checkpointer=saver, interrupt_after=["controller"])
        graph.add_node("tools", wrapped)
        graph.add_node("capture", capture)
        graph.add_conditional_edges("controller", lambda s:
            END if s["controller_decision"].decision_type == Decision.DISPATCH_BRAIN else "tools")
        graph.add_edge("tools", "capture")
        graph.add_edge("capture", "controller")
        return graph.compile(checkpointer=saver, interrupt_before=["tools"])

    saver = InMemorySaver()
    graph = compile_graph(saver)
    config = {"configurable": {"thread_id": f"batch-resume-{mode}"}}
    graph.invoke({"execution_state": initial_state(), "brain_result": action, "messages": []}, config)
    if mode == "deferred":
        graph.invoke(None, config)  # A captured; B authorized; pause before B.
    assert seen == [action.tool_requests[0].request_id]
    # Reconstruct the graph and adapters against the existing checkpoint store.
    graph = compile_graph(saver)
    while graph.get_state(config).next:
        graph.invoke(None, config)
    restored = graph.get_state(config).values["execution_state"]
    assert seen == [r.request_id for r in action.tool_requests]
    assert [r.result.request_id for r in restored.working.tool_execution_history] == seen
    assert restored.protocol_visible.pending_tool_request is None
    assert restored.protocol_visible.tool_request_continuation is None
    if mode == "portable":
        brain.assert_called_once()
