"""Successful evidence prevents repeated authorization without changing tool semantics."""

from contextlib import contextmanager
from functools import partial
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from core.brain import BrainService, _build_execution_messages
from core.brain_normalization import normalize_brain_output
from core.brain_provider import LangChainBrainProvider
from core.finalizer import Finalizer
from core.protocol.bridge import build_brain_input, build_controller_input
from core.protocol.controller import CortexController, apply_controller_decision_to_state
from core.protocol.enums import AsyncJobStatus, BrainOutcomeKind as Kind, ControllerDecisionType as Decision
from core.protocol.models import (
    BrainOutcome, ExecutionState, ProtocolVisibleState, RetryMetadata, ToolExecutionRecord,
    ToolRequest, ToolRequestContinuation, ToolResult,
)
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver, WorkerDispatchError
from core.runtime.tool_result_integration import SerializedToolRuntimePort, integrate_tool_result
from core.tool_call_guard import DuplicateSuccessfulCall, reject_duplicate_successful_calls
from test_brain_outcomes import SequenceModel, brain_input, native_action
from tools.discovery_ops import get_discovery_tools
from tools.file_ops import get_file_tools
from tools.registry import ToolRegistry, get_tool_argument_schema


def input_for(state, outcome=None):
    value = build_controller_input({"execution_state": state, "brain_result": outcome})
    return value.model_copy(update={"tool_result": None}) if outcome is not None else value


def proposal(name, arguments, *, request_id="new"):
    return BrainOutcome(outcome=Kind.TOOL_REQUESTED, step_id="s1", tool_request=ToolRequest(
        request_id=request_id, tool_name=name, arguments=arguments,
    ))


@pytest.fixture
def setup(tmp_path):
    tools = [*get_file_tools(str(tmp_path)), *get_discovery_tools(str(tmp_path))]
    registry = ToolRegistry.from_tools(tools)
    lookup = partial(get_tool_argument_schema, registry=registry)
    context = brain_input()
    context = context.model_copy(update={"active_plan": context.active_plan.model_copy(update={
        "available_tools": tuple(registry.names),
    })})
    state = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=context.identity, cursor=context.cursor, active_plan=context.active_plan,
        active_step=context.active_step, retry=context.retry,
        original_user_request="Find the Snake UI implementation.",
    ))
    calls = []

    @contextmanager
    def observe(request):
        calls.append(request)
        yield

    controller = CortexController(24, argument_schema_for=lookup)
    runtime = SerializedToolRuntimePort(tools, observe=observe)
    driver = ExecutionDriver(coordinator=ControllerCoordinator(controller), planner=Mock(), brain=Mock(),
                             tool_runtime=runtime, finalizer=Finalizer())
    return SimpleNamespace(tools=tools, lookup=lookup, state=state, calls=calls, controller=controller,
                           runtime=runtime, driver=driver, root=tmp_path)


def execute_once(case, name, arguments):
    turn = case.driver.turn(case.state, input_for(case.state, proposal(name, arguments)))
    assert turn.decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME
    captured = integrate_tool_result(turn.execution_state, turn.decision, turn.worker_result)
    # Apply the result's authorization without invoking the fixture's dummy Brain.
    continuation = case.driver.turn(captured, input_for(captured), dispatch_worker=False)
    return continuation.execution_state, turn.worker_result


@pytest.mark.parametrize("name,arguments", [
    ("list_files", {"path": "Snake/"}),
    ("search_text", {"query": "Snake", "file_pattern": "*.py", "recursive": True}),
])
def test_second_successful_execution_is_blocked_including_empty_search(setup, name, arguments):
    case = setup
    snake = case.root / "Snake"
    snake.mkdir()
    for child in ("app", "core", "data", "ui"):
        (snake / child).mkdir()
    (snake / "CMakeLists.txt").write_text("project(Game)", encoding="utf-8")
    state, result = execute_once(case, name, arguments)
    assert result.success
    if name == "search_text":
        assert result.data["matches"] == [] and not result.pagination.has_more
    checkpoint = state.model_dump_json()
    # Deliberately bypass native normalization: Controller must enforce the rule.
    rejected = case.driver.turn(state, input_for(state, proposal(name, arguments, request_id="fresh")),
                                dispatch_worker=False)
    assert rejected.decision.decision_type == Decision.DISPATCH_BRAIN
    assert rejected.decision.retry.last_error_code == "duplicate_successful_call"
    assert rejected.decision.retry.retry_count == 1
    assert rejected.decision.pending_tool_request is None
    assert rejected.decision.completed_step_id is None and not rejected.decision.requires_replan
    assert len(case.calls) == 1
    assert rejected.execution_state.working.tool_execution_history == state.working.tool_execution_history
    assert state.model_dump_json() == checkpoint
    feedback = _build_execution_messages(system_prompt="Execute", brain_input=build_brain_input({
        "execution_state": rejected.execution_state,
    }))
    rejection = next(message.content for message in feedback if message.content.startswith("Tool-call rejection"))
    assert "duplicate_successful_call" in rejection and "Reuse the prior evidence" in rejection
    repeated = case.driver.turn(rejected.execution_state, input_for(rejected.execution_state,
        proposal(name, arguments, request_id="another-fresh-id")), dispatch_worker=False)
    assert repeated.decision.terminal and repeated.decision.reason == "step_failed_retries_exhausted"
    assert repeated.decision.retry.last_error_code == "duplicate_successful_call"
    assert len(case.calls) == 1


def test_native_normalization_compares_reordered_arguments_and_schema_defaults(setup):
    case = setup
    state, _ = execute_once(case, "search_text", {"query": "Snake", "file_pattern": "*.py", "recursive": True})
    context = build_brain_input({"execution_state": state})
    reordered = {"recursive": True, "file_pattern": "*.py", "query": "Snake", "offset": 0, "limit": 100}
    before = context.model_dump_json()
    rejected = normalize_brain_output(native_action("search_text", reordered), context,
                                      {"search_text"}, argument_schema_for=case.lookup)
    assert rejected.kind == Kind.INVALID_OUTPUT and rejected.error_code == "duplicate_successful_call"
    assert rejected.tool_request is rejected.tool_requests is None
    assert context.model_dump_json() == before


def test_ignored_extra_arguments_do_not_escape_effective_identity(setup):
    case = setup
    state, _ = execute_once(case, "list_files", {"path": "."})
    context = build_brain_input({"execution_state": state})
    # The executable's inferred schema ignores this key, so it changes no input.
    repeated = native_action("list_files", {"path": ".", "unused": "different"})
    outcome = normalize_brain_output(repeated, context, {"list_files"}, argument_schema_for=case.lookup)
    assert outcome.error_code == "duplicate_successful_call"
    assert len(case.calls) == 1


def test_materially_different_arguments_and_changed_pagination_are_allowed(setup):
    case = setup
    for name in ("a.py", "b.py"):
        (case.root / name).write_text("Snake", encoding="utf-8")
    state, first = execute_once(case, "find_files", {"pattern": "*.py", "limit": 1})
    assert first.pagination.has_more
    for arguments in ({"pattern": "*.py", "limit": 1, "offset": 1}, {"pattern": "*.txt", "limit": 1}):
        turn = case.driver.turn(state, input_for(state, proposal("find_files", arguments, request_id="next")))
        assert turn.decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME
        assert turn.worker_result.success
    assert len(case.calls) == 3
    # Repeating the original partial page also makes no forward progress.
    rejected = case.controller.decide(input_for(state, proposal("find_files", {"pattern": "*.py", "limit": 1})))
    assert rejected.retry.last_error_code == "duplicate_successful_call"


@pytest.mark.parametrize("field,old,new", [
    ("cursor", "a", "b"), ("page", 1, 2), ("continuation_token", "a", "b"),
    ("range", [0, 9], [10, 19]), ("offset", 0, 10),
])
def test_explicit_continuation_arguments_remain_part_of_identity(setup, field, old, new):
    class ContinuationArguments(BaseModel):
        query: str
        cursor: str | None = None
        page: int | None = None
        continuation_token: str | None = None
        range: list[int] | None = None
        offset: int = 0

    context = build_brain_input({"execution_state": setup.state})
    source = scoped_record(context, "future_tool", {"query": "Snake", field: old})
    context = context.model_copy(update={"tool_execution_history": (source,)})
    lookup = lambda name: ContinuationArguments
    reject_duplicate_successful_calls([("future_tool", {"query": "Snake", field: new})],
                                     context, argument_schema_for=lookup)
    with pytest.raises(DuplicateSuccessfulCall):
        reject_duplicate_successful_calls([("future_tool", {field: old, "query": "Snake"})],
                                         context, argument_schema_for=lookup)


def test_failed_prior_call_can_retry_identically(setup):
    case = setup
    case.state = case.state.model_copy(update={"protocol_visible": case.state.protocol_visible.model_copy(
        update={"retry": RetryMetadata(max_retries=2)},
    )})
    state, failed = execute_once(case, "list_files", {"path": "missing"})
    assert not failed.success
    (case.root / "missing").mkdir()
    retry = case.driver.turn(state, input_for(state, proposal("list_files", {"path": "missing"}, request_id="retry")))
    assert retry.decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME and retry.worker_result.success
    assert len(case.calls) == 2


@pytest.mark.parametrize("scope", ["step", "revision", "execution", "plan"])
def test_same_call_in_different_active_scope_is_allowed(setup, scope):
    case = setup
    state, _ = execute_once(case, "list_files", {"path": "."})
    protocol = state.protocol_visible
    changes = {}
    if scope == "step":
        step = protocol.active_step.model_copy(update={"step_id": "s2"})
        changes = {"active_step": step, "active_plan": protocol.active_plan.model_copy(update={"steps": (step,)}),
                   "cursor": protocol.cursor.model_copy(update={"step_id": "s2"})}
    elif scope == "revision":
        changes = {"active_plan": protocol.active_plan.model_copy(update={"revision": 2}),
                   "cursor": protocol.cursor.model_copy(update={"plan_revision": 2})}
    elif scope == "execution":
        changes = {"identity": protocol.identity.model_copy(update={"execution_id": "new-execution"})}
    else:
        changes = {"active_plan": protocol.active_plan.model_copy(update={"plan_id": "new-plan"})}
    changed = state.model_copy(update={"protocol_visible": protocol.model_copy(update=changes)})
    action = proposal("list_files", {"path": "."}, request_id="new-scope").model_copy(update={
        "step_id": changed.protocol_visible.active_step.step_id,
    })
    turn = case.driver.turn(changed, input_for(changed, action))
    assert turn.decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME and turn.worker_result.success
    assert len(case.calls) == 2


def test_batch_with_one_historical_duplicate_is_rejected_atomically(setup):
    case = setup
    (case.root / "new").mkdir()
    state, _ = execute_once(case, "list_files", {"path": "."})
    requests = (ToolRequest(request_id="fresh-1", tool_name="list_files", arguments={"path": "new"}),
                ToolRequest(request_id="fresh-2", tool_name="list_files", arguments={"path": "."}))
    action = BrainOutcome(outcome=Kind.TOOL_REQUESTED, step_id="s1", tool_requests=requests)
    rejected = case.driver.turn(state, input_for(state, action), dispatch_worker=False)
    assert rejected.decision.retry.last_error_code == "duplicate_successful_call"
    assert rejected.decision.tool_request_continuation is None
    assert rejected.execution_state.working.tool_execution_history == state.working.tool_execution_history
    assert len(case.calls) == 1


def test_checkpoint_restores_history_and_blocks_duplicate_without_reexecution(setup):
    case = setup
    state, _ = execute_once(case, "list_files", {"path": "."})
    restored = ExecutionState.model_validate_json(state.model_dump_json())
    rejected = case.driver.turn(restored, input_for(restored, proposal("list_files", {"path": "."}, request_id="replay")),
                                dispatch_worker=False)
    assert rejected.decision.retry.last_error_code == "duplicate_successful_call"
    assert rejected.execution_state.working.tool_execution_history == restored.working.tool_execution_history
    assert len(case.calls) == 1


def test_replaying_a_captured_authorization_cannot_execute_success_again(setup):
    case = setup
    value = input_for(case.state, proposal("list_files", {"path": "."}))
    executed = case.driver.turn(case.state, value)
    captured = integrate_tool_result(executed.execution_state, executed.decision, executed.worker_result)
    restored = ExecutionState.model_validate_json(captured.model_dump_json())
    assert restored.protocol_visible.pending_tool_request == executed.decision.pending_tool_request
    with pytest.raises(WorkerDispatchError, match="duplicate_successful_call"):
        case.driver.dispatch_authorized(restored, executed.decision, value)
    assert len(case.calls) == 1 and len(restored.working.tool_execution_history) == 1


def test_native_batch_duplicate_uses_one_correction_and_can_choose_a_fresh_member(setup):
    case = setup
    (case.root / "new").mkdir()
    state, _ = execute_once(case, "list_files", {"path": "."})
    batch = AIMessage(content="", tool_calls=[
        {"id": "fresh", "name": "list_files", "args": {"path": "new"}},
        {"id": "duplicate", "name": "list_files", "args": {"path": "."}},
    ])
    context = build_brain_input({"execution_state": state})
    rejected = normalize_brain_output(batch, context, {"list_files"}, argument_schema_for=case.lookup)
    assert rejected.error_code == "duplicate_successful_call" and rejected.tool_requests is None
    model = SequenceModel(batch, native_action("list_files", {"path": "new"}))
    provider = LangChainBrainProvider(brain_llm=model, executable_tools=case.tools)
    alternative = provider.generate(context, ())
    assert alternative.kind == Kind.TOOL_REQUESTED and alternative.tool_request.arguments == {"path": "new"}
    assert len(model.calls) == 2 and "duplicate_successful_call" in model.calls[1][-1].content
    assert len(case.calls) == 1
    turn = case.driver.turn(state, input_for(state, alternative))
    assert turn.worker_result.success and len(case.calls) == 2


def test_resumed_batch_authorization_is_rechecked_against_successful_history(setup):
    case = setup
    state, _ = execute_once(case, "list_files", {"path": "."})
    protocol = state.protocol_visible
    pending = ToolRequest(request_id="not-the-prior-id", tool_name="list_files", arguments={"path": "."})
    continuation = ToolRequestContinuation(
        execution_id=protocol.identity.execution_id, plan_id=protocol.active_plan.plan_id,
        plan_revision=protocol.active_plan.revision, step_id="s1", step_attempt=protocol.active_step.attempt,
    )
    checkpoint = state.model_copy(update={"protocol_visible": protocol.model_copy(update={
        "pending_tool_request": pending, "tool_request_continuation": continuation,
    }), "working": state.working.model_copy(update={"last_tool_result": None})})
    restored = ExecutionState.model_validate_json(checkpoint.model_dump_json())
    decision = case.controller.decide(input_for(restored))
    assert decision.retry.last_error_code == "duplicate_successful_call"
    updated = apply_controller_decision_to_state(restored, decision)
    assert updated.protocol_visible.pending_tool_request is None
    assert updated.protocol_visible.tool_request_continuation is None
    assert len(case.calls) == 1


@pytest.mark.parametrize("name,arguments", [
    ("list_files", {"path": "Snake/"}),
    ("search_text", {"query": "Snake", "file_pattern": "*.py", "recursive": True}),
])
@pytest.mark.parametrize("obeys_correction", [True, False])
def test_live_loop_shape_projects_success_then_rejects_duplicate_with_one_correction(setup, name, arguments, obeys_correction):
    case = setup
    (case.root / "Snake").mkdir()
    model = SequenceModel(native_action(name, arguments), native_action(name, arguments),
                          native_action("brain_step_completed", {"message": "Used prior evidence."})
                          if obeys_correction else native_action(name, arguments))
    brain = BrainService(provider=LangChainBrainProvider(brain_llm=model, executable_tools=case.tools),
                         agent_system_prompt="Execute the active step.")
    driver = ExecutionDriver(coordinator=ControllerCoordinator(case.controller), planner=Mock(), brain=brain,
                             tool_runtime=case.runtime, finalizer=Finalizer())
    initial = case.state.model_copy(update={"protocol_visible": case.state.protocol_visible.model_copy(update={
        "retry": RetryMetadata(max_retries=0),
    })})
    requested = brain.run(build_brain_input({"execution_state": initial}))
    executed = driver.turn(initial, input_for(initial, requested))
    captured = integrate_tool_result(executed.execution_state, executed.decision, executed.worker_result)
    after_success = driver.turn(captured, input_for(captured))
    evidence = next(message.content for message in model.calls[1] if message.content.startswith("Execution evidence v1:"))
    attempt = json.loads(evidence.split("\n", 1)[1])["current_attempts"][0]
    assert attempt["tool"] == name and attempt["success"]
    if name == "search_text":
        assert attempt["evidence"]["matches"] == []
    assert "duplicate_successful_call" in model.calls[2][-1].content
    assert "no tool was executed" in model.calls[2][-1].content
    assert len(case.calls) == 1 and len(model.calls) == 3
    assert len(after_success.execution_state.working.tool_execution_history) == 1
    if obeys_correction:
        assert after_success.worker_result.kind == Kind.STEP_COMPLETED
    else:
        assert after_success.worker_result.kind == Kind.INVALID_OUTPUT
        assert after_success.worker_result.error_code == "duplicate_successful_call"
        failed = driver.turn(after_success.execution_state, input_for(after_success.execution_state, after_success.worker_result))
        assert failed.decision.terminal and failed.decision.reason == "step_failed_retries_exhausted"
        assert failed.decision.cursor.controller_iteration < 24
        assert len(case.calls) == 1


def scoped_record(context, name, arguments, **result_fields):
    return ToolExecutionRecord(
        execution_id=context.identity.execution_id, plan_id=context.active_plan.plan_id,
        plan_revision=context.active_plan.revision, step_id=context.active_step.step_id,
        tool_name=name, arguments=arguments,
        result=ToolResult(request_id="prior", success=True, message="Observed", **result_fields),
    )


@pytest.mark.parametrize("terminal", [False, True])
def test_unfinished_async_observation_remains_pollable_but_terminal_success_is_duplicate(setup, terminal):
    context = build_brain_input({"execution_state": setup.state})
    source = scoped_record(context, "future_poll", {"job": "123"}, is_async_job=True,
                           async_job_id="123", async_terminal=terminal,
                           async_job_status=AsyncJobStatus.COMPLETED if terminal else AsyncJobStatus.RUNNING)
    context = context.model_copy(update={"tool_execution_history": (source,)})
    if terminal:
        with pytest.raises(DuplicateSuccessfulCall):
            reject_duplicate_successful_calls([("future_poll", {"job": "123"})], context)
    else:
        reject_duplicate_successful_calls([("future_poll", {"job": "123"})], context)


def test_argument_identity_preserves_nested_values_types_case_and_order(setup):
    context = build_brain_input({"execution_state": setup.state})
    arguments = {"value": {"b": True, "a": ["Snake", "snake"]}}
    source = scoped_record(context, "future_tool", arguments)
    context = context.model_copy(update={"tool_execution_history": (source,)})
    with pytest.raises(DuplicateSuccessfulCall):
        reject_duplicate_successful_calls([("future_tool", {"value": {"a": ["Snake", "snake"], "b": True}})], context)
    for different in ({"value": {"b": 1, "a": ["Snake", "snake"]}},
                      {"value": {"b": True, "a": ["snake", "Snake"]}}):
        reject_duplicate_successful_calls([("future_tool", different)], context)
