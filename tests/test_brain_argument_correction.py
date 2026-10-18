"""Executable schema failures share the single bounded provider correction."""

from copy import deepcopy
from enum import Enum
from functools import partial
import json
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from ollama._types import ChatRequest
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.v1 import BaseModel as LegacyBaseModel

import core.brain_normalization as normalization
import core.brain_provider as provider_module
from core.brain import BrainService
from core.brain_provider import LangChainBrainProvider
from core.logging.live_status import LiveStatus
from core.protocol.bridge import build_controller_input
from core.protocol.controller import CortexController, apply_controller_decision_to_state
from core.protocol.enums import BrainOutcomeKind as Kind, ControllerDecisionType as Decision
from core.protocol.models import ExecutionState, ProtocolVisibleState, RetryMetadata
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver
from core.runtime.tool_result_integration import SerializedToolRuntimePort, integrate_tool_result
from test_brain_outcomes import SequenceModel, brain_input, native_action
from test_brain_read_file_batch import native_batch
from test_duplicate_successful_call import scoped_record
from test_web_ops import dns_records, html_response
from tools import web_ops as web
from tools.file_ops import get_file_tools
from tools.registry import ToolRegistry, get_tool_argument_schema


class NestedArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    count: int = Field(ge=1, le=5)
    mode: Literal["brief", "full"]


class ProbeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    limit: int = Field(default=2, ge=1, le=10)
    mode: Literal["brief", "full"] = "brief"
    nested: list[NestedArguments] = Field(default_factory=list)


def context_for(*names):
    context = brain_input()
    return context.model_copy(update={"active_plan": context.active_plan.model_copy(update={"available_tools": names})})


def state_for(context):
    return ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=context.identity, cursor=context.cursor, active_plan=context.active_plan,
        active_step=context.active_step, retry=context.retry,
    ))


def feedback_in(model):
    return json.loads(model.calls[1][-1].content.split("\n")[1])


def setup_probe(*replies, schema=ProbeArguments):
    model = SequenceModel(*replies)
    model.bind_tools = Mock(wraps=model.bind_tools)
    tool = SimpleNamespace(name="schema_probe", args_schema=schema, invoke=Mock())
    provider = LangChainBrainProvider(brain_llm=model, executable_tools=[tool])
    return provider, model, tool


@pytest.mark.parametrize("arguments,field,error_type", [
    ({}, ["path"], "missing"),
    ({"path": "a", "extra": "never echo me"}, ["extra"], "extra_forbidden"),
    ({"path": 12}, ["path"], "string_type"),
    ({"path": "a", "limit": "2"}, ["limit"], "int_type"),
    ({"path": "a", "limit": True}, ["limit"], "int_type"),
    ({"path": "a", "limit": 0}, ["limit"], "greater_than_equal"),
    ({"path": "a", "limit": 11}, ["limit"], "less_than_equal"),
    ({"path": "a", "mode": "unknown"}, ["mode"], "literal_error"),
    ({"path": "a", "nested": [{"count": 6, "mode": "brief"}]}, ["nested", 0, "count"], "less_than_equal"),
])
@pytest.mark.parametrize("content", ["", "Incidental model explanation"])
def test_schema_errors_use_correction_without_executing_or_creating_requests(
    arguments, field, error_type, content, monkeypatch, caplog, capsys,
):
    first = native_action("schema_probe", arguments).model_copy(update={"content": content})
    corrected = native_action("schema_probe", {"path": "repaired"})
    provider, model, tool = setup_probe(first, corrected)
    context = context_for(tool.name)
    snapshot = context.model_dump_json()
    create_request = Mock(wraps=normalization._tool_request)
    monkeypatch.setattr(normalization, "_tool_request", create_request)
    invoke = model.invoke

    def inspect(messages):
        if model.calls:
            assert create_request.call_count == 0 and context.tool_execution_history == ()
            tool.invoke.assert_not_called()
        return invoke(messages)

    model.invoke = inspect
    with caplog.at_level("DEBUG", logger="core.brain_provider"):
        outcome = provider.generate(context, ())
    assert outcome.kind == Kind.TOOL_REQUESTED
    assert outcome.tool_request.arguments == {"path": "repaired"}
    assert len(model.calls) == 2 and model.calls[1][:-2] == model.calls[0]
    assert model.calls[1][-2] is first
    model.bind_tools.assert_called_once()
    create_request.assert_called_once()
    assert create_request.call_args.args[:2] == (tool.name, {"path": "repaired"})
    feedback = feedback_in(model)
    assert feedback["code"] == "invalid_native_tool_arguments" and not feedback["issues_truncated"]
    issue, = feedback["issues"]
    assert issue["tool"] == tool.name and issue["field"] == field and issue["type"] == error_type
    assert issue["call_index"] == 0 and issue["reason"]
    assert "never echo me" not in model.calls[1][-1].content
    assert context.model_dump_json() == snapshot
    tool.invoke.assert_not_called()
    assert any("correction_reason=invalid_native_tool_arguments" in record.message for record in caplog.records)
    assert capsys.readouterr().out == ""


def test_web_fetch_upper_bound_repairs_then_executes_once_and_survives_checkpoint(monkeypatch):
    url = "https://docs.python.org/3.15/whatsnew/3.15.html"
    monkeypatch.setattr(web.socket, "getaddrinfo", Mock(return_value=dns_records("93.184.216.34")))
    send = Mock(return_value=html_response(b"<title>Python 3.15</title><article>Supported page evidence.</article>"))
    monkeypatch.setattr(web._PublicHTTPAdapter, "send", send)
    tools = web.get_web_tools()
    registry = ToolRegistry.from_tools(tools)
    lookup = partial(get_tool_argument_schema, registry=registry)
    first = native_action("web_fetch", {"url": url, "max_chars": 1_000_000})
    corrected = native_action("web_fetch", {"url": url, "max_chars": 200_000})
    model = SequenceModel(first, corrected)
    context = context_for("web_fetch")
    state = state_for(context)
    provider = LangChainBrainProvider(brain_llm=model, executable_tools=tools)
    status = LiveStatus(enabled=False)
    status.start("brain")
    try:
        outcome = BrainService(provider=provider, agent_system_prompt="Read supported evidence.").run(context)
        assert status.provider_invocations_by_worker["brain"] == 2
        assert status.usage_by_worker["brain"]["calls"] == 2
    finally:
        status.stop()
    assert outcome.kind == Kind.TOOL_REQUESTED
    issue, = feedback_in(model)["issues"]
    assert issue["field"] == ["max_chars"] and "200000" in issue["reason"]
    assert "less than or equal" in issue["reason"]
    send.assert_not_called()
    assert state.working.tool_execution_history == ()
    assert registry.by_name["web_fetch"].args_schema.model_json_schema()["properties"]["max_chars"]["maximum"] == 200_000
    controller = CortexController(24, argument_schema_for=lookup)
    driver = ExecutionDriver(coordinator=ControllerCoordinator(controller), planner=Mock(), brain=Mock(),
                             tool_runtime=SerializedToolRuntimePort(tools), finalizer=Mock())
    # A replayed corrected proposal has the ordinary deterministic identity.
    expected = normalization.normalize_brain_output(corrected, context, {"web_fetch"}, argument_schema_for=lookup)
    assert outcome.tool_request == expected.tool_request
    restored = ExecutionState.model_validate_json(state.model_dump_json())
    value = build_controller_input({"execution_state": restored, "brain_result": outcome})
    turn = driver.turn(restored, value)
    captured = integrate_tool_result(turn.execution_state, turn.decision, turn.worker_result)
    assert send.call_count == 1 and len(captured.working.tool_execution_history) == 1
    assert turn.worker_result.success and turn.worker_result.data["text"] == "Supported page evidence."
    replay = ExecutionState.model_validate_json(captured.model_dump_json())
    repeated = build_controller_input({"execution_state": replay, "brain_result": outcome}).model_copy(update={"tool_result": None})
    rejected = driver.turn(replay, repeated, dispatch_worker=False)
    assert rejected.decision.retry.last_error_code == "duplicate_successful_call" and send.call_count == 1
    assert replay.working.tool_execution_history == captured.working.tool_execution_history


@pytest.mark.parametrize("first", [
    native_action("schema_probe", {}), AIMessage(content="No native call"),
])
def test_second_invalid_response_fails_without_requests_or_third_invocation(first, monkeypatch, caplog):
    provider, model, tool = setup_probe(first, native_action("schema_probe", {"path": 12}))
    create_request = Mock(wraps=normalization._tool_request)
    monkeypatch.setattr(normalization, "_tool_request", create_request)
    context = context_for(tool.name)
    with caplog.at_level("DEBUG", logger="core.brain_provider"):
        outcome = provider.generate(context, ())
    assert len(model.calls) == 2 and outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.error_code == "invalid_native_tool_arguments"
    assert outcome.message == "Brain returned invalid native tool arguments."
    assert outcome.tool_request is outcome.tool_requests is None
    create_request.assert_not_called()
    tool.invoke.assert_not_called()
    assert context.tool_execution_history == ()
    assert any("attempt=2" in record.message and "correction_exhausted=True" in record.message for record in caplog.records)
    # Controller keeps its existing configured failure policy (no new retries).
    state = state_for(context)
    state = state.model_copy(update={"protocol_visible": state.protocol_visible.model_copy(update={"retry": RetryMetadata(max_retries=0)})})
    decision = CortexController(24).decide(build_controller_input({"execution_state": state, "brain_result": outcome}))
    assert decision.terminal and decision.reason == "step_failed_retries_exhausted"


def test_valid_arguments_have_no_correction_invocation_or_identity_change():
    raw = native_action("schema_probe", {"path": "valid", "limit": 10})
    provider, model, tool = setup_probe(raw)
    context = context_for(tool.name)
    outcome = provider.generate(context, ())
    assert len(model.calls) == 1 and outcome.kind == Kind.TOOL_REQUESTED
    expected = normalization.normalize_brain_output(raw, context, {tool.name})
    assert outcome.tool_request == expected.tool_request
    tool.invoke.assert_not_called()


@pytest.mark.parametrize("still_invalid", [False, True])
def test_invalid_batch_member_keeps_entire_response_atomic(tmp_path, monkeypatch, still_invalid):
    tools = get_file_tools(str(tmp_path))
    registry = ToolRegistry.from_tools(tools)
    lookup = partial(get_tool_argument_schema, registry=registry)
    first = native_batch(args=[{"path": "good.txt"}, {"path": "repaired.txt", "limit": 0}])
    corrected = first if still_invalid else native_batch(args=[{"path": "good.txt"}, {"path": "repaired.txt", "limit": 10}])
    model = SequenceModel(first, corrected)
    context = context_for("read_file")
    create_request = Mock(wraps=normalization._tool_request)
    monkeypatch.setattr(normalization, "_tool_request", create_request)
    execute = Mock(wraps=SerializedToolRuntimePort(tools).execute)
    invoke = model.invoke

    def inspect(messages):
        create_request.assert_not_called()
        execute.assert_not_called()
        assert context.tool_execution_history == ()
        return invoke(messages)

    model.invoke = inspect
    outcome = LangChainBrainProvider(brain_llm=model, executable_tools=tools).generate(context, ())
    assert len(model.calls) == 2
    issue, = feedback_in(model)["issues"]
    assert issue["call_index"] == 1 and issue["field"] == ["limit"]
    execute.assert_not_called()
    if still_invalid:
        assert outcome.kind == Kind.INVALID_OUTPUT and outcome.error_code == "invalid_native_tool_arguments"
        create_request.assert_not_called()
        return
    assert outcome.kind == Kind.TOOL_REQUESTED and len(outcome.tool_requests) == 2
    assert create_request.call_count == 2
    for name in ("good.txt", "repaired.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    state = state_for(context)
    controller = CortexController(24, argument_schema_for=lookup)
    for index, request in enumerate(outcome.tool_requests):
        value = build_controller_input({"execution_state": state, **({"brain_result": outcome} if index == 0 else {})})
        decision = controller.decide(value)
        assert decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME
        state = apply_controller_decision_to_state(state, decision)
        result = execute(request)
        assert result.success
        state = integrate_tool_result(state, decision, result)
    assert execute.call_count == 2 and len(state.working.tool_execution_history) == 2


def test_corrected_arguments_cannot_bypass_successful_duplicate_guard():
    context = context_for("schema_probe")
    record = scoped_record(context, "schema_probe", {"path": "already-read"})
    context = context.model_copy(update={"tool_execution_history": (record,)})
    provider, model, tool = setup_probe(
        native_action("schema_probe", {}), native_action("schema_probe", {"path": "already-read"}),
    )
    outcome = provider.generate(context, ())
    assert len(model.calls) == 2 and outcome.error_code == "duplicate_successful_call"
    assert outcome.tool_request is None and context.tool_execution_history == (record,)
    tool.invoke.assert_not_called()


@pytest.mark.parametrize("name,arguments", [
    ("brain_step_completed", {}), ("brain_step_completed", {"message": "Done", "step_id": "spoof"}),
    ("brain_step_failed", {"message": 12}),
    ("brain_replan_requested", {"reason": "Changed", "constraints": "wrong"}),
])
def test_invalid_lifecycle_arguments_preserve_strict_no_correction_behavior(name, arguments):
    provider, model, tool = setup_probe(native_action(name, arguments))
    outcome = provider.generate(context_for(tool.name), ())
    assert outcome.kind == Kind.INVALID_OUTPUT and len(model.calls) == 1
    assert outcome.tool_request is outcome.completion_evidence is outcome.replan_request is None
    tool.invoke.assert_not_called()


def test_invalid_lifecycle_after_argument_correction_is_not_accepted():
    provider, model, tool = setup_probe(
        native_action("schema_probe", {}), native_action("brain_step_completed", {"message": "Done", "extra": True}),
    )
    outcome = provider.generate(context_for(tool.name), ())
    assert outcome.kind == Kind.INVALID_OUTPUT and outcome.error_code == "unexpected_envelope_fields"
    assert len(model.calls) == 2 and outcome.completion_evidence is None
    tool.invoke.assert_not_called()


def test_bounded_feedback_covers_many_nested_errors_without_echoing_inputs():
    bad = {"path": "a", "nested": [{"count": 0, "mode": "secret input" * 1000} for _ in range(20)]}
    provider, model, tool = setup_probe(native_action("schema_probe", bad), native_action("schema_probe", {"path": "fixed"}))
    assert provider.generate(context_for(tool.name), ()).kind == Kind.TOOL_REQUESTED
    feedback = feedback_in(model)
    assert 1 <= len(feedback["issues"]) <= 8 and feedback["issues_truncated"]
    assert len(json.dumps(feedback, ensure_ascii=True)) <= 4000
    assert "secret input" not in model.calls[1][-1].content
    assert all("input" not in issue and "ctx" not in issue and "url" not in issue for issue in feedback["issues"])


def test_custom_validation_messages_do_not_expose_exception_internals():
    class CustomArguments(ProbeArguments):
        @field_validator("path")
        @classmethod
        def check_path(cls, value):
            if value == "bad":
                raise ValueError("private validation internals" * 1000)
            return value

    provider, model, tool = setup_probe(native_action("schema_probe", {"path": "bad"}),
                                       native_action("schema_probe", {"path": "good"}), schema=CustomArguments)
    assert provider.generate(context_for(tool.name), ()).kind == Kind.TOOL_REQUESTED
    assert "private validation internals" not in model.calls[1][-1].content


def test_legacy_pydantic_validation_uses_same_correction_boundary():
    class LegacyArguments(LegacyBaseModel):
        path: str

    provider, model, tool = setup_probe(native_action("schema_probe", {}),
                                       native_action("schema_probe", {"path": "valid"}), schema=LegacyArguments)
    assert provider.generate(context_for(tool.name), ()).kind == Kind.TOOL_REQUESTED
    assert feedback_in(model)["issues"][0]["field"] == ["path"]
    assert len(model.calls) == 2


def test_equivalent_schema_validation_failure_has_safe_generic_feedback():
    class EquivalentArguments:
        @classmethod
        def model_validate(cls, arguments):
            if not isinstance(arguments.get("path"), str):
                raise ValueError("private validator internals")

    provider, model, tool = setup_probe(native_action("schema_probe", {}),
                                       native_action("schema_probe", {"path": "valid"}), schema=EquivalentArguments)
    assert provider.generate(context_for(tool.name), ()).kind == Kind.TOOL_REQUESTED
    issue, = feedback_in(model)["issues"]
    assert issue["type"] == "schema_validation_failed" and issue["field"] == []
    assert "private validator internals" not in model.calls[1][-1].content


def test_enum_argument_violation_uses_specific_correction_feedback():
    class Mode(str, Enum):
        BRIEF = "brief"
        FULL = "full"

    class EnumArguments(BaseModel):
        mode: Mode

    provider, model, tool = setup_probe(native_action("schema_probe", {"mode": "wrong"}),
                                       native_action("schema_probe", {"mode": "brief"}), schema=EnumArguments)
    assert provider.generate(context_for(tool.name), ()).kind == Kind.TOOL_REQUESTED
    issue, = feedback_in(model)["issues"]
    assert issue["field"] == ["mode"] and issue["type"] == "enum"
    assert "brief" in issue["reason"] and "full" in issue["reason"]


def test_argument_correction_provider_failure_does_not_start_another_retry():
    provider, model, tool = setup_probe(native_action("schema_probe", {}), RuntimeError("offline"))
    outcome = provider.generate(context_for(tool.name), ())
    assert outcome.kind == Kind.PROVIDER_FAILURE and outcome.error_code == "RuntimeError"
    assert len(model.calls) == 2 and outcome.tool_request is None
    tool.invoke.assert_not_called()


def test_argument_correction_can_choose_a_valid_lifecycle_action():
    provider, model, tool = setup_probe(native_action("schema_probe", {}), native_action("brain_replan_requested", {
        "reason": "Need a revised strategy", "constraints": ["Keep current evidence"],
    }))
    outcome = provider.generate(context_for(tool.name), ())
    assert outcome.kind == Kind.REPLAN_REQUESTED and len(model.calls) == 2
    assert outcome.replan_request.constraints == ("Keep current evidence",)
    tool.invoke.assert_not_called()


def test_argument_correction_survives_real_ollama_binding_with_identical_schemas():
    requests = []
    replies = iter([{"path": 12}, {"path": "corrected"}])

    def chat(**kwargs):
        requests.append(ChatRequest(**deepcopy(kwargs)).model_dump(exclude_none=True))
        return iter([{"message": {"role": "assistant", "content": "", "tool_calls": [{
            "function": {"name": "schema_probe", "arguments": next(replies)},
        }]}, "done": True}])

    model = ChatOllama(model="offline-test", temperature=0)
    model._client = SimpleNamespace(chat=chat)
    from langchain_core.tools import StructuredTool
    tool = StructuredTool.from_function(lambda **kwargs: "must not execute", name="schema_probe",
                                        description="Probe strict argument validation.", args_schema=ProbeArguments)
    outcome = LangChainBrainProvider(brain_llm=model, executable_tools=[tool]).generate(context_for(tool.name), ())
    assert outcome.kind == Kind.TOOL_REQUESTED and len(requests) == 2
    assert requests[0]["tools"] == requests[1]["tools"]
    assert requests[0]["messages"] == requests[1]["messages"][:-2]
    assert "invalid_native_tool_arguments" in requests[1]["messages"][-1]["content"]
    assert not any(message["role"] == "tool" for message in requests[1]["messages"])
