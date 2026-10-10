"""Metadata, executable schemas and ordered execution share one batch policy."""

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import tools.registry as registry_module
from core.brain_batch_policy import validate_read_only_batch, is_oversized_read_only_batch
from core.brain_normalization import normalize_brain_output
from core.brain_provider import LangChainBrainProvider, _is_permitted_native_batch
from core.protocol.completion_identity import evidence_identity
from core.protocol.controller import CortexController, apply_controller_decision_to_state
from core.protocol.enums import BrainOutcomeKind as Kind, ControllerDecisionType as Decision
from core.protocol.models import BrainOutcome, ExecutionState, ToolRequest, ToolResult
from core.runtime.tool_result_integration import SerializedToolRuntimePort, integrate_tool_result
from test_brain_outcomes import SequenceModel, brain_input, native_action
from test_brain_read_file_batch import initial_state, input_for, native_batch
from tools.discovery_ops import get_discovery_tools
from tools.file_ops import get_file_tools
from tools.registry import ToolDefinition, ToolRegistry, get_tool_argument_schema, get_tool_definition


def context_for(names=("read_file", "find_files", "list_files", "write_file", "search_text")):
    context = brain_input()
    return context.model_copy(update={"active_plan": context.active_plan.model_copy(
        update={"available_tools": names},
    )})


def state_for(context):
    state = initial_state()
    return state.model_copy(update={"protocol_visible": state.protocol_visible.model_copy(
        update={"active_plan": context.active_plan, "active_step": context.active_step},
    )})


def group(name, count=2, arguments=None):
    arguments = arguments if arguments is not None else [
        {"path": f"file-{i}.txt"} if name in {"read_file", "list_files"} else {"pattern": f"*.ext{i}"}
        for i in range(count)
    ]
    return native_batch(count, names=[name] * count, args=arguments)


def injected_proposal(raw):
    # Bypass normalization to exercise Controller's independent rejection boundary.
    requests = tuple(ToolRequest(request_id=f"injected-{i}", tool_name="read_file", arguments={})
                     .model_copy(update={"tool_name": call["name"], "arguments": call["args"]})
                     for i, call in enumerate(raw.tool_calls))
    return BrainOutcome(outcome=Kind.TOOL_REQUESTED, step_id="s1", tool_requests=requests)


@pytest.fixture
def executables(tmp_path):
    tools = [*get_file_tools(str(tmp_path)), *get_discovery_tools(str(tmp_path))]
    registry = ToolRegistry.from_tools(tools)
    return tools, partial(get_tool_argument_schema, registry=registry)


def test_only_explicit_metadata_enables_batches():
    assert ToolDefinition("future", "workspace").max_batch_calls == 1
    assert {definition.name: definition.max_batch_calls
            for definition in registry_module.TOOL_DEFINITIONS if definition.max_batch_calls > 1} == {
        "read_file": 24, "find_files": 24, "list_files": 24,
    }
    assert all(definition.max_batch_calls == 1 for definition in registry_module.TOOL_DEFINITIONS
               if definition.name not in {"read_file", "find_files", "list_files"})


@pytest.mark.parametrize("name,count", [
    (name, count) for name in ("read_file", "find_files") for count in (2, 24)
] + [("list_files", count) for count in range(2, 25)])
def test_batches_use_bound_schemas_at_all_boundaries(executables, name, count):
    tools, schema_for = executables
    context = context_for()
    raw = group(name, count)
    model = SequenceModel(raw)
    output = LangChainBrainProvider(brain_llm=model, executable_tools=tools).generate(context, ())
    normalized = normalize_brain_output(raw, context, set(context.active_plan.available_tools),
                                        argument_schema_for=schema_for)
    assert output.kind == normalized.kind == Kind.TOOL_REQUESTED
    assert output.tool_requests == normalized.tool_requests
    assert len(output.tool_requests) == count and len(model.calls) == 1
    assert [request.arguments for request in output.tool_requests] == [call["args"] for call in raw.tool_calls]
    controller = CortexController(24, argument_schema_for=schema_for)
    decision = controller.decide(input_for(state_for(context), output))
    assert decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME
    assert decision.pending_tool_request == output.tool_requests[0]
    assert decision.tool_request_continuation.remaining == output.tool_requests[1:]
    if name == "find_files":
        assert get_tool_definition(name).args_schema is None
        assert schema_for(name).model_validate({}).model_dump()["path"] == "."


@pytest.mark.parametrize("name", ["read_file", "find_files", "list_files"])
def test_25_calls_are_rejected_at_all_boundaries_without_third_model_call(executables, name):
    tools, schema_for = executables
    context = context_for()
    raw = group(name, 25)
    assert not _is_permitted_native_batch(raw, {tool.name: tool for tool in tools})
    model = SequenceModel(raw, raw)
    output = LangChainBrainProvider(brain_llm=model, executable_tools=tools).generate(context, ())
    assert output.kind == Kind.INVALID_OUTPUT and len(model.calls) == 2
    assert "at most 24" in model.calls[1][-1].content
    normalized = normalize_brain_output(raw, context, set(context.active_plan.available_tools),
                                        argument_schema_for=schema_for)
    assert normalized.kind == Kind.INVALID_OUTPUT
    controller = CortexController(24, argument_schema_for=schema_for)
    decision = controller.decide(input_for(state_for(context), injected_proposal(raw)))
    assert decision.decision_type == Decision.TERMINATE
    assert decision.pending_tool_request is decision.tool_request_continuation is None


@pytest.mark.parametrize("name", ["read_file", "find_files", "list_files"])
def test_changed_metadata_drives_guidance_correction_and_all_limits(monkeypatch, executables, name):
    monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", tuple(
        replace(definition, max_batch_calls=3) if definition.name == name else definition
        for definition in registry_module.TOOL_DEFINITIONS
    ))
    tools, schema_for = executables
    context = context_for((name,))
    raw = group(name, 4)
    corrected = group(name, 3)
    model = SequenceModel(raw, corrected)
    output = LangChainBrainProvider(brain_llm=model, executable_tools=tools).generate(context, ())
    assert output.kind == Kind.TOOL_REQUESTED and len(output.tool_requests) == 3
    assert any(f"- {name}: up to 3" in message.content for message in model.calls[0])
    assert f"Per-tool call limits: {name}: 3." in model.calls[1][-1].content
    assert normalize_brain_output(raw, context, {name}, argument_schema_for=schema_for).kind == Kind.INVALID_OUTPUT
    controller = CortexController(24, argument_schema_for=schema_for)
    assert controller.decide(input_for(state_for(context), injected_proposal(raw))).decision_type == Decision.TERMINATE
    assert controller.decide(input_for(state_for(context), output)).decision_type == Decision.DISPATCH_TOOL_RUNTIME


@pytest.mark.parametrize("authorized,bound,missing_schema,expected", [
    (("read_file", "find_files"), ("read_file", "find_files"), None, ("read_file", "find_files")),
    (("read_file",), ("read_file", "find_files"), None, ("read_file",)),
    (("find_files",), ("read_file", "find_files"), None, ("find_files",)),
    (("find_files",), ("read_file",), None, ()),
    ((), ("read_file", "find_files"), None, ()),
    (("read_file", "find_files"), ("read_file", "find_files"), "find_files", ("read_file",)),
    (("read_file", "find_files"), ("read_file", "find_files"), "read_file", ("find_files",)),
    (("list_files",), ("list_files",), None, ("list_files",)),
    (("search_text",), ("search_text",), None, ()),
])
def test_guidance_only_advertises_authorized_bound_schema_valid_tools(executables, authorized, bound,
                                                                    missing_schema, expected):
    tools, _ = executables
    tools = [SimpleNamespace(name=tool.name, args_schema=None) if tool.name == missing_schema else tool
             for tool in tools if tool.name in bound]
    model = SequenceModel(native_action("brain_step_failed", {"message": "Finished test"}))
    LangChainBrainProvider(brain_llm=model, executable_tools=tools).generate(context_for(authorized), ())
    guidance = next(message.content for message in model.calls[0]
                    if message.content.endswith("lifecycle actions require one call."))
    lines = [f"- {name}: up to {get_tool_definition(name).max_batch_calls}" for name in expected]
    expected_guidance = ("Independent multiple calls allowed (up to 24 total; tools may be mixed):\n" + "\n".join(lines) +
                         "\nAll other tools and lifecycle actions require one call." if lines else
                         "All tools and lifecycle actions require one call.")
    assert guidance == expected_guidance
    assert {tool.name for tool in model.bound_tools if hasattr(tool, "name")} == set(authorized) & set(bound)


@pytest.mark.parametrize("invalid", ["unauthorized", "malformed", "duplicate", "mutating",
                                     "singleton_only", "schema_missing", "invalid_args"])
def test_invalid_batches_fail_closed_at_all_boundaries(executables, monkeypatch, invalid):
    tools, schema_for = executables
    context = context_for()
    raw = group("find_files")
    if invalid == "unauthorized":
        context = context_for(("read_file",))
    elif invalid == "malformed":
        raw.tool_calls[1]["args"] = []
    elif invalid == "duplicate":
        raw = group("find_files", arguments=[{"pattern": "*.py"},
                    {"path": ".", "pattern": "*.py", "recursive": True, "offset": "0", "limit": 100}])
    elif invalid == "mutating":
        monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", tuple(
            replace(definition, max_batch_calls=24) if definition.name == "write_file" else definition
            for definition in registry_module.TOOL_DEFINITIONS
        ))
        raw = group("write_file", arguments=[{"path": "a", "content": "x"}, {"path": "b", "content": "y"}])
    elif invalid == "singleton_only":
        raw = group("search_text", arguments=[{"query": "a"}, {"query": "b"}])
    elif invalid == "schema_missing":
        tools = [SimpleNamespace(name=tool.name, args_schema=None) if tool.name == "find_files" else tool
                 for tool in tools]
        schema_for = partial(get_tool_argument_schema, registry=ToolRegistry.from_tools(tools))
    else:
        raw.tool_calls[1]["args"] = {"offset": "not an integer"}
    authorized = {tool.name: tool for tool in tools if tool.name in context.active_plan.available_tools}
    assert not _is_permitted_native_batch(raw, authorized, argument_schema_for=schema_for)
    model = SequenceModel(raw, raw)
    assert LangChainBrainProvider(brain_llm=model, executable_tools=tools).generate(context, ()).kind == Kind.INVALID_OUTPUT
    assert len(model.calls) <= 2
    assert normalize_brain_output(raw, context, set(authorized), argument_schema_for=schema_for).kind == Kind.INVALID_OUTPUT
    decision = CortexController(24, argument_schema_for=schema_for).decide(
        input_for(state_for(context), injected_proposal(raw)),
    )
    assert decision.decision_type == Decision.TERMINATE
    assert decision.pending_tool_request is decision.tool_request_continuation is None


def test_explicit_missing_schema_never_falls_back_to_read_file_declaration(executables):
    tools, _ = executables
    raw = group("read_file")
    missing = lambda name: None
    assert get_tool_definition("read_file").args_schema is not None
    assert not _is_permitted_native_batch(raw, {tool.name: tool for tool in tools}, argument_schema_for=missing)
    assert normalize_brain_output(raw, context_for(), {"read_file"}, argument_schema_for=missing).kind == Kind.INVALID_OUTPUT
    decision = CortexController(24, argument_schema_for=missing).decide(
        input_for(state_for(context_for()), injected_proposal(raw)),
    )
    assert decision.decision_type == Decision.TERMINATE


@pytest.mark.parametrize("unusable", ["json_schema", "mutating"])
def test_guidance_excludes_unsupported_schema_or_mutating_metadata(executables, monkeypatch, unusable):
    tools, _ = executables
    if unusable == "json_schema":
        tools = [SimpleNamespace(name="read_file", args_schema={"type": "object"})]
    else:
        monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", tuple(
            replace(definition, mutating=True) if definition.name == "read_file" else definition
            for definition in registry_module.TOOL_DEFINITIONS
        ))
    model = SequenceModel(native_action("brain_step_failed", {"message": "Finished test"}))
    LangChainBrainProvider(brain_llm=model, executable_tools=tools).generate(context_for(("read_file",)), ())
    assert model.calls[0][0].content == "All tools and lifecycle actions require one call."


@pytest.mark.parametrize("duplicate", [False, True])
def test_duplicate_detection_supports_pathless_tools(monkeypatch, duplicate):
    class Query(BaseModel):
        query: str
        limit: int = 10

    monkeypatch.setattr(registry_module, "TOOL_DEFINITIONS", (
        *registry_module.TOOL_DEFINITIONS, ToolDefinition("lookup", "general", max_batch_calls=4),
    ))
    schema_for = lambda name: Query
    raw = group("lookup", arguments=[{"query": "a"}, {"limit": "10", "query": "a" if duplicate else "b"}])
    context = context_for(("lookup",))
    normalized = normalize_brain_output(raw, context, {"lookup"}, argument_schema_for=schema_for)
    if duplicate:
        with pytest.raises(ValueError, match="duplicate_effective"):
            validate_read_only_batch([(call["name"], call["args"]) for call in raw.tool_calls], {"lookup"},
                                     argument_schema_for=schema_for)
        assert normalized.kind == Kind.INVALID_OUTPUT
    else:
        assert normalized.kind == Kind.TOOL_REQUESTED
        assert normalized.tool_requests[0].arguments == {"query": "a"}
        assert normalized.tool_requests[1].arguments == {"limit": "10", "query": "b"}
    controller = CortexController(24, argument_schema_for=schema_for)
    decision = controller.decide(input_for(state_for(context), injected_proposal(raw)))
    assert decision.decision_type == (Decision.TERMINATE if duplicate else Decision.DISPATCH_TOOL_RUNTIME)
    tool = SimpleNamespace(name="lookup", args_schema=Query)
    assert _is_permitted_native_batch(raw, {"lookup": tool}) is not duplicate
    if not duplicate:
        model = SequenceModel(raw)
        output = LangChainBrainProvider(brain_llm=model, executable_tools=[tool]).generate(context, ())
        assert output.tool_requests == normalized.tool_requests
        assert "- lookup: up to 4" in model.calls[0][0].content


def test_duplicate_identity_does_not_assume_filesystem_path_normalization():
    raw = group("read_file", arguments=[{"path": "a"}, {"path": "./a"}])
    output = normalize_brain_output(raw, context_for(), {"read_file"})
    assert output.kind == Kind.TOOL_REQUESTED
    assert [request.arguments for request in output.tool_requests] == [{"path": "a"}, {"path": "./a"}]


@pytest.mark.parametrize("captured", [False, True])
def test_find_files_checkpoint_resume_continuation_preserves_order_and_provenance(executables, tmp_path, captured):
    tools, schema_for = executables
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("pass", encoding="utf-8")
    context = context_for(("find_files",))
    raw = group("find_files", 3, [{"pattern": "*.py", "offset": i, "limit": 1} for i in range(3)])
    output = normalize_brain_output(raw, context, {"find_files"}, argument_schema_for=schema_for)
    controller = CortexController(24, argument_schema_for=schema_for)
    runtime = SerializedToolRuntimePort(tools)
    state = state_for(context)
    observed = []

    def execute(state, decision):
        assert decision.pending_tool_request == output.tool_requests[len(observed)]
        observed.append(decision.pending_tool_request)
        result = runtime.execute(decision.pending_tool_request)
        assert result.success
        return integrate_tool_result(state, decision, result)

    first = controller.decide(input_for(state, output))
    state = execute(apply_controller_decision_to_state(state, first), first)
    second = controller.decide(input_for(state))
    state = apply_controller_decision_to_state(state, second)
    if captured:
        state = execute(state, second)
    state = ExecutionState.model_validate_json(state.model_dump_json())
    controller = CortexController(24, argument_schema_for=schema_for)
    resumed = controller.decide(input_for(state))
    if not captured:
        assert resumed.pending_tool_request == second.pending_tool_request
        state = execute(apply_controller_decision_to_state(state, resumed), resumed)
        resumed = controller.decide(input_for(state))
    state = execute(apply_controller_decision_to_state(state, resumed), resumed)
    brain_dispatch = controller.decide(input_for(state))
    assert brain_dispatch.decision_type == Decision.DISPATCH_BRAIN
    state = apply_controller_decision_to_state(state, brain_dispatch)
    records = state.working.tool_execution_history
    assert observed == list(output.tool_requests)
    assert [record.result.data["files"] for record in records] == [["a.py"], ["b.py"], ["c.py"]]
    assert all((record.execution_id, record.plan_id, record.plan_revision, record.step_id, record.tool_name) ==
               (context.identity.execution_id, "p1", 1, "s1", "find_files") for record in records)
    completion = controller.decide(input_for(state, BrainOutcome(
        outcome=Kind.STEP_COMPLETED, step_id="s1", message="Found three Python files",
    )))
    assert completion.completion_evidence.tool_request_ids == tuple(request.request_id for request in observed)
    assert completion.completion_evidence.evidence_id == evidence_identity(records)


@pytest.mark.parametrize("captured", [False, True])
def test_find_files_schema_is_rechecked_on_resume_and_continuation(executables, captured):
    _, schema_for = executables
    context = context_for(("find_files",))
    output = normalize_brain_output(group("find_files"), context, {"find_files"}, argument_schema_for=schema_for)
    controller = CortexController(24, argument_schema_for=schema_for)
    state = state_for(context)
    first = controller.decide(input_for(state, output))
    state = apply_controller_decision_to_state(state, first)
    if captured:
        state = integrate_tool_result(state, first, ToolResult(
            request_id=first.pending_tool_request.request_id, success=True, message="Found", data={"files": []},
        ))
    resumed = CortexController(24, argument_schema_for=lambda name: None).decide(input_for(state))
    assert resumed.decision_type == Decision.TERMINATE
    assert "batch_argument_schema_unavailable" in resumed.reason


def test_singleton_only_tool_is_not_classified_as_size_only(executables):
    _, schema_for = executables
    assert not is_oversized_read_only_batch([("search_text", {"query": "a"}), ("search_text", {"query": "b"})],
                                           {"search_text"}, argument_schema_for=schema_for)


def test_graph_composition_supplies_inferred_schema_to_controller(executables, monkeypatch):
    import core.graph_nodes as module

    tools, schema_for = executables
    monkeypatch.setattr(module, "create_planner_node", lambda **kwargs: None)
    monkeypatch.setattr(module, "create_brain_node", lambda **kwargs: None)
    monkeypatch.setattr(module, "create_capture_tool_output_node", lambda: None)
    controller_node, *_ = module.create_graph_nodes(
        brain_llm=None, executable_tools=tools, planner_llm=None, rag_service=None,
        rag_top_k=4, agent_system_prompt="test", sap_system_prompt=None,
        tools_set={"find_files"}, show_raw_llm=False,
    )
    context = context_for(("find_files",))
    output = normalize_brain_output(group("find_files"), context, {"find_files"}, argument_schema_for=schema_for)
    update = controller_node({"execution_state": state_for(context), "brain_result": output})
    assert update["controller_decision"].decision_type == Decision.DISPATCH_TOOL_RUNTIME
    assert update["controller_decision"].pending_tool_request == output.tool_requests[0]


@pytest.mark.parametrize("invalid", ["duplicate", "duplicate_default",
                                     "unauthorized", "invalid_args", "schema_missing"])
def test_list_files_batches_reject_invalid_groups_at_all_boundaries(executables, invalid):
    tools, schema_for = executables
    context = context_for()
    raw = group("list_files")
    if invalid == "duplicate":
        raw = group("list_files", arguments=[{"path": "Snake/ui"}, {"path": "Snake/ui"}])
    elif invalid == "duplicate_default":
        raw = group("list_files", arguments=[{}, {"path": "."}])
    elif invalid == "unauthorized":
        context = context_for(("read_file", "find_files"))
    elif invalid == "invalid_args":
        raw.tool_calls[1]["args"] = {"path": []}
    else:
        tools = [SimpleNamespace(name=tool.name, args_schema=None) if tool.name == "list_files" else tool
                 for tool in tools]
        schema_for = partial(get_tool_argument_schema, registry=ToolRegistry.from_tools(tools))
    authorized = {tool.name: tool for tool in tools if tool.name in context.active_plan.available_tools}
    candidates = [(call["name"], call["args"]) for call in raw.tool_calls]
    with pytest.raises(ValueError):
        validate_read_only_batch(candidates, authorized, argument_schema_for=schema_for)
    assert not _is_permitted_native_batch(raw, authorized, argument_schema_for=schema_for)
    model = SequenceModel(raw, raw)
    output = LangChainBrainProvider(brain_llm=model, executable_tools=tools).generate(context, ())
    assert output.kind == Kind.INVALID_OUTPUT
    assert output.tool_request is output.tool_requests is None
    assert normalize_brain_output(raw, context, set(authorized), argument_schema_for=schema_for).kind == Kind.INVALID_OUTPUT
    decision = CortexController(24, argument_schema_for=schema_for).decide(
        input_for(state_for(context), injected_proposal(raw)),
    )
    assert decision.decision_type == Decision.TERMINATE
    assert decision.pending_tool_request is decision.tool_request_continuation is None


def test_snake_list_files_batch_executes_sequentially_with_ordered_provenance(executables, tmp_path):
    tools, schema_for = executables
    paths = ("Snake/ui", "Snake/app", "Snake/core")
    for index, path in enumerate(paths):
        target = tmp_path / path
        target.mkdir(parents=True)
        (target / f"module-{index}.py").write_text("pass", encoding="utf-8")
    assert get_tool_definition("list_files").args_schema is None
    assert schema_for("list_files").model_validate({}).model_dump(mode="json") == {"path": "."}
    raw = group("list_files", 3, [{"path": path} for path in paths])
    context = context_for(("list_files",))
    model = SequenceModel(raw)
    output = LangChainBrainProvider(brain_llm=model, executable_tools=tools).generate(context, ())
    assert output.kind == Kind.TOOL_REQUESTED and len(model.calls) == 1
    assert [request.arguments for request in output.tool_requests] == [{"path": path} for path in paths]
    controller = CortexController(24, argument_schema_for=schema_for)
    runtime = SerializedToolRuntimePort(tools)
    state = state_for(context)
    for index, request in enumerate(output.tool_requests):
        decision = controller.decide(input_for(state, output if index == 0 else None))
        assert decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME
        assert decision.pending_tool_request == request
        assert decision.tool_request_continuation.remaining == output.tool_requests[index + 1:]
        state = apply_controller_decision_to_state(state, decision)
        if index == 1:
            state = ExecutionState.model_validate_json(state.model_dump_json())
            controller = CortexController(24, argument_schema_for=schema_for)
            decision = controller.decide(input_for(state))
            assert decision.pending_tool_request == request
            state = apply_controller_decision_to_state(state, decision)
        result = runtime.execute(decision.pending_tool_request)
        assert result.success and result.data["entries"] == [f"module-{index}.py"]
        state = integrate_tool_result(state, decision, result)
        assert len(state.working.tool_execution_history) == index + 1
    brain_dispatch = controller.decide(input_for(state))
    assert brain_dispatch.decision_type == Decision.DISPATCH_BRAIN
    state = apply_controller_decision_to_state(state, brain_dispatch)
    assert state.protocol_visible.pending_tool_request is state.protocol_visible.tool_request_continuation is None
    records = state.working.tool_execution_history
    assert [record.result.request_id for record in records] == [request.request_id for request in output.tool_requests]
    assert [record.result.data["path"] for record in records] == list(paths)
    assert all((record.execution_id, record.plan_id, record.plan_revision, record.step_id, record.tool_name) ==
               (context.identity.execution_id, "p1", 1, "s1", "list_files") for record in records)
    completion = controller.decide(input_for(state, BrainOutcome(
        outcome=Kind.STEP_COMPLETED, step_id="s1", message="Listed Snake UI, app and core directories",
    )))
    assert completion.completion_evidence.tool_request_ids == tuple(request.request_id for request in output.tool_requests)
    assert completion.completion_evidence.evidence_id == evidence_identity(records)
