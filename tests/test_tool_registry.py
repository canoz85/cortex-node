from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from core.brain_provider import LangChainBrainProvider
from core.graph_constants import MUTATING_TOOLS
from core.planner import PlannerService
from core.protocol.controller import CortexController
from core.protocol.enums import ControllerDecisionType, PlannerOutcome, BrainOutcomeKind
from core.protocol.models import (
    ControllerInput, ExecutionContext, ExecutionCursor, ExecutionIdentity, PlanningCapabilities,
)
from test_brain_outcomes import FakeModel, brain_input
from test_planner_service import FakeProvider, FakeRouter
from tools.registry import (
    TOOL_DEFINITIONS, ToolDefinition, ToolEntry, ToolRegistry,
    build_tool_registry, default_enabled_tools,
)


def test_registry_rejects_duplicates_mismatches_and_unknown_names():
    tool = SimpleNamespace(name="example")
    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry.from_tools([tool, tool])
    with pytest.raises(ValueError, match="mismatch"):
        ToolRegistry([ToolEntry("other", tool)])
    registry = ToolRegistry.from_tools([tool])
    with pytest.raises(ValueError, match="Unregistered"):
        registry.resolve(["missing"])
    with pytest.raises(TypeError):
        registry.by_name["other"] = tool


def test_production_tools_all_registered_and_enabled_explicitly(tmp_path):
    registry = build_tool_registry(str(tmp_path), str(tmp_path), SimpleNamespace(), "test")
    assert len(registry.names) == len(TOOL_DEFINITIONS) == 28
    assert registry.names == {entry.name for entry in TOOL_DEFINITIONS}
    assert {tool.name for tool in default_enabled_tools(registry)} == registry.names
    assert {"find_files", "search_text", "git_changed_files"} <= registry.names
    assert MUTATING_TOOLS == {
        "write_file", "make_directory", "install_package", "execute_abap_report",
        "run_python", "run_comfy_workflow", "rag_refresh_index", "download_comfy_output_image",
    }


def test_registration_alone_does_not_enable_tool_or_broaden_controller_snapshot(monkeypatch):
    import tools.registry as module

    tool = SimpleNamespace(name="read_file")
    registry = ToolRegistry.from_tools([tool])
    capabilities = PlanningCapabilities(available_tools=tuple(tool.name for tool in default_enabled_tools(registry)))
    controller = CortexController(24, planning_capabilities=capabilities)
    value = ControllerInput(identity=ExecutionIdentity(execution_id="registry", protocol_version="1"),
                            cursor=ExecutionCursor(), context=ExecutionContext(user_request="Inspect workspace"))
    before = controller.decide(value).planning_request.capabilities
    extra = SimpleNamespace(name="new_registered_tool")
    extended = ToolRegistry.from_tools([tool, extra])
    definition = ToolDefinition(extra.name, "workspace")
    monkeypatch.setattr(module, "TOOL_DEFINITIONS", (*TOOL_DEFINITIONS, definition))
    assert extra.name in extended.names
    assert {tool.name for tool in default_enabled_tools(extended)} == {"read_file"}
    assert controller.decide(value).planning_request.capabilities == before == capabilities
    # Even explicitly enabling the deployment for future executions cannot change
    # the ceiling already supplied to this Controller.
    monkeypatch.setattr(module, "TOOL_DEFINITIONS", (*TOOL_DEFINITIONS, ToolDefinition(extra.name, "workspace", enabled_by_default=True)))
    assert extra.name in {tool.name for tool in default_enabled_tools(extended)}
    assert controller.decide(value).planning_request.capabilities == capabilities


def test_graph_validates_controller_names_and_preserves_authorized_projection(monkeypatch):
    import core.graph_nodes as module

    captured = {}
    monkeypatch.setattr(module, "create_controller_node", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(module, "create_planner_node", lambda **kwargs: None)
    monkeypatch.setattr(module, "create_brain_node", lambda **kwargs: None)
    monkeypatch.setattr(module, "create_capture_tool_output_node", lambda: None)
    arguments = dict(brain_llm=None, executable_tools=[SimpleNamespace(name="read_file"), SimpleNamespace(name="find_files")],
                     planner_llm=None, rag_service=None, rag_top_k=4, agent_system_prompt="test",
                     sap_system_prompt=None, tools_set={"read_file"}, show_raw_llm=False)
    module.create_graph_nodes(**arguments)
    assert captured["planning_capabilities"].available_tools == ("read_file",)
    with pytest.raises(ValueError, match="Unregistered"):
        module.create_graph_nodes(**{**arguments, "tools_set": {"missing"}})


def test_planner_projects_controller_authorization_not_registered_inventory():
    registry = ToolRegistry.from_tools([SimpleNamespace(name="read_file"), SimpleNamespace(name="find_files")])
    controller = CortexController(24, planning_capabilities=PlanningCapabilities(available_tools=("read_file",)))
    value = ControllerInput(identity=ExecutionIdentity(execution_id="registry-planner", protocol_version="1"),
                            cursor=ExecutionCursor(), context=ExecutionContext(user_request="Read a file"))
    decision = controller.decide(value)
    assert decision.decision_type == ControllerDecisionType.DISPATCH_PLANNER
    registry.resolve(decision.planning_request.capabilities.available_tools)
    provider = FakeProvider({"result": "PLAN_PROPOSED", "objective": "Read", "steps": [
        {"step_id": "read", "title": "Read", "description": "Read the file", "primary_tool": "read_file", "dependencies": []},
    ], "message": ""})
    planner = PlannerService(provider=provider, router=FakeRouter("info"), mutating_tools=MUTATING_TOOLS)
    result = planner.run(decision.planning_request)
    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    assert result.proposed_plan.available_tools == ("read_file",)
    assert "- read_file" in provider.messages[0][0].content
    assert "- find_files" not in provider.messages[0][0].content


def test_brain_binds_only_authorized_tools_from_registry():
    registry = ToolRegistry.from_tools([SimpleNamespace(name="read_file"), SimpleNamespace(name="find_files")])
    context = brain_input()
    context = context.model_copy(update={"active_plan": context.active_plan.model_copy(update={"available_tools": ("read_file",)})})
    model = FakeModel(AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "read"}]))
    provider = LangChainBrainProvider(brain_llm=model, executable_tools=registry.resolve(registry.names))
    assert provider.generate(context, ()).kind == BrainOutcomeKind.TOOL_REQUESTED
    assert {tool.name for tool in model.bound_tools if hasattr(tool, "name")} == {"read_file"}
    model.reply = AIMessage(content="", tool_calls=[{"name": "find_files", "args": {}, "id": "unauthorized"}])
    # Binding, rather than registration, is the boundary: the unused registered
    # executable never becomes an authorized native response.
    rejected = provider.generate(context, ())
    assert rejected.kind == BrainOutcomeKind.INVALID_OUTPUT
    assert rejected.error_code == "unknown_tool"
