from pathlib import Path

from langchain_core.messages import HumanMessage

from core.graph import _load_sap_system_prompt, build_app
from core.protocol.enums import ExecutionPhase, ExecutionStatus
from core.protocol.models import ExecutionCursor, ExecutionIdentity, ExecutionState, ProtocolVisibleState, WorkingState


class DummyTool:
    def __init__(self, name: str):
        self.name = name


class FakeChatModel:
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature

    def bind_tools(self, tools):
        return {
            "kind": "bound",
            "model": self.model,
            "temperature": self.temperature,
            "tools": list(tools),
        }


class InvokeOnlyNode:
    def __init__(self, handler):
        self._handler = handler

    def invoke(self, state):
        return self._handler(state)


def _sample_execution_state(execution_id: str = "run-1") -> ExecutionState:
    return ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=ExecutionIdentity(
                execution_id=execution_id,
                protocol_version="1.0",
            ),
            status=ExecutionStatus.NON_TERMINAL,
            cursor=ExecutionCursor(phase=ExecutionPhase.INITIALIZING),
        ),
        working=WorkingState(),
    )


def test_load_sap_system_prompt_returns_none_when_file_missing(tmp_path):
    assert _load_sap_system_prompt(tmp_path) is None


def test_build_app_uses_injected_factories(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "systemprompts_sap.md").write_text("custom sap prompt", encoding="utf-8")

    call_log: dict[str, object] = {}

    def rag_factory(knowledge_root: Path, embedding_model: str, rag_top_k: int):
        call_log["rag"] = (knowledge_root, embedding_model, rag_top_k)
        return {"kind": "rag-service", "knowledge_root": str(knowledge_root)}

    def tool_list_factory(workspace_root: str, knowledge_root: str, rag_service, model: str):
        call_log["tools"] = (workspace_root, knowledge_root, rag_service, model)
        return [DummyTool("list_files")]

    def chat_model_factory(model: str, temperature: float):
        return FakeChatModel(model, temperature)

    def graph_nodes_factory(**kwargs):
        call_log["graph_nodes_kwargs"] = kwargs

        def planner_node(_state):
            return {}

        def brain_node(_state):
            return {}

        def capture_tool_output_node(_state):
            return {}

        def route_after_brain(_state):
            return "__end__"

        def summarize_memory_node(_state):
            return {}

        return planner_node, brain_node, capture_tool_output_node, route_after_brain, summarize_memory_node

    def tool_node_factory(tools):
        call_log["tool_node_tools"] = list(tools)

        def _tool_node(_state):
            return {}

        return _tool_node

    app = build_app(
        workspace_dir=str(tmp_path / "workspace"),
        knowledge_dir=str(tmp_path / "knowledge"),
        model="test-model",
        model_planner="planner-model",
        embedding_model="embed-model",
        rag_top_k=7,
        rag_factory=rag_factory,
        tool_list_factory=tool_list_factory,
        chat_model_factory=chat_model_factory,
        graph_nodes_factory=graph_nodes_factory,
        tool_node_factory=tool_node_factory,
        project_root=tmp_path,
    )

    assert app is not None

    rag_call = call_log["rag"]
    assert rag_call[1] == "embed-model"
    assert rag_call[2] == 7

    graph_nodes_kwargs = call_log["graph_nodes_kwargs"]
    assert graph_nodes_kwargs["sap_system_prompt"] == "custom sap prompt"
    assert graph_nodes_kwargs["rag_top_k"] == 7
    assert graph_nodes_kwargs["tool_name_set"] == {"list_files"}

    llm = graph_nodes_kwargs["tool_brain_llm"]
    brain_llm = graph_nodes_kwargs["brain_llm"]
    planner_llm = graph_nodes_kwargs["planner_llm"]
    assert llm["kind"] == "bound"
    assert llm["model"] == "test-model"
    assert brain_llm.model == "test-model"
    assert planner_llm.model == "planner-model"

    tool_node_tools = call_log["tool_node_tools"]
    assert len(tool_node_tools) == 1
    assert tool_node_tools[0].name == "list_files"


def test_build_app_propagates_same_execution_state_across_graph_nodes(tmp_path):
    observed: dict[str, list[ExecutionState]] = {
        "planner": [],
        "brain": [],
        "tools": [],
        "capture": [],
        "summary": [],
    }

    def tool_list_factory(_workspace_root: str, _knowledge_root: str, _rag_service, _model: str):
        return [DummyTool("noop")]

    def chat_model_factory(model: str, temperature: float):
        return FakeChatModel(model, temperature)

    def graph_nodes_factory(**_kwargs):
        def planner_node(state):
            observed["planner"].append(state["execution_state"])
            return {"plan": "noop"}

        def brain_node(state):
            observed["brain"].append(state["execution_state"])
            return {"steps": state.get("steps", 0) + 1}

        def capture_tool_output_node(state):
            observed["capture"].append(state["execution_state"])
            return {"last_tool_signature": "capture"}

        def route_after_brain(_state):
            return "tools" if len(observed["brain"]) == 1 else "summarize_memory"

        def summarize_memory_node(state):
            observed["summary"].append(state["execution_state"])
            return {"rolling_summary": "done"}

        return planner_node, brain_node, capture_tool_output_node, route_after_brain, summarize_memory_node

    def tool_node_factory(_tools):
        def _tool_node(state):
            observed["tools"].append(state["execution_state"])
            return {"last_tool_output": "ok"}

        return _tool_node

    app = build_app(
        workspace_dir=str(tmp_path / "workspace"),
        knowledge_dir=str(tmp_path / "knowledge"),
        chat_model_factory=chat_model_factory,
        tool_list_factory=tool_list_factory,
        graph_nodes_factory=graph_nodes_factory,
        tool_node_factory=tool_node_factory,
        project_root=tmp_path,
    )

    execution_state = _sample_execution_state()
    result = app.invoke({
        "messages": [HumanMessage(content="start")],
        "steps": 0,
        "execution_state": execution_state,
    })

    assert result["execution_state"] is execution_state
    assert observed["planner"] == [execution_state]
    assert observed["brain"][0] is execution_state
    assert observed["brain"][1] is execution_state
    assert observed["tools"] == [execution_state]
    assert observed["capture"] == [execution_state]
    assert observed["summary"] == [execution_state]


def test_build_app_preserves_explicit_execution_state_replacement(tmp_path):
    observed: dict[str, list[ExecutionState]] = {
        "brain": [],
        "tools": [],
        "capture": [],
        "summary": [],
    }
    original_execution_state = _sample_execution_state("run-1")
    replacement_execution_state = _sample_execution_state("run-2")

    def tool_list_factory(_workspace_root: str, _knowledge_root: str, _rag_service, _model: str):
        return [DummyTool("noop")]

    def chat_model_factory(model: str, temperature: float):
        return FakeChatModel(model, temperature)

    def graph_nodes_factory(**_kwargs):
        def planner_node(_state):
            return {"plan": "noop"}

        def brain_node(state):
            observed["brain"].append(state["execution_state"])
            if len(observed["brain"]) == 1:
                return {"execution_state": replacement_execution_state, "steps": 1}
            return {"steps": state.get("steps", 0) + 1}

        def capture_tool_output_node(state):
            observed["capture"].append(state["execution_state"])
            return {"last_tool_signature": "capture"}

        def route_after_brain(_state):
            return "tools" if len(observed["brain"]) == 1 else "summarize_memory"

        def summarize_memory_node(state):
            observed["summary"].append(state["execution_state"])
            return {"rolling_summary": "done"}

        return planner_node, brain_node, capture_tool_output_node, route_after_brain, summarize_memory_node

    def tool_node_factory(_tools):
        def _tool_node(state):
            observed["tools"].append(state["execution_state"])
            return {"last_tool_output": "ok"}

        return _tool_node

    app = build_app(
        workspace_dir=str(tmp_path / "workspace"),
        knowledge_dir=str(tmp_path / "knowledge"),
        chat_model_factory=chat_model_factory,
        tool_list_factory=tool_list_factory,
        graph_nodes_factory=graph_nodes_factory,
        tool_node_factory=tool_node_factory,
        project_root=tmp_path,
    )

    result = app.invoke({
        "messages": [HumanMessage(content="start")],
        "steps": 0,
        "execution_state": original_execution_state,
    })

    assert observed["brain"][0] is original_execution_state
    assert observed["brain"][1] is replacement_execution_state
    assert observed["tools"] == [replacement_execution_state]
    assert observed["capture"] == [replacement_execution_state]
    assert observed["summary"] == [replacement_execution_state]
    assert result["execution_state"] is replacement_execution_state


def test_build_app_supports_invoke_only_tool_nodes(tmp_path):
    observed: dict[str, list[ExecutionState]] = {
        "tools": [],
        "capture": [],
    }
    execution_state = _sample_execution_state()

    def tool_list_factory(_workspace_root: str, _knowledge_root: str, _rag_service, _model: str):
        return [DummyTool("noop")]

    def chat_model_factory(model: str, temperature: float):
        return FakeChatModel(model, temperature)

    def graph_nodes_factory(**_kwargs):
        def planner_node(_state):
            return {"plan": "noop"}

        def brain_node(state):
            return {"steps": state.get("steps", 0) + 1}

        def capture_tool_output_node(state):
            observed["capture"].append(state["execution_state"])
            return {"last_tool_signature": "capture"}

        def route_after_brain(_state):
            return "tools" if not observed["tools"] else "summarize_memory"

        def summarize_memory_node(_state):
            return {"rolling_summary": "done"}

        return planner_node, brain_node, capture_tool_output_node, route_after_brain, summarize_memory_node

    def tool_node_factory(_tools):
        return InvokeOnlyNode(lambda state: observed["tools"].append(state["execution_state"]) or {"last_tool_output": "ok"})

    app = build_app(
        workspace_dir=str(tmp_path / "workspace"),
        knowledge_dir=str(tmp_path / "knowledge"),
        chat_model_factory=chat_model_factory,
        tool_list_factory=tool_list_factory,
        graph_nodes_factory=graph_nodes_factory,
        tool_node_factory=tool_node_factory,
        project_root=tmp_path,
    )

    result = app.invoke({
        "messages": [HumanMessage(content="start")],
        "steps": 0,
        "execution_state": execution_state,
    })

    assert observed["tools"] == [execution_state]
    assert observed["capture"] == [execution_state]
    assert result["execution_state"] is execution_state
