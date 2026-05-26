from pathlib import Path

from core.graph import _load_sap_system_prompt, build_app


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

        return planner_node, brain_node, capture_tool_output_node, route_after_brain

    def tool_node_factory(tools):
        call_log["tool_node_tools"] = list(tools)

        def _tool_node(_state):
            return {}

        return _tool_node

    app = build_app(
        workspace_dir=str(tmp_path / "workspace"),
        knowledge_dir=str(tmp_path / "knowledge"),
        model="test-model",
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

    llm = graph_nodes_kwargs["llm"]
    planner_llm = graph_nodes_kwargs["planner_llm"]
    assert llm["kind"] == "bound"
    assert llm["model"] == "test-model"
    assert planner_llm.model == "test-model"

    tool_node_tools = call_log["tool_node_tools"]
    assert len(tool_node_tools) == 1
    assert tool_node_tools[0].name == "list_files"
