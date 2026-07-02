from pathlib import Path
from typing import Any, Callable

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.graph.state import END, CompiledStateGraph
from langgraph.prebuilt import ToolNode

from core.graph_constants import CASUAL_SYSTEM_PROMPT_TEMPLATE, MAX_REASONING_STEPS, SYSTEM_PROMPT_TEMPLATE
from core.graph_nodes import create_graph_nodes
from core.graph_runner import run_prompt
from core.rag import WorkspaceRAG
from core.state import AgentState
from tools.exec_ops import get_exec_tools
from tools.file_ops import get_file_tools
from tools.git_ops import get_git_tools
from tools.info_ops import get_info_tools
from tools.rag_ops import get_rag_tools
from tools.sap_ops import get_sap_tools
from tools.scada_ops import get_scada_tools
from tools.vision_ops import get_vision_tools


ToolListFactory = Callable[[str, str, WorkspaceRAG, str], list[Any]]
ChatModelFactory = Callable[[str, float], Any]
RAGFactory = Callable[[Path, str, int], WorkspaceRAG]


def _default_rag_factory(knowledge_root: Path, embedding_model: str, rag_top_k: int) -> WorkspaceRAG:
    return WorkspaceRAG(
        knowledge_root,
        embed_model=embedding_model,
        top_k=rag_top_k,
    )


def _default_tool_list_factory(workspace_root: str, knowledge_root: str, rag_service: WorkspaceRAG, model: str) -> list[Any]:
    return [
        *get_file_tools(workspace_root, knowledge_dir=knowledge_root),
        *get_exec_tools(workspace_root),
        *get_git_tools(workspace_root),
        *get_info_tools(model=model, workspace_dir=workspace_root),
        *get_rag_tools(rag_service),
        *get_sap_tools(workspace_root),
        *get_scada_tools(workspace_root),
        *get_vision_tools(workspace_root),
    ]


def _default_chat_model_factory(model: str, temperature: float) -> Any:
    return ChatOllama(model=model, temperature=temperature)


def _load_sap_system_prompt(project_root: Path) -> str | None:
    """Load SAP-specific system prompt from prompts folder if present."""
    sap_prompt_path = project_root / "prompts" / "systemprompts_sap.md"
    if not sap_prompt_path.exists():
        return None
    content = sap_prompt_path.read_text(encoding="utf-8").strip()
    return content or None


def build_app(
    workspace_dir: str = "workspace",
    model: str = "gpt-oss:20b", #"qwen2.5-coder:14b",
    model_planner: str = "gpt-oss:20b",
    knowledge_dir: str = "knowledge",
    embedding_model: str = "nomic-embed-text",
    rag_top_k: int = 4,
    rag_factory: RAGFactory = _default_rag_factory,
    tool_list_factory: ToolListFactory = _default_tool_list_factory,
    chat_model_factory: ChatModelFactory = _default_chat_model_factory,
    graph_nodes_factory: Callable[..., tuple[Any, Any, Any, Any]] = create_graph_nodes,
    tool_node_factory: Callable[[list[Any]], Any] = ToolNode,
    project_root: Path | None = None,
    show_raw_llm: bool = False,
) -> CompiledStateGraph:
    app_root = project_root or Path(__file__).resolve().parents[1]
    workspace_root = Path(workspace_dir).resolve()
    workspace_root_str = str(workspace_root)
    knowledge_root = Path(knowledge_dir).resolve()

    rag_service = rag_factory(knowledge_root, embedding_model, rag_top_k)
    sap_system_prompt = _load_sap_system_prompt(app_root)

    tools = tool_list_factory(workspace_root_str, str(knowledge_root), rag_service, model)
    tool_name_set = {getattr(tool, "name", "") for tool in tools}

    # Format the available tools into a clean, scannable string block
    tools_list_str = "\n".join([f"- {name}" for name in sorted(tool_name_set) if name])

    agent_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        model=model,
        workspace_dir=workspace_root_str,
        knowledge_dir=str(knowledge_root),
        max_steps=MAX_REASONING_STEPS,
        available_tools=tools_list_str,
    )

    casual_system_prompt = CASUAL_SYSTEM_PROMPT_TEMPLATE

    planner_llm = chat_model_factory(model_planner, 0)
    brain_llm = chat_model_factory(model, 0)
    tool_brain_llm = chat_model_factory(model, 0).bind_tools(tools)

    planner_node, brain_node, capture_tool_output_node, route_after_brain, summarize_memory_node = graph_nodes_factory(
        brain_llm=brain_llm,
        tool_brain_llm=tool_brain_llm,
        planner_llm=planner_llm,
        rag_service=rag_service,
        rag_top_k=rag_top_k,
        agent_system_prompt=agent_system_prompt,
        casual_system_prompt=casual_system_prompt,
        sap_system_prompt=sap_system_prompt,
        tool_name_set=tool_name_set,
        show_raw_llm=show_raw_llm,
    )

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("brain", brain_node)
    workflow.add_node("tools", tool_node_factory(tools))
    workflow.add_node("capture_tool_output", capture_tool_output_node)
    workflow.add_node("summarize_memory", summarize_memory_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "brain")
    workflow.add_conditional_edges("brain", route_after_brain)
    workflow.add_edge("tools", "capture_tool_output")
    workflow.add_edge("capture_tool_output", "brain")
    workflow.add_edge("summarize_memory", END)

    return workflow.compile()
