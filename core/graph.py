from pathlib import Path

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from core.graph_constants import MAX_REASONING_STEPS, SYSTEM_PROMPT_TEMPLATE
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


def _load_sap_system_prompt(project_root: Path) -> str | None:
    """Load SAP-specific system prompt from prompts folder if present."""
    sap_prompt_path = project_root / "prompts" / "systemprompts_sap.md"
    if not sap_prompt_path.exists():
        return None
    content = sap_prompt_path.read_text(encoding="utf-8").strip()
    return content or None


def build_app(
    workspace_dir: str = "workspace",
    model: str = "qwen2.5-coder:14b",
    knowledge_dir: str = "knowledge",
    embedding_model: str = "nomic-embed-text",
    rag_top_k: int = 4,
) -> CompiledStateGraph:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = Path(workspace_dir).resolve()
    workspace_root_str = str(workspace_root)
    knowledge_root = Path(knowledge_dir).resolve()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        model=model,
        workspace_dir=workspace_root_str,
        knowledge_dir=str(knowledge_root),
        max_steps=MAX_REASONING_STEPS,
    )

    rag_service = WorkspaceRAG(
        knowledge_root,
        embed_model=embedding_model,
        top_k=rag_top_k,
    )
    sap_system_prompt = _load_sap_system_prompt(project_root)

    tools = [
        *get_file_tools(workspace_root_str, knowledge_dir=str(knowledge_root)),
        *get_exec_tools(workspace_root_str),
        *get_git_tools(workspace_root_str),
        *get_info_tools(model=model, workspace_dir=workspace_root_str),
        *get_rag_tools(rag_service),
        *get_sap_tools(workspace_root_str),
        *get_scada_tools(workspace_root_str),
    ]
    tool_name_set = {getattr(tool, "name", "") for tool in tools}

    llm = ChatOllama(model=model, temperature=0).bind_tools(tools)
    planner_llm = ChatOllama(model=model, temperature=0)

    planner_node, brain_node, capture_tool_output_node, route_after_brain = create_graph_nodes(
        llm=llm,
        planner_llm=planner_llm,
        rag_service=rag_service,
        rag_top_k=rag_top_k,
        system_prompt=system_prompt,
        sap_system_prompt=sap_system_prompt,
        tool_name_set=tool_name_set,
    )

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("brain", brain_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("capture_tool_output", capture_tool_output_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "brain")
    workflow.add_conditional_edges("brain", route_after_brain)
    workflow.add_edge("tools", "capture_tool_output")
    workflow.add_edge("capture_tool_output", "brain")

    return workflow.compile()
