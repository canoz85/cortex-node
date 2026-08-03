import uuid
from typing import Set
from langchain_ollama import ChatOllama

from core.graph_brain import create_brain_node
from core.graph_capture import create_capture_tool_output_node
from core.graph_constants import ANSI_BLUE, ANSI_ITALIC, ANSI_RED, ANSI_GREEN, ANSI_YELLOW, ANSI_RESET, MAX_PSEUDO_RETRIES, RECENT_MESSAGE_WINDOW

from core.graph_controller import create_controller_node
from core.graph_planner import create_planner_node
from core.graph_summarize import create_summarize_memory_node

from core.rag import WorkspaceRAG

def create_graph_nodes(
    *,
    brain_llm: ChatOllama,
    tool_brain_llm: ChatOllama,
    planner_llm: ChatOllama,
    rag_service: WorkspaceRAG,
    rag_top_k: int,
    agent_system_prompt: str,
    final_answer_system_prompt: str,
    casual_system_prompt: str,
    sap_system_prompt: str | None,
    tools_set: Set[str],
    show_raw_llm: bool,
):

    controller_node = create_controller_node()
    
    planner_node = create_planner_node(
        planner_llm=planner_llm,
        router_llm=planner_llm,
        rag_service=rag_service,
        rag_top_k=rag_top_k,
        tools_set=tools_set,
    )
    capture_tool_output_node = create_capture_tool_output_node()

    summarize_memory_node = create_summarize_memory_node(summarize_llm=planner_llm,)

    brain_node = create_brain_node(
        brain_llm=brain_llm,
        tool_brain_llm=tool_brain_llm,
        agent_system_prompt=agent_system_prompt,
        final_answer_system_prompt=final_answer_system_prompt,
        casual_system_prompt=casual_system_prompt,
        tools_set=tools_set,
        show_raw_llm=show_raw_llm,
    )



    return controller_node, planner_node, brain_node, capture_tool_output_node, summarize_memory_node
