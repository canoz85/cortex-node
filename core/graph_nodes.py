import uuid

from langchain_ollama import ChatOllama

from core.graph_brain import create_brain_node
from core.graph_capture import create_capture_tool_output_node
from core.graph_constants import ANSI_BLUE, ANSI_ITALIC, ANSI_RED, ANSI_GREEN, ANSI_YELLOW, ANSI_RESET, MAX_PSEUDO_RETRIES, RECENT_MESSAGE_WINDOW

from core.graph_planner import create_planner_node
from core.graph_summarize import create_summarize_memory_node
from core.graph_routing import route_after_brain

from core.rag import WorkspaceRAG


# def _build_pre_messages(
#     *,
#     active_system_prompt: str,
#     retrieval_messages: list[SystemMessage],
#     rolling_summary: str,
#     route: str,
#     preferred_required_tool_name: str | None,
#     preferred_required_tool_pending: bool,
#     planner_plan_source: str,
#     planner_plan: str,
#     planner_domain: str,
#     planner_domain_enforced: bool,
#     planner_confidence: float,
#     file_generation_requested: bool,
#     successful_tool_result_in_turn: bool,
#     current_filegen_issue: str | None,
#     read_only_file_request: bool,
#     action_required: bool,
#     state: AgentState,
# ) -> tuple[list[SystemMessage], AIMessage | None]:
#     pre_messages = [
#         SystemMessage(content=active_system_prompt),
#         *retrieval_messages,
#         *rolling_summary_message(rolling_summary),
#     ]

#     if route.startswith("action"):
#         planner_brief = planner_execution_brief(route, planner_plan_source, planner_plan)
#         if planner_brief:
#             pre_messages.append(SystemMessage(content=planner_brief))

#     if route == "action:sap" or planner_domain == "sap":
#         sap_enforcement_suffix = (
#             "This domain is explicitly enforced by the user."
#             if planner_domain_enforced
#             else f"Planner confidence for SAP domain: {planner_confidence:.2f}."
#         )
#         pre_messages.append(
#             SystemMessage(
#                 content=(
#                     "SAP domain execution rules: Prefer SAP tools first "
#                     "(`query_abap_table`, `execute_abap_report`, `lookup_material`, `get_report_data`). "
#                     "Do not use Python file-generation or runtime tools unless the user explicitly requests Python code generation. "
#                     f"{sap_enforcement_suffix}"
#                 )
#             )
#         )
#         pre_messages.append(
#             SystemMessage(
#                 content=(
#                     "SAP ABAP correctness checklist before finalizing any code: "
#                     "(1) Do not use SELECT *. Select only required fields. "
#                     "(2) For PO vendor filtering, use EKKO-LIFNR via join EKPO<->EKKO on EBELN; do not use EKPO-LIFNR. "
#                     "(3) Use modern Open SQL host variables with @ for select-options and constants. "
#                     "(4) Avoid obsolete ABAP patterns such as OCCURS/header lines/implicit work areas. "
#                     "(5) For open PO logic, enforce MENGE > WEMNG, LOEKZ = space, ELIKZ = space, and compute remaining quantity in SQL when possible. "
#                     "If any checklist item is not satisfied, correct the solution before responding."
#                 )
#             )
#         )

#     if preferred_required_tool_name and preferred_required_tool_pending:
#         pre_messages.append(
#             SystemMessage(
#                 content=(
#                     "This request must be answered by calling the "
#                     f"{preferred_required_tool_name} tool first. "
#                     "Do not answer from memory. Use the tool and then respond from its output."
#                 )
#             )
#         )

#     if file_generation_requested and not successful_tool_result_in_turn:
#         pre_messages.append(_file_generation_initial_guidance())
#     if file_generation_requested and current_filegen_issue:
#         pre_messages.append(_file_generation_gap_rewrite_guidance(str(current_filegen_issue)))

#     if read_only_file_request:
#         pre_messages.append(_read_only_analysis_guidance())

#     missing_module = detect_missing_dependency(state.get("last_tool_output", ""))
#     if missing_module:
#         return pre_messages, _missing_dependency_response(missing_module)

#     if file_generation_requested and state.get("last_tool_success") is False and last_tool_missing_required_args(state.get("last_tool_output", "")):
#         pre_messages.append(_missing_required_args_guidance())
#     if route in {"coding_discussion", "conversation"}:
#         pre_messages.append(
#             SystemMessage(
#                 content=(
#                     "This turn is a discussion request, not an implementation request. "
#                     "Answer directly and do not create files, run tools, or execute code unless the user explicitly asks for concrete actions."
#                 )
#             )
#         )
#     if state.get("last_tool_success") is False and state.get("last_tool_signature"):
#         signature = state.get("last_tool_signature", "")
#         pre_messages.append(_failed_signature_advisory(signature))
#         stderr = last_tool_stderr(state.get("last_tool_output", ""))
#         if stderr:
#             pre_messages.append(_stderr_repair_guidance(stderr))
#         if last_tool_has_args_nameerror(state.get("last_tool_output", "")):
#             pre_messages.append(_args_scope_repair_guidance())
#     elif action_required and state.get("last_tool_success") is True and state.get("last_tool_signature"):
#         signature = state.get("last_tool_signature", "")
#         pre_messages.append(_successful_signature_advisory(signature))
#     elif action_required and successful_tool_result_in_turn and not (file_generation_requested and current_filegen_issue):
#         pre_messages.append(
#             SystemMessage(
#                 content=(
#                     "A tool has already succeeded during this user turn. "
#                     "If that successful result satisfies the request, provide the final concise answer now "
#                     "instead of calling more tools. Only call another tool if a specific remaining gap still exists."
#                 )
#             )
#         )
#     elif file_generation_requested and current_filegen_issue:
#         pre_messages.append(_file_generation_still_incomplete_guidance(str(current_filegen_issue)))

#     return pre_messages, None

def create_graph_nodes(
    *,
    brain_llm: ChatOllama,
    tool_brain_llm: ChatOllama,
    planner_llm: ChatOllama,
    rag_service: WorkspaceRAG,
    rag_top_k: int,
    agent_system_prompt: str,
    casual_system_prompt: str,
    sap_system_prompt: str | None,
    tool_name_set: set[str],
):
    planner_node = create_planner_node(
        planner_llm=planner_llm,
        router_llm=planner_llm,
        rag_service=rag_service,
        rag_top_k=rag_top_k,
        tool_name_set=tool_name_set,
    )
    capture_tool_output_node = create_capture_tool_output_node()

    summarize_memory_node = create_summarize_memory_node(summarize_llm=planner_llm,)

    brain_node = create_brain_node(
        brain_llm=brain_llm,
        tool_brain_llm=tool_brain_llm,
        agent_system_prompt=agent_system_prompt,
        casual_system_prompt=casual_system_prompt,
        tool_name_set=tool_name_set,
    )

    return planner_node, brain_node, capture_tool_output_node, route_after_brain, summarize_memory_node
