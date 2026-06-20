import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from core.graph_brain import create_brain_node
from core.graph_capture import create_capture_tool_output_node
from core.graph_constants import ANSI_BLUE, ANSI_ITALIC, ANSI_RED, ANSI_GREEN, ANSI_YELLOW, ANSI_RESET, MAX_PSEUDO_RETRIES, RECENT_MESSAGE_WINDOW
from core.graph_filegen_policy import (
    last_failed_verification_rewrite_info,
    message_repeats_write_content,
)

from core.graph_messages import normalize_message_content
from core.graph_planner import create_planner_node
from core.graph_summarize import create_summarize_memory_node, rolling_summary_message
from core.graph_pseudo_tools import finalize_action_response, is_pseudo_tool_response, looks_like_pseudo_tool_text, recover_pseudo_tool_response
from core.graph_routing import route_after_brain

from core.rag import WorkspaceRAG
from core.state import AgentState


def _makes_workspace_analysis_claim(response: AIMessage) -> bool:
    response_text = normalize_message_content(response).lower()
    return "workspace" in response_text and (
        "reviewed" in response_text
        or "analyzed" in response_text
        or "analysed" in response_text
        or "read all" in response_text
        or "files in the workspace" in response_text
    )


def _file_generation_initial_guidance() -> SystemMessage:
    return SystemMessage(
        content=(
            "This is a concrete file-generation task inside the sandbox workspace. "
            "Your next response should start with executable tool calls only. "
            "Prefer write_file or make_directory for implementation, then run_python to verify when possible. "
            "Do not explain planned code before taking action. "
            "Python code structure rules: "
            "All business logic must be inside named functions. "
            "The entry point must be a main() function called from 'if __name__ == \"__main__\"'. "
            "Do NOT place logic or variable assignments at module level outside of functions."
        )
    )


def _file_generation_gap_rewrite_guidance(current_filegen_issue: str) -> SystemMessage:
    return SystemMessage(
        content=(
            "The generated file is not complete for this request yet. "
            f"Detected gap: {current_filegen_issue} "
            "You must call write_file now with a corrected, fully implemented version of the file that fixes this gap. "
            "Do NOT call run_python again until you have first rewritten the file with the missing implementation."
        )
    )


def _file_generation_still_incomplete_guidance(current_filegen_issue: str) -> SystemMessage:
    return SystemMessage(
        content=(
            "A tool succeeded, but the implementation is still incomplete. "
            f"Detected gap: {current_filegen_issue} "
            "Call write_file now with a corrected, fully implemented version. "
            "Do NOT call run_python again before rewriting the file."
        )
    )


def _file_generation_enforcement_prompt() -> str:
    return (
        "This file-generation request must continue with executable tool calls now. "
        "Return tool calls only. Start by creating or updating files in the sandbox, then verify the result."
    )


def _file_generation_incomplete_enforcement_prompt(current_filegen_issue: str) -> str:
    return (
        "This file-generation request is still incomplete. "
        f"Detected gap: {current_filegen_issue}. "
        "Call write_file now with a corrected, fully implemented version of the file. "
        "Do NOT call run_python again before rewriting the file."
    )


def _required_tool_enforcement_prompt(tool_name: str, kind: str) -> str:
    return (
        f"You ignored the required {kind} tool. "
        f"Call {tool_name} now. "
        "Do not answer from memory or provide a prose-only response."
    )


def _action_required_enforcement_prompt() -> str:
    return (
        "The user requested concrete actions. "
        "Return at least one executable tool call now. "
        "Do not return a prose-only response."
    )


def _pseudo_tool_retry_prompt() -> SystemMessage:
    return SystemMessage(
        content=(
            "Your previous response included pseudo tool invocation text. "
            "Do not output code blocks that look like tool calls. "
            "If actions are needed, emit real tool calls only. "
            "If no action is needed, provide only the final concise answer."
        )
    )


def _discussion_tool_call_correction_prompt() -> SystemMessage:
    return SystemMessage(
        content=(
            "You started taking actions for a discussion-only request. "
            "Do not call tools. Provide a concise direct answer only."
        )
    )

def _read_only_analysis_guidance() -> SystemMessage:
    return SystemMessage(
        content=(
            "This is a read-only file analysis request. "
            "Allowed tools: list_files/read_file only (and knowledge retrieval helpers). "
            "Never call write_file, make_directory, or run_python for this request. "
            "If a file was already read successfully this turn, provide the final explanation now instead of additional tool calls."
        )
    )

def _missing_required_args_guidance() -> SystemMessage:
    return SystemMessage(
        content=(
            "The CLI verification failed because required command-line arguments were missing. "
            "Do not rerun the same bare command. Instead, create a minimal sample input file in the workspace if needed, "
            "then call run_python again with the required argument values using v__args."
        )
    )

def _stderr_repair_guidance(stderr: str) -> SystemMessage:
    return SystemMessage(
        content=(
            "Latest Python/tool error to fix before re-verification:\n"
            f"{stderr[:1200]}"
        )
    )

def _args_scope_repair_guidance() -> SystemMessage:
    return SystemMessage(
        content=(
            "The failure indicates args scope is broken. "
            "Repair the script so parse_args() result is defined and used inside main(), "
            "and avoid referencing args at module top level. "
            "Write a corrected file version before running run_python again."
        )
    )

def _read_only_guard_correction_prompt() -> SystemMessage:
    return SystemMessage(
        content=(
            "Read-only request guard: the previous response attempted a mutating or execution tool call. "
            "Do not modify files. "
            "Use list_files/read_file only, or provide the final analysis answer now if sufficient context is already available."
        )
    )

def _unchanged_write_retry_prompt() -> SystemMessage:
    return SystemMessage(
        content=(
            "Your proposed write_file call rewrites the file with identical content after a failed verification. "
            "That is a no-op and will loop. "
            "Produce a corrected write_file call that changes the file to address the failing error, then verify again. "
            "Do not emit an unchanged write_file call."
        )
    )

def _apply_failed_rewrite_guard(
    *,
    llm: ChatOllama,
    pre_messages: list,
    recent_history: list,
    history: list,
    response: AIMessage,
    state: AgentState,
) -> AIMessage:
    failed_rewrite_context = last_failed_verification_rewrite_info(history, state)
    if not (failed_rewrite_context and getattr(response, "tool_calls", None)):
        return response

    failed_path, failed_write_content = failed_rewrite_context
    if not (failed_write_content and message_repeats_write_content(response, failed_path, failed_write_content)):
        return response

    last_tool_output = state.get("last_tool_output", "")
    failure_details = ""
    if isinstance(last_tool_output, dict):
        data = last_tool_output.get("data")
        if isinstance(data, dict):
            failure_details = str(data.get("stderr", "") or data.get("stdout", "") or "")

    return llm.invoke(
        [
            *pre_messages,
            *recent_history,
            response,
            SystemMessage(
                content=(
                    "You rewrote the same file content that already failed verification. "
                    f"File: {failed_path}. "
                    "Do not write the same content again. Produce a corrected write_file call that changes the file to address this failure before any further verification. "
                    f"Failure details: {failure_details}"
                )
            ),
        ]
    )


def _apply_action_enforcement(
    *,
    llm: ChatOllama,
    pre_messages: list,
    recent_history: list,
    response: AIMessage,
    tool_name_set: set[str],
    file_generation_requested: bool,
    successful_tool_result_in_turn: bool,
    current_filegen_issue: str | None,
    preferred_required_tool_name: str | None,
    preferred_required_tool_kind: str | None,
    preferred_required_tool_pending: bool,
    action_required: bool,
    skip_action_enforcement: bool,
) -> AIMessage:
    if getattr(response, "tool_calls", None):
        return response

    enforcement_prompt = ""
    if file_generation_requested and not successful_tool_result_in_turn:
        enforcement_prompt = _file_generation_enforcement_prompt()
    elif file_generation_requested and current_filegen_issue:
        enforcement_prompt = _file_generation_incomplete_enforcement_prompt(str(current_filegen_issue))
    elif preferred_required_tool_name and preferred_required_tool_pending and preferred_required_tool_kind:
        enforcement_prompt = _required_tool_enforcement_prompt(preferred_required_tool_name, preferred_required_tool_kind)
    elif action_required and not skip_action_enforcement and not successful_tool_result_in_turn:
        enforcement_prompt = _action_required_enforcement_prompt()

    if not enforcement_prompt:
        return response

    response = llm.invoke(
        [
            *pre_messages,
            *recent_history,
            response,
            SystemMessage(content=enforcement_prompt),
        ]
    )
    return finalize_action_response(response, tool_name_set)


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
        rag_service=rag_service,
        rag_top_k=rag_top_k,
        agent_system_prompt=agent_system_prompt,
        casual_system_prompt=casual_system_prompt,
        tool_name_set=tool_name_set,
    )

    return planner_node, brain_node, capture_tool_output_node, route_after_brain, summarize_memory_node
