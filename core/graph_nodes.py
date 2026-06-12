import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from core import state
from core.graph_capture import create_capture_tool_output_node
from core.graph_constants import ANSI_BLUE, ANSI_ITALIC, ANSI_RED, ANSI_GREEN, ANSI_YELLOW, ANSI_RESET, MAX_PSEUDO_RETRIES, RECENT_MESSAGE_WINDOW
from core.graph_context import retrieval_message, rolling_summary_message
from core.graph_filegen_policy import (
    file_generation_quality_issue,
    file_generation_verification_failures,
    last_failed_verification_rewrite_info,
    last_read_file_snapshot,
    last_tool_has_args_nameerror,
    last_tool_missing_required_args,
    last_tool_stderr,
    message_repeats_write_content,
    next_args_scope_autofix_call,
    next_file_generation_repair_call,
    next_file_generation_verification_call,
    response_has_unchanged_write,
    should_finalize_action_turn,
)
from core.graph_intents import (
    preferred_file_tool,
    preferred_info_tool,
)
from core.graph_messages import current_turn_messages, is_effectively_empty_response, latest_human_message, latest_human_message_str, normalize_message_content, recent_messages
from core.graph_node_helpers import (
    detect_missing_dependency,
    direct_discussion_response,
    planner_execution_brief,
    response_with_usage,
)
from core.graph_planner import create_planner_node
from core.graph_pseudo_tools import finalize_action_response, is_pseudo_tool_response, looks_like_pseudo_tool_text, recover_pseudo_tool_response
from core.graph_response_formatters import format_action_completion_response, format_preferred_tool_response
from core.graph_routing import route_after_brain
from core.graph_tool_events import (
    current_turn_tool_events,
    message_repeats_signature,
    parse_tool_signature,
)
from core.rag import WorkspaceRAG
from core.state import AgentState


READ_ONLY_TOOL_NAMES = {"list_files", "read_file", "read_knowledge_file", "rag_search", "rag_refresh_index"}


def _latest_tool_context(history: list[BaseMessage]) -> list[BaseMessage]:
    """Return latest AI tool-call + ToolMessage pair for grounding when needed."""
    if not history:
        return []

    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if not isinstance(message, ToolMessage):
            continue

        context: list[BaseMessage] = []
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            for prior in range(index - 1, -1, -1):
                candidate = history[prior]
                if not isinstance(candidate, AIMessage):
                    continue
                tool_calls = getattr(candidate, "tool_calls", None) or []
                if any(call.get("id") == tool_call_id for call in tool_calls):
                    context.append(candidate)
                    break
        context.append(message)
        return context
    return []


def _ensure_tool_message_visible(history: list[BaseMessage], recent_history: list[BaseMessage]) -> list[BaseMessage]:
    """Ensure brain prompt always carries latest tool result context."""
    if any(isinstance(message, ToolMessage) for message in recent_history):
        return recent_history

    stitched = list(recent_history)
    for message in _latest_tool_context(history):
        if message not in stitched:
            stitched.append(message)
    return stitched


def _successful_preferred_tool_fast_path(state: AgentState, latest_user_prompt: str) -> AIMessage | None:
    """Finalize simple preferred-tool requests without another tool-enabled LLM pass."""
    if state.get("last_tool_success") is not True:
        return None

    last_tool_output = state.get("last_tool_output", "")
    if not isinstance(last_tool_output, dict):
        return None

    preferred_tool_name = preferred_file_tool(latest_user_prompt) or preferred_info_tool(latest_user_prompt)
    if not preferred_tool_name:
        return None

    signature = parse_tool_signature(str(state.get("last_tool_signature", "") or ""))
    if not signature:
        return None

    tool_name, _ = signature
    if tool_name != preferred_tool_name:
        return None

    rendered = str(state.get("last_tool_rendered", "") or "").strip()
    if not rendered:
        rendered = format_preferred_tool_response(last_tool_output).strip()
    if not rendered:
        return None

    return AIMessage(content=rendered)


def _required_first_tool_response(required_first_tool: str, latest_user_prompt: str = "") -> AIMessage:
    if required_first_tool == "list_files":
        tool_args = {"path": "."}
    elif required_first_tool == "solve_math":
        tool_args = {"question": latest_user_prompt}
    else:
        tool_args = {}
    return AIMessage(
        content=f"Calling {required_first_tool} to answer your request.",
        tool_calls=[
            {
                "name": required_first_tool,
                "args": tool_args,
                "id": f"guard-required-tool-{uuid.uuid4().hex}",
                "type": "tool_call",
            }
        ],
    )


def _disallowed_read_only_tool_calls(response: AIMessage) -> list[dict]:
    return [call for call in getattr(response, "tool_calls", None) or [] if call.get("name") not in READ_ONLY_TOOL_NAMES]


def _read_only_guard_fallback_response() -> AIMessage:
    return AIMessage(
        content="Read-only request guard forced a safe file listing before further analysis.",
        tool_calls=[
            {
                "name": "list_files",
                "args": {"path": "."},
                "id": f"guard-readonly-{uuid.uuid4().hex}",
                "type": "tool_call",
            }
        ],
    )


def _makes_workspace_analysis_claim(response: AIMessage) -> bool:
    response_text = normalize_message_content(response).lower()
    return "workspace" in response_text and (
        "reviewed" in response_text
        or "analyzed" in response_text
        or "analysed" in response_text
        or "read all" in response_text
        or "files in the workspace" in response_text
    )


def _has_successful_file_events(tool_events: list[dict]) -> bool:
    return any(event.get("success") and event.get("name") in {"list_files", "read_file"} for event in tool_events)


def _workspace_claim_guard_response() -> AIMessage:
    return AIMessage(
        content="Listing workspace files first to avoid fabricated file-analysis claims.",
        tool_calls=[
            {
                "name": "list_files",
                "args": {"path": "."},
                "id": f"guard-{uuid.uuid4().hex}",
                "type": "tool_call",
            }
        ],
    )


def _missing_dependency_response(missing_module: str) -> AIMessage:
    return AIMessage(
        content=(
            f"The code failed with a missing dependency: '{missing_module}'. "
            f"Please install it using:\n\n"
            f"pip install {missing_module}\n\n"
            f"After installation, you can retry the code execution."
        )
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


def _pseudo_tool_fallback_response() -> AIMessage:
    return AIMessage(
        content=(
            "I produced pseudo tool-call text instead of executable tool calls, so no action was taken. "
            "Please retry with a task phrased as file changes inside the sandbox workspace."
        )
    )


def _empty_response_retry_prompt() -> SystemMessage:
    return SystemMessage(
        content=(
            "Your previous response was empty. "
            "Respond again with one of the following: "
            "(1) concrete tool calls to progress the task, or "
            "(2) a concise final answer."
        )
    )


def _empty_response_fallback() -> AIMessage:
    return AIMessage(
        content=(
            "I could not produce a valid action or answer from the model. "
            "Please retry the prompt or check model availability in Ollama."
        )
    )


def _discussion_tool_call_correction_prompt() -> SystemMessage:
    return SystemMessage(
        content=(
            "You started taking actions for a discussion-only request. "
            "Do not call tools. Provide a concise direct answer only."
        )
    )


def _repeated_signature_correction_prompt(repeated_signature: str, repeat_reason: str) -> SystemMessage:
    return SystemMessage(
        content=(
            "You repeated the exact same tool call again. "
            f"Repeated signature: {repeated_signature}. "
            f"That signature {repeat_reason}. "
            "Do not emit that same tool call again. "
            "Choose the next distinct step now, such as fixing the file, reading the error output, creating sample input, verifying with different arguments, or giving the final answer if the task is complete."
        )
    )


def _repeated_success_final_answer_prompt(repeated_signature: str) -> SystemMessage:
    return SystemMessage(
        content=(
            "The repeated tool call already succeeded earlier in this turn and must not be emitted again. "
            f"Repeated signature: {repeated_signature}. "
            "Do not call any tools now. "
            "Return a concise final answer using the tool outputs already in context."
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


def _apply_workspace_claim_guard(*, history: list, response: AIMessage, tool_name_set: set[str]) -> AIMessage:
    if getattr(response, "tool_calls", None):
        return response

    if not _makes_workspace_analysis_claim(response):
        return response

    tool_events = current_turn_tool_events(history)
    if _has_successful_file_events(tool_events):
        return response

    if "list_files" not in tool_name_set:
        return response

    return _workspace_claim_guard_response()

def _apply_repeated_signature_guard(
    *,
    llm: ChatOllama,
    pre_messages: list,
    response: AIMessage,
    action_required: bool,
    state: AgentState,
) -> AIMessage:
    
    last_tool_signature = str(state.get("last_tool_signature", ""))
    if not (action_required and last_tool_signature):
        return response

    if not message_repeats_signature(response, last_tool_signature):
        return response

    repeat_reason = "already succeeded" if state.get("last_tool_success") is True else "already failed"
    corrected_response = llm.invoke(
        [
            *pre_messages,
            response,
            _repeated_signature_correction_prompt(last_tool_signature, repeat_reason),
        ]
    )

    _print_raw_llm_request_response(color=ANSI_YELLOW, messages=[*pre_messages, 
        response,_repeated_signature_correction_prompt(last_tool_signature, repeat_reason)], raw_text=response.content)


    if state.get("last_tool_success") is True and message_repeats_signature(corrected_response, last_tool_signature):
        final_response = llm.invoke(
            [
                *pre_messages,
                corrected_response,
                _repeated_success_final_answer_prompt(last_tool_signature),
            ]
        )

        _print_raw_llm_request_response(color=ANSI_GREEN, messages=[*pre_messages,
             corrected_response, _repeated_success_final_answer_prompt(last_tool_signature)], 
                                         raw_text=response.content)


        if getattr(final_response, "tool_calls", None):
            last_tool_output = state.get("last_tool_output", "")
            if isinstance(last_tool_output, dict):
                return AIMessage(content=format_preferred_tool_response(last_tool_output))
            return AIMessage(
                content=(
                    "I already ran this tool call successfully and will not repeat it in this turn. "
                    "Using the existing tool output, I can continue with a direct answer next."
                )
            )
        return final_response

    return corrected_response


def _apply_response_recovery(
    *,
    response_llm: ChatOllama,
    pre_messages: list,
    response: AIMessage,
    action_required: bool,
    tool_name_set: set[str],
) -> AIMessage:
  
    # Always sanitize pseudo tool text on action-required turns so raw JSON/call text
    # is never emitted as a final brain answer.
    if (action_required) and is_pseudo_tool_response(response):
        response = recover_pseudo_tool_response(response, tool_name_set)
        if not getattr(response, "tool_calls", None):
            response = _pseudo_tool_fallback_response()

    if is_effectively_empty_response(response):
        response = response_llm.invoke(
            [
                    *pre_messages,
                    _empty_response_retry_prompt(),
            ]
        )
        _print_raw_llm_request_response(color=ANSI_GREEN, messages=[*pre_messages, _empty_response_retry_prompt()], raw_text=response.content)

    if is_effectively_empty_response(response):
        response = _empty_response_fallback()

    return response


def _finalize_from_successful_tool_context(
    *,
    llm: ChatOllama,
    pre_messages: list,
    state: AgentState,
    history: list,
    response: AIMessage,
    tool_name_set: set[str],
) -> AIMessage:
    """Ask the model to decide: final answer now or distinct next tool call.

    This prevents raw tool output from being echoed as the final brain response.
    """
    if getattr(response, "tool_calls", None):
        return response

    completion = format_action_completion_response(history)
    if completion:
        return AIMessage(content=completion)

    decision = llm.invoke(
        [
            *pre_messages,
            response,
            SystemMessage(
                content=(
                    "A tool already succeeded in this turn. Decide one of two outcomes only: "
                    "(1) If the user request is satisfied, provide the final concise answer now. "
                    "(2) If not satisfied, emit exactly one executable next tool call with distinct arguments. "
                    "Do not repeat the same successful tool call signature. "
                    "Do not restate raw tool output verbatim unless it is the final user-facing answer."
                )
            ),
        ]
    )

    _print_raw_llm_request_response(color=ANSI_RED, messages=[*pre_messages, response, 
            SystemMessage(
                content=(
                    "A tool already succeeded in this turn. Decide one of two outcomes only: "
                    "(1) If the user request is satisfied, provide the final concise answer now. "
                    "(2) If not satisfied, emit exactly one executable next tool call with distinct arguments. "
                    "Do not repeat the same successful tool call signature. "
                    "Do not restate raw tool output verbatim unless it is the final user-facing answer."
                ))], raw_text=response.content)


    decision = finalize_action_response(decision, tool_name_set)
    if getattr(decision, "tool_calls", None):
        return decision

    # Last-resort fallback keeps the flow deterministic when model output is unusable.
    if "Action-required run stopped" in normalize_message_content(decision):
        last_tool_output = state.get("last_tool_output", "")
        if isinstance(last_tool_output, dict):
            rendered = format_preferred_tool_response(last_tool_output)
            if rendered.strip():
                return AIMessage(content=rendered)

    return decision


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

def _build_pre_messages(
    *,
    active_system_prompt: str,
    retrieval_messages: list[SystemMessage],
    state: AgentState,
) -> tuple[list[SystemMessage], AIMessage | None]:

    route = str(state.get("planner_route", ""))

    last_tool_success = state.get("last_tool_success")
    last_tool_output = state.get("last_tool_output", "")
    last_tool_rendered = str(state.get("last_tool_rendered", "") or "")
    last_tool_signature = str(state.get("last_tool_signature", "") or "")

    history = state.get("messages", [])
    rolling_summary = state.get("rolling_summary", "")
    #recent_history = recent_messages(history, RECENT_MESSAGE_WINDOW)
    #execution_history = _ensure_tool_message_visible(history, recent_history)


    # 1. Core system context
    pre_messages = [
        SystemMessage(content=active_system_prompt),
        *retrieval_messages,
    ]

    # 2. Planner / execution context (optional)
    #if route.startswith("action") and state.get("plan") and state.get("last_tool_success") is None:
    if state.get("plan"):

        if route.startswith("action"):
            planner_brief = planner_execution_brief(
                route,
                state.get("plan", "")
            )
            if planner_brief:
                pre_messages.append(SystemMessage(content=planner_brief))
        
        state["plan"] = None  # only at END OF TURN

    pre_messages.append(SystemMessage(content=rolling_summary))

    if route.startswith("action"):
        pre_messages.append(current_turn_messages(history))
    else:
        pre_messages.append(latest_human_message(history))


    # 3. Runtime instruction buffer
    guidance = []

    # ------------------------------------------------------------
    # A) TOOL FAILURE HANDLING
    # ------------------------------------------------------------
    if last_tool_success is False:
        guidance.append("The previous tool execution FAILED.")

        if last_tool_signature:
            guidance.append(f"Failed tool signature: {last_tool_signature}")

        stderr = last_tool_stderr(last_tool_output)
        if stderr:
            guidance.append(f"Runtime error detected:\n{stderr}")

        if last_tool_missing_required_args(last_tool_output):
            guidance.append("Fix missing required tool arguments based on tool schema.")

        if last_tool_has_args_nameerror(last_tool_output):
            guidance.append("Fix NameError: ensure all variables/functions are defined or imported.")

    # ------------------------------------------------------------
    # B) TOOL SUCCESS HANDLING (IMPORTANT PART)
    # ------------------------------------------------------------
    if last_tool_success is True and last_tool_rendered.strip():

        guidance.append(
            "A tool has already succeeded during this user turn. "
            "Use the tool output as the authoritative source of truth for the current state of the world. "
            "If that successful result satisfies the request, provide the final concise answer now instead of calling more tools. "
            "Only call another tool if a specific remaining gap still exists that can be directly addressed by one more tool call."
        )
        
    # ------------------------------------------------------------
    # 4. FINAL SYSTEM INSTRUCTION BLOCK (single injection)
    # ------------------------------------------------------------
    if guidance:
        combined = "\n".join([
            "### RUNTIME INSTRUCTIONS (HIGHEST PRIORITY):",
            *guidance
        ])
        pre_messages.append(SystemMessage(content=combined))

    return pre_messages, None

def _apply_file_generation_fast_path(
    *,
    history: list,
    state: AgentState,
    route: str,
    file_generation_requested: bool,
) -> AIMessage | None:
    if not file_generation_requested:
        return None

    failed_verifications, latest_verification_error = file_generation_verification_failures(history)
    if (
        failed_verifications >= 2
        and not should_finalize_action_turn(history, route)
        and not last_tool_missing_required_args(latest_verification_error)
    ):
        return AIMessage(
            content=(
                "Action-required run stopped after repeated verification failures while generating the file. "
                f"Latest verification error: {latest_verification_error} "
                "I prevented further read/write/run looping. Please retry and I will apply a different repair strategy immediately."
            )
        )

    args_scope_fix_call = next_args_scope_autofix_call(history)
    if args_scope_fix_call is not None:
        return AIMessage(content="Applying deterministic args-scope repair before re-verification.", tool_calls=[args_scope_fix_call])

    repair_tool_call = next_file_generation_repair_call(state)
    if repair_tool_call is not None:
        return AIMessage(content="Inspecting the generated Python file before another verification attempt.", tool_calls=[repair_tool_call])

    verification_tool_call = next_file_generation_verification_call(history)
    if verification_tool_call is not None:
        return AIMessage(content="Proceeding to verify the generated Python file.", tool_calls=[verification_tool_call])

    return None

def _print_raw_llm_request_response(color, messages: list[BaseMessage], raw_text: str) -> None:
    text = (raw_text or "").strip()
    if not text:
        text = "<empty>"
    print(f"{color}{ANSI_ITALIC}[raw-llm]{ANSI_RESET}")
    print(f"{color}{ANSI_ITALIC}Messages:{ANSI_RESET}")
    for msg in messages:
        role = (
            "human"
            if isinstance(msg, HumanMessage)
            else "ai"
            if isinstance(msg, AIMessage)
            else "system"
            if isinstance(msg, SystemMessage)
            else "tool"
            if isinstance(msg, ToolMessage)
            else "other"
        )
        print(f"{color}{ANSI_ITALIC}[{role}]{ANSI_RESET}")
        print(f"{color}{ANSI_ITALIC}{normalize_message_content(msg)}{ANSI_RESET}")
    print(f"{color}{ANSI_ITALIC}Raw LLM response content:{ANSI_RESET}")
    print(f"{color}{ANSI_ITALIC}{text}{ANSI_RESET}")


def _run_main_execution_branch(
    *,
    llm: ChatOllama,
    state: AgentState,
    active_system_prompt: str,
    retrieval_messages: list[SystemMessage],
    action_required: bool,
    tool_name_set: set[str],
) -> AIMessage:
    
    response_llm = llm

    pre_messages, early_response = _build_pre_messages(
        active_system_prompt=active_system_prompt,
        retrieval_messages=retrieval_messages,
        state=state,
    )
    if early_response is not None:
        return early_response

    response = response_llm.invoke([*pre_messages])

    _print_raw_llm_request_response(color=ANSI_BLUE, messages=[*pre_messages], raw_text=response.content)

    response = _apply_response_recovery(
        response_llm=response_llm,
        pre_messages=pre_messages,
        response=response,
        action_required=action_required,
        tool_name_set=tool_name_set,
    )

    response = _apply_repeated_signature_guard(
        llm=llm,
        pre_messages=pre_messages,
        response=response,
        action_required=action_required,
        state=state,
    )

    if action_required and state.get("last_tool_success") is True and not getattr(response, "tool_calls", None):
        content = normalize_message_content(response)
        if content and content.strip():
            return response

        response = _finalize_from_successful_tool_context(
            llm=llm,
            pre_messages=pre_messages,
            state=state,
            history=history,
            response=response,
            tool_name_set=tool_name_set,
        )

    return response


def create_graph_nodes(
    *,
    llm: ChatOllama,
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
        rag_service=rag_service,
        rag_top_k=rag_top_k,
        tool_name_set=tool_name_set,
    )
    capture_tool_output_node = create_capture_tool_output_node()

    def brain_node(state: AgentState):
        history = state.get("messages", [])
        latest_user_prompt = latest_human_message_str(history)       

        preferred_tool_response = _successful_preferred_tool_fast_path(state, latest_user_prompt)
        if preferred_tool_response is not None:
            response = preferred_tool_response
        else:

            action_required = False
            retrieval_messages = []
            system_prompt = casual_system_prompt

            planner_route = str(state.get("planner_route", ""))
            if planner_route.startswith("action"):
                action_required = True
                system_prompt = agent_system_prompt
                retrieval_messages = retrieval_message(rag_service, latest_user_prompt, rag_top_k)
            elif planner_route.startswith("info"):
                action_required = True

            response = _run_main_execution_branch(
                llm=llm,
                state=state,
                active_system_prompt=system_prompt,
                retrieval_messages=retrieval_messages,
                action_required=action_required,
                tool_name_set=tool_name_set,
            )

        return response_with_usage(state, response)

    return planner_node, brain_node, capture_tool_output_node, route_after_brain
