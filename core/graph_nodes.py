from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END

from core.graph_constants import MAX_PSEUDO_RETRIES, MAX_REASONING_STEPS, RECENT_MESSAGE_WINDOW
from core.graph_context import retrieval_message, rolling_summary_message, update_rolling_summary
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
from core.graph_intents import is_file_generation_request, planner_routing_decision, preferred_info_tool, requires_action
from core.graph_messages import is_effectively_empty_response, latest_user_message, normalize_message_content, recent_messages
from core.graph_pseudo_tools import finalize_action_response, is_pseudo_tool_response, looks_like_pseudo_tool_text, recover_pseudo_tool_response
from core.graph_response_formatters import format_action_completion_response, format_info_tool_response
from core.graph_tool_events import (
    current_turn_has_successful_tool_result,
    extract_tool_signature,
    info_tool_already_called,
    message_repeats_signature,
)
from core.models import TokenUsage
from core.rag import WorkspaceRAG
from core.state import AgentState
from core.tool_output import parse_tool_result, unwrap_tool_output
from tools.info_ops import update_token_usage


def _direct_discussion_response(
    planner_llm: ChatOllama,
    system_prompt: str,
    retrieval_messages: list[SystemMessage],
    rolling_summary: str,
    recent_history: list,
) -> AIMessage:
    def _finalize_llm_text(source: AIMessage, content: str) -> AIMessage:
        metadata = getattr(source, "response_metadata", None)
        if isinstance(metadata, dict) and metadata:
            return AIMessage(content=content, response_metadata=metadata)
        return AIMessage(content=content)

    messages = [
        SystemMessage(content=system_prompt),
        *retrieval_messages,
        *rolling_summary_message(rolling_summary),
        *recent_history,
        SystemMessage(
            content=(
                "This turn is discussion-only. Answer directly in concise prose. "
                "Do not call tools, do not propose tool syntax, and do not create or modify files."
            )
        ),
    ]
    response = planner_llm.invoke(messages)
    content = normalize_message_content(response).strip()
    if getattr(response, "tool_calls", None) or looks_like_pseudo_tool_text(content) or not content:
        fallback = planner_llm.invoke(
            [
                *messages,
                SystemMessage(
                    content=(
                        "Your previous reply was not a direct discussion answer. "
                        "Reply with plain prose only, no code blocks and no tool-like syntax."
                    )
                ),
            ]
        )
        fallback_content = normalize_message_content(fallback).strip()
        if fallback_content and not looks_like_pseudo_tool_text(fallback_content):
            return _finalize_llm_text(fallback, fallback_content)
        return AIMessage(
            content=(
                "Describe the error message, the JSON input, and the code path that fails, and I will help isolate the parsing bug directly."
            )
        )
    return _finalize_llm_text(response, content)


def _detect_missing_dependency(tool_output_raw: str) -> str | None:
    """Extract package name from ModuleNotFoundError or ImportError in stderr. Returns package name or None."""
    if not tool_output_raw:
        return None
    stderr = last_tool_stderr(tool_output_raw)
    if not stderr:
        return None
    stderr_lower = stderr.lower()
    if "modulenotfounderror" in stderr_lower or "importerror" in stderr_lower:
        # Try to extract the module name from error message
        # e.g., "No module named 'Crypto'" -> "Crypto"
        if "no module named" in stderr_lower:
            start = stderr.find("'") + 1
            end = stderr.find("'", start)
            if start > 0 and end > start:
                return stderr[start:end]
        return "unknown_module"
    return None


def _planner_execution_brief(route: str, plan_source: str, plan_text: str) -> str:
    """Return a concise planner brief for the brain LLM on action routes."""
    normalized_plan = str(plan_text or "").strip()
    if not normalized_plan:
        return ""

    if len(normalized_plan) > 1500:
        normalized_plan = f"{normalized_plan[:1500]}..."

    normalized_source = str(plan_source or "").strip() or "unknown"
    return (
        "Planner execution brief below. Use it as guidance, but produce executable next steps now.\n"
        f"Route: {route}\n"
        f"Plan source: {normalized_source}\n"
        "Plan:\n"
        f"{normalized_plan}"
    )


def create_graph_nodes(
    *,
    llm: ChatOllama,
    planner_llm: ChatOllama,
    rag_service: WorkspaceRAG,
    rag_top_k: int,
    system_prompt: str,
    sap_system_prompt: str | None,
    tool_name_set: set[str],
):
    def planner_node(state: AgentState):
        """First pass: analyze prompt and create a plan WITHOUT taking actions."""
        history = state.get("messages", [])
        recent_history = recent_messages(history, RECENT_MESSAGE_WINDOW)
        previous_summary = state.get("rolling_summary", "")
        updated_summary = update_rolling_summary(
            planner_llm=planner_llm,
            existing_summary=previous_summary,
            recent_history=recent_history,
        )

        latest_user_prompt = latest_user_message(history)
        routing = planner_routing_decision(latest_user_prompt)
        route = routing.route
        preferred_tool = preferred_info_tool(latest_user_prompt)
        plan_source = "synthetic"

        if route == "info":
            plan_text = f"Info query detected: call {preferred_tool} tool and report the result."
        elif route == "clarify_domain":
            plan_text = (
                "Ambiguous domain detected: ask the user to choose SAP or Python before taking actions. "
                "Do not call tools until clarified."
            )
        elif route == "casual":
            plan_text = (
                "Casual conversation detected: respond directly without tools. "
                "Use conversation context for personal facts already shared and keep the reply brief."
            )
        elif route == "coding_discussion":
            plan_text = (
                "Coding discussion detected: answer directly unless a targeted tool becomes necessary. "
                "Use conversation context, retrieved knowledge, and keep the reply concise."
            )
        elif route == "conversation":
            plan_text = "Conversation detected: respond directly and briefly without tools unless the user asks for concrete action."
        else:
            retrieval_messages = retrieval_message(rag_service, latest_user_prompt, rag_top_k)
            summary_message = rolling_summary_message(updated_summary)

            planning_system = """You are a strategic planner. Analyze the user's request and create a clear step-by-step plan.
DO NOT take any actions yet. Just output:
1. What needs to be done (list of 2-4 key tasks)
2. File/tool sequence required
3. Expected outcome

Be concise. Format as a numbered list."""

            pre_messages = [
                SystemMessage(content=planning_system),
                *retrieval_messages,
                *summary_message,
            ]

            plan_response = planner_llm.invoke([*pre_messages, *recent_history])
            plan_text = str(plan_response.content)
            plan_source = "llm"

        return {
            "plan": plan_text,
            "planner_plan_source": plan_source,
            "planner_route": route,
            "planner_domain": routing.domain,
            "planner_confidence": routing.confidence,
            "planner_domain_enforced": routing.enforced,
            "rolling_summary": updated_summary,
            "steps": 0,
            "last_tool_success": True,
            "repeat_fail_count": 0,
            "tool_text_retry_used": False,
        }

    def brain_node(state: AgentState):
        history = state.get("messages", [])
        recent_history = recent_messages(history, RECENT_MESSAGE_WINDOW)
        rolling_summary = state.get("rolling_summary", "")
        route = str(state.get("planner_route", ""))
        planner_domain = str(state.get("planner_domain", "general") or "general")
        planner_confidence = float(state.get("planner_confidence", 0.0) or 0.0)
        planner_domain_enforced = bool(state.get("planner_domain_enforced", False))
        planner_plan = str(state.get("plan", "") or "")
        planner_plan_source = str(state.get("planner_plan_source", "") or "")
        latest_user_prompt = latest_user_message(history)
        action_required = requires_action(latest_user_prompt)
        preferred_tool = preferred_info_tool(latest_user_prompt)
        use_sap_prompt = route == "action:sap" or planner_domain == "sap"
        active_system_prompt = sap_system_prompt if use_sap_prompt and sap_system_prompt else system_prompt
        file_generation_requested = route == "action:file_generation" or is_file_generation_request(latest_user_prompt)
        successful_tool_result_in_turn = current_turn_has_successful_tool_result(history)
        current_filegen_issue = (
            file_generation_quality_issue(history, latest_user_prompt)
            if file_generation_requested and successful_tool_result_in_turn
            else None
        )
        action_completion_summary = format_action_completion_response(history)
        retrieval_messages = retrieval_message(rag_service, latest_user_message(history), rag_top_k)

        if route == "clarify_domain":
            response = AIMessage(
                content=(
                    "Your request could map to both SAP and Python workflows. "
                    "Please choose one so I can execute correctly: `SAP` or `Python`. "
                    "Tip: you can force routing with `[domain:sap]` or `[domain:python]` in your prompt."
                )
            )
            meta = getattr(response, "response_metadata", {}) or {}
            usage = TokenUsage.from_response_metadata(meta)
            update_token_usage(usage.model_dump())
            return {
                "messages": [response],
                "steps": state.get("steps", 0) + 1,
                "token_usage": usage,
                "tool_text_retry_used": False,
            }

        if route in {"casual", "coding_discussion", "conversation"} and not preferred_tool:
            response = _direct_discussion_response(
                planner_llm=planner_llm,
                system_prompt=active_system_prompt,
                retrieval_messages=retrieval_messages,
                rolling_summary=rolling_summary,
                recent_history=recent_history,
            )
            meta = getattr(response, "response_metadata", {}) or {}
            usage = TokenUsage.from_response_metadata(meta)
            update_token_usage(usage.model_dump())
            return {
                "messages": [response],
                "steps": state.get("steps", 0) + 1,
                "token_usage": usage,
                "tool_text_retry_used": False,
            }

        if file_generation_requested:
            failed_verifications, latest_verification_error = file_generation_verification_failures(history)
            if (
                failed_verifications >= 2
                and not should_finalize_action_turn(history, route)
                and not last_tool_missing_required_args(latest_verification_error)
            ):
                response = AIMessage(
                    content=(
                        "Action-required run stopped after repeated verification failures while generating the file. "
                        f"Latest verification error: {latest_verification_error} "
                        "I prevented further read/write/run looping. Please retry and I will apply a different repair strategy immediately."
                    )
                )
                meta = getattr(response, "response_metadata", {}) or {}
                usage = TokenUsage.from_response_metadata(meta)
                update_token_usage(usage.model_dump())
                return {
                    "messages": [response],
                    "steps": state.get("steps", 0) + 1,
                    "token_usage": usage,
                    "tool_text_retry_used": False,
                }

            args_scope_fix_call = next_args_scope_autofix_call(history)
            if args_scope_fix_call is not None:
                response = AIMessage(content="Applying deterministic args-scope repair before re-verification.", tool_calls=[args_scope_fix_call])
                meta = getattr(response, "response_metadata", {}) or {}
                usage = TokenUsage.from_response_metadata(meta)
                update_token_usage(usage.model_dump())
                return {
                    "messages": [response],
                    "steps": state.get("steps", 0) + 1,
                    "token_usage": usage,
                    "tool_text_retry_used": False,
                }

            repair_tool_call = next_file_generation_repair_call(state)
            if repair_tool_call is not None:
                response = AIMessage(content="Inspecting the generated Python file before another verification attempt.", tool_calls=[repair_tool_call])
                meta = getattr(response, "response_metadata", {}) or {}
                usage = TokenUsage.from_response_metadata(meta)
                update_token_usage(usage.model_dump())
                return {
                    "messages": [response],
                    "steps": state.get("steps", 0) + 1,
                    "token_usage": usage,
                    "tool_text_retry_used": False,
                }

            verification_tool_call = next_file_generation_verification_call(history)
            if verification_tool_call is not None:
                response = AIMessage(content="Proceeding to verify the generated Python file.", tool_calls=[verification_tool_call])
                meta = getattr(response, "response_metadata", {}) or {}
                usage = TokenUsage.from_response_metadata(meta)
                update_token_usage(usage.model_dump())
                return {
                    "messages": [response],
                    "steps": state.get("steps", 0) + 1,
                    "token_usage": usage,
                    "tool_text_retry_used": False,
                }

        if (
            action_required
            and action_completion_summary
            and should_finalize_action_turn(history, route)
            and not (file_generation_requested and current_filegen_issue)
        ):
            response = AIMessage(content=action_completion_summary)
            meta = getattr(response, "response_metadata", {}) or {}
            usage = TokenUsage.from_response_metadata(meta)
            update_token_usage(usage.model_dump())
            return {
                "messages": [response],
                "steps": state.get("steps", 0) + 1,
                "token_usage": usage,
                "tool_text_retry_used": False,
            }

        info_tool_called = preferred_tool and info_tool_already_called(history, preferred_tool)

        if info_tool_called:
            last_tool_result = state.get("last_tool_output", "")
            formatted_response = format_info_tool_response(preferred_tool, last_tool_result)
            response = AIMessage(content=formatted_response)
        else:
            pre_messages = [
                SystemMessage(content=active_system_prompt),
                *retrieval_messages,
                *rolling_summary_message(rolling_summary),
            ]
            skip_action_enforcement = False
            response_llm = planner_llm if route in {"casual", "coding_discussion", "conversation"} else llm
            allow_tool_recovery = route not in {"casual", "coding_discussion", "conversation"}
            if route.startswith("action") and not preferred_tool:
                planner_brief = _planner_execution_brief(route, planner_plan_source, planner_plan)
                if planner_brief:
                    pre_messages.append(SystemMessage(content=planner_brief))
            if route == "action:sap" or planner_domain == "sap":
                sap_enforcement_suffix = (
                    "This domain is explicitly enforced by the user."
                    if planner_domain_enforced
                    else f"Planner confidence for SAP domain: {planner_confidence:.2f}."
                )
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "SAP domain execution rules: Prefer SAP tools first "
                            "(`query_abap_table`, `execute_abap_report`, `lookup_material`, `get_report_data`). "
                            "Do not use Python file-generation or runtime tools unless the user explicitly requests Python code generation. "
                            f"{sap_enforcement_suffix}"
                        )
                    )
                )
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "SAP ABAP correctness checklist before finalizing any code: "
                            "(1) Do not use SELECT *. Select only required fields. "
                            "(2) For PO vendor filtering, use EKKO-LIFNR via join EKPO<->EKKO on EBELN; do not use EKPO-LIFNR. "
                            "(3) Use modern Open SQL host variables with @ for select-options and constants. "
                            "(4) Avoid obsolete ABAP patterns such as OCCURS/header lines/implicit work areas. "
                            "(5) For open PO logic, enforce MENGE > WEMNG, LOEKZ = space, ELIKZ = space, and compute remaining quantity in SQL when possible. "
                            "If any checklist item is not satisfied, correct the solution before responding."
                        )
                    )
                )
            if preferred_tool:
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "This request must be answered by calling the "
                            f"{preferred_tool} tool first. "
                            "Do not answer from memory. Use the tool and then respond from its output."
                        )
                    )
                )
            if file_generation_requested and not successful_tool_result_in_turn:
                pre_messages.append(
                    SystemMessage(
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
                )
            if file_generation_requested and current_filegen_issue:
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "The generated file is not complete for this request yet. "
                            f"Detected gap: {current_filegen_issue} "
                            "You must call write_file now with a corrected, fully implemented version of the file that fixes this gap. "
                            "Do NOT call run_python again until you have first rewritten the file with the missing implementation."
                        )
                    )
                )
            # Check for missing dependencies (ModuleNotFoundError, ImportError)
            missing_module = _detect_missing_dependency(state.get("last_tool_output", ""))
            if missing_module:
                response = AIMessage(
                    content=(
                        f"The code failed with a missing dependency: '{missing_module}'. "
                        f"Please install it using:\n\n"
                        f"pip install {missing_module}\n\n"
                        f"After installation, you can retry the code execution."
                    )
                )
                meta = getattr(response, "response_metadata", {}) or {}
                usage = TokenUsage.from_response_metadata(meta)
                update_token_usage(usage.model_dump())
                return {
                    "messages": [response],
                    "steps": state.get("steps", 0) + 1,
                    "token_usage": usage,
                    "tool_text_retry_used": False,
                }

            if file_generation_requested and state.get("last_tool_success") is False and last_tool_missing_required_args(state.get("last_tool_output", "")):
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "The CLI verification failed because required command-line arguments were missing. "
                            "Do not rerun the same bare command. Instead, create a minimal sample input file in the workspace if needed, "
                            "then call run_python again with the required argument values using v__args."
                        )
                    )
                )
            if route in {"coding_discussion", "conversation"}:
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "This turn is a discussion request, not an implementation request. "
                            "Answer directly and do not create files, run tools, or execute code unless the user explicitly asks for concrete actions."
                        )
                    )
                )
            if state.get("last_tool_success") is False and state.get("last_tool_signature"):
                signature = state.get("last_tool_signature", "")
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "The previous tool call failed. "
                            f"Failed signature: {signature}. "
                            "Do not repeat that same tool call with identical arguments. "
                            "Choose a different corrective next action."
                        )
                    )
                )
                stderr = last_tool_stderr(state.get("last_tool_output", ""))
                if stderr:
                    pre_messages.append(
                        SystemMessage(
                            content=(
                                "Latest Python/tool error to fix before re-verification:\n"
                                f"{stderr[:1200]}"
                            )
                        )
                    )
                if last_tool_has_args_nameerror(state.get("last_tool_output", "")):
                    pre_messages.append(
                        SystemMessage(
                            content=(
                                "The failure indicates args scope is broken. "
                                "Repair the script so parse_args() result is defined and used inside main(), "
                                "and avoid referencing args at module top level. "
                                "Write a corrected file version before running run_python again."
                            )
                        )
                    )
            elif action_required and state.get("last_tool_success") is True and state.get("last_tool_signature"):
                signature = state.get("last_tool_signature", "")
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "The previous tool call already succeeded. "
                            f"Successful signature: {signature}. "
                            "Do not repeat that same tool call with identical arguments. "
                            "Choose the next distinct step, such as verification, creating required sample input, or giving the final answer if the task is done."
                        )
                    )
                )
            elif action_required and successful_tool_result_in_turn and not (file_generation_requested and current_filegen_issue):
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "A tool has already succeeded during this user turn. "
                            "If that successful result satisfies the request, provide the final concise answer now "
                            "instead of calling more tools. Only call another tool if a specific remaining gap still exists."
                        )
                    )
                )
            elif file_generation_requested and current_filegen_issue:
                pre_messages.append(
                    SystemMessage(
                        content=(
                            "A tool succeeded, but the implementation is still incomplete. "
                            f"Detected gap: {current_filegen_issue} "
                            "Call write_file now with a corrected, fully implemented version. "
                            "Do NOT call run_python again before rewriting the file."
                        )
                    )
                )

            response = response_llm.invoke([*pre_messages, *recent_history])

            pseudo_retry_count = 0
            while allow_tool_recovery and is_pseudo_tool_response(response) and pseudo_retry_count < MAX_PSEUDO_RETRIES:
                response = response_llm.invoke(
                    [
                        *pre_messages,
                        *recent_history,
                        response,
                        SystemMessage(
                            content=(
                                "Your previous response included pseudo tool invocation text. "
                                "Do not output code blocks that look like tool calls. "
                                "If actions are needed, emit real tool calls only. "
                                "If no action is needed, provide only the final concise answer."
                            )
                        ),
                    ]
                )
                pseudo_retry_count += 1

            if allow_tool_recovery and is_pseudo_tool_response(response):
                response = recover_pseudo_tool_response(response, tool_name_set)
                if not getattr(response, "tool_calls", None):
                    response = AIMessage(
                        content=(
                            "I produced pseudo tool-call text instead of executable tool calls, so no action was taken. "
                            "Please retry with a task phrased as file changes inside the sandbox workspace."
                        )
                    )

            if is_effectively_empty_response(response):
                response = response_llm.invoke(
                    [
                        *pre_messages,
                        *recent_history,
                        SystemMessage(
                            content=(
                                "Your previous response was empty. "
                                "Respond again with one of the following: "
                                "(1) concrete tool calls to progress the task, or "
                                "(2) a concise final answer."
                            )
                        ),
                    ]
                )

            if is_effectively_empty_response(response):
                response = AIMessage(
                    content=(
                        "I could not produce a valid action or answer from the model. "
                        "Please retry the prompt or check model availability in Ollama."
                    )
                )

            if route in {"coding_discussion", "conversation"} and getattr(response, "tool_calls", None):
                response = planner_llm.invoke(
                    [
                        *pre_messages,
                        *recent_history,
                        SystemMessage(
                            content=(
                                "You started taking actions for a discussion-only request. "
                                "Do not call tools. Provide a concise direct answer only."
                            )
                        ),
                    ]
                )

            if (
                action_required
                and state.get("last_tool_signature")
                and message_repeats_signature(response, str(state.get("last_tool_signature", "")))
            ):
                repeated_signature = str(state.get("last_tool_signature", ""))
                repeat_reason = (
                    "already succeeded"
                    if state.get("last_tool_success") is True
                    else "already failed"
                )
                response = llm.invoke(
                    [
                        *pre_messages,
                        *recent_history,
                        response,
                        SystemMessage(
                            content=(
                                "You repeated the exact same tool call again. "
                                f"Repeated signature: {repeated_signature}. "
                                f"That signature {repeat_reason}. "
                                "Do not emit that same tool call again. "
                                "Choose the next distinct step now, such as fixing the file, reading the error output, creating sample input, verifying with different arguments, or giving the final answer if the task is complete."
                            )
                        ),
                    ]
                )

            if file_generation_requested and state.get("last_tool_success") is True:
                read_snapshot = last_read_file_snapshot(state)
                unchanged_retry_count = 0
                while response_has_unchanged_write(response, read_snapshot) and unchanged_retry_count < 2:
                    response = llm.invoke(
                        [
                            *pre_messages,
                            *recent_history,
                            response,
                            SystemMessage(
                                content=(
                                    "Your proposed write_file call rewrites the file with identical content after a failed verification. "
                                    "That is a no-op and will loop. "
                                    "Produce a corrected write_file call that changes the file to address the failing error, then verify again. "
                                    "Do not emit an unchanged write_file call."
                                )
                            ),
                        ]
                    )
                    response = finalize_action_response(response, tool_name_set)
                    unchanged_retry_count += 1

                if response_has_unchanged_write(response, read_snapshot):
                    response = AIMessage(
                        content=(
                            "Action-required run stopped because repair attempts kept rewriting identical file content after a failed verification. "
                            "Please retry so I can apply a different repair strategy."
                        )
                    )
                    skip_action_enforcement = True

            failed_rewrite_context = last_failed_verification_rewrite_info(history, state)
            if failed_rewrite_context and getattr(response, "tool_calls", None):
                failed_path, failed_write_content = failed_rewrite_context
                if failed_write_content and message_repeats_write_content(response, failed_path, failed_write_content):
                    last_tool_output = state.get("last_tool_output", "")
                    failure_details = ""
                    if isinstance(last_tool_output, dict):
                        data = last_tool_output.get("data")
                        if isinstance(data, dict):
                            failure_details = str(data.get("stderr", "") or data.get("stdout", "") or "")
                    response = llm.invoke(
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

            if not getattr(response, "tool_calls", None):
                enforcement_prompt = ""
                if file_generation_requested and not successful_tool_result_in_turn:
                    enforcement_prompt = (
                        "This file-generation request must continue with executable tool calls now. "
                        "Return tool calls only. Start by creating or updating files in the sandbox, then verify the result."
                    )
                elif file_generation_requested and current_filegen_issue:
                    enforcement_prompt = (
                        "This file-generation request is still incomplete. "
                        f"Detected gap: {current_filegen_issue}. "
                        "Call write_file now with a corrected, fully implemented version of the file. "
                        "Do NOT call run_python again before rewriting the file."
                    )
                elif preferred_tool:
                    enforcement_prompt = (
                        "You ignored the required info tool. "
                        f"Call {preferred_tool} now. "
                        "Do not answer from memory or provide a prose-only response."
                    )
                elif action_required and not skip_action_enforcement and not successful_tool_result_in_turn:
                    enforcement_prompt = (
                        "The user requested concrete actions. "
                        "Return at least one executable tool call now. "
                        "Do not return a prose-only response."
                    )

                if enforcement_prompt:
                    response = llm.invoke(
                        [
                            *pre_messages,
                            *recent_history,
                            response,
                            SystemMessage(content=enforcement_prompt),
                        ]
                    )
                    response = finalize_action_response(response, tool_name_set)

        meta = getattr(response, "response_metadata", {}) or {}
        usage = TokenUsage.from_response_metadata(meta)
        update_token_usage(usage.model_dump())
        return {
            "messages": [response],
            "steps": state.get("steps", 0) + 1,
            "token_usage": usage,
            "tool_text_retry_used": False,
        }

    def capture_tool_output_node(state: AgentState):
        history = state.get("messages", [])
        if not history:
            return {
                "last_tool_output": "",
                "last_tool_signature": "",
                "last_tool_success": True,
                "repeat_fail_count": 0,
            }

        last_message = history[-1]
        if isinstance(last_message, ToolMessage):
            raw_content = str(last_message.content)
            parsed = parse_tool_result(raw_content)
            unwrapped = unwrap_tool_output(raw_content)
            success = parsed.success if parsed is not None else bool(isinstance(unwrapped, dict) and unwrapped.get("success") is True)
            current_signature = extract_tool_signature(history[:-1], getattr(last_message, "tool_call_id", None))

            previous_signature = state.get("last_tool_signature", "")
            previous_success = state.get("last_tool_success", True)
            previous_repeat_count = state.get("repeat_fail_count", 0)
            if not success and current_signature and previous_signature == current_signature and not previous_success:
                repeat_fail_count = previous_repeat_count + 1
            elif not success and current_signature:
                repeat_fail_count = 1
            else:
                repeat_fail_count = 0

            if isinstance(unwrapped, dict):
                return {
                    "last_tool_output": unwrapped,
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            if isinstance(unwrapped, list):
                return {
                    "last_tool_output": {"message": str(unwrapped), "data": unwrapped, "success": success},
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            if isinstance(unwrapped, str):
                return {
                    "last_tool_output": {"message": unwrapped, "data": None, "success": success},
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            return {
                "last_tool_output": raw_content,
                "last_tool_signature": current_signature,
                "last_tool_success": success,
                "repeat_fail_count": repeat_fail_count,
            }
        return {
            "last_tool_output": state.get("last_tool_output", ""),
            "last_tool_signature": state.get("last_tool_signature", ""),
            "last_tool_success": state.get("last_tool_success", True),
            "repeat_fail_count": state.get("repeat_fail_count", 0),
        }

    def route_after_brain(state: AgentState):
        history = state.get("messages", [])
        if not history:
            return END

        if state.get("steps", 0) >= MAX_REASONING_STEPS:
            return END

        last_message = history[-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    return planner_node, brain_node, capture_tool_output_node, route_after_brain
