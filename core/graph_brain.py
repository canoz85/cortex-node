
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from core.graph_pseudo_tools import finalize_action_response, is_generic_json_tool_response, is_pseudo_tool_response, recover_pseudo_tool_response
from langchain_ollama import ChatOllama

from core.graph_constants import ANSI_BLUE, ANSI_GREEN, ANSI_ITALIC, ANSI_RED, ANSI_RESET, ANSI_YELLOW
from core.graph_context import retrieval_message
from core.graph_filegen_policy import last_tool_has_args_nameerror, last_tool_missing_required_args, last_tool_stderr
from core.graph_intents import preferred_file_tool, preferred_info_tool
from core.graph_messages import current_turn_messages, is_effectively_empty_response, latest_human_message, latest_human_message_str, normalize_message_content

from core.graph_node_helpers import (
    planner_execution_brief,
    response_with_usage,
)

from core.graph_response_formatters import format_action_completion_response, format_preferred_tool_response
from core.graph_summarize import rolling_summary_message
from core.graph_tool_events import (
    message_repeats_signature,
    parse_tool_signature,
)

from core.rag import WorkspaceRAG
from core.state import AgentState

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

def _invoke_with_trace(
    *,
    llm: ChatOllama,
    messages: list[BaseMessage],
    color: str,
) -> AIMessage:
    response = llm.invoke(messages)
    _print_raw_llm_request_response(color=color, messages=messages, raw_text=response.content)
    return response

def create_brain_node(
    *,
    brain_llm: ChatOllama,
    tool_brain_llm: ChatOllama,
    rag_service: WorkspaceRAG,
    rag_top_k: int,
    agent_system_prompt: str,
    casual_system_prompt: str,
    tool_name_set: set[str],
):
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
    
    def _pseudo_tool_fallback_response() -> AIMessage:
        return AIMessage(
            content=(
                "I produced pseudo tool-call text instead of executable tool calls, so no action was taken. "
                "Please retry with a task phrased as file changes inside the sandbox workspace."
            )
        )


    def _apply_response_recovery(
        *,
        response_llm: ChatOllama,
        pre_messages: list[BaseMessage],
        response: AIMessage,
        action_required: bool,
        state: AgentState,
        tool_name_set: set[str],
    ) -> AIMessage:
    
        # Always sanitize pseudo tool text on action-required turns so raw JSON/call text
        # is never emitted as a final brain answer.
        if (action_required) and is_pseudo_tool_response(response):
            response = recover_pseudo_tool_response(response, tool_name_set)
            if not getattr(response, "tool_calls", None):
                response = _pseudo_tool_fallback_response()

        if (action_required) and is_generic_json_tool_response(response):
            response = _finalize_from_successful_tool_context(
                llm=response_llm,
                pre_messages=pre_messages,
                state=state,
                response=response,
                tool_name_set=tool_name_set,
         )

 
        if is_effectively_empty_response(response):
            response = _invoke_with_trace(
                llm=response_llm,
                messages=[*pre_messages, _empty_response_retry_prompt()],
                color=ANSI_GREEN,
            )

        if is_effectively_empty_response(response):
            response = _empty_response_fallback()

        return response
    
    def _repeated_success_final_answer_prompt(repeated_signature: str) -> SystemMessage:
        return SystemMessage(
            content=(
                "The repeated tool call already succeeded earlier in this turn and must not be emitted again. "
                f"Repeated signature: {repeated_signature}. "
                "Do not call any tools now. "
                "Return a concise final answer using the tool outputs already in context."
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

    def _apply_repeated_signature_guard(
        *,
        llm: ChatOllama,
        pre_messages: list[BaseMessage],
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
        corrected_response = _invoke_with_trace(
            llm=llm,
            messages=[*pre_messages, response, _repeated_signature_correction_prompt(last_tool_signature, repeat_reason)],
            color=ANSI_YELLOW,
        )

        if state.get("last_tool_success") is True and message_repeats_signature(corrected_response, last_tool_signature):
            
            final_response = _invoke_with_trace(
                llm=llm,
                messages=[*pre_messages, corrected_response, _repeated_success_final_answer_prompt(last_tool_signature)],
                color=ANSI_GREEN,
            )

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
    
    def _finalize_from_successful_tool_context(
        *,
        llm: ChatOllama,
        pre_messages: list[BaseMessage],
        state: AgentState,
        response: AIMessage,
        tool_name_set: set[str],
    ) -> AIMessage:
        """Ask the model to decide: final answer now or distinct next tool call.

        This prevents raw tool output from being echoed as the final brain response.
        """
        if getattr(response, "tool_calls", None):
            return response
        
        history = state.get("messages", [])

        completion = format_action_completion_response(history)
        if completion:
            return AIMessage(content=completion)
        
        decision = _invoke_with_trace(
            llm=llm,
            messages=[*pre_messages, response, SystemMessage(
                content=(
                    "Decide one of two outcomes only: "
                    "(1) If the user request is satisfied, provide the final concise answer now. "
                    "(2) If not satisfied, emit exactly one executable next tool call with distinct arguments. "
                    "Do not repeat the same successful tool call signature. "
                    "Do not restate raw tool output verbatim unless it is the final user-facing answer."
                )
            )],
            color=ANSI_RED,
        )

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

    def _is_action_route(route: str) -> bool:
        return route.startswith("action")
    
    def _is_info_route(route: str) -> bool:
        return route.startswith("info")
    
    def _build_context_messages(
        *,
        active_system_prompt: str,
        retrieval_messages: list[SystemMessage],
        route: str,
        history: list,
        plan_text: str,
        rolling_summary: str,
    ) -> list[BaseMessage]:
    
        context_messages: list[BaseMessage] = [
            SystemMessage(content=active_system_prompt),
            *retrieval_messages,
            *rolling_summary_message(rolling_summary),
        ]

        if plan_text and _is_action_route(route):
            planner_brief = planner_execution_brief(route, plan_text)
            if planner_brief:
                context_messages.append(SystemMessage(content=planner_brief))

        if _is_action_route(route):
            context_messages.extend(current_turn_messages(history))
        else:
            latest = latest_human_message(history)
            if latest is not None:
                context_messages.append(latest)

        return context_messages

    def _build_runtime_guidance_messages(
        *,
        last_tool_success: object,
        last_tool_output: object,
        last_tool_rendered: str,
        last_tool_signature: str,
    ) -> list[SystemMessage]:
        guidance: list[str] = []

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

        has_usable_success_output = bool(last_tool_rendered.strip()) or isinstance(last_tool_output, dict)
        if last_tool_success is True and has_usable_success_output:
            guidance.append(
            "A tool has already succeeded during this user turn. "
            "Use the tool output as the authoritative source of truth for the current state of the world. "
            "If that successful result satisfies the request, provide the final concise answer now instead of calling more tools. "
            "Only call another tool if a specific remaining gap still exists that can be directly addressed by one more tool call."
            )

        if not guidance:
            return []

        combined = "\n".join([
            "### RUNTIME INSTRUCTIONS (HIGHEST PRIORITY):",
            *guidance,
        ])
        return [SystemMessage(content=combined)]
    
    def _build_pre_messages(
        *,
        active_system_prompt: str,
        retrieval_messages: list[SystemMessage],
        state: AgentState,
    ) -> list[BaseMessage]:
        
        route = str(state.get("planner_route", ""))
        history = state.get("messages", [])
        plan_text = str(state.get("plan", "") or "")
        rolling_summary = state.get("rolling_summary", "")

        last_tool_success = state.get("last_tool_success")
        last_tool_output = state.get("last_tool_output", "")
        last_tool_rendered = str(state.get("last_tool_rendered", "") or "")
        last_tool_signature = str(state.get("last_tool_signature", "") or "")

        # Route guard:
        # action routes are execution-first, info* routes are tool-oriented but not full action context,
        # everything else falls back to direct latest-user context.
        if not (_is_action_route(route) or _is_info_route(route) or route in {"", "casual", "conversation", "coding_discussion", "clarify_domain"}):
        # Unknown planner routes are treated as non-action fallback for safety.
            pass

        pre_messages = _build_context_messages(
            active_system_prompt=active_system_prompt,
            retrieval_messages=retrieval_messages,
            route=route,
            history=history,
            plan_text=plan_text,
            rolling_summary=rolling_summary,
        )

        pre_messages.extend(
            _build_runtime_guidance_messages(
            last_tool_success=last_tool_success,
            last_tool_output=last_tool_output,
            last_tool_rendered=last_tool_rendered,
            last_tool_signature=last_tool_signature,
            )
        )

        return pre_messages
    
    def _run_main_execution_branch(
        *,
        llm: ChatOllama,
        state: AgentState,
        active_system_prompt: str,
        retrieval_messages: list[SystemMessage],
        action_required: bool,
        tool_name_set: set[str],
    ) -> AIMessage:
        
        pre_messages = _build_pre_messages(
            active_system_prompt=active_system_prompt,
            retrieval_messages=retrieval_messages,
            state=state,
        )
        
        response = _invoke_with_trace(
            llm=llm,
            messages=[*pre_messages],
            color=ANSI_BLUE,
        )

        response = _apply_response_recovery(
            response_llm=llm,
            pre_messages=pre_messages,
            response=response,
            action_required=action_required,
            tool_name_set=tool_name_set,
            state=state,
        )

        response = _apply_repeated_signature_guard(
            llm=llm,
            pre_messages=pre_messages,
            response=response,
            action_required=action_required,
            state=state,
        )

        return response

    def brain_node(state: AgentState):
        history = state.get("messages", [])
        latest_user_prompt = latest_human_message_str(history)       

        preferred_tool_response = _successful_preferred_tool_fast_path(state, latest_user_prompt)
        if preferred_tool_response is not None:
            response = preferred_tool_response
        else:

            planner_route = str(state.get("planner_route", ""))
            action_required = planner_route.startswith("action") or planner_route.startswith("info")
            llm = tool_brain_llm if planner_route.startswith("action") else brain_llm
            system_prompt = agent_system_prompt if planner_route.startswith("action") else casual_system_prompt
            retrieval_messages = (retrieval_message(rag_service, latest_user_prompt, rag_top_k) 
                                  if planner_route.startswith("action")
                                  else [])

            response = _run_main_execution_branch(
                llm=llm,
                state=state,
                active_system_prompt=system_prompt,
                retrieval_messages=retrieval_messages,
                action_required=action_required,
                tool_name_set=tool_name_set,
            )

        return response_with_usage(state, response)

    return brain_node