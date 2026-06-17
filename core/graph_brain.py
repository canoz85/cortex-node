
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from core.graph_pseudo_tools import finalize_action_response, is_generic_json_tool_response, is_pseudo_tool_response, recover_pseudo_tool_response
from langchain_ollama import ChatOllama

from core.graph_constants import ANSI_BLUE, ANSI_GREEN, ANSI_ITALIC, ANSI_RED, ANSI_RESET, ANSI_YELLOW
from core.graph_context import retrieval_message
from core.graph_filegen_policy import last_tool_has_args_nameerror, last_tool_missing_required_args, last_tool_stderr
from core.graph_intents import preferred_file_tool, preferred_info_tool
from core.graph_messages import current_turn_messages, is_effectively_empty_response, latest_human_message, latest_human_message_str, normalize_message_content, recent_messages

from core.graph_node_helpers import (
    planner_execution_brief,
    response_with_usage,
)

from core.graph_response_formatters import format_action_completion_response, format_preferred_tool_response
from core.graph_summarize import rolling_summary_message
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
        pre_messages: list,
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
    
    def _finalize_from_successful_tool_context(
        *,
        llm: ChatOllama,
        pre_messages: list,
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

        # 1. Core system context
        pre_messages = [
            SystemMessage(content=active_system_prompt),
            *retrieval_messages,
            *rolling_summary_message(rolling_summary),
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
            
        if route.startswith("action"):
            pre_messages.extend(current_turn_messages(history))
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
            state=state,
        )

        response = _apply_repeated_signature_guard(
            llm=response_llm,
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

            action_required = False
            retrieval_messages = []
            llm = brain_llm
            system_prompt = casual_system_prompt

            planner_route = str(state.get("planner_route", ""))
            if planner_route.startswith("action"):
                action_required = True
                llm = tool_brain_llm
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

    return brain_node