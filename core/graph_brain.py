from typing import Any, Set

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    AIMessage, BaseMessage, 
    HumanMessage, SystemMessage, ToolMessage
)
from uuid_utils import uuid4

from core.protocol.enums import (
    BrainOutcome,
    StepStatus,
    WorkerRole,
)

from core.protocol.models import (
    BrainInput,
    BrainResult,
    ToolRequest,
)

from core.protocol.bridge import brain_result_to_legacy, build_execution_state, tool_result_to_legacy, with_cursor
from core.graph_pseudo_tools import (
    finalize_action_response,
    is_generic_json_tool_response,
    is_pseudo_tool_response,
    recover_action_response,
)

from core.graph_constants import ANSI_BLUE, ANSI_GREEN, ANSI_ITALIC, ANSI_RED, ANSI_RESET, ANSI_YELLOW, FINAL_ANSWER_SYSTEM_PROMPT
from core.graph_filegen_policy import last_tool_has_args_nameerror, last_tool_missing_required_args, last_tool_stderr
from core.graph_messages import current_turn_messages, is_effectively_empty_response, latest_human_message, normalize_message_content

from core.graph_node_helpers import response_with_usage

from core.graph_response_formatters import format_action_completion_response, format_tool_result_response
from core.graph_state_machine import (
    decide_action_recovery,
    decide_brain_execution,
    decide_repeated_signature,
    should_fallback_after_empty_response,
    should_retry_after_empty_response,
    BrainExecutionDecision,
)
from core.graph_summarize import rolling_summary_message
from core.graph_tool_events import message_repeats_signature
from core.protocol.bridge import build_brain_input

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
    show_raw_llm: bool = False,
    color: str,
) -> AIMessage:
    response = llm.invoke(messages)
    if show_raw_llm:
        _print_raw_llm_request_response(color=color, messages=messages, raw_text=response.content)
    return response

def create_brain_node(
    *,
    brain_llm: ChatOllama,
    tool_brain_llm: ChatOllama,
    agent_system_prompt: str,
    final_answer_system_prompt: str,
    casual_system_prompt: str,
    tools_set: Set[str],
    show_raw_llm: bool,
):

    
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

    def _recover_response_for_action_flow(
        *,
        llm: ChatOllama,
        pre_messages: list[BaseMessage],
        response: AIMessage,
        action_required: bool,
        state: AgentState,
        brain_input,
        tools_set: set[str],
    ) -> AIMessage:
    
        # Always sanitize pseudo tool text on action-required turns so raw JSON/call text
        # is never emitted as a final brain answer.
        recovered_action_response = recover_action_response(response, tools_set) if action_required else None
        action_recovery_decision = decide_action_recovery(
            action_required=action_required,
            recovered_action_response_exists=(recovered_action_response is not None),
            pseudo_tool_response_detected=is_pseudo_tool_response(response),
            generic_json_tool_response_detected=is_generic_json_tool_response(response),
        )

        if action_recovery_decision.use_recovered_action_response and recovered_action_response is not None:
            response = recovered_action_response
        elif action_recovery_decision.use_pseudo_fallback:
            response = _pseudo_tool_fallback_response()

        if action_recovery_decision.finalize_generic_json:
            response = _finalize_from_successful_tool_context(
                llm=llm,
                pre_messages=pre_messages,
                state=state,
                brain_input=brain_input,
                response=response,
                tools_set=tools_set,
            )

 
        if should_retry_after_empty_response(is_effectively_empty_response(response)):
            response = _invoke_with_trace(
                llm=llm,
                messages=[*pre_messages, _empty_response_retry_prompt()],
                show_raw_llm=show_raw_llm,
                color=ANSI_GREEN,
            )

        if should_fallback_after_empty_response(is_effectively_empty_response(response)):
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

    def _enforce_repeated_signature_policy(
        *,
        llm: ChatOllama,
        pre_messages: list[BaseMessage],
        response: AIMessage,
        action_required: bool,
        state: AgentState,
        brain_input,
    ) -> AIMessage:
        last_tool_signature = _resolved_tool_signature(state=state, brain_input=brain_input)
        has_last_signature = bool(last_tool_signature)
        initial_repeats_signature = (
            message_repeats_signature(response, last_tool_signature)
            if has_last_signature
            else False
        )
        repeated_signature_guard_decision = decide_repeated_signature(
            action_required=action_required,
            has_last_tool_signature=has_last_signature,
            response_repeats_signature=initial_repeats_signature,
            last_tool_success=(_resolved_tool_success(state=state, brain_input=brain_input) is True),
            corrected_repeats_signature=False,
        )
        if not repeated_signature_guard_decision.apply_guard:
            return response

        corrected_response = _invoke_with_trace(
            llm=llm,
            messages=[*pre_messages, response, _repeated_signature_correction_prompt(last_tool_signature, repeated_signature_guard_decision.repeat_reason)],
            show_raw_llm=show_raw_llm,
            color=ANSI_YELLOW,
        )

        corrected_repeats_signature = message_repeats_signature(corrected_response, last_tool_signature)
        repeated_signature_followup_decision = decide_repeated_signature(
            action_required=action_required,
            has_last_tool_signature=has_last_signature,
            response_repeats_signature=initial_repeats_signature,
            last_tool_success=(_resolved_tool_success(state=state, brain_input=brain_input) is True),
            corrected_repeats_signature=corrected_repeats_signature,
        )

        if repeated_signature_followup_decision.request_final_answer:
            
            final_response = _invoke_with_trace(
                llm=llm,
                messages=[*pre_messages, corrected_response, _repeated_success_final_answer_prompt(last_tool_signature)],
                show_raw_llm=show_raw_llm,
                color=ANSI_GREEN,
            )

            if getattr(final_response, "tool_calls", None):
                last_tool_output = _resolved_tool_output_payload(state=state, brain_input=brain_input)
                if isinstance(last_tool_output, dict):
                    return AIMessage(content=format_tool_result_response(last_tool_output))
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
        brain_input,
        response: AIMessage,
        tools_set: set[str],
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
            show_raw_llm=show_raw_llm,
            color=ANSI_RED,
        )

        decision = finalize_action_response(decision, tools_set)
        if getattr(decision, "tool_calls", None):
            return decision

        # Last-resort fallback keeps the flow deterministic when model output is unusable.
        if "Action-required run stopped" in normalize_message_content(decision):
            last_tool_output = _resolved_tool_output_payload(state=state, brain_input=brain_input)
            if isinstance(last_tool_output, dict):
                rendered = format_tool_result_response(last_tool_output)
                if rendered.strip():
                    return AIMessage(content=rendered)

        return decision
    
    def _build_context_messages(
        *,
        system_prompt: str,
        execution_policy: BrainExecutionDecision,
        retrieval_messages: list[SystemMessage],
        history: list,
        rolling_summary: str,
    ) -> list[BaseMessage]:
    
        context_messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            *(retrieval_messages if execution_policy.include_retrieval else []),
            *rolling_summary_message(rolling_summary),
        ]

        if execution_policy.action_required:
            if execution_policy.planner_brief:
                context_messages.append(SystemMessage(content=execution_policy.planner_brief))

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
        system_prompt: str,
        state: AgentState,
        brain_input,
        execution_policy: BrainExecutionDecision,

    ) -> list[BaseMessage]:
        
        history = state.get("messages", [])
        retrieval_messages = state.get("retrieval_messages", [])
        rolling_summary = state.get("rolling_summary", "")

        # Prefer protocol ToolResult, with legacy fallbacks for compatibility.
        last_tool_output = _resolved_tool_output_payload(state=state, brain_input=brain_input)
        last_tool_success = _resolved_tool_success(state=state, brain_input=brain_input)
        last_tool_rendered = str(state.get("last_tool_rendered", "") or "")
        last_tool_signature = _resolved_tool_signature(state=state, brain_input=brain_input)

        # Route guard: think to add something


        pre_messages = _build_context_messages(
            system_prompt=system_prompt,
            execution_policy=execution_policy,
            retrieval_messages=retrieval_messages,
            history=history,
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

    def _resolved_tool_output_payload(*, state: AgentState, brain_input) -> object:
        last_tool_result = getattr(brain_input, "last_tool_result", None)
        if last_tool_result is not None:
            return tool_result_to_legacy(last_tool_result)
        return state.get("last_tool_output", "")

    def _resolved_tool_success(*, state: AgentState, brain_input) -> object:
        last_tool_result = getattr(brain_input, "last_tool_result", None)
        if last_tool_result is not None:
            return last_tool_result.success
        return state.get("last_tool_success")

    def _resolved_tool_signature(*, state: AgentState, brain_input) -> str:
        last_tool_result = getattr(brain_input, "last_tool_result", None)
        if last_tool_result is not None:
            return str(last_tool_result.request_id or "")
        return str(state.get("last_tool_signature", "") or "")
    
    def _finalize_brain_response(
        *,
        llm: ChatOllama,
        pre_messages: list[BaseMessage],
        response: AIMessage,
        execution_policy: BrainExecutionDecision,
        state: AgentState,
        brain_input,
        tools_set: set[str],
    ) -> AIMessage:
        response = _recover_response_for_action_flow(
            llm=llm,
            pre_messages=pre_messages,
            response=response,
            action_required=execution_policy.action_required,
            state=state,
            brain_input=brain_input,
            tools_set=tools_set,
        )
        response = _enforce_repeated_signature_policy(
            llm=llm,
            pre_messages=pre_messages,
            response=response,
            action_required=execution_policy.action_required,
            state=state,
            brain_input=brain_input,
        )
        return response
    
    def _resolve_brain_response(
        brain_input,
        response: AIMessage,
    ) -> BrainResult:
                
        if getattr(response, "tool_calls", None):
            return _tool_request_result(
                brain_input=brain_input,
                response=response,
            )

        return _final_answer_result(response)
    
    def _resolve_execution_context(
        *, 
        brain_input,
    ):
        
        execution_policy = decide_brain_execution(brain_input)
        if execution_policy.final_answer:
            llm = brain_llm
            system_prompt = final_answer_system_prompt

        elif execution_policy.action_required:
            llm = tool_brain_llm
            system_prompt = agent_system_prompt

        else:
            llm = brain_llm
            system_prompt = casual_system_prompt

        return llm, system_prompt, execution_policy

    # def brain_node(state: AgentState):
        
    #     brain_input = build_brain_input(state)

    #     #
    #     # Controller already has a successful tool result.
    #     # The current step is finished.
    #     #
    #     if (
    #         brain_input.last_tool_result is not None
    #         and brain_input.last_tool_result.success
    #     ):
    #         brain_result = BrainResult(
    #             outcome=BrainOutcome.STEP_COMPLETED,
    #             message="Current execution step completed successfully.",
    #             proposed_step_status=StepStatus.COMPLETED,
    #         )

    #     else:

    #         llm, system_prompt, action_required, planner_brief = _resolve_execution_context(
    #             brain_input=brain_input,
    #             brain_llm=brain_llm,
    #             tool_brain_llm=tool_brain_llm,
    #             agent_system_prompt=agent_system_prompt,
    #             casual_system_prompt=casual_system_prompt
    #         )

    #         response = _resolve_brain_response(
    #             state=state,
    #             brain_input=brain_input,
    #             tools_set=tools_set,
    #             llm=llm,
    #             active_system_prompt=system_prompt,
    #             action_required=action_required,
    #             planner_brief=planner_brief
    #         )

    #         if getattr(response, "tool_calls", None):

    #             tool_call = response.tool_calls[0]

    #             brain_result = BrainResult(
    #                 outcome=BrainOutcome.TOOL_REQUEST,
    #                 message="Brain requested tool execution.",
    #                 tool_request=ToolRequest(
    #                     request_id=f"{brain_input.identity.execution_id}:tool:{uuid4().hex}",
    #                     tool_name=tool_call["name"],
    #                     arguments=tool_call.get("args", {}),
    #                     requested_by=WorkerRole.BRAIN,
    #                 ),
    #             )
    #         else:

    #             brain_result = BrainResult(
    #                 outcome=BrainOutcome.FINAL_ANSWER,
    #                 message=str(response.content),
    #                 proposed_step_status=StepStatus.COMPLETED,
    #             )

    #     legacy = brain_result_to_legacy(brain_result)
    #     legacy["brain_result"] = brain_result

    #     execution_state = build_execution_state(state)

    #     legacy["execution_state"] = with_cursor(
    #         execution_state,
    #         current_worker=WorkerRole.BRAIN,
    #     )

    #     if "response" in locals():
    #         legacy.update(
    #             response_with_usage(state, response)
    #         )

    #     return legacy

    def _handle_tool_result(
        *,
        state: AgentState,
        brain_input: BrainInput,
        llm: ChatOllama,
        system_prompt: str,
        execution_policy: BrainExecutionDecision,
    ) -> tuple[Any, BrainResult]:

        tool_result = brain_input.last_tool_result
        assert tool_result is not None

        if tool_result.success:
            return None, _step_completed()

        return _execute_step(
            state=state,
            brain_input=brain_input,
            llm=llm,
            system_prompt=system_prompt,
            execution_policy=execution_policy
        )

    def _step_completed() -> BrainResult:
        return BrainResult(
            outcome=BrainOutcome.STEP_COMPLETED,
            message="Current execution step completed successfully.",
            proposed_step_status=StepStatus.COMPLETED,
        )


    def _tool_request_result(
        brain_input: BrainInput,
        tool_call: dict,
    ) -> BrainResult:
        return BrainResult(
            outcome=BrainOutcome.TOOL_REQUEST,
            message="Brain requested tool execution.",
            tool_request=ToolRequest(
                request_id=f"{brain_input.identity.execution_id}:tool:{uuid4().hex}",
                tool_name=tool_call["name"],
                arguments=tool_call.get("args", {}),
                requested_by=WorkerRole.BRAIN,
            ),
        )


    def _final_answer_result(response: AIMessage) -> BrainResult:
        return BrainResult(
            outcome=BrainOutcome.FINAL_ANSWER,
            message="Execution completed successfully.",
            final_answer=str(response.content),
            proposed_step_status=StepStatus.COMPLETED,
        )


    def _resolve_brain_response(
        *,
        brain_input: BrainInput,
        response: AIMessage,
    ) -> BrainResult:

        if getattr(response, "tool_calls", None):
            return _tool_request_result(
                brain_input=brain_input,
                tool_call=response.tool_calls[0],
            )

        return _final_answer_result(response)

    def _invoke_brain_llm(
        *,
        state: AgentState,
        brain_input: BrainInput,
        tools_set: set[str],
        llm: ChatOllama,
        system_prompt: str,
        execution_policy: BrainExecutionDecision
    ) -> AIMessage:
        """
        Build the Brain prompt, invoke the LLM and return the raw AIMessage.

        This function owns prompt construction and model invocation only.
        """

        pre_messages = _build_pre_messages(
            system_prompt=system_prompt,
            state=state,
            brain_input=brain_input,
            execution_policy=execution_policy,
        )

        response = _invoke_with_trace(
            llm=llm,
            messages=[*pre_messages],
            show_raw_llm=show_raw_llm,
            color=ANSI_BLUE,
        )

        response = _finalize_brain_response(
            llm=llm,
            pre_messages=pre_messages,
            response=response,
            execution_policy=execution_policy,
            state=state,
            brain_input=brain_input,
            tools_set=tools_set,
        )

        return response

    def _execute_step(
        *,
        state: AgentState,
        brain_input: BrainInput,
        llm: ChatOllama,
        system_prompt: str,
        execution_policy: BrainExecutionDecision,
    ) -> tuple[AIMessage, BrainResult]:

        response = _invoke_brain_llm(
            state=state,
            brain_input=brain_input,
            tools_set=tools_set,
            llm=llm,
            system_prompt=system_prompt,
            execution_policy=execution_policy
        )

        return (
            response,
            _resolve_brain_response(
                brain_input=brain_input,
                response=response,
            ),
        )


    def _generate_final_answer(
        *,
        state: AgentState,
        brain_input: BrainInput,
        llm: ChatOllama,
        system_prompt: str,
        execution_policy: BrainExecutionDecision,
    ) -> tuple[AIMessage, BrainResult]:

        response = _invoke_brain_llm(
            state=state,
            brain_input=brain_input,
            tools_set=tools_set,
            llm=llm,
            system_prompt=system_prompt,
            execution_policy=execution_policy,
        )

        return response, _final_answer_result(response)

    def _handle_completed_tool(
        brain_input: BrainInput,
    ) -> BrainResult:
        return BrainResult(
            outcome=BrainOutcome.STEP_COMPLETED,
            message="Step completed successfully.",
            proposed_step_status=StepStatus.COMPLETED,
        )

    def brain_node(state: AgentState):

        brain_input = build_brain_input(state)

        llm, system_prompt, execution_policy = _resolve_execution_context(
            brain_input=brain_input,
        )

        response = None

        if brain_input.last_tool_result is not None:

            response, brain_result = _handle_tool_result(
                state=state,
                brain_input=brain_input,
                llm=llm,
                system_prompt=system_prompt,
                execution_policy=execution_policy,
            )

        elif execution_policy.final_answer:

            response, brain_result = _generate_final_answer(
                state=state,
                brain_input=brain_input,
                llm=llm,
                system_prompt=system_prompt,
                execution_policy=execution_policy,
            )

        else:

            response, brain_result = _execute_step(
                state=state,
                brain_input=brain_input,
                llm=llm,
                system_prompt=system_prompt,
                execution_policy=execution_policy,
            )

        legacy = brain_result_to_legacy(brain_result)
        legacy["brain_result"] = brain_result

        execution_state = build_execution_state(state)

        legacy["execution_state"] = with_cursor(
            execution_state,
            current_worker=WorkerRole.BRAIN,
        )

        if response is not None:
            legacy.update(
                response_with_usage(state, response)
            )

        # print("\n===== BRAIN RESULT =====")
        # print(brain_result)
        # print("========================\n")
        # print("\n===== BRAIN RETURN =====")
        # print(legacy)
        # print("========================\n")


        return legacy

    return brain_node