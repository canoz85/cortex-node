import json
from typing import Any, Set

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    AIMessage, BaseMessage, 
    HumanMessage, SystemMessage, ToolMessage
)
from dataclasses import dataclass
from uuid_utils import uuid4

from core.protocol.enums import (
    BrainOutcome,
    StepStatus,
    WorkerRole,
)

from core.protocol.models import (
    BrainInput,
    BrainResult,
    ToolResult,
)

from core.protocol.bridge import build_execution_state, build_tool_request, with_cursor
from core.graph_pseudo_tools import (
    finalize_action_response,
    is_generic_json_tool_response,
    is_pseudo_tool_response,
    recover_action_response,
)

from core.graph_constants import ANSI_BLUE, ANSI_GREEN, ANSI_ITALIC, ANSI_RED, ANSI_RESET
from core.graph_messages import current_turn_messages, is_effectively_empty_response, latest_human_message, normalize_message_content

from core.graph_node_helpers import response_with_usage

from core.graph_response_formatters import format_action_completion_response
from core.graph_state_machine import (
    _build_brain_execution_brief,
    decide_action_recovery,
    decide_brain_execution,
    should_fallback_after_empty_response,
    should_retry_after_empty_response,
    BrainExecutionDecision,
)
from core.graph_summarize import rolling_summary_message
from core.protocol.bridge import build_brain_input

from core.state import AgentState

@dataclass(slots=True)
class _BrainConversationContext:
    history: list[BaseMessage]
    retrieval_messages: list[BaseMessage]
    rolling_summary: str

@dataclass(slots=True)
class _BrainExecutionContext:
    llm: ChatOllama
    messages: list[BaseMessage]
    decision: BrainExecutionDecision

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
    tool_completed_system_prompt: str,
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
        brain_input: BrainInput,
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
                brain_input=brain_input,
                response=response,
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
    
    def _finalize_from_successful_tool_context(
        *,
        llm: ChatOllama,
        pre_messages: list[BaseMessage],
        brain_input: BrainInput,
        response: AIMessage,
    ) -> AIMessage:
        """Ask the model to decide: final answer now or distinct next tool call.

        This prevents raw tool output from being echoed as the final brain response.
        """
        if getattr(response, "tool_calls", None):
            return response
        
        history = brain_input.context.recent_history

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
            tool_result = brain_input.last_tool_result

            if tool_result is not None:
                rendered = tool_result.rendered_output
                if rendered.strip():
                    return AIMessage(content=rendered)

        return decision
    
    def _build_execution_context(
        *, 
        brain_input: BrainInput,
    ) -> _BrainExecutionContext:
        
        execution_policy = decide_brain_execution(brain_input)
        if execution_policy.final_answer:
            llm = brain_llm
            system_prompt = final_answer_system_prompt
            messages = _build_final_answer_messages(
                system_prompt=system_prompt,
                brain_input=brain_input,
            )

        elif execution_policy.tool_completed:
            llm = brain_llm
            system_prompt = tool_completed_system_prompt
            messages = _build_tool_completed_messages(
                system_prompt=system_prompt,
                brain_input=brain_input,
            )
        else:
            execution_brief = ""
            if execution_policy.action_required:
                llm = tool_brain_llm
                system_prompt = agent_system_prompt

                if execution_policy.execution_brief:
                    execution_brief = execution_policy.execution_brief
            else:
                llm = brain_llm
                system_prompt = casual_system_prompt    

            retrieval_messages = (
                brain_input.context.retrieval_messages
                if execution_policy.include_retrieval
                else ()
            )
            
            messages = _build_execution_messages(
                system_prompt=system_prompt,
                brain_input=brain_input,
                retrieval_messages=retrieval_messages,
                execution_brief=execution_brief,
            )

        return _BrainExecutionContext(
            llm=llm,
            messages=messages,
            decision=execution_policy,

        )
    
    def _build_context_messages(
        *,
        system_prompt: str,
        execution_brief: str | None,
        retrieval_messages: list[SystemMessage],
        user_request: str,
    ) -> list[BaseMessage]:
    
        context_messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            *retrieval_messages,
            # *rolling_summary_message(rolling_summary),
        ]

        if execution_brief:
            context_messages.append(SystemMessage(content=execution_brief))

        if user_request:
            context_messages.append(
                HumanMessage(content=user_request)
            )

        return context_messages
    
    def _build_runtime_guidance_messages(
        *,
        tool_result: ToolResult | None,
    ) -> list[SystemMessage]:

        if tool_result is None:
            return []

        guidance: list[str] = []

        if tool_result.success is False:
            guidance.append("The previous tool execution FAILED.")

            if tool_result.request_id:
                guidance.append(f"Failed tool signature: {tool_result.request_id}")

            stderr = ""

            if isinstance(tool_result.data, dict):
                stderr = str(tool_result.data.get("stderr") or "").strip()


            if stderr:
                guidance.append(f"Runtime error detected:\n{stderr}")

                lowered = stderr.lower()

                if "the following arguments are required" in lowered:
                    guidance.append(
                        "Correct the missing required tool arguments "
                        "using the available tool schema."
                    )

                if (
                    "nameerror" in lowered
                    and "args" in lowered
                    and "not defined" in lowered
                ):
                    guidance.append(
                        "Correct the NameError by ensuring required "
                        "variables or functions are defined or imported."
                    )
        elif tool_result.success:

            has_output = (
                bool((tool_result.rendered_output or "").strip())
                or tool_result.data is not None
            )

            if has_output:
                guidance.append(
                    "The previous tool execution SUCCEEDED."
                )
                guidance.append(
                    "Treat its output as authoritative evidence for the "
                    "current execution state."
                )
                guidance.append(
                    "Inspect the result against the current active step "
                    "and continue that step only if required work remains."
                )

        if not guidance:
            return []

        return [
            SystemMessage(
                content="\n".join(
                    [
                        "### RUNTIME GUIDANCE:",
                        *guidance,
                    ]
                )
            )
        ]

    def _build_final_answer_messages(
        *,
        system_prompt: str,
        brain_input: BrainInput,
    ) -> list[BaseMessage]:
        """Build the message list used only for FINAL ANSWER generation."""

        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=brain_input.context.user_request),
        ]

        if brain_input.active_plan is not None and brain_input.active_step is None:
            messages.append(
                SystemMessage(
                    content=(
                        "Execution plan:\n"
                        f"{brain_input.active_plan.objective}"
                    )
                )
            )

            execution_records = [
                {
                    "request_id": record.result.request_id,
                    "step_id": record.step_id,
                    "tool_name": record.tool_name,
                    "arguments": record.arguments,
                    "signature": record.result.signature,
                    "success": record.result.success,
                    "message": record.result.message,
                    "rendered_output": record.result.rendered_output,
                    "data": record.result.data,
                    "error_code": record.result.error_code,
                }
                for record in brain_input.tool_execution_history
            ]

            if execution_records:
                messages.append(
                    SystemMessage(
                        content=(
                            "Execution evidence (structured):\n"
                            f"{json.dumps(execution_records, ensure_ascii=True)}"
                        )
                    )
                )

        return messages

    def _build_tool_completed_messages(
        *,
        system_prompt: str,
        brain_input: BrainInput,
    ) -> list[BaseMessage]:

        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=brain_input.context.user_request),
        ]

        execution_brief = _build_brain_execution_brief(brain_input)
        if execution_brief:
            messages.append(SystemMessage(content=execution_brief))

        messages.extend(
            _build_tool_progress_messages(
                brain_input=brain_input,
            )
        )

        return messages

    def _build_tool_progress_messages(
        *,
        brain_input: BrainInput,
    ) -> list[SystemMessage]:
        history = brain_input.tool_execution_history
        if not history:
            return []

        active_step_id = brain_input.active_step.step_id if brain_input.active_step is not None else ""

        current_step_records: list[dict[str, Any]] = []
        previous_successful_records: list[dict[str, Any]] = []

        for record in history:
            normalized = {
                "request_id": record.result.request_id,
                "step_id": record.step_id,
                "tool_name": record.tool_name,
                "arguments": record.arguments,
                "signature": record.result.signature,
                "success": record.result.success,
                "message": record.result.message,
                "rendered_output": record.result.rendered_output,
                "data": record.result.data,
                "error_code": record.result.error_code,
            }

            if record.step_id == active_step_id:
                current_step_records.append(normalized)
            elif record.result.success:
                previous_successful_records.append(normalized)

        payload = {
            "active_step_id": active_step_id,
            "current_step_tool_records": current_step_records,
            "previous_successful_tool_records": previous_successful_records,
        }

        return [
            SystemMessage(
                content=(
                    "Tool execution progress (structured):\n"
                    f"{json.dumps(payload, ensure_ascii=True)}"
                )
            )
        ]

    def _build_execution_messages(
        *,
        system_prompt: str,
        brain_input: BrainInput,
        retrieval_messages: list[SystemMessage],
        execution_brief: str | None,

    ) -> list[BaseMessage]:
        """Build the message list used for tool execution and action-required turns."""

        pre_messages = _build_context_messages(
            system_prompt=system_prompt,
            execution_brief=execution_brief,
            retrieval_messages=retrieval_messages,
            user_request=brain_input.context.user_request,
        )

        pre_messages.extend(
            _build_runtime_guidance_messages(tool_result=brain_input.last_tool_result)
        )

        pre_messages.extend(
            _build_tool_progress_messages(
                brain_input=brain_input,
            )
        )

        return pre_messages

    # def _build_brain_messages(
    #     *,
    #     system_prompt: str,
    #     conversation: _BrainConversationContext,
    #     brain_input: BrainInput,
    #     execution_policy: BrainExecutionDecision,
    # ) -> list[BaseMessage]:
        
    #     if execution_policy.final_answer:
    #         return _build_final_answer_messages(
    #             system_prompt=system_prompt,
    #             brain_input=brain_input,
    #         )

    #     if execution_policy.tool_completed:
    #         return _build_tool_completed_messages(
    #             system_prompt=system_prompt,
    #             brain_input=brain_input,
    #         )
        
    #     return _build_execution_messages(
    #         system_prompt=system_prompt,
    #         conversation=conversation,
    #         brain_input=brain_input,
    #         execution_policy=execution_policy,
    #     )

    def _build_tool_completed_result(
        *,
        response: AIMessage,
    ) -> BrainResult:

        answer = response.content.strip().upper()

        if answer == "YES":
            return BrainResult(
                outcome=BrainOutcome.STEP_COMPLETED,
                message=response.content,
                proposed_step_status=StepStatus.COMPLETED,
            )

        return BrainResult(
            outcome=BrainOutcome.CONTINUE,
            message="Execution continues after tool completion.",
        )
        

    def _build_answer_result(response: AIMessage) -> BrainResult:
        return BrainResult(
            outcome=BrainOutcome.FINAL_ANSWER,
            message="Execution completed successfully.",
            final_answer=str(response.content),
            proposed_step_status=StepStatus.COMPLETED,
        )

    def _build_brain_result(
        *,
        brain_input: BrainInput,
        response: AIMessage,
    ) -> BrainResult:

        if getattr(response, "tool_calls", None):
            return BrainResult(
                outcome=BrainOutcome.TOOL_REQUEST,
                message="Brain requested tool execution.",
                tool_request=build_tool_request(brain_input=brain_input, response=response),
            )

        return _build_answer_result(response)

    def _execute_tool_completed(
        *,
        execution_context: _BrainExecutionContext,
        brain_input: BrainInput,
    ) -> tuple[Any, BrainResult]:

        assert brain_input.last_tool_result is not None

        response = _run_brain_llm(
                conversation=None,
                brain_input=brain_input,
                execution_context=execution_context,
            )

        return response, _build_tool_completed_result(
            response=response,
        )

    def _execute_step(
        *,
        execution_context: _BrainExecutionContext,
        conversation: _BrainConversationContext,
        brain_input: BrainInput,
    ) -> tuple[AIMessage, BrainResult]:

        response = _run_brain_llm(
            conversation=conversation,
            brain_input=brain_input,
            execution_context=execution_context,
        )

        return (
            response,
            _build_brain_result(
                brain_input=brain_input,
                response=response,
            ),
        )

    def _execute_final_answer(
        *,
        brain_input: BrainInput,
        execution_context: _BrainExecutionContext,
    ) -> tuple[AIMessage, BrainResult]:

        response = _run_brain_llm(
            conversation=None,
            brain_input=brain_input,
            execution_context=execution_context,
        )

        return response, _build_answer_result(response)

    def _run_brain_llm(
        *,
        execution_context: _BrainExecutionContext,
        brain_input: BrainInput,
    ) -> AIMessage:
        """
        Build the Brain prompt, invoke the LLM and return the raw AIMessage.

        This function owns prompt construction and model invocation only.
        """

        # pre_messages = _build_brain_messages(
        #     system_prompt=execution_context.system_prompt,
        #     conversation=conversation,
        #     brain_input=brain_input,
        #     execution_policy=execution_context.decision,
        # )

        messages = execution_context.messages

        response = _invoke_with_trace(
            llm=execution_context.llm,
            messages=[*messages],
            show_raw_llm=show_raw_llm,
            color=ANSI_BLUE,
        )

        response = _finalize_brain_response(
            llm=execution_context.llm,
            pre_messages=messages,
            response=response,
            execution_policy=execution_context.decision,
            brain_input=brain_input,
        )

        return response

    
    def _execute_brain(
        *,
        brain_input: BrainInput,
        execution_context: _BrainExecutionContext,
    ):
        brain_result = None

        response = _run_brain_llm(
            brain_input=brain_input,
            execution_context=execution_context,
        )

        if execution_context.decision.final_answer:
            brain_result = _build_answer_result(response)

        elif execution_context.decision.tool_completed:
            brain_result = _build_tool_completed_result(response=response)
            response = None

            # response, brain_result = _execute_tool_completed(
            #     execution_context=execution_context,
            #     brain_input=brain_input,
            # )
        else:
            brain_result = _build_brain_result(
                brain_input=brain_input,
                response=response
            )

            # response, brain_result = _execute_step(
            #     execution_context=execution_context,
            #     conversation=conversation,
            #     brain_input=brain_input,
            # )

        return response, brain_result

    def _finalize_brain_response(
            *,
            llm: ChatOllama,
            pre_messages: list[BaseMessage],
            response: AIMessage,
            execution_policy: BrainExecutionDecision,
            brain_input: BrainInput,
        ) -> AIMessage:
            
            response = _recover_response_for_action_flow(
                llm=llm,
                pre_messages=pre_messages,
                response=response,
                action_required=execution_policy.action_required,
                brain_input=brain_input,
            )
            # todo: implement repeated signature policy enforcement for action-required turns
            # response = _enforce_repeated_signature_policy(
            #     llm=llm,
            #     pre_messages=pre_messages,
            #     response=response,
            #     action_required=execution_policy.action_required,
            #     state=state,
            #     brain_input=brain_input,
            # )
            return response
        

    def brain_node(state: AgentState):

        # conversation = _BrainConversationContext(
        #     history=state.get("messages", []),
        #     retrieval_messages=state.get("retrieval_messages", []),
        #     rolling_summary=state.get("rolling_summary", ""),
        # )

        brain_input = build_brain_input(state)

        # print("=== BRAIN INPUT ===")
        # print("cursor:", brain_input.cursor)
        # print("active_step:", brain_input.active_step)
        # print("last_tool_result:", brain_input.last_tool_result)
        # print("plan:", brain_input.active_plan)
        # print("===================")

        execution_context = _build_execution_context(
            brain_input=brain_input,
        )

        response, brain_result = _execute_brain(
            brain_input=brain_input,
            execution_context=execution_context,
        )

        execution_state = with_cursor(
            build_execution_state(state),
            current_worker=WorkerRole.BRAIN,
        )

        # ToolResult has been consumed by Brain.
        if brain_input.last_tool_result is not None:
            execution_state = execution_state.model_copy(
                update={
                    "working": execution_state.working.model_copy(
                        update={
                            "last_tool_result": None,
                        }
                    )
                }
            )

        update = {
            "brain_result": brain_result,
            "execution_state": execution_state,
        }

        if response is not None:
            update.update(
                response_with_usage(state, response)
            )

        return update

    return brain_node