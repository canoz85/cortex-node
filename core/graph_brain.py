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
    system_prompt: str
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
        conversation: _BrainConversationContext,
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
                conversation=conversation,
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
        conversation: _BrainConversationContext,
        brain_input: BrainInput,
        response: AIMessage,
    ) -> AIMessage:
        """Ask the model to decide: final answer now or distinct next tool call.

        This prevents raw tool output from being echoed as the final brain response.
        """
        if getattr(response, "tool_calls", None):
            return response
        
        history = conversation.messages

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

            lowered = stderr.lower()

            if stderr:
                guidance.append(f"Runtime error detected:\n{stderr}")

            if "the following arguments are required" in lowered:
                guidance.append(
                    "Fix missing required tool arguments based on tool schema."
                )

            if (
                "nameerror" in lowered
                and "args" in lowered
                and "not defined" in lowered
            ):
                guidance.append(
                    "Fix NameError: ensure all variables/functions are defined or imported."
                )

        has_usable_success_output = (
            bool((tool_result.rendered_output or "").strip())
            or tool_result.data is not None
        )

        if tool_result.success and has_usable_success_output:
            guidance.append(
                "A tool has already succeeded during this user turn. "
                "Use the tool output as the authoritative source of truth for the current state of the world. "
                "If that successful result satisfies the request, provide the final concise answer now instead of calling more tools. "
                "Only call another tool if a specific remaining gap still exists that can be directly addressed by one more tool call."
            )

        if not guidance:
            return []

        return [
            SystemMessage(
                content="\n".join(
                    [
                        "### RUNTIME INSTRUCTIONS (HIGHEST PRIORITY):",
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

        if brain_input.active_plan is not None:
            messages.append(
                SystemMessage(
                    content=(
                        "Execution plan:\n"
                        f"{brain_input.active_plan.objective}"
                    )
                )
            )

        if brain_input.last_tool_result is not None:
            messages.append(
                SystemMessage(
                    content=(
                        "Latest tool result:\n"
                        f"{brain_input.last_tool_result.rendered_output}"
                    )
                )
            )

        return messages

    def _build_execution_messages(
        *,
        conversation: _BrainConversationContext,
        system_prompt: str,
        brain_input: BrainInput,
        execution_policy: BrainExecutionDecision,
    ) -> list[BaseMessage]:
        """Build the message list used for tool execution and action-required turns."""

        pre_messages = _build_context_messages(
            system_prompt=system_prompt,
            execution_policy=execution_policy,
            retrieval_messages=conversation.retrieval_messages,
            history=conversation.history,
            rolling_summary=conversation.rolling_summary,
        )

        pre_messages.extend(
            _build_runtime_guidance_messages(tool_result=brain_input.last_tool_result)
        )

        return pre_messages

    def _build_brain_messages(
        *,
        system_prompt: str,
        conversation: _BrainConversationContext,
        brain_input: BrainInput,
        execution_policy: BrainExecutionDecision,
    ) -> list[BaseMessage]:
        
        if execution_policy.final_answer:
            return _build_final_answer_messages(
                system_prompt=system_prompt,
                brain_input=brain_input,
            )
        
        return _build_execution_messages(
            system_prompt=system_prompt,
            conversation=conversation,
            brain_input=brain_input,
            execution_policy=execution_policy,
        )
    
    def _finalize_brain_response(
        *,
        llm: ChatOllama,
        pre_messages: list[BaseMessage],
        response: AIMessage,
        execution_policy: BrainExecutionDecision,
        conversation: _BrainConversationContext,
        brain_input: BrainInput,
    ) -> AIMessage:
        
        response = _recover_response_for_action_flow(
            llm=llm,
            pre_messages=pre_messages,
            response=response,
            action_required=execution_policy.action_required,
            conversation=conversation,
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
    
    def _build_execution_context(
        *, 
        brain_input: BrainInput,
    ) -> _BrainExecutionContext:
        
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

        return _BrainExecutionContext(
            llm=llm,
            system_prompt=system_prompt,
            decision=execution_policy,
        )

    def _handle_tool_result(
        *,
        execution_context: _BrainExecutionContext,
        conversation: _BrainConversationContext,
        brain_input: BrainInput,
    ) -> tuple[Any, BrainResult]:

        tool_result = brain_input.last_tool_result
        assert tool_result is not None

        if tool_result.success:
            return None, _build_step_completed_result()

        return _execute_step(
            conversation=conversation,
            brain_input=brain_input,
            execution_context=execution_context
        )

    def _build_step_completed_result() -> BrainResult:
        return BrainResult(
            outcome=BrainOutcome.STEP_COMPLETED,
            message="Current execution step completed successfully.",
            proposed_step_status=StepStatus.COMPLETED,
        )

    def _final_answer_result(response: AIMessage) -> BrainResult:
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

        return _final_answer_result(response)

    def _run_brain_llm(
        *,
        execution_context: _BrainExecutionContext,
        conversation: _BrainConversationContext | None,
        brain_input: BrainInput,
    ) -> AIMessage:
        """
        Build the Brain prompt, invoke the LLM and return the raw AIMessage.

        This function owns prompt construction and model invocation only.
        """

        pre_messages = _build_brain_messages(
            system_prompt=execution_context.system_prompt,
            conversation=conversation,
            brain_input=brain_input,
            execution_policy=execution_context.decision,
        )

        response = _invoke_with_trace(
            llm=execution_context.llm,
            messages=[*pre_messages],
            show_raw_llm=show_raw_llm,
            color=ANSI_BLUE,
        )

        response = _finalize_brain_response(
            llm=execution_context.llm,
            conversation=conversation,
            pre_messages=pre_messages,
            response=response,
            execution_policy=execution_context.decision,
            brain_input=brain_input,
        )

        return response

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

    def _generate_final_answer(
        *,
        brain_input: BrainInput,
        execution_context: _BrainExecutionContext,
    ) -> tuple[AIMessage, BrainResult]:

        response = _run_brain_llm(
            conversation=None,
            brain_input=brain_input,
            execution_context=execution_context,
        )

        return response, _final_answer_result(response)

    def _execute_brain(
        *,
        conversation: _BrainConversationContext,
        brain_input: BrainInput,
        execution_context: _BrainExecutionContext,
    ):
        response = None

        if execution_context.decision.final_answer:

            response, brain_result = _generate_final_answer(
                brain_input=brain_input,
                execution_context=execution_context,
            )
        elif brain_input.last_tool_result is not None:

            response, brain_result = _handle_tool_result(
                execution_context=execution_context,
                brain_input=brain_input,
                conversation=conversation,
            )
        else:

            response, brain_result = _execute_step(
                execution_context=execution_context,
                conversation=conversation,
                brain_input=brain_input,
            )

        return response, brain_result

    def brain_node(state: AgentState):

        conversation = _BrainConversationContext(
            history=state.get("messages", []),
            retrieval_messages=state.get("retrieval_messages", []),
            rolling_summary=state.get("rolling_summary", ""),
        )

        brain_input = build_brain_input(state)

        execution_context = _build_execution_context(
            brain_input=brain_input,
        )

        response, brain_result = _execute_brain(
            conversation=conversation,
            brain_input=brain_input,
            execution_context=execution_context,
        )

        update = {
            "brain_result": brain_result,
            "execution_state": with_cursor(
                build_execution_state(state),
                current_worker=WorkerRole.BRAIN,
            ),
        }

        if response is not None:
            update.update(
                response_with_usage(state, response)
            )

        return update

    return brain_node