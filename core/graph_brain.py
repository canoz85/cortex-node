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
    ActionRecoveryKind,
    _build_brain_execution_brief,
    decide_action_recovery,
    decide_brain_execution,
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
    step_completed_system_prompt: str,
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

    def _retry_empty_response(
        *,
        llm: ChatOllama,
        pre_messages: list[BaseMessage],
        action_required: bool,
        brain_input: BrainInput,
    ) -> AIMessage:
        retry_response = _invoke_with_trace(
            llm=llm,
            messages=[*pre_messages, _empty_response_retry_prompt()],
            show_raw_llm=show_raw_llm,
            color=ANSI_GREEN,
        )

        return _recover_response_for_action_flow(
            llm=llm,
            pre_messages=pre_messages,
            response=retry_response,
            action_required=action_required,
            brain_input=brain_input,
            allow_empty_retry=False,
        )

    def _recover_response_for_action_flow(
        *,
        llm: ChatOllama,
        pre_messages: list[BaseMessage],
        response: AIMessage,
        action_required: bool,
        brain_input: BrainInput,
        allow_empty_retry: bool = True,
    ) -> AIMessage:
        recovered_action_response = (
            recover_action_response(response, tools_set)
            if action_required
            else None
        )
        response_is_empty = is_effectively_empty_response(response)
        action_recovery_decision = decide_action_recovery(
            action_required=action_required,
            recovered_action_response_exists=recovered_action_response is not None,
            pseudo_tool_response_detected=is_pseudo_tool_response(response),
            generic_json_tool_response_detected=is_generic_json_tool_response(response),
            response_is_empty=response_is_empty,
        )

        if action_recovery_decision.kind is ActionRecoveryKind.RECOVERED_ACTION:
            if recovered_action_response is not None:
                return recovered_action_response
            return response

        if action_recovery_decision.kind is ActionRecoveryKind.PSEUDO_TOOL_FALLBACK:
            return _pseudo_tool_fallback_response()

        if action_recovery_decision.kind is ActionRecoveryKind.GENERIC_JSON_FINALIZATION:
            finalized_response = _finalize_from_successful_tool_context(
                llm=llm,
                pre_messages=pre_messages,
                brain_input=brain_input,
                response=response,
            )
            if not is_effectively_empty_response(finalized_response):
                return finalized_response

            if not allow_empty_retry:
                return _empty_response_fallback()

            return _retry_empty_response(
                llm=llm,
                pre_messages=pre_messages,
                action_required=action_required,
                brain_input=brain_input,
            )

        if action_recovery_decision.kind is not ActionRecoveryKind.RETRY_EMPTY:
            return response

        if not allow_empty_retry:
            return _empty_response_fallback()

        return _retry_empty_response(
            llm=llm,
            pre_messages=pre_messages,
            action_required=action_required,
            brain_input=brain_input,
        )
    
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

        instruction_brief: str | None = None
        execution_policy = decide_brain_execution(brain_input)
        if execution_policy.is_final_answer:
            llm = brain_llm
            system_prompt = final_answer_system_prompt
            messages = _build_final_answer_messages(
                system_prompt=system_prompt,
                brain_input=brain_input,
            )

        elif execution_policy.is_step_completed:
            llm = brain_llm
            system_prompt = step_completed_system_prompt
            if execution_policy.instruction_brief:
                instruction_brief = execution_policy.instruction_brief  

            messages = _build_step_completed_messages(
                system_prompt=system_prompt,
                brain_input=brain_input,
                instruction_brief=instruction_brief
            )
        else:
            if execution_policy.has_action:
                llm = tool_brain_llm
                system_prompt = agent_system_prompt

                if execution_policy.instruction_brief:
                    instruction_brief = execution_policy.instruction_brief
            else:
                llm = brain_llm
                system_prompt = casual_system_prompt    

            retrieval_messages = (
                brain_input.context.retrieval_messages
                if execution_policy.needs_retrieval
                else ()
            )
            
            messages = _build_execution_messages(
                system_prompt=system_prompt,
                brain_input=brain_input,
                retrieval_messages=retrieval_messages,
                instruction_brief=instruction_brief,
            )

        return _BrainExecutionContext(
            llm=llm,
            messages=messages,
            decision=execution_policy,

        )
    
    def _build_context_messages(
        *,
        system_prompt: str,
        instruction_brief: str | None,
        retrieval_messages: list[SystemMessage],
        user_request: str,
    ) -> list[BaseMessage]:
    
        context_messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            *retrieval_messages,
            # *rolling_summary_message(rolling_summary),
        ]

        if instruction_brief:
            context_messages.append(SystemMessage(content=instruction_brief))

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

    def _build_step_completed_messages(
        *,
        system_prompt: str,
        brain_input: BrainInput,
        instruction_brief: str | None = None,
    ) -> list[BaseMessage]:

        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=brain_input.context.user_request),
        ]

        if instruction_brief:
            messages.append(SystemMessage(content=instruction_brief))

        messages.extend(
            _build_step_progress_messages(
                brain_input=brain_input,
            )
        )

        return messages

    # def _build_tool_progress_messages(
    #     *,
    #     brain_input: BrainInput,
    # ) -> list[SystemMessage]:
    #     history = brain_input.tool_execution_history
    #     if not history:
    #         return []

    #     active_step_id = brain_input.active_step.step_id if brain_input.active_step is not None else ""

    #     current_step_records: list[dict[str, Any]] = []
    #     previous_successful_records: list[dict[str, Any]] = []

    #     for record in history:
    #         normalized = {
    #             "request_id": record.result.request_id,
    #             "step_id": record.step_id,
    #             "tool_name": record.tool_name,
    #             "arguments": record.arguments,
    #             "signature": record.result.signature,
    #             "success": record.result.success,
    #             "message": record.result.message,
    #             "rendered_output": record.result.rendered_output,
    #             "data": record.result.data,
    #             "error_code": record.result.error_code,
    #         }

    #         if record.step_id == active_step_id:
    #             current_step_records.append(normalized)
    #         elif record.result.success:
    #             previous_successful_records.append(normalized)

    #     payload = {
    #         "active_step_id": active_step_id,
    #         "current_step_tool_records": current_step_records,
    #         "previous_successful_tool_records": previous_successful_records,
    #     }

    #     return [
    #         SystemMessage(
    #             content=(
    #                 "Tool execution progress (structured):\n"
    #                 f"{json.dumps(payload, ensure_ascii=True)}"
    #             )
    #         )
    #     ]


    def _build_step_progress_messages(
        *,
        brain_input: BrainInput,
    ) -> list[SystemMessage]:
        history = brain_input.tool_execution_history
        if not history:
            return []

        max_current_records = 24
        max_prior_records = 36
        max_text_chars = 4000
        max_list_items = 100

        active_step = brain_input.active_step
        active_step_id = active_step.step_id if active_step is not None else None

        def truncate_text(value: str) -> tuple[str, bool]:
            text = value.strip()
            if len(text) <= max_text_chars:
                return text, False

            return (
                f"{text[:max_text_chars]}\n...[truncated]",
                True,
            )

        def sanitize_stderr(value: str) -> str:
            lines = value.splitlines()
            useful_lines: list[str] = []

            for line in lines:
                lowered = line.lower()

                if (
                    "debugpy" in lowered
                    or "pydevd" in lowered
                    or "debugpy._vendored" in lowered
                    or "pydevd_frame_evaluator" in lowered
                ):
                    continue

                useful_lines.append(line)

            sanitized = "\n".join(useful_lines).strip()
            truncated, _ = truncate_text(sanitized)
            return truncated

        def bounded_value(
            value: Any,
            *,
            field_name: str | None = None,
        ) -> Any:
            if isinstance(value, str):
                text, was_truncated = truncate_text(value)

                if field_name == "content" and was_truncated:
                    return {
                        "value": text,
                        "content_chars": len(value),
                        "content_truncated": True,
                    }

                return text

            if isinstance(value, dict):
                bounded: dict[str, Any] = {}

                for key, item in value.items():
                    key_text = str(key)

                    if key_text == "stderr" and isinstance(item, str):
                        bounded[key_text] = sanitize_stderr(item)
                    else:
                        bounded[key_text] = bounded_value(
                            item,
                            field_name=key_text,
                        )

                return bounded

            if isinstance(value, (list, tuple)):
                bounded_items = [
                    bounded_value(item)
                    for item in value[:max_list_items]
                ]

                if len(value) > max_list_items:
                    bounded_items.append(
                        f"... {len(value) - max_list_items} additional items omitted"
                    )

                return bounded_items

            return value

        def evidence_for(record: Any) -> Any:
            result = record.result

            # Prefer structured data. Fall back to rendered output only when
            # the tool did not produce structured data.
            if result.data is not None:
                return bounded_value(result.data)

            rendered_output = (result.rendered_output or "").strip()
            if rendered_output:
                return bounded_value(rendered_output)

            return None

        def success_record(record: Any) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "tool": record.tool_name,
                "args": bounded_value(record.arguments),
            }

            if record.tool_name == "read_file":
                result = record.result
                rendered = (result.rendered_output or "").strip()

                if rendered:
                    payload["evidence"] = rendered
                elif result.data is not None:
                    payload["evidence"] = result.data
            else:
                evidence = evidence_for(record)
                if evidence is not None:
                    payload["evidence"] = evidence

            return payload

        def failure_record(record: Any) -> dict[str, Any]:
            result = record.result
            error: dict[str, Any] = {}

            if result.error_code:
                error["code"] = result.error_code

            if result.message:
                message, _ = truncate_text(result.message)
                error["message"] = message

            if isinstance(result.data, dict):
                stderr = result.data.get("stderr")
                if isinstance(stderr, str) and stderr.strip():
                    error["stderr"] = sanitize_stderr(stderr)

                details = {
                    key: value
                    for key, value in result.data.items()
                    if key not in {"stderr", "stdout", "traceback"}
                }

                if details:
                    error["details"] = bounded_value(details)

            payload: dict[str, Any] = {
                "step": record.step_id,
                "tool": record.tool_name,
                "args": bounded_value(record.arguments),
                "error": error,
            }

            return payload

        current_attempts: list[dict[str, Any]] = []
        prior_facts: list[dict[str, Any]] = []
        prior_failures: list[dict[str, Any]] = []

        for record in history:
            result = record.result

            if record.step_id == active_step_id:
                if result.success:
                    current_attempts.append(success_record(record))
                else:
                    current_attempts.append(failure_record(record))
                continue

            if result.success:
                prior_facts.append(
                    {
                        "step": record.step_id,
                        **success_record(record),
                    }
                )
            else:
                prior_failures.append(failure_record(record))

        payload = {
            "schema": 1,
            "active_step": (
                {
                    "id": active_step.step_id,
                    "title": active_step.title,
                }
                if active_step is not None
                else None
            ),
            "current_attempts": current_attempts[-max_current_records:],
            "prior_facts": prior_facts[-max_prior_records:],
            "prior_failures": prior_failures[-max_prior_records:],
        }

        return [
            SystemMessage(
                content=(
                    "Execution evidence v1:\n"
                    f"{json.dumps(payload, ensure_ascii=True, separators=(',', ':'))}"
                )
            )
        ]

    def _build_execution_messages(
        *,
        system_prompt: str,
        brain_input: BrainInput,
        retrieval_messages: list[SystemMessage],
        instruction_brief: str | None,

    ) -> list[BaseMessage]:
        """Build the message list used for tool execution and action-required turns."""

        pre_messages = _build_context_messages(
            system_prompt=system_prompt,
            instruction_brief=instruction_brief,
            retrieval_messages=retrieval_messages,
            user_request=brain_input.context.user_request,
        )

        pre_messages.extend(
            _build_runtime_guidance_messages(tool_result=brain_input.last_tool_result)
        )

        pre_messages.extend(
            _build_step_progress_messages(
                brain_input=brain_input,
            )
        )

        return pre_messages

    def _build_step_completed_result(
        *,
        response: AIMessage,
    ) -> BrainResult:

        answer = response.content.strip().upper()

        if "STEP COMPLETED" in answer:
            return BrainResult(
                outcome=BrainOutcome.STEP_COMPLETED,
                message=response.content,
                proposed_step_status=StepStatus.COMPLETED,
            )

        return BrainResult(
            outcome=BrainOutcome.STEP_FAILED,
            message=response.content,
            proposed_step_status=StepStatus.FAILED,

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

        return _build_step_completed_result(response=response)

    def _run_brain_llm(
        *,
        execution_context: _BrainExecutionContext,
        brain_input: BrainInput,
    ) -> AIMessage:
        """
        Build the Brain prompt, invoke the LLM and return the raw AIMessage.

        This function owns prompt construction and model invocation only.
        """

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

        if execution_context.decision.is_final_answer:
            brain_result = _build_answer_result(response)

        # elif execution_context.decision.is_step_completed:
        #     brain_result = _build_step_completed_result(response=response)
        #     response = None

        else:
            brain_result = _build_brain_result(
                brain_input=brain_input,
                response=response
            )

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
                action_required=execution_policy.has_action,
                brain_input=brain_input,
            )
            # todo: implement repeated signature policy enforcement for action-required turns
            # response = _enforce_repeated_signature_policy(
            #     llm=llm,
            #     pre_messages=pre_messages,
            #     response=response,
            #     action_required=execution_policy.has_action,
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