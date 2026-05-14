import json
import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from core.models import TokenUsage, ToolResult
from core.state import AgentState
from core.tool_output import parse_tool_result, unwrap_tool_output
from tools.exec_ops import get_exec_tools
from tools.file_ops import get_file_tools
from tools.git_ops import get_git_tools
from tools.info_ops import get_info_tools, update_token_usage
from tools.scada_ops import get_scada_tools

SYSTEM_PROMPT_TEMPLATE = """You are CortexNode, a local-first autonomous software engineering agent.
You can reason, use tools, and iterate until the task is complete.
Runtime info:
- Model: {model}
- Context window: ~128k tokens
- Sandbox workspace: {workspace_dir}
- Max reasoning steps per prompt: 12
Constraints:
- Operate only inside the sandbox workspace directory.
- Prefer Python solutions with clear, testable code.
- For time/date requests, use the current_time tool instead of generating guessed values.
- After writing code, run it to verify behavior when possible.
- Do not print pseudo tool calls like write_file(...). If an action is needed, emit actual tool calls.
- If a tool call fails, do not repeat the same tool with identical arguments; choose a different next action.
- Keep responses concise and action-oriented.
"""

PSEUDO_TOOL_CALL_PATTERN = re.compile(
    r"\b(?:list_files|read_file|write_file|make_directory|run_python|git_status|git_diff|git_log|git_show|agent_info|token_usage|current_time|scada_status)\s*\(",
    re.IGNORECASE,
)


def _tool_signature(name: str, args: object) -> str:
    """Create a stable signature for duplicate tool-call detection."""
    try:
        args_json = json.dumps(args, sort_keys=True, ensure_ascii=True)
    except TypeError:
        args_json = str(args)
    return f"{name}:{args_json}"


def _looks_like_pseudo_tool_text(content: str) -> bool:
    return bool(PSEUDO_TOOL_CALL_PATTERN.search(content or ""))


def _extract_tool_signature(history: list, tool_call_id: str | None) -> str:
    """Find the matching tool call in message history and derive a signature."""
    if not history:
        return ""

    for message in reversed(history):
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            continue

        if tool_call_id:
            for call in tool_calls:
                if call.get("id") == tool_call_id:
                    return _tool_signature(call.get("name", "unknown"), call.get("args", {}))

        call = tool_calls[-1]
        return _tool_signature(call.get("name", "unknown"), call.get("args", {}))

    return ""


def build_app(workspace_dir: str = "workspace", model: str = "qwen2.5:7b"):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(model=model, workspace_dir=workspace_dir)

    tools = [
        *get_file_tools(workspace_dir),
        *get_exec_tools(workspace_dir),
        *get_git_tools(workspace_dir),
        *get_info_tools(model=model, workspace_dir=workspace_dir),
        *get_scada_tools(workspace_dir),
    ]

    llm = ChatOllama(model=model, temperature=0).bind_tools(tools)
    planner_llm = ChatOllama(model=model, temperature=0)  # No tools for planning

    def planner_node(state: AgentState):
        """First pass: analyze prompt and create a plan WITHOUT taking actions."""
        history = state.get("messages", [])
        
        planning_system = """You are a strategic planner. Analyze the user's request and create a clear step-by-step plan.
DO NOT take any actions yet. Just output:
1. What needs to be done (list of 2-4 key tasks)
2. File/tool sequence required
3. Expected outcome

Be concise. Format as a numbered list."""
        
        pre_messages = [
            SystemMessage(content=planning_system),
        ]
        
        plan_response = planner_llm.invoke([*pre_messages, *history])
        
        return {
            "messages": [*history, plan_response],
            "plan": str(plan_response.content),
            "steps": 0,  # Reset step counter for execution phase
            "last_tool_success": True,
            "repeat_fail_count": 0,
            "tool_text_retry_used": False,
        }

    def brain_node(state: AgentState):
        history = state.get("messages", [])
        pre_messages = [SystemMessage(content=system_prompt)]
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

        response = llm.invoke([*pre_messages, *history])

        retry_used = state.get("tool_text_retry_used", False)
        if (
            not getattr(response, "tool_calls", None)
            and _looks_like_pseudo_tool_text(str(getattr(response, "content", "")))
            and not retry_used
        ):
            response = llm.invoke(
                [
                    *pre_messages,
                    *history,
                    response,
                    SystemMessage(
                        content=(
                            "Your previous response included pseudo tool invocation text. "
                            "Respond again. If actions are needed, emit real tool calls only. "
                            "If no action is needed, provide only the final concise answer."
                        )
                    ),
                ]
            )
            retry_used = True

        meta = getattr(response, "response_metadata", {}) or {}
        usage = TokenUsage.from_response_metadata(meta)
        update_token_usage(usage.model_dump())
        return {
            "messages": [response],
            "steps": state.get("steps", 0) + 1,
            "token_usage": usage,
            "tool_text_retry_used": retry_used,
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
            success = parsed.success if parsed is not None else False
            current_signature = _extract_tool_signature(history[:-1], getattr(last_message, "tool_call_id", None))

            previous_signature = state.get("last_tool_signature", "")
            previous_success = state.get("last_tool_success", True)
            previous_repeat_count = state.get("repeat_fail_count", 0)
            if not success and current_signature and previous_signature == current_signature and not previous_success:
                repeat_fail_count = previous_repeat_count + 1
            elif not success and current_signature:
                repeat_fail_count = 1
            else:
                repeat_fail_count = 0

            unwrapped = unwrap_tool_output(raw_content)
            if isinstance(unwrapped, dict):
                return {
                    "last_tool_output": unwrapped.get("message", raw_content),
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            if isinstance(unwrapped, list):
                return {
                    "last_tool_output": str(unwrapped),
                    "last_tool_signature": current_signature,
                    "last_tool_success": success,
                    "repeat_fail_count": repeat_fail_count,
                }
            if isinstance(unwrapped, str):
                return {
                    "last_tool_output": unwrapped,
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

        if state.get("steps", 0) >= 12:
            return END

        last_message = history[-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("brain", brain_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("capture_tool_output", capture_tool_output_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "brain")
    workflow.add_conditional_edges("brain", route_after_brain)
    workflow.add_edge("tools", "capture_tool_output")
    workflow.add_edge("capture_tool_output", "brain")

    return workflow.compile()


def run_prompt(app, prompt: str, history: list | None = None) -> list:
    """Run a single prompt and return the updated message history."""
    prior_messages = history or []
    initial_state: AgentState = {
        "messages": [*prior_messages, HumanMessage(content=prompt)],
        "steps": 0,
        "plan": "",
        "last_tool_output": "",
        "last_tool_signature": "",
        "last_tool_success": True,
        "repeat_fail_count": 0,
        "tool_text_retry_used": False,
    }

    final_messages = list(initial_state["messages"])
    last_usage: dict = {}
    events = app.stream(initial_state)
    for event in events:
        for node_name, value in event.items():
            # if isinstance(value, dict) and "token_usage" in value:
            #     last_usage = value["token_usage"]

            messages = value.get("messages") if isinstance(value, dict) else None
            if not messages:
                continue

            message = messages[-1]
            final_messages.append(message)
            print(f"\n[{node_name}]")
            if getattr(message, "tool_calls", None):
                print(f"Tool calls: {message.tool_calls}")
            else:
                raw_content = str(message.content)
                summary, _ = ToolResult.split_tool_output(raw_content)
                parsed = parse_tool_result(raw_content)
                if parsed is not None:
                    print(parsed.to_pretty_text())
                elif summary:
                    print(summary)
                else:
                    print(raw_content)

            # if node_name == "brain" and last_usage:
            #     print(
            #         f"  [tokens] prompt={last_usage.get('prompt_tokens', '?')}  "
            #         f"completion={last_usage.get('completion_tokens', '?')}  "
            #         f"total={last_usage.get('total_tokens', '?')}"
            #     )

    return final_messages
