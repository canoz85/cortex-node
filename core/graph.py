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
- Keep responses concise and action-oriented.
"""


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

    def brain_node(state: AgentState):
        history = state.get("messages", [])
        response = llm.invoke([SystemMessage(content=system_prompt), *history])

        meta = getattr(response, "response_metadata", {}) or {}
        usage = TokenUsage.from_response_metadata(meta)
        update_token_usage(usage.model_dump())
        return {"messages": [response], "steps": state.get("steps", 0) + 1, "token_usage": usage}

    def capture_tool_output_node(state: AgentState):
        history = state.get("messages", [])
        if not history:
            return {"last_tool_output": ""}

        last_message = history[-1]
        if isinstance(last_message, ToolMessage):
            raw_content = str(last_message.content)
            unwrapped = unwrap_tool_output(raw_content)
            if isinstance(unwrapped, dict):
                return {"last_tool_output": unwrapped.get("message", raw_content)}
            if isinstance(unwrapped, list):
                return {"last_tool_output": str(unwrapped)}
            if isinstance(unwrapped, str):
                return {"last_tool_output": unwrapped}
            return {"last_tool_output": raw_content}
        return {"last_tool_output": state.get("last_tool_output", "")}

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
    workflow.add_node("brain", brain_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("capture_tool_output", capture_tool_output_node)

    workflow.set_entry_point("brain")
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
        "last_tool_output": "",
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
