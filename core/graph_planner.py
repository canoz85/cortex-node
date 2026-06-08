from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from core.graph_constants import RECENT_MESSAGE_WINDOW
from core.graph_context import retrieval_message, rolling_summary_message, update_rolling_summary
from core.graph_intents import planner_routing_decision, preferred_info_tool
from core.graph_messages import latest_user_message, recent_messages
from core.rag import WorkspaceRAG
from core.state import AgentState


PLANNING_SYSTEM_PROMPT_TEMP = """You are a strategic planner. Analyze the user's request and create a clear step-by-step plan.
DO NOT take any actions yet. Just output:
1. What needs to be done (list of 2-4 key tasks)
2. File/tool sequence required
3. Expected outcome

Be concise. Format as a numbered list."""

PLANNING_SYSTEM_PROMPT = """You are a strategic planner for an execution agent. 
Analyze the user's request and create a step-by-step plan using only the agent's available tools.

AVAILABLE AGENT TOOLS:
{available_tools}

CRITICAL RULES:
1. If the user asks to create code/scripts and run or execute them, your plan MUST explicitly divide this into separate steps using the agent's tools:
   - A step to save the code to disk using the `write_file` tool.
   - A subsequent step to execute that script using the `run_python` tool.
2. Do not describe the internal libraries of the script (like os or pathlib) in the tool sequence. Describe what the AGENT must do with its tools.
3. DO NOT generate or output any code blocks in this phase.

Format as a numbered list:
1. Tasks to be done
2. Agent tool sequence required (e.g., write_file -> run_python)
3. Expected outcome"""

def create_planner_node(
    *,
    planner_llm: ChatOllama,
    rag_service: WorkspaceRAG,
    rag_top_k: int,
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
        routing_decision = planner_routing_decision(latest_user_prompt)
        planner_route = routing_decision.route

        if planner_route == "info":
            plan_text = f"Info query detected: call {routing_decision.reason} tool and report the result."
        elif planner_route == "clarify_domain":
            plan_text = (
                "Ambiguous domain detected: ask the user to choose SAP or Python before taking actions. "
                "Do not call tools until clarified."
            )
        elif planner_route == "casual":
            plan_text = (
                "Casual conversation detected: respond directly without tools. "
                "Use conversation context for personal facts already shared and keep the reply brief."
            )
        elif planner_route == "coding_discussion":
            plan_text = (
                "Coding discussion detected: answer directly unless a targeted tool becomes necessary. "
                "Use conversation context, retrieved knowledge, and keep the reply concise."
            )
        elif planner_route == "conversation":
            plan_text = "Conversation detected: respond directly and briefly without tools unless the user asks for concrete action."
        else:
            retrieval_messages = retrieval_message(rag_service, latest_user_prompt, rag_top_k)
            summary_message = rolling_summary_message(updated_summary)

            # Format the available tools into a clean, scannable string block
            tools_list_str = "\n".join([f"- {name}" for name in sorted(tool_name_set) if name])

            # Dynamically build the definitive prompt for this instance
            runtime_planning_prompt = PLANNING_SYSTEM_PROMPT.format(available_tools=tools_list_str)

            pre_messages = [
                SystemMessage(content=runtime_planning_prompt),
                *retrieval_messages,
                *summary_message,
            ]

            plan_response = planner_llm.invoke([*pre_messages, *recent_history])
            plan_text = str(plan_response.content)

        return {
            "plan": plan_text,
            "planner_route": routing_decision.route,
            "planner_domain": routing_decision.domain,
            "planner_confidence": routing_decision.confidence,
            "planner_domain_enforced": routing_decision.enforced,
            "rolling_summary": updated_summary,
            "steps": 0,
            "last_tool_rendered": "",
            "last_tool_success": None,
            "repeat_fail_count": 0,
            "tool_text_retry_used": False,
        }

    return planner_node