from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from core.graph_constants import RECENT_MESSAGE_WINDOW
from core.graph_context import retrieval_message, rolling_summary_message, update_rolling_summary
from core.graph_intents import planner_routing_decision, preferred_info_tool
from core.graph_messages import latest_user_message, recent_messages
from core.rag import WorkspaceRAG
from core.state import AgentState


PLANNING_SYSTEM_PROMPT = """You are a strategic planner. Analyze the user's request and create a clear step-by-step plan.
DO NOT take any actions yet. Just output:
1. What needs to be done (list of 2-4 key tasks)
2. File/tool sequence required
3. Expected outcome

Be concise. Format as a numbered list."""


def create_planner_node(
    *,
    planner_llm: ChatOllama,
    rag_service: WorkspaceRAG,
    rag_top_k: int,
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

            pre_messages = [
                SystemMessage(content=PLANNING_SYSTEM_PROMPT),
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

    return planner_node