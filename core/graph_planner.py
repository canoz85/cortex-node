from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from typing import Dict, Any, Set

from core.protocol.bridge import build_planner_input, planner_result_to_legacy

from core.graph_constants import RECENT_MESSAGE_WINDOW
from core.graph_context import retrieval_message
from core.graph_intents import planner_routing_decision
from core.protocol.models import ExecutionPlan, ExecutionStep, PlannerResult
from core.rag import WorkspaceRAG
from core.state import AgentState

CONCRETE_PLANNING_SYSTEM_PROMPT = """
You are a strategic execution planner for an autonomous AI agent.
Your goal is to turn the user's request into a strict, executable plan based on the upstream classification.

ROUTER CONTEXT:
- Target Route: {route}
- Target Domain: {domain}
- Intent Justification: {reason}

AVAILABLE AGENT TOOLS:
{available_tools}

PLANNING RULES:
1. Generate between 3 and 5 logical, sequential steps.
2. If Target Route is "conversation", do NOT assign tools to steps. Write conversational step titles (e.g., "Synthesize explanation").
3. If Target Route is "info" or "action", EVERY step MUST explicitly name the tool to use in the description (e.g., "Use `rag_search` to...").
4. Keep steps strictly aligned with the "{domain}" domain. Do not invoke unrelated tools.
5. Keep execution safe: perform read/validation steps before state-changing write actions.

FORMAT REQUIREMENT:
Output ONLY a numbered list in this exact format (no markdown code blocks, no introduction, no conversational wrap-up):

1. <title (≤10 words)> – <description (≤30 words)>
2. <title (≤10 words)> – <description (≤30 words)>
3. <title (≤10 words)> – <description (≤30 words)>
"""

# Static set for non-tool/direct response routes
DIRECT_RESPONSE_ROUTES: Set[str] = {
    "conversation",
    "casual",
    "coding_discussion",
    "info",
    "clarify_domain",
}

def create_planner_node(
    *,
    planner_llm: ChatOllama,
    router_llm: ChatOllama | None = None,
    rag_service: WorkspaceRAG,
    rag_top_k: int,
    tool_name_set: set[str],
):

    def _decide_route_and_plan(
        latest_user_prompt: str,
        router_llm: ChatOllama | None,
        planner_llm: ChatOllama,
        tool_name_set: set[str],
        rag_service: WorkspaceRAG,
        rag_top_k: int,
    ) -> tuple[Any, str, list]:
        """Decides the routing direction and constructs the initial objective plan text."""
        
        routing_decision = planner_routing_decision(
            latest_user_prompt, 
            router_llm=router_llm, 
            tool_name_set=tool_name_set
        )
        planner_route = routing_decision.route
        retrieval_messages = []

        # 1. Direct Conversational / Non-Tool Routes
        if planner_route == "conversation":
            plan_text = (
                "You are in standard conversation mode. Under no circumstances should you "
                "invoke any function, plugin, or external tool. Answer the user directly using text only."
                f"ROUTER CONTEXT: {routing_decision.reason}"
            )
        elif planner_route == "clarify_domain" or routing_decision.needs_clarification:
            plan_text = (
                "The request is ambiguous or spans multiple domains. "
                "Ask a targeted clarifying question to determine the user's explicit intent."
            )
        # 2. Tool / Execution Required Routes
        else:
            tools_list_str = "\n".join([f"- {name}" for name in sorted(tool_name_set) if name])
            runtime_planning_prompt = CONCRETE_PLANNING_SYSTEM_PROMPT.format(
                available_tools=tools_list_str,
                route=planner_route,
                domain=routing_decision.domain,
                reason=routing_decision.reason,
            )

            retrieval_messages = retrieval_message(rag_service, latest_user_prompt, rag_top_k)
            pre_messages = [
                SystemMessage(content=runtime_planning_prompt),
                *retrieval_messages,
                HumanMessage(content=latest_user_prompt),
            ]

            plan_response = planner_llm.invoke(pre_messages)
            plan_text = str(plan_response.content)

        return routing_decision, plan_text, retrieval_messages
    
    def planner_node(state: AgentState):
        """First pass: analyze prompt and create a plan WITHOUT taking actions."""

        planner_input = build_planner_input(state)
        latest_user_prompt = planner_input.context.user_request

        # Run decision layer
        routing_decision, plan_text, retrieval_messages = _decide_route_and_plan(
            latest_user_prompt=latest_user_prompt,
            router_llm=router_llm,
            planner_llm=planner_llm,
            tool_name_set=tool_name_set,
            rag_service=rag_service,
            rag_top_k=rag_top_k,
        )

        planner_route = routing_decision.route
        active_plan = planner_input.active_plan

        # Construct single vs multi-step plan depending on route
        if planner_route in DIRECT_RESPONSE_ROUTES:
            steps = (
                ExecutionStep(
                    step_id="step-1",
                    title="Respond",
                    description="Respond directly to the user.",
                ),
            )
        else:
            steps = (
                ExecutionStep(
                    step_id="step-1",
                    title="Execute plan",
                    description="Perform the first planned action.",
                ),
                ExecutionStep(
                    step_id="step-2",
                    title="Finalize",
                    description="Complete the request and return the final answer.",
                    depends_on_step_ids=("step-1",),
                ),
            )

        # Build structured execution plan
        planner_result = PlannerResult(
            proposed_plan=ExecutionPlan(
                plan_id=(
                    active_plan.plan_id
                    if active_plan
                    else f"{planner_input.identity.execution_id}:plan"
                ),
                revision=(active_plan.revision + 1 if active_plan else 1),
                objective=plan_text,
                steps=steps,
            ),
            message="Plan generated successfully.",
            planning_rationale=(
                f"Route '{planner_route}' selected with "
                f"confidence {routing_decision.confidence:.2f}."
            ),
        )

        # TODO(CEP-006):
        # Remove legacy bridge after Controller consumes PlannerResult directly.
        legacy = planner_result_to_legacy(planner_result)

        legacy.update({
            "planner_result": planner_result,
            "retrieval_messages": retrieval_messages,
            "planner_route": routing_decision.route,
            "planner_domain": routing_decision.domain,
            "planner_confidence": routing_decision.confidence,
            "planner_domain_enforced": routing_decision.enforced,
            "planner_route_source": routing_decision.source,
            "planner_needs_clarification": routing_decision.needs_clarification,
            "steps": 0, # TODO: remove after controller migration
            "last_tool_rendered": "",
            "last_tool_success": None,
            "repeat_fail_count": 0,
            "tool_text_retry_used": False,
        })

        print(
            "PLANNER RETURN:",
            planner_result.proposed_plan.plan_id,
            id(planner_result),
        )

        return legacy

    return planner_node