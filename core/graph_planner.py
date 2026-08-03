import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from typing import Any, Set

from core.protocol.bridge import build_execution_state, build_planner_input, planner_result_to_legacy, with_cursor

from core.graph_constants import MUTATING_TOOLS, DOMAIN_TOOL_MAP, SYSTEM_CAPABILITIES_TEXT
from core.graph_context import retrieval_message
from core.graph_intents import RoutingDecision, planner_routing_decision
from core.protocol.enums import PlannerOutcome, StepStatus, WorkerRole
from core.protocol.models import ExecutionPlan, ExecutionStep, PlannerInput, PlannerResult
from core.rag import WorkspaceRAG
from core.state import AgentState


CONCRETE_PLANNING_SYSTEM_PROMPT_v1 = f"""
You are a strategic execution planner for an autonomous AI agent.
Your goal is to turn the user's request into a strict, executable plan based on the upstream classification.

ROUTER CONTEXT:
- Target Route: {{route}}
- Target Domain: {{domain}}
- Intent Justification: {{reason}}  

SYSTEM CAPABILITIES & AVAILABLE TOOLS:
{SYSTEM_CAPABILITIES_TEXT}

FILTERED TOOL SET FOR THIS TURN:
{{available_tools}}

PLANNING RULES:
1. Generate between 3 and 5 logical, sequential steps.
2. If Target Route is "conversation", do NOT assign tools to steps. Write conversational step titles (e.g., "Synthesize explanation").
3. If Target Route is "info" or "action", EVERY step MUST explicitly name the tool to use in the description (e.g., "Use `rag_search` to...").
4. Keep steps strictly aligned with the "{{domain}}" domain. Do not invoke unrelated tools.
5. Keep execution safe: perform read/validation steps before state-changing write actions.

FORMAT REQUIREMENT:
Output ONLY a numbered list in this exact format (no markdown code blocks, no introduction, no conversational wrap-up):

1. <title (≤10 words)> – <description (≤30 words)>
2. <title (≤10 words)> – <description (≤30 words)>
3. <title (≤10 words)> – <description (≤30 words)>
"""

CONCRETE_PLANNING_SYSTEM_PROMPT = f"""
You are the Planner worker of CortexNode.

Your responsibility is to transform a user request into a deterministic execution plan.
You NEVER execute tools.
You NEVER answer the user.
You ONLY produce the execution plan.

ROUTER CONTEXT
--------------
Route: {{route}}
Domain: {{domain}}
Reason: {{reason}}

SYSTEM CAPABILITIES
-------------------
{SYSTEM_CAPABILITIES_TEXT}

AVAILABLE TOOLS FOR THIS REQUEST
--------------------------------
{{available_tools}}

PLANNING RULES
--------------
1. Produce the minimum number of sequential execution steps required to complete the request.

2. Every step must represent ONE logical objective.

3. If Route == "conversation":
   - produce conversational reasoning steps
   - do not mention tools

4. If Route == "info" or "action":
   - every executable step MUST specify the primary tool that should be used.
   - use only tools from Available Tools.
   - do not invent tool names.
   - never introduce additional tool calls solely to increase the number of steps.

5. Prefer safe execution order:
   validate → inspect → modify → verify

6. Do not merge unrelated actions into one step.

7. Describe WHAT should be accomplished with the tool, not HOW to invoke it.
   - Do not include tool arguments.
   - Do not include filenames unless explicitly required by the user.
   - Do not include code, commands, JSON, queries, prompts, or parameter values.
   - Leave execution details to the Brain worker.
   
8. Do not explain the plan.

9. Do not include any text outside the numbered steps.

10. Do not include maintenance, setup, or initialization tools unless the user explicitly requested them or they are strictly required to complete the task.


OUTPUT FORMAT
-------------
Return ONLY:

1. <Short title> – <Short description including tool if applicable>
2. <Short title> – <Short description including tool if applicable>
3. <Short title> – <Short description including tool if applicable>

Example:

1. Create directories – Use `make_directory` to create the required folders.
2. Create README files – Use `write_file` to add README.md files.
3. Verify structure – Use `list_files` to verify the created layout.
"""

# Static set for non-tool/direct response routes
DIRECT_RESPONSE_ROUTES: Set[str] = {
    "conversation",
    "clarify_domain",
}

STEP_RE = re.compile(
    r"^\s*(\d+)\.\s+(.*?)\s+–\s+(.*)$"
)

TOOL_RE = re.compile(
    r"Use\s+`([^`]+)`",
    re.IGNORECASE,
)

def create_planner_node(
    *,
    planner_llm: ChatOllama,
    router_llm: ChatOllama | None = None,
    rag_service: WorkspaceRAG,
    rag_top_k: int,
    tools_set: Set[str],
):



    def _build_execution_steps(
        planner_text: str,
    ) -> tuple[ExecutionStep, ...]:
        """
        Convert planner output into ExecutionStep objects.

        Expected planner format:

        1. Create project structure – Use `run_python` to create ...
        2. Verify structure – Use `list_files` to verify ...
        """

        steps: list[ExecutionStep] = []

        previous_step_id: str | None = None

        for line in planner_text.splitlines():
            line = line.strip()

            if not line:
                continue

            match = STEP_RE.match(line)
            if not match:
                continue

            number, title, description = match.groups()

            step_id = f"step-{number}"

            tool_match = TOOL_RE.search(description)
            primary_tool = tool_match.group(1) if tool_match else None

            depends_on: tuple[str, ...] = ()
            if previous_step_id:
                depends_on = (previous_step_id,)

            steps.append(
                ExecutionStep(
                    step_id=step_id,
                    title=title.strip(),
                    description=description.strip(),
                    primary_tool=primary_tool,
                    status=StepStatus.PENDING,
                    attempt=0,
                    depends_on_step_ids=depends_on,
                )
            )

            previous_step_id = step_id

        if not steps:
            raise ValueError("Planner produced no executable steps.")

        return tuple(steps)

    def _get_filtered_tools(
        all_tools: Set[str], 
        route: str, 
        domain: str
    ) -> Set[str]:
        """Filters tools based on the active domain and safety constraints."""
        
        # 1. Filter by Domain (Default to all if domain unknown)
        domain_allowed = DOMAIN_TOOL_MAP.get(domain, all_tools)
        filtered = all_tools.intersection(domain_allowed)
        
        # Always include ubiquitous environment tools
        filtered.update({"current_time", "agent_info", "token_usage"})
        
        # 2. Safety Rule: If route is read-only ('info'), strip mutating tools
        if route == "info":
            filtered = filtered - MUTATING_TOOLS

        return filtered

    def _decide_route_and_plan(
        latest_user_prompt: str,
        router_llm: ChatOllama | None,
        planner_llm: ChatOllama,
        tools_set: Set[str],
        rag_service: WorkspaceRAG,
        rag_top_k: int,
    ) -> tuple[Any, str, list]:
        """Decides the routing direction and constructs the initial objective plan text."""
        
        routing_decision = planner_routing_decision(
            latest_user_prompt, 
            router_llm=router_llm
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
            filtered_tools_set = _get_filtered_tools(
                all_tools=tools_set,
                route=planner_route,
                domain=routing_decision.domain,
            )
            tools_list_str = (
                "\n".join([f"- {name}" for name in sorted(filtered_tools_set) if name])
                or "- No tool access allowed for this step"
)
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

    def _build_planner_result(
        *,
        planner_input: PlannerInput,
        routing_decision: RoutingDecision,
        active_plan: ExecutionPlan | None,
        plan_text: str,
    ) -> PlannerResult:

        route = routing_decision.route

        rationale = (
            f"Route '{route}' selected with "
            f"confidence {routing_decision.confidence:.2f}."
        )

        if route in DIRECT_RESPONSE_ROUTES:
            return PlannerResult(
                outcome=PlannerOutcome.DIRECT_RESPONSE,
                proposed_plan=None,
                message="No execution plan required.",
                planning_rationale=rationale,
            )

        steps = _build_execution_steps(plan_text)
        if not steps:
            steps = (
                ExecutionStep(
                    step_id="step-1",
                    title="Complete request",
                    description="Complete the user request.",
                ),
            )

        plan = ExecutionPlan(
            plan_id=(
                active_plan.plan_id
                if active_plan
                else f"{planner_input.identity.execution_id}:plan"
            ),
            revision=(
                active_plan.revision + 1
                if active_plan
                else 1
            ),
            objective=plan_text,
            steps=steps,
        )

        return PlannerResult(
            outcome=PlannerOutcome.EXECUTION_PLAN,
            proposed_plan=plan,
            message="Plan generated successfully.",
            planning_rationale=rationale,
        )



    def planner_node(state: AgentState):
        """First pass: analyze prompt and create a plan WITHOUT taking actions."""

        planner_input = build_planner_input(state)

        # Run decision layer
        routing_decision, plan_text, retrieval_messages = _decide_route_and_plan(
            latest_user_prompt=planner_input.context.user_request,
            router_llm=router_llm,
            planner_llm=planner_llm,
            tools_set=tools_set,
            rag_service=rag_service,
            rag_top_k=rag_top_k,
        )

        planner_result = _build_planner_result(
            planner_input=planner_input,
            routing_decision=routing_decision,
            active_plan=planner_input.active_plan,
            plan_text=plan_text,
        )

        # TODO(CEP-006):
        # Remove legacy bridge after Controller consumes PlannerResult directly.
        legacy = planner_result_to_legacy(planner_result)

        execution_state = build_execution_state(state)

        legacy["execution_state"] = with_cursor(
            execution_state,
            current_worker=WorkerRole.PLANNER,
        )

        legacy.update({
            "planner_result": planner_result,
            "retrieval_messages": retrieval_messages,

             # Legacy runtime fields
            "steps": 0, 
            "last_tool_rendered": "",
            "last_tool_success": None,
            "repeat_fail_count": 0,
            "tool_text_retry_used": False,
        })

        return legacy

    return planner_node