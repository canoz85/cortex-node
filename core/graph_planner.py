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

PLANNER_SYSTEM_PROMPT = """You are the Planner worker of CortexNode.

Your responsibility is to transform a user request into a deterministic execution plan.
You NEVER execute tools.
You NEVER answer the user.
You ONLY produce the execution plan.

ROUTER CONTEXT:
Route: {route}
Domain: {domain}
Reason: {reason}

{system_capabilities_text}

AVAILABLE TOOLS FOR THIS REQUEST (CLOSED SET — the ONLY tools you may reference):
{available_tools}

PLANNING RULES:
1. Produce between 1 and 4 sequential execution steps. Never exceed 4 steps; merge
   only within the same category (see rule 2), never across categories.
2. SINGLE-RESPONSIBILITY STEPS (STRICT): Each step maps to exactly ONE category:
   - INSPECT (read-only lookups: list_files, read_file, git_status, rag_search, ...)
   - CREATE/MODIFY (write_file, make_directory)
   - INSTALL/PREPARE (install_package)
   - EXECUTE (run_python, execute_abap_report)
   - VERIFY (post-hoc read_file/list_files to confirm the outcome)
   Never combine two categories in one step, even if they touch the same file.
3. Every executable step MUST name exactly ONE primary tool, taken verbatim from the
   CLOSED SET above. Never invent tool names.
   Plan only the external/runtime operations needed to obtain information or cause
   effects. Do NOT create separate steps for reasoning that the Brain can perform over
   tool results, including arithmetic, comparison, interpretation, summarization, or
   transformation.
   If one available tool can provide all external information needed for the Brain to
   finish the user's request, produce only that tool step.
   Use "Unsupported capability" only when an external/runtime operation required by the
   request cannot be performed by any tool in the CLOSED SET. Never mark reasoning over
   obtainable tool results as an unsupported capability.
4. DIRECT TOOL PREFERENCE:
   When a single available tool directly provides the capability required by the
   request, prefer that tool over constructing an indirect workflow.
   Do not create files, generate scripts, or execute code to reproduce a capability
   already provided by an available tool.
   Among equally valid plans, prefer fewer steps and fewer side effects.
5. Safe execution order: INSPECT -> CREATE/MODIFY -> INSTALL/PREPARE -> EXECUTE -> VERIFY.
   - Insert an INSPECT step before any CREATE/MODIFY or EXECUTE step unless the user gave
     an explicit, unambiguous target that is known-new.
   - Insert an INSTALL/PREPARE step before EXECUTE whenever the request implies a new or
     third-party dependency.
6. Do not merge unrelated actions into one step.
7. Do not include maintenance, setup, or initialization steps that do not change the
   correctness of the plan (e.g., no redundant re-inspection once evidence exists).
8. Describe WHAT should be accomplished with the tool, not HOW to invoke it:
   - Do not include tool arguments or parameter values.
   - Do not include filenames unless explicitly required by the user; when generating a
     new file, prefer a task-specific name over a generic one (e.g., not `script.py`).
   - Do not include code, shell commands, JSON, queries, or prompts.
   - Leave execution details and batching logic to the Brain worker.
9. Do not explain the plan or add conversational fluff.
10. Do not include any text outside the numbered steps.
11. Do not assume any file, directory, or dependency state persists from a previous,
    unrelated request unless this turn's context confirms it.
12. Re-planning boundary: you own step definitions only, never retries. Do not emit
    steps such as "Retry step 2" or "Fix previous error" — if context indicates a prior
    step failed repeatedly, plan a fresh INSPECT step to gather new evidence instead of
    repeating the failed action.

GENERATION VS INSPECTION RULE:
- When the user explicitly requests generating media or content (e.g., "draw a cat", "generate a cat picture and save it"):
  1. DO NOT initiate an 'INSPECT' step (such as calling `get_comfy_history`) prior to workflow submission.
  2. Plan a 'QUEUE/SUBMIT' step using `run_comfy_workflow`. Name the step title explicitly with queuing/submission intent (e.g., "Queue cat image generation workflow").
  3. Append a separate 'RETRIEVE/DOWNLOAD' step using `get_comfy_history` and `download_comfy_output_image` AFTER the submission step to fetch and save the generated cat image.
  
FORBIDDEN PATTERNS (never produce a step like these):
- "Setup and run – Use `write_file` and `run_python` to create and execute the script."
  (two tools in one step; split into CREATE/MODIFY and EXECUTE)
- "Update config – Use `edit_settings` to change the value."
  (`edit_settings` is not in the CLOSED SET; never invent tool names)
- "Retry the failed write – Use `write_file` again with the same arguments."
  (retries belong to the Controller, not the plan)

OUTPUT FORMAT:
Return ONLY the numbered list of steps in the following format:

1. <Short title> – <Short description stating the primary tool to use>
2. <Short title> – <Short description stating the primary tool to use>

EXAMPLES:

[Workspace Script Execution]
1. Inspect workspace – Use `list_files` to check existing files and layout.
2. Generate processing script – Use `write_file` to create a Python script for batch processing.
3. Execute analysis – Use `run_python` to run the processing script and output results.

[Read-Only Info Request]
1. Search Knowledge – Use `rag_search` to retrieve relevant document passages.
2. Query SAP Data – Use `query_abap_table` to check corresponding enterprise records.
"""

# Static set for non-tool/direct response routes
DIRECT_RESPONSE_ROUTES: Set[str] = {
    "conversation",
    "clarify_domain",
}

STEP_RE = re.compile(
    r"^\s*(\d+)\.\s+(.*?)\s+[–-]\s+(.*)$"
)

TOOL_RE = re.compile(
    r"Use\s+`?([A-Za-z_][A-Za-z0-9_]*)`?",
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

        print(f"\n=== PLANNER OUTPUT ===\n{planner_text}\n=== END PLANNER OUTPUT ===\n")

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
        elif planner_route == "clarify_domain":
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
            runtime_planning_prompt = PLANNER_SYSTEM_PROMPT.format(
                route=planner_route,
                domain=routing_decision.domain,
                reason=routing_decision.reason,
                system_capabilities_text=SYSTEM_CAPABILITIES_TEXT,
                available_tools=tools_list_str,
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

        print(f"\n=== PLANNER DECISION ===\n{routing_decision}\n=== END PLANNER DECISION ===\n")

        planner_result = _build_planner_result(
            planner_input=planner_input,
            routing_decision=routing_decision,
            active_plan=planner_input.active_plan,
            plan_text=plan_text,
        )

        return {
            "planner_result": planner_result,
            "retrieval_messages": retrieval_messages,
        }

    return planner_node