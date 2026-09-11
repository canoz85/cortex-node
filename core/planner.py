"""Framework-neutral Planner service; numbered prose is P1 compatibility only."""

from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass
from typing import Protocol

from core.planner_normalization import DIRECT_RESPONSE_ROUTES, normalize_planner_output, planner_failure
from core.protocol.models import PlannerInput, PlannerResult


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


@dataclass(frozen=True)
class PlannerRouting:
    """Internal routing value, not a new execution outcome contract."""

    route: str
    domain: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class PlannerMessage:
    role: str
    content: str


class PlannerProvider(Protocol):
    def route(self, user_request: str) -> PlannerRouting:
        """Run the existing router policy and return transport-free routing data."""
        ...

    def generate(self, messages: tuple[PlannerMessage, ...]) -> object:
        """Invoke once and return content for legacy normalization. Never retry."""
        ...


def filter_planner_tools(
    all_tools: Set[str], *, route: str, domain: str,
    domain_tool_map: Mapping[str, Set[str]], mutating_tools: Set[str],
) -> set[str]:
    """Preserve P1 prompt filtering, including unregistered ubiquitous tools."""
    filtered = set(all_tools).intersection(domain_tool_map.get(domain, all_tools))
    filtered.update({"current_time", "agent_info", "token_usage"})
    if route == "info":
        filtered.difference_update(mutating_tools)
    return filtered


class PlannerService:
    def __init__(
        self, *, provider: PlannerProvider, tools_set: Set[str],
        domain_tool_map: Mapping[str, Set[str]], mutating_tools: Set[str],
        system_capabilities_text: str,
    ):
        self.provider = provider
        self.tools_set = frozenset(tools_set)
        self.domain_tool_map = {key: frozenset(value) for key, value in domain_tool_map.items()}
        self.mutating_tools = frozenset(mutating_tools)
        self.system_capabilities_text = system_capabilities_text

    def run(
        self, planner_input: PlannerInput, *,
        retrieve: Callable[[str], tuple[str, ...]] | None = None,
    ) -> PlannerResult:
        """Produce only a proposal/result; Controller owns acceptance and state.

        Retrieval is supplied by context assembly and requested only for tool
        routes, preserving the current runtime's lazy retrieval behavior.
        """
        user_request = planner_input.context.user_request
        try:
            routing = self.provider.route(user_request)
        except Exception as exc:
            return planner_failure("provider", exc)

        if routing.route in DIRECT_RESPONSE_ROUTES:
            return normalize_planner_output(
                "", planner_input, route=routing.route, confidence=routing.confidence,
            )

        filtered = filter_planner_tools(
            self.tools_set, route=routing.route, domain=routing.domain,
            domain_tool_map=self.domain_tool_map, mutating_tools=self.mutating_tools,
        )
        prompt = PLANNER_SYSTEM_PROMPT.format(
            route=routing.route, domain=routing.domain, reason=routing.reason,
            system_capabilities_text=self.system_capabilities_text,
            available_tools="\n".join(f"- {name}" for name in sorted(filtered) if name)
            or "- No tool access allowed for this step",
        )
        try:
            retrieval = retrieve(user_request) if retrieve is not None else planner_input.context.retrieval_messages
        except Exception as exc:
            return planner_failure("context retrieval", exc)
        messages = (
            PlannerMessage("system", prompt),
            *(PlannerMessage("system", text) for text in retrieval),
            PlannerMessage("human", user_request),
        )
        try:
            content = self.provider.generate(messages)
        except Exception as exc:
            return planner_failure("provider", exc)
        return normalize_planner_output(
            content, planner_input, route=routing.route, confidence=routing.confidence,
        )


