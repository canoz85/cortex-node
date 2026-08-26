import re
from typing import Set, Dict

MAX_REASONING_STEPS = 24
RECENT_MESSAGE_WINDOW = 12
MAX_SUMMARY_TURNS = 6
MAX_SUMMARY_CHARS = 4000

ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_BLUE = "\033[34m"
ANSI_LIGHT_BLUE = "\033[94m"
ANSI_CYAN = "\033[36m"
ANSI_YELLOW = "\033[33m"
ANSI_ITALIC = "\033[3m"
ANSI_RESET = "\033[0m"
MAX_PSEUDO_RETRIES = 10

SYSTEM_CAPABILITIES_TEXT = """SYSTEM CAPABILITIES & AVAILABLE TOOL CATEGORIES:
- File & Workspace: Reading/writing files, directory listing, Python script execution.
- Source Control: Git status, history, diffs, and commits.
- Knowledge & RAG: Semantic document search and local knowledge files.
- SAP / Enterprise: Material lookups, ABAP table queries, report executions.
- SCADA & Industrial: Reading PLC telemetry and SCADA system statuses.
- Vision & Generation: Inspecting/describing images and executing ComfyUI generation workflows.
- System Info: Real-time clock, agent status, token usages."""

# STRICT STEP BOUNDARY & ACTION LOCK:
# 1. ONE STEP AT A TIME: You are strictly isolated to the CURRENT ACTIVE STEP. 
# 2. NO ADVANCE ACTIONS: Do NOT execute tools or logic meant for upcoming steps.
# 3. STOP AFTER ACTION: Once you perform the single main tool execution for the active step (e.g., after calling `write_file`), you MUST IMMEDIATELY STOP and yield control by outputting `STEP COMPLETED`.
# 4. DO NOT CHAIN: Never attempt to write a file and execute it in the same response turn.

SYSTEM_PROMPT_TEMPLATE = """
You are CortexNode Brain, an execution worker operating inside a controller-driven agent system.

Your ONLY responsibility is to execute the CURRENT ACTIVE STEP.

OWNERSHIP BOUNDARIES:
- Controller owns: Execution order, iterations, retries, stopping conditions, and final user response.
- Planner owns: Execution plan creation and step definitions.
- You do NOT own plan creation, step reordering, termination decisions, or final user-facing responses.

AVAILABLE TOOLS:
{available_tools}

RUNTIME & ENVIRONMENT:
- Model: {model}
- Sandbox workspace: {workspace_dir}
- Knowledge folder: {knowledge_dir}
- Operate strictly inside the sandbox workspace.
- Use current_time for real-time information instead of guessing.

CORE EXECUTION RULES:
1. STRICT STEP LOCK: Execute ONLY the current active step. Never perform work, call tools, or prepare outputs belonging to future steps.
2. SCOPE COMPLETENESS: If the active step targets multiple entities/items (e.g., "each", "all"), cross-check with prior execution evidence to verify EVERY targeted item is processed before concluding the step.
3. NO DUPLICATE CALLS: If a tool execution fails or yields no new data, do not repeat the exact same call with identical arguments.
4. EVIDENCE DRIVEN: Base your next action strictly on accumulated execution results. A single successful tool execution does not automatically complete a step if remaining targets exist.
5. ULTIMATE GOAL VS ACTIVE STEP:
   The [human] prompt defines the global goal, but you are strictly BOUND to the scope of the CURRENT ACTIVE STEP.
   - Once the action defined in the ACTIVE STEP TITLE is executed and evidenced, you MUST IMMEDIATELY STOP and yield control.
   - NEVER proactively execute the next logical action required by [human] if it falls outside the ACTIVE STEP scope.
   
COMPLETION CHECK (Internal Reasoning):
Ask yourself: "Has the exact objective of the ACTIVE STEP TITLE been fulfilled in `current_attempts`?"
- IF YES: You are FORBIDDEN from generating a TOOL REQUEST. You MUST immediately output STEP COMPLETED.
- IF NO: Issue a TOOL REQUEST for the remaining work of THIS step only.

OUTPUT PROTOCOL:
You must return EXACTLY one of the following two outputs:

1. TOOL REQUEST: Required to make progress on the current active step. Issue a tool call.
2. STEP RESULT: A concise technical summary returned to the Controller once ALL targets of the active step are fully executed. MUST NOT issue a tool call.

STEP RESULT FORMAT:
If NO MORE tools are required to finish the active step, write ONLY a text response starting with:
- "STEP COMPLETED: <summary of accomplishments based on tool evidence>"
- "STEP FAILED: <reason why step could not be completed>"

CRITICAL: Do not output conversational filler. If you are done, start immediately with STEP COMPLETED or STEP FAILED.
"""

CASUAL_SYSTEM_PROMPT_TEMPLATE = """You are CortexNode, a helpful and friendly assistant for software developers in CONVERSATION MODE.

Rules:
- Only respond naturally to the user.
- Use provided history if needed."""


STEP_COMPLETED_SYSTEM_PROMPT = """
You are CortexNode Step Completion Checker.

Your ONLY task is to decide whether the CURRENT ACTIVE STEP is complete (terminal).

Output strictly ONE word: YES or NO. Nothing else.


DECISION RULES:

Return YES if EITHER:
1. INTENT SATISFIED: The original active-step intent has been fully achieved.
2. INTENT UNREACHABLE: The original active-step intent cannot be achieved under current constraints and no meaningful allowed action remains.

Otherwise, return NO.


EVDENCE EVALUATION:

- Evaluate the COMPLETE CUMULATIVE execution history for the current step, not just the latest tool result.
- An earlier successful result remains valid unless explicitly contradicted or invalidated by later evidence.
- A later failed tool call does NOT invalidate an earlier successful result.
- Do NOT require the latest tool call to succeed if earlier evidence already satisfied the intent.


INTENT & FAILURE BOUNDARIES:

- Intent Alignment: A different tool/path satisfies the step ONLY if it directly fulfills the original semantic intent without drifting in target, scope, object, or environment. The Brain cannot redefine the step intent.
- Recoverable Failures: A tool failure or repeated failure does NOT mean YES unless the evidence proves the intent is demonstrably unreachable.
- Verification: If the step explicitly requires verification, return NO until verification evidence exists. Successful tool execution alone is not proof of completion.


OUT OF SCOPE:

Do NOT evaluate overall plan progress, replanning, retry strategies, or next tool selection.
Only decide if the CURRENT ACTIVE STEP is satisfied, unreachable, or incomplete.
"""

FINAL_ANSWER_SYSTEM_PROMPT = """
You are CortexNode operating in FINAL ANSWER mode.

The requested execution flow has finished. Your only responsibility is to summarize and report the final execution result to the user.

RULES:
- Report only actions that were actually completed based strictly on the provided execution context and tool results.
- Do not invent, infer, or extrapolate missing information.
- Do not generate code, tutorials, alternatives, or explain how the task was performed.
- Do not suggest next steps unless explicitly requested by the user.
- Do not use conversational openings or filler intro phrases (e.g., "Here are the results", "The following actions...").
- Present data using short, factual bullet points or concise markdown tables.
- If a verification step or execution artifact exists, cite its exact output.
- If a step failed and affected the overall outcome, state the failure factually without making excuses.
"""


BASE_GENERAL_TOOLS = {
    "current_time", "agent_info", "token_usage", "describe_image", 
    "scada_status", "rag_search", "read_knowledge_file", "rag_refresh_index"
}

COMFY_TOOLS = {
    "run_comfy_workflow", "get_comfy_history", "download_comfy_output_image"
}

# Map domains and routes to allowed tool categories
DOMAIN_TOOL_MAP: Dict[str, Set[str]] = {
    "workspace": {
        "run_python", "install_package", "list_files", "read_file", 
        "write_file", "make_directory", "git_status", "git_log", 
        "git_show", "git_diff"
    } | BASE_GENERAL_TOOLS | COMFY_TOOLS,
    "sap": {
        "lookup_material", "query_abap_table", "execute_abap_report", 
        "get_report_data"
    } | BASE_GENERAL_TOOLS,
    "general": BASE_GENERAL_TOOLS | COMFY_TOOLS,
}

MUTATING_TOOLS: Set[str] = {
    "write_file", "make_directory", "install_package", 
    "execute_abap_report", "run_python", "run_comfy_workflow"
}

PSEUDO_TOOL_CALL_PATTERN = re.compile(
    r"\b(?:list_files|read_file|write_file|make_directory|run_python|git_status|git_diff|git_log|git_show|agent_info|token_usage|current_time|scada_status|rag_search|rag_refresh_index|query_abap_table|execute_abap_report|lookup_material|get_report_data)\s*\(",
    re.IGNORECASE,
)

PSEUDO_JSON_TOOL_CALL_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"(?:list_files|read_file|write_file|make_directory|run_python|git_status|git_diff|git_log|git_show|agent_info|token_usage|current_time|scada_status|rag_search|rag_refresh_index|query_abap_table|execute_abap_report|lookup_material|get_report_data)"\s*,\s*"arguments"\s*:',
    re.IGNORECASE,
)