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

SYSTEM_CAPABILITIES_TEXT = """SYSTEM CAPABILITIES & AVAILABLE TOOL CATEGORIES:
- File & Workspace: Reading/writing files, directory listing, Python script execution.
- Source Control: Git status, history, diffs, and commits.
- Knowledge & RAG: Semantic document search and local knowledge files.
- SAP / Enterprise: Material lookups, ABAP table queries, report executions.
- SCADA & Industrial: Reading PLC telemetry and SCADA system statuses.
- Vision & Generation: Inspecting/describing images and executing ComfyUI generation workflows.
- System Info: Real-time clock, agent status, token usages."""

SYSTEM_PROMPT_TEMPLATE = """You are CortexNode Brain, an execution worker for the current active step.
Evaluate whether the available evidence satisfies the step's objective and requirements.

AVAILABLE TOOLS:
{available_tools}

ENVIRONMENT:
Model: {model}
Sandbox workspace: {workspace_dir}
Knowledge folder: {knowledge_dir}

Output capability is specified in the BRAIN OUTCOME CONTRACT.
"""

CASUAL_SYSTEM_PROMPT_TEMPLATE = """You are CortexNode, a helpful and friendly assistant for software developers in CONVERSATION MODE.

Rules:
- Respond to the user following the BRAIN OUTCOME CONTRACT.
- Use provided history if needed."""


STEP_COMPLETED_SYSTEM_PROMPT = """
You are CortexNode Step Completion Checker.

Your ONLY task is to decide whether the CURRENT ACTIVE STEP is complete (terminal).

Follow the BRAIN OUTCOME CONTRACT, using typed JSON for non-tool outcomes, never YES/NO text.


DECISION RULES:

Return STEP_COMPLETED only if the original active-step intent has been fully achieved.
Return STEP_FAILED if the intent cannot be achieved under current constraints.
If allowed work remains, use the configured tool invocation mechanism for that active step only.


EVDENCE EVALUATION:

- Evaluate the COMPLETE CUMULATIVE execution history for the current step, not just the latest tool result.
- An earlier successful result remains valid unless explicitly contradicted or invalidated by later evidence.
- A later failed tool call does NOT invalidate an earlier successful result.
- Do NOT require the latest tool call to succeed if earlier evidence already satisfied the intent.


INTENT & FAILURE BOUNDARIES:

- Intent Alignment: A different tool/path satisfies the step ONLY if it directly fulfills the original semantic intent without drifting in target, scope, object, or environment. The Brain cannot redefine the step intent.
- Recoverable Failures: A tool failure or repeated failure does NOT establish unreachable intent unless the evidence proves the intent is demonstrably unreachable.
- Verification: If the step explicitly requires verification, do not return STEP_COMPLETED until verification evidence exists. Successful tool execution alone is not proof of completion.


OUT OF SCOPE:

Do NOT evaluate overall plan progress, select future steps, or decide retry policy.
Only decide if the CURRENT ACTIVE STEP is satisfied, unreachable, or incomplete.
"""

FINAL_ANSWER_SYSTEM_PROMPT = """
You are CortexNode operating in FINAL ANSWER mode.

The requested execution flow has finished. Your only responsibility is to summarize and report the final execution result to the user under the BRAIN OUTCOME CONTRACT.

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
