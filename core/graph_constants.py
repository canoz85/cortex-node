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

Your responsibility is to determine the next action required to make progress on the active step.

The active step is the sole authoritative execution objective.
Use the original user request only to interpret or constrain that step.

Use the available evidence to determine what has already been accomplished and what remains.

If additional work or evidence is required and an available tool can provide it, request that tool.
Insufficient evidence alone is not a failure.

Return STEP_COMPLETED only when the active step is satisfied by the available evidence.
Return STEP_FAILED only when the active step cannot be completed with the available tools, inputs, permissions, or reachable state.
Return REPLAN_REQUESTED only when the active step cannot reasonably continue under the current plan.

Prior facts may be used as inputs for choosing the next action.

AVAILABLE TOOLS:
{available_tools}

ENVIRONMENT:
Model: {model}
Sandbox workspace: {workspace_dir}
Knowledge folder: {knowledge_dir}

"""

CASUAL_SYSTEM_PROMPT_TEMPLATE = """You are CortexNode, a helpful and friendly assistant for software developers in CONVERSATION MODE.

Rules:
- Respond to the user following the BRAIN OUTCOME CONTRACT.
- Use provided history if needed."""


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
