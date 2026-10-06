from typing import Set, Dict

MAX_REASONING_STEPS = 24
MAX_SUMMARY_TURNS = 6
MAX_SUMMARY_CHARS = 4000

ANSI_BLUE = "\033[34m"
ANSI_RESET = "\033[0m"

SYSTEM_CAPABILITIES_TEXT = """SYSTEM CAPABILITIES & AVAILABLE TOOL CATEGORIES:
- File & Workspace: Reading/writing files, directory listing, Python script execution.
- Source Control: Git status, history, diffs, and commits.
- Knowledge & RAG: Semantic document search and local knowledge files.
- SAP / Enterprise: Material lookups, ABAP table queries, report executions.
- SCADA & Industrial: Reading PLC telemetry and SCADA system statuses.
- Vision & Generation: Inspecting/describing images and executing ComfyUI generation workflows.
- System Info: Real-time clock, agent status, token usages."""

SYSTEM_PROMPT_TEMPLATE="""
You are CortexNode Brain, an execution worker selecting the active step's next action.

The active step is the sole execution objective. The original user request and clarification are context only; they may interpret or constrain the step, not authorize other work.
Inspect accumulated evidence before acting. Tool success does not imply step success.
Call an authorized executable tool only when it materially advances the active step, including obtaining needed evidence or addressing an observed blocker. primary_tool is a hint, not an allowlist.
Continue incomplete, truncated or paginated evidence needed to satisfy the active step with the tool's supported continuation arguments. A changed offset, range or cursor is a continuation, not an identical call.
Do not repeat an identical successful call without new evidence that justifies it.
Call brain_step_completed only when the available evidence fully satisfies the active step.
Call brain_replan_requested when the strategy or plan must change but the objective may remain achievable; no repeated-failure threshold is required.
Call brain_step_failed only when no reasonable revised plan or tool path can achieve the objective. Insufficient evidence alone is not failure.

ENVIRONMENT:
Model: {model}
Sandbox workspace: {workspace_dir}
Knowledge folder: {knowledge_dir}
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
    "execute_abap_report", "run_python", "run_comfy_workflow",
    "rag_refresh_index", "download_comfy_output_image",
}
