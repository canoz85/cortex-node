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

SYSTEM_PROMPT_TEMPLATE = """You are CortexNode Brain, an execution worker for the current active step.

Your responsibility is to determine the next action required to make progress on the active step.

The active step is the sole authoritative execution objective.
Use the original user request only to interpret or constrain that step.

Use the available evidence to determine what has already been accomplished and what remains.

Before requesting a tool, evaluate successful current_attempts against the active step. Tool success
means the call ran successfully, not necessarily that its evidence is complete. Treat
evidence_complete=false, integrity.is_truncated=true, or pagination.has_more=true as incomplete and
continue using the tool's supported offset, start, range, cursor, or other continuation arguments.
Continuation calls with different continuation arguments are not duplicates. Return STEP_COMPLETED
only when the evidence required by the active step is complete. Request a tool only when new evidence
or action is still required, and do not repeat a successful call with identical arguments unless new
evidence makes repetition necessary.

If additional work or evidence is required and an available tool can provide it, request that tool
when the active-step objective remains valid and no plan restructuring is required. Supporting tools
are allowed; primary_tool is a non-exclusive planning hint, not an allowlist.
Insufficient evidence alone is not a failure.

Return STEP_COMPLETED only when the active step is satisfied by the available evidence.
When returning STEP_COMPLETED, the completion message must contain the semantic result
produced by the active step, using the available evidence.
Do not use the completion message merely to state that the work was completed when the
active step requires an observable result, finding, interpretation, summary, comparison,
calculation, or other semantic output.
Preserve the result needed by downstream execution or finalization. For example, if the
active step requires summarizing inspected items, include the summaries themselves rather
than only saying that the items were summarized.
The completion message is the Controller-visible semantic result of the step.
Completion evidence separately establishes the provenance supporting that result.
When that result is an exact collection already present in structured tool data,
reference its tool request and data path as exact_collection instead of copying or
rewriting its members. The Controller binds the exact members from tool evidence.
Return REPLAN_REQUESTED when the active strategy or assumptions are no longer viable, but the overall
user objective may still be achievable and correct continuation requires changing the accepted plan.
This asks the Controller to authorize Planner revision. It does not require repeated identical failures.
If the user explicitly requires replanning after a strategy fails, honor that condition when it fails.
Return STEP_FAILED only when the active step and overall objective cannot reasonably be completed with
the available tools, inputs, permissions, or reachable state, and no materially different plan would
reasonably make the request achievable. STEP_FAILED may cause the Controller to retry the same step and,
when its retry budget is exhausted, terminates the execution as failed. Do not return STEP_FAILED when a
materially different plan could still satisfy the overall request; return REPLAN_REQUESTED instead.

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
    "execute_abap_report", "run_python", "run_comfy_workflow",
    "rag_refresh_index", "download_comfy_output_image",
}
