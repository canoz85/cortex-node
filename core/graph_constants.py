from tools.registry import TOOL_DEFINITIONS

MAX_REASONING_STEPS = 24
MAX_SUMMARY_TURNS = 6
MAX_SUMMARY_CHARS = 4000

ANSI_BLUE = "\033[34m"
ANSI_RESET = "\033[0m"

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


BASE_GENERAL_TOOLS = {entry.name for entry in TOOL_DEFINITIONS if entry.family == "general"}
COMFY_TOOLS = {entry.name for entry in TOOL_DEFINITIONS if entry.family == "comfy"}

# Map domains and routes to allowed tool categories
DOMAIN_TOOL_MAP = {
    "workspace": {entry.name for entry in TOOL_DEFINITIONS if entry.family == "workspace"}
    | BASE_GENERAL_TOOLS | COMFY_TOOLS,
    "sap": {entry.name for entry in TOOL_DEFINITIONS if entry.family == "sap"} | BASE_GENERAL_TOOLS,
    "general": BASE_GENERAL_TOOLS | COMFY_TOOLS,
}

MUTATING_TOOLS = {entry.name for entry in TOOL_DEFINITIONS if entry.mutating}
