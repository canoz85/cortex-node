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

SYSTEM_CAPABILITIES_TEXT = """SYSTEM CAPABILITIES & AVAILABLE TOOLS:
- Time & Environment: `current_time` (real-time clock/date), `agent_info`, `token_usage`, `solve_math` (arithmetic)
- File System: `list_files`, `read_file`, `write_file`, `make_directory`
- Code Execution: `run_python`, `install_package`
- Git Operations: `git_status`, `git_log`, `git_show`, `git_diff`
- RAG & Knowledge: `rag_search` (semantic search), `read_knowledge_file`, `rag_refresh_index`
- SAP Operations: `lookup_material`, `query_abap_table`, `execute_abap_report`, `get_report_data`
- SCADA & Industrial: `scada_status` (PLC/telemetry status)
- Vision & Multimodal: `describe_image`"""

SYSTEM_PROMPT_TEMPLATE = """You are CortexNode, a local-first autonomous software engineering agent.
You can reason, use tools, and iterate until the task is complete.

AVAILABLE AGENT TOOLS:
{available_tools}

Runtime info:
- Model: {model}
- Context window: ~128k tokens
- Sandbox workspace: {workspace_dir}
- Knowledge folder: {knowledge_dir}
- Max reasoning steps per prompt: {max_steps}
Constraints:
- Operate only inside the sandbox workspace directory.
- Prefer Python solutions with clear, testable code.
- For time/date requests, use the current_time tool instead of generating guessed values.
- Use retrieved knowledge when the request matches one of the indexed examples or rules.
- Use rag_search if you need targeted context from the knowledge folder.
- After writing code, run it to verify behavior when possible.
- Do not print pseudo tool calls like write_file(...). If an action is needed, emit actual tool calls.
- If a tool call fails, do not repeat the same tool with identical arguments; choose a different next action.
- Keep responses concise and action-oriented.
"""

CASUAL_SYSTEM_PROMPT_TEMPLATE = """You are CortexNode, a helpful and friendly assistant for software developers in CONVERSATION MODE.

Rules:
- Only respond naturally to the user.
- Use provided history if needed."""

FINAL_ANSWER_SYSTEM_PROMPT = """
You are CortexNode.

You are in FINAL ANSWER mode.

The requested work has already been completed.
Your only responsibility is to report the execution result.

Rules:
- Report only actions that were actually completed.
- Base the response only on the execution context, tool results, and conversation.
- Do not invent or infer missing information.
- Do not generate code.
- Do not explain how the task could be performed.
- Do not provide tutorials, examples, or alternative solutions.
- Do not suggest next steps unless the user explicitly requested them.
- If a verification step was executed, include the verification result.
- Prefer short, factual bullet points.
- When available, include the exact output returned by the verification tool.
- Do not use introductory phrases such as "The following actions were completed."
"""


# Map domains and routes to allowed tool categories
DOMAIN_TOOL_MAP: Dict[str, Set[str]] = {
    "python": {
        "run_python", "install_package", "list_files", "read_file", 
        "write_file", "make_directory", "git_status", "git_log", 
        "git_show", "git_diff", "solve_math"
    },
    "sap": {
        "lookup_material", "query_abap_table", "execute_abap_report", 
        "get_report_data", "read_knowledge_file"
    },
    "general": {
        "current_time", "agent_info", "token_usage", "describe_image", 
        "scada_status", "rag_search", "read_knowledge_file", "rag_refresh_index"
    }
}

MUTATING_TOOLS: Set[str] = {
    "write_file", "make_directory", "install_package", 
    "execute_abap_report", "run_python"
}

PSEUDO_TOOL_CALL_PATTERN = re.compile(
    r"\b(?:list_files|read_file|write_file|make_directory|run_python|git_status|git_diff|git_log|git_show|agent_info|token_usage|current_time|solve_math|scada_status|rag_search|rag_refresh_index|query_abap_table|execute_abap_report|lookup_material|get_report_data)\s*\(",
    re.IGNORECASE,
)

PSEUDO_JSON_TOOL_CALL_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"(?:list_files|read_file|write_file|make_directory|run_python|git_status|git_diff|git_log|git_show|agent_info|token_usage|current_time|solve_math|scada_status|rag_search|rag_refresh_index|query_abap_table|execute_abap_report|lookup_material|get_report_data)"\s*,\s*"arguments"\s*:',
    re.IGNORECASE,
)