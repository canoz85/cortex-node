import re

MAX_REASONING_STEPS = 24
RECENT_MESSAGE_WINDOW = 12
MAX_SUMMARY_CHARS = 1800
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_BLUE = "\033[34m"
ANSI_ITALIC = "\033[3m"
ANSI_RESET = "\033[0m"
MAX_PSEUDO_RETRIES = 2

SYSTEM_PROMPT_TEMPLATE = """You are CortexNode, a local-first autonomous software engineering agent.
You can reason, use tools, and iterate until the task is complete.
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

PSEUDO_TOOL_CALL_PATTERN = re.compile(
    r"\b(?:list_files|read_file|write_file|make_directory|run_python|git_status|git_diff|git_log|git_show|agent_info|token_usage|current_time|scada_status|rag_search|rag_refresh_index|query_abap_table|execute_abap_report|lookup_material|get_report_data)\s*\(",
    re.IGNORECASE,
)

PSEUDO_JSON_TOOL_CALL_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"(?:list_files|read_file|write_file|make_directory|run_python|git_status|git_diff|git_log|git_show|agent_info|token_usage|current_time|scada_status|rag_search|rag_refresh_index|query_abap_table|execute_abap_report|lookup_material|get_report_data)"\s*,\s*"arguments"\s*:',
    re.IGNORECASE,
)

ACTION_INTENT_PATTERN = re.compile(
    r"\b(create|write|edit|update|modify|generate|implement|fix|refactor|run|execute|test|build|add|remove|delete)\b",
    re.IGNORECASE,
)
TOKEN_USAGE_INTENT_PATTERN = re.compile(
    r"\b(token|tokens|usage|consumed|consume|spent|prompt tokens|completion tokens)\b",
    re.IGNORECASE,
)
CURRENT_TIME_INTENT_PATTERN = re.compile(
    r"\b(time|date|datetime|today|now|current time|current date)\b",
    re.IGNORECASE,
)
AGENT_INFO_INTENT_PATTERN = re.compile(
    r"\b(model|runtime|context window|max steps|agent info|configuration)\b",
    re.IGNORECASE,
)
CASUAL_CHAT_PATTERN = re.compile(
    r"\b(hi|hello|hey|thanks|thank you|how are you|what'?s up|good morning|good afternoon|good evening|my name is|i am|call me|what is my name|who am i|nice to meet you|bye|goodbye|see you)\b",
    re.IGNORECASE,
)
CODE_DISCUSSION_PATTERN = re.compile(
    r"\b(code|coding|bug|debug|error|issue|python|javascript|typescript|file|function|class|test|build|repo|repository|project|app|script|refactor|stack trace|traceback|api|database|sql|json|yaml|docker|git|langgraph|ollama|tool|workspace)\b",
    re.IGNORECASE,
)
CODING_DISCUSSION_QUESTION_PATTERN = re.compile(
    r"^(how|why|what|when|where|can|could|would|should|do)\b|\b(explain|help me|walk me through|show me how)\b",
    re.IGNORECASE,
)
FILE_GENERATION_PATTERN = re.compile(
    r"\b(cli|command line|script|tool|file|module|program|utility|app|sensor|json file)\b",
    re.IGNORECASE,
)
SAP_INTENT_PATTERN = re.compile(
    r"\b(sap|abap|material master|material|mm|fi|purchase order|po|vendor|mara|marc|table|query|report|rmmg|mfbf|transaction|tcode)\b",
    re.IGNORECASE,
)

MODIFYING_TOOL_NAMES = {"write_file", "make_directory"}
VERIFICATION_TOOL_NAMES = {"run_python"}
