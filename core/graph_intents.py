import re
from dataclasses import dataclass

from core.graph_constants import (
    ACTION_INTENT_PATTERN,
    AGENT_INFO_INTENT_PATTERN,
    CASUAL_CHAT_PATTERN,
    CODE_DISCUSSION_PATTERN,
    CODING_DISCUSSION_QUESTION_PATTERN,
    CURRENT_TIME_INTENT_PATTERN,
    FILE_GENERATION_PATTERN,
    MATH_INTENT_PATTERN,
    TOKEN_USAGE_INTENT_PATTERN,
)


@dataclass(frozen=True)
class RoutingDecision:
    route: str
    domain: str
    confidence: float
    enforced: bool
    reason: str
    needs_clarification: bool


EXPLICIT_SAP_PATTERN = re.compile(r"\[(?:domain\s*:\s*)?sap\]|\bdomain\s*:\s*sap\b", re.IGNORECASE)
EXPLICIT_PYTHON_PATTERN = re.compile(r"\[(?:domain\s*:\s*)?python\]|\bdomain\s*:\s*python\b", re.IGNORECASE)
STRONG_SAP_PATTERN = re.compile(
    r"\b(sap|abap|tcode|se38|se11|se16|ekko|ekpo|mara|marc|makt|vbak|vbap|bkpf|bseg|lifnr|matnr|werks|ebeln|ebelp)\b",
    re.IGNORECASE,
)
STRONG_PYTHON_PATTERN = re.compile(
    r"\b(python|py|pandas|numpy|pip|venv|pytest|fastapi|flask|django|script|module|package|traceback|import|def|class|json|csv|parse|parsing)\b",
    re.IGNORECASE,
)
FILE_PATH_HINT_PATTERN = re.compile(
    r"(?:^|[\s'\"`(\[])(?:[\w.-]+[\\/])*[\w.-]+\.[a-z0-9]{1,8}(?:$|[\s'\"`)\]])",
    re.IGNORECASE,
)
FILE_READ_INTENT_PATTERN = re.compile(
    r"\b(read|open|show|inspect|review|check|analy[sz]e|debug|fix|explain)\b",
    re.IGNORECASE,
)
FILE_MUTATION_INTENT_PATTERN = re.compile(
    r"\b(create|write|edit|update|modify|generate|implement|refactor|delete|remove|rename)\b",
    re.IGNORECASE,
)

READ_AUDIT_INTENT_PATTERN = re.compile(
    r"\b(which|what|where)\b.*\b(files?|file)\b.*\b(read|reviewed|analy[sz]e(?:d)?)\b|\bdid you read\b",
    re.IGNORECASE,
)
FILE_FACT_EXTRACTION_INTENT_PATTERN = re.compile(
    r"\b(what|which|show|tell)\b.*\b(device[_\s-]?id|id|temperature|pressure|value|status|field|json|data|latest)\b"
    r"|\b(device[_\s-]?id|temperature|pressure)\b",
    re.IGNORECASE,
)


###
LIST_WORKSPACE_INTENT_PATTERN = re.compile(
    r"^\s*(list|show|display|ls|dir)\s+(all\s+)?(workspace\s+)?(files|folders|directories)\s*$",
    re.IGNORECASE,
)


def _domain_decision(user_text: str) -> tuple[str, float, bool, bool, str]:
    text = (user_text or "").strip()
    if not text:
        return "general", 0.0, False, False, "empty input"

    if EXPLICIT_SAP_PATTERN.search(text):
        return "sap", 1.0, True, False, "explicit sap domain override"

    if EXPLICIT_PYTHON_PATTERN.search(text):
        return "python", 1.0, True, False, "explicit python domain override"

    sap_strong_hits = len(STRONG_SAP_PATTERN.findall(text))
    python_hits = len(STRONG_PYTHON_PATTERN.findall(text))

    sap_score = sap_strong_hits * 0.45
    python_score = python_hits * 0.35

    if sap_score <= 0 and python_score <= 0:
        return "general", 0.35, False, False, "no domain indicators"

    if abs(sap_score - python_score) <= 0.2 and sap_score > 0 and python_score > 0:
        confidence = min(0.74, 0.45 + max(sap_score, python_score) * 0.1)
        return "general", confidence, False, True, "mixed sap and python signals"

    if sap_score > python_score:
        confidence = min(0.95, 0.55 + sap_score * 0.1)
        return "sap", confidence, False, False, "sap indicators dominate"

    confidence = min(0.95, 0.55 + python_score * 0.1)
    return "python", confidence, False, False, "python indicators dominate"


def requests_workspace_file_access(user_text: str) -> bool:
    text = (user_text or "").strip()
    if not text:
        return False
    if not FILE_PATH_HINT_PATTERN.search(text):
        return False
    return bool(FILE_READ_INTENT_PATTERN.search(text))


def is_read_only_file_request(user_text: str) -> bool:
    text = (user_text or "").strip()
    if not requests_workspace_file_access(text):
        return False
    return not bool(FILE_MUTATION_INTENT_PATTERN.search(text))


def preferred_info_tool(user_text: str) -> str | None:
    text = (user_text or "").strip()
    if not text:
        return None
    if TOKEN_USAGE_INTENT_PATTERN.search(text):
        return "token_usage"
    if CURRENT_TIME_INTENT_PATTERN.search(text):
        return "current_time"
    if MATH_INTENT_PATTERN.search(text):
        return "solve_math"
    if AGENT_INFO_INTENT_PATTERN.search(text):
        return "agent_info"
    return None


def preferred_file_tool(user_text: str) -> str | None:
    text = (user_text or "").strip()
    if not text:
        return None
    if LIST_WORKSPACE_INTENT_PATTERN.search(text):
        return "list_files"
    return None


def _is_casual_chat(user_text: str) -> bool:
    """Return True for social/identity chat that should skip planning/tool routing."""
    text = (user_text or "").strip()
    if not text:
        return False
    if preferred_info_tool(text):
        return False
    if CODE_DISCUSSION_PATTERN.search(text):
        return False
    if CASUAL_CHAT_PATTERN.search(text):
        return True

    word_count = len(text.split())
    if word_count <= 6 and text.endswith("?"):
        return True
    return False


def is_file_generation_request(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    return bool(FILE_GENERATION_PATTERN.search(text))

def workspace_intent(user_text: str) -> str | None:
    text = (user_text or "").lower()

    intent_scores = {
        "LIST": 0,
        "READ": 0,
        "ANALYZE": 0,
        "GENERATE": 0,
    }

    # LIST
    if any(k in text for k in [
        "list workspace", "list files", "show workspace", "directory structure", "ls"
    ]):
        intent_scores["LIST"] += 3

    # READ
    if any(k in text for k in [
        "read workspace", "open file", "show file content", "inspect files", "read files"
    ]):
        intent_scores["READ"] += 3

    # ANALYZE
    if any(k in text for k in [
        "analyze", "count characters", "summarize", "explain project", "what is inside"
    ]):
        intent_scores["ANALYZE"] += 3

    # GENERATE
    if any(k in text for k in [
        "generate", "create project", "write script", "build project", "implement", "execute code", "run code"
    ]):
        intent_scores["GENERATE"] += 3

    # soft signals (important improvement)
    if "workspace" in text:
        intent_scores["LIST"] += 1
        intent_scores["READ"] += 1
        intent_scores["ANALYZE"] += 1

    best = max(intent_scores, key=intent_scores.get)

    return best if intent_scores[best] > 0 else None

def planner_routing_decision(user_text: str) -> RoutingDecision:
    text = (user_text or "").strip()
    if not text:
        return RoutingDecision("conversation", "general", 0.0, False, "empty input", False)

    # 1. info tools first
    info_tool = preferred_info_tool(text)
    if info_tool:
        return RoutingDecision("info", "general", 1.0, False, info_tool, False)

    # 2. workspace actions (single source of truth)
    intent = workspace_intent(text)
    if intent:
        route_map = {
            "LIST": "action:list_workspace",
            "READ": "action:read_workspace",
            "ANALYZE": "action:analyze_workspace",
            "GENERATE": "action:generate_workspace",
        }
        return RoutingDecision(route_map[intent], "general", 1.0, False, f"{intent} workspace intent", False)

    # 3. casual
    if _is_casual_chat(text):
        return RoutingDecision("casual", "general", 1.0, False, "casual chat", False)

    # 4. domain logic fallback
    domain, confidence, enforced, ambiguous, reason = _domain_decision(text)

    # if ambiguous and requires_action(text):
    #     return RoutingDecision("clarify_domain", "general", confidence, enforced, reason, True)

    # if domain == "sap" and requires_action(text):
    #     return RoutingDecision("action:sap", domain, confidence, enforced, reason, False)

    # if requires_action(text):
    #     return RoutingDecision("action", domain, max(confidence, 0.65), enforced, reason, False)

    return RoutingDecision("conversation", domain, confidence, enforced, reason, False)