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


def requires_action(user_text: str) -> bool:
    text = (user_text or "").strip()
    if not text:
        return False
    return bool(ACTION_INTENT_PATTERN.search(text))


def preferred_info_tool(user_text: str) -> str | None:
    text = (user_text or "").strip()
    if not text:
        return None
    if TOKEN_USAGE_INTENT_PATTERN.search(text):
        return "token_usage"
    if CURRENT_TIME_INTENT_PATTERN.search(text):
        return "current_time"
    if AGENT_INFO_INTENT_PATTERN.search(text):
        return "agent_info"
    return None


def _is_casual_chat(user_text: str) -> bool:
    """Return True for social/identity chat that should skip planning/tool routing."""
    text = (user_text or "").strip()
    if not text:
        return False
    if preferred_info_tool(text) or requires_action(text):
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
    text = (user_text or "").strip()
    if not text or not requires_action(text):
        return False
    return bool(FILE_GENERATION_PATTERN.search(text))


def planner_route(user_text: str) -> str:
    return planner_routing_decision(user_text).route


def planner_routing_decision(user_text: str) -> RoutingDecision:
    text = (user_text or "").strip()
    if not text:
        return RoutingDecision(
            route="conversation",
            domain="general",
            confidence=0.0,
            enforced=False,
            reason="empty input",
            needs_clarification=False,
        )
    if preferred_info_tool(text):
        return RoutingDecision("info", "general", 1.0, False, "info intent", False)
    if _is_casual_chat(text):
        return RoutingDecision("casual", "general", 1.0, False, "casual chat", False)

    domain, confidence, enforced, ambiguous, reason = _domain_decision(text)

    if ambiguous and requires_action(text):
        return RoutingDecision("clarify_domain", "general", confidence, enforced, reason, True)

    if domain == "sap" and requires_action(text):
        return RoutingDecision("action:sap", domain, confidence, enforced, reason, False)

    if CODE_DISCUSSION_PATTERN.search(text) and CODING_DISCUSSION_QUESTION_PATTERN.search(text):
        return RoutingDecision("coding_discussion", domain, max(confidence, 0.7), enforced, reason, False)
    if is_file_generation_request(text):
        return RoutingDecision("action:file_generation", domain, max(confidence, 0.75), enforced, reason, False)
    if requires_action(text):
        return RoutingDecision("action", domain, max(confidence, 0.65), enforced, reason, False)
    if CODE_DISCUSSION_PATTERN.search(text):
        return RoutingDecision("coding_discussion", domain, max(confidence, 0.6), enforced, reason, False)
    return RoutingDecision("conversation", domain, confidence, enforced, reason, False)
