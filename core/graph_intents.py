import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from core.graph_constants import (
    AGENT_INFO_INTENT_PATTERN,
    CASUAL_CHAT_PATTERN,
    CODE_DISCUSSION_PATTERN,
    CURRENT_TIME_INTENT_PATTERN,
    MATH_INTENT_PATTERN,
    TOKEN_USAGE_INTENT_PATTERN,
)

PLANNER_ROUTER_PROMPT_OLD = """
    You are a strict router.
    Return ONLY valid JSON with keys: route, domain, confidence, enforced, reason, needs_clarification.
    Allowed routes: info, action, conversation, clarify_domain.
    Allowed domains: general, python, sap.
    Confidence must be 0.0..1.0.
    If uncertain, choose conversation with low confidence.
"""

PLANNER_ROUTER_PROMPT = """
You are a fast, high-precision intent router for an AI agent system.

YOUR TASK:
Analyze the user's input and classify it into a route and domain to guide downstream planning.

ROUTE DEFINITIONS:
- "info": Read-only requests requiring tool lookup (reading files, searching docs/RAG, git status/log, querying SCADA state, checking SAP records).
- "action": State-changing operations requiring execution tools (creating/modifying files, running python scripts, git commit/push, executing SCADA actions or SAP updates).
- "conversation": General Q&A, explanations, chit-chat, or reasoning that requires NO tool execution.
- "clarify_domain": Ambiguous requests where the domain or intent cannot be safely determined without user input.

DOMAIN DEFINITIONS:
- "python": Code generation, debugging, file system operations, execution, or git management.
- "sap": Anything related to SAP systems, BAPIs, enterprise data, or SAP tool execution.
- "general": General queries, SCADA/hardware control, vision tasks, or cross-domain requests.

OUTPUT REQUIREMENTS:
Return ONLY a valid JSON object matching this exact schema:
{
  "route": "info" | "action" | "conversation" | "clarify_domain",
  "domain": "python" | "sap" | "general",
  "confidence": float (0.0 to 1.0),
  "enforced": boolean (true if safety rules or explicit user instruction force this route),
  "needs_clarification": boolean,
  "reason": "Short 1-sentence justification for this route and domain decision"
}
"""


@dataclass(frozen=True)
class RoutingDecision:
    route: str
    domain: str
    confidence: float
    enforced: bool
    reason: str
    needs_clarification: bool
    source: str = "hard_rule"
    
ALLOWED_ROUTES = {
    "info",
    "action",
    "casual",
    "coding_discussion",
    "conversation",
    "clarify_domain",
}


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

def _workspace_intent(user_text: str) -> str | None:
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

def _llm_route_decision(   
    user_text: str,
    llm,
    tool_name_set: set[str],
) -> RoutingDecision | None:

    try:
        resp = llm.invoke(
            [
                SystemMessage(content=PLANNER_ROUTER_PROMPT),
                HumanMessage(content=user_text),
            ]
        )

        raw = str(getattr(resp, "content", "") or "").strip()
        payload = json.loads(raw)

        route = str(payload.get("route", "")).strip()
        domain = str(payload.get("domain", "general")).strip().lower()
        confidence = float(payload.get("confidence", 0.0))
        enforced = bool(payload.get("enforced", False))
        reason = str(payload.get("reason", "llm_router")).strip()
        needs_clarification = bool(payload.get("needs_clarification", False))

        if route not in ALLOWED_ROUTES:
            return None
        if domain not in {"general", "python", "sap"}:
            return None
        if confidence < 0.0 or confidence > 1.0:
            return None

        return RoutingDecision(
            route=route,
            domain=domain,
            confidence=confidence,
            enforced=enforced,
            reason=reason or "llm_router",
            needs_clarification=needs_clarification,
            source="llm_router",
        )
    except Exception:
        return None
    
def _arbiter_route(
    user_text: str,
    hard_decision: RoutingDecision | None,
    llm_decision: RoutingDecision | None,
) -> RoutingDecision:
        
    if hard_decision is not None:
        return hard_decision

    if llm_decision is not None:
        route = llm_decision.route
        conf = llm_decision.confidence

        # Strong confidence for potentially mutating/action routes
        if route.startswith("action") and route not in ALLOWED_ROUTES:
            if conf >= 0.80:
                return llm_decision
        else:
            # Lower threshold for non-mutating routes
            if conf >= 0.65:
                return llm_decision

    # Safe fallback
    return RoutingDecision(
        route="conversation",
        domain="general",
        confidence=0.35,
        enforced=False,
        reason="arbiter fallback",
        needs_clarification=False,
        source="fallback",
    )

def planner_routing_decision(
    user_text: str,
    router_llm: ChatOllama | None = None,
    tool_name_set: set[str] | None = None,
) -> RoutingDecision:

    text = (user_text or "").strip()
    if not text:
        return RoutingDecision(
            route="conversation",
            domain="general",
            confidence=0.0,
            enforced=False,
            reason="empty input",
            needs_clarification=False,
            source="fallback",
        )

    hard_decision: RoutingDecision | None = None

    # info_tool = preferred_info_tool(text)
    # if info_tool:
    #     hard_decision = RoutingDecision(
    #         route="info",
    #         domain="general",
    #         confidence=1.0,
    #         enforced=False,
    #         reason=info_tool,
    #         needs_clarification=False,
    #         source="hard_rule",
    #     )

    # if hard_decision is None:
    #     # Optional: explicit domain override hard rule
    #     # Reuse your existing _domain_decision behavior if needed.
    #     intent = _workspace_intent(text)
    #     if intent:
    #         route_map = {
    #             "LIST": "action:list_workspace",
    #             "READ": "action:read_workspace",
    #             "ANALYZE": "action:analyze_workspace",
    #             "GENERATE": "action:generate_workspace",
    #         }
    #         hard_decision = RoutingDecision(
    #             route=route_map[intent],
    #             domain="general",
    #             confidence=1.0,
    #             enforced=False,
    #             reason=f"{intent} workspace intent",
    #             needs_clarification=False,
    #             source="hard_rule",
    #         )

    # if hard_decision is None and _is_casual_chat(text):
    #     hard_decision = RoutingDecision(
    #         route="casual",
    #         domain="general",
    #         confidence=1.0,
    #         enforced=False,
    #         reason="casual chat",
    #         needs_clarification=False,
    #         source="hard_rule",
    #     )

    llm_decision = None
    if hard_decision is None and router_llm is not None:
        llm_decision = _llm_route_decision(
            user_text=text,
            llm=router_llm,
            tool_name_set=tool_name_set or set(),
        )

    return _arbiter_route(
        user_text=text,
        hard_decision=hard_decision,
        llm_decision=llm_decision,
    )