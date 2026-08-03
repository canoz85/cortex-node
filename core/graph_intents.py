import json
import re
from typing import Dict, Any, Set

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from core.graph_constants import SYSTEM_CAPABILITIES_TEXT

ALLOWED_ROUTES = {"info", "action", "conversation", "clarify_domain"}
ALLOWED_DOMAINS = {"python", "sap", "general"}

ROUTES_SCHEMA_STR = " | ".join(f'"{r}"' for r in sorted(ALLOWED_ROUTES))
DOMAINS_SCHEMA_STR = " | ".join(f'"{d}"' for d in sorted(ALLOWED_DOMAINS))

PLANNER_ROUTER_PROMPT = f"""
You are a fast, high-precision intent router for an AI agent system.

YOUR TASK:
Analyze the user's input and classify it into a route and domain to guide downstream planning.

{SYSTEM_CAPABILITIES_TEXT}

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
{{
  "route": {ROUTES_SCHEMA_STR},
  "domain": {DOMAINS_SCHEMA_STR},
  "confidence": float (0.0 to 1.0),
  "enforced": boolean (true if safety rules or explicit user instruction force this route),
  "needs_clarification": boolean,
  "reason": "Short 1-sentence justification for this route and domain decision"
}}
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
    

def _llm_route_decision(   
    user_text: str,
    llm,
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
        )

    return _arbiter_route(
        user_text=text,
        hard_decision=hard_decision,
        llm_decision=llm_decision,
    )