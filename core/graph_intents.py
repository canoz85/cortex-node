import re
from typing import Dict, Any, Set

from dataclasses import dataclass
from typing_extensions import Literal

from core.logging_utils import get_logger


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict

from core.graph_constants import SYSTEM_CAPABILITIES_TEXT

ALLOWED_ROUTES = {"info", "action", "conversation", "clarify_domain"}
ALLOWED_DOMAINS = {"workspace", "sap", "general"}

ROUTES_SCHEMA_STR = " | ".join(f'"{r}"' for r in sorted(ALLOWED_ROUTES))
DOMAINS_SCHEMA_STR = " | ".join(f'"{d}"' for d in sorted(ALLOWED_DOMAINS))

PLANNER_ROUTER_PROMPT = f"""
You are a fast, high-precision intent router for an AI agent system.

YOUR TASK:
Analyze the user's input and categorize it into exactly ONE route and ONE domain to guide execution.

{SYSTEM_CAPABILITIES_TEXT}

ROUTE DEFINITIONS:
- "action": State-changing operations, writing/modifying files, running Python code, or executing system tasks.
- "info": Read-only operations requiring tools (e.g., reading files, searching RAG/knowledge, checking Git status, querying SAP/SCADA).
- "conversation": Questions that can be answered directly using internal LLM knowledge WITHOUT calling any tools (e.g., explanations, coding assistance, chit-chat).
- "clarify_domain": The request is too ambiguous or critical information is missing to decide safely.

DOMAIN DEFINITIONS:
- "workspace": File system operations, Python code/execution, Git management, workspace scripts.
- "sap": SAP system queries, ABAP reports, material lookups, enterprise tables.
- "general": SCADA/telemetry, vision tasks, general knowledge, or cross-domain queries.

DECISION RULES:
1. Direct answers without tools -> "conversation".
2. If the user explicitly asks to run, write, or execute something -> "action".
3. If the user asks to inspect, read, or search existing data -> "info".
4. Do not infer tool usage from simple verbs like "find", "check", or "calculate" unless tool/workspace execution is explicitly required.

OUTPUT REQUIREMENTS:
Return a JSON object with the following fields:
- "route": One of "action", "info", "conversation", "clarify".
- "domain": One of "workspace", "sap", "general".
- "confidence": Float between 0.0 and 1.0.
- "enforced": Boolean (true if safety rules or explicit user instructions force this route).
- "reason": Short 1-sentence justification for this route and domain decision.
"""

logger = get_logger(__name__)



@dataclass(frozen=True)
class RoutingDecision:
    route: str
    domain: str
    confidence: float
    enforced: bool
    reason: str
    source: str = "hard_rule"


class RouterDecisionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: str
    domain: str
    confidence: float
    enforced: bool
    reason: str
    

def _llm_route_decision(   
    user_text: str,
    llm,
) -> RoutingDecision | None:

    try:
        structured_router = llm.with_structured_output(
            RouterDecisionSchema,
            method="json_schema",
        )
        payload = structured_router.invoke(
            [
                SystemMessage(content=PLANNER_ROUTER_PROMPT),
                HumanMessage(content=user_text),
            ]
        )

        route = str(payload.route).strip()
        domain = str(payload.domain).strip().lower()
        confidence = float(payload.confidence)
        enforced = bool(payload.enforced)
        reason = str(payload.reason).strip()

        if route not in ALLOWED_ROUTES:
            return None
        if domain not in ALLOWED_DOMAINS:
            return None
        if confidence < 0.0 or confidence > 1.0:
            return None

        return RoutingDecision(
            route=route,
            domain=domain,
            confidence=confidence,
            enforced=enforced,
            reason=reason or "llm_router",
            source="llm_router",
        )
    except Exception as exc:
        logger.warning(f"LLM Routing failed: {str(exc)}")
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
        if route.startswith("action"):
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