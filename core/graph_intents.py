import re
from typing import Dict, Any, Set

from dataclasses import dataclass
from typing_extensions import Literal

from core.logging_utils import get_logger
from core.planner_debug import log_planner


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
- "info": Read-only operations requiring tools, including current runtime/system
  state, files, knowledge retrieval, Git status, SAP/SCADA queries, etc.
- "conversation": Questions that can be answered directly using internal LLM knowledge WITHOUT calling any tools (e.g., explanations, coding assistance, chit-chat).
- "clarify_domain": The request is too ambiguous or critical information is missing to decide safely.

DOMAIN DEFINITIONS:
- "workspace": File system operations, Python code/execution, Git management, workspace scripts.
- "sap": SAP system queries, ABAP reports, material lookups, enterprise tables.
- "general": SCADA/telemetry, vision tasks, general knowledge, or cross-domain queries.

DECISION RULES:
1. Direct answers without tools -> "conversation".
2. "conversation" is valid only when all information required to answer is already
   provided by the user or available from stable internal knowledge. If the answer
   depends on current, live, runtime, environment, or system state, use a tool-based route.
3. If the user explicitly asks to run, write, or execute something -> "action".
4. If the user asks to inspect, read, search, or obtain current runtime/system data
   without changing state -> "info".
5. Do not infer tool usage from verbs like "find", "check", or "calculate" alone.
   Tool usage is required only when the requested answer depends on information
   unavailable without a configured capability.

OUTPUT REQUIREMENTS:
Return a JSON object with the following fields:
- "route": One of "action", "info", "conversation", "clarify_domain".
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
    *,
    propagate_errors: bool = False,
    show_raw_llm: bool = False,
) -> RoutingDecision | None:

    try:
        structured_router = llm.with_structured_output(
            RouterDecisionSchema,
            method="json_schema",
        )
        log_planner("router][system", PLANNER_ROUTER_PROMPT, enabled=show_raw_llm)
        log_planner("router][human", user_text, enabled=show_raw_llm)
        payload = structured_router.invoke(
            [
                SystemMessage(content=PLANNER_ROUTER_PROMPT),
                HumanMessage(content=user_text),
            ]
        )

        if show_raw_llm:
            log_planner("router][structured", {
                key: getattr(payload, key, None)
                for key in ("route", "domain", "confidence", "enforced", "reason")
            })
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
        # PlannerService opts in so model failures become PlannerResult.FAILED.
        # Keep the historical fallback for other callers of this router helper.
        if propagate_errors:
            raise
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
    *,
    propagate_errors: bool = False,
    show_raw_llm: bool = False,
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

    llm_decision = None
    if hard_decision is None and router_llm is not None:
        llm_decision = _llm_route_decision(
            user_text=text,
            llm=router_llm,
            propagate_errors=propagate_errors,
            show_raw_llm=show_raw_llm,
        )

    return _arbiter_route(
        user_text=text,
        hard_decision=hard_decision,
        llm_decision=llm_decision,
    )
