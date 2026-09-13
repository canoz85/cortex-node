from dataclasses import dataclass
from typing_extensions import Literal

from core.logging_utils import get_logger
from core.planner_debug import log_planner


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict

ALLOWED_ROUTES = {"info", "action", "conversation", "clarify"}

PLANNER_ROUTER_PROMPT = """Classify the user's request by execution mode only.

Routes:
- conversation: no external/runtime information or side effect is required.
- info: external/runtime information is required, read-only.
- action: any state-changing operation is required or requested.
- clarify: the user's intent itself is too ambiguous to determine a route.

Rules:
- Do not plan, select tools, or decompose the request.
- Runtime-discoverable details are not grounds for clarify.
- Missing execution details are not grounds for clarify when intent is known; the Planner handles NEEDS_INPUT.
- Mixed read and write requests are action.
- Mixed inspect, modify, and verify requests are action.

Return only the structured route.
"""

logger = get_logger(__name__)



@dataclass(frozen=True)
class RoutingDecision:
    route: str


class RouterDecisionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal["conversation", "info", "action", "clarify"]
    

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
                for key in ("route",)
            })
        route = str(payload.route).strip()

        if route not in ALLOWED_ROUTES:
            return None

        return RoutingDecision(route=route)
    except Exception as exc:
        # PlannerService opts in so model failures become PlannerResult.FAILED.
        # Keep the historical fallback for other callers of this router helper.
        if propagate_errors:
            raise
        logger.warning(f"LLM Routing failed: {str(exc)}")
        return None
    
def _arbiter_route(
    hard_decision: RoutingDecision | None,
    llm_decision: RoutingDecision | None,
) -> RoutingDecision:
        
    if hard_decision is not None:
        return hard_decision

    if llm_decision is not None:
        return llm_decision

    # Safe fallback
    return RoutingDecision(
        route="conversation",
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
            route="clarify",
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
        hard_decision=hard_decision,
        llm_decision=llm_decision,
    )
