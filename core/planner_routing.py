from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal

from core.logging_utils import get_logger
from core.debug import log_llm_exchange
from core.logging.live_status import add_response_usage
from core.logging.live_status import current_live_status


PLANNER_ROUTER_PROMPT = """Classify the user's request by execution mode only.

Routes:
- conversation: no runtime tool execution is required. This includes requests answerable from the model's own knowledge, reasoning, conversation context, or Planner memory.
- info: one or more read-only runtime tool executions are required to satisfy the request.
- action: any state-changing operation is required or requested.
- clarify: the user's intent itself is too ambiguous to determine a route.

Rules:
- Do not plan, select tools, or decompose the request.
- Classify by required runtime tool execution, not by whether the user is asking for information.
- Questions about the user, prior conversation, or Planner memory are conversation,
  even when the requested fact may be unavailable. Missing answer data is not ambiguous
  intent; the Planner handles NEEDS_INPUT when necessary.
- Runtime-discoverable details are not grounds for clarify.
- Missing execution details are not grounds for clarify when intent is known; the Planner handles NEEDS_INPUT.
- Mixed read and write requests are action.
- Mixed inspect, modify, and verify requests are action.

Return only the structured route.
"""


logger = get_logger(__name__)

class LangChainPlannerRouter:
    def __init__(
        self,
        *,
        router_llm=None,
        show_raw_llm: bool = False,
    ):
        self.router_llm = router_llm
        self.show_raw_llm = show_raw_llm

    def route(self, user_request: str) -> RoutingDecision:
        return planner_routing_decision(
            user_request,
            router_llm=self.router_llm,
            propagate_errors=True,
            show_raw_llm=self.show_raw_llm,
        )


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
        status = current_live_status()
        if status is not None:
            status.update("planner", "routing")

        structured_router = llm.with_structured_output(
            RouterDecisionSchema,
            method="json_schema",
            include_raw=True,
        )

        messages = [
            SystemMessage(content=PLANNER_ROUTER_PROMPT),
            HumanMessage(content=user_text),
        ]

        result = structured_router.invoke(messages)

        add_response_usage(result, worker="planner")

        log_llm_exchange(
            worker="planner",
            operation="route",
            messages=messages,
            response=result.get("raw"),
            execution_id=None,
            enabled=show_raw_llm,
        )

        # 1. Normal structured-output path
        payload = result.get("parsed")

        if payload is not None:
            return RoutingDecision(route=payload.route)

        # 2. Compatibility fallback from raw content
        raw = result.get("raw")
        content = (raw.content or "").strip() if raw is not None else ""

        # Proper JSON, but LangChain failed to parse it
        if content:
            try:
                payload = RouterDecisionSchema.model_validate_json(content)
                return RoutingDecision(route=payload.route)
            except Exception:
                pass

        # Model returned just the enum token, e.g. "clarify"
        if content in {"conversation", "info", "action", "clarify"}:
            return RoutingDecision(route=content)

        # 3. Nothing recoverable: now respect the original parsing error
        parsing_error = result.get("parsing_error")

        if parsing_error is not None:
            if propagate_errors:
                raise parsing_error

            logger.warning(f"LLM Routing parse failed: {parsing_error}")
            return None

        return None

    except Exception as exc:
        if propagate_errors:
            raise

        logger.warning(f"LLM Routing failed: {exc}")
        return None


def planner_routing_decision(
    user_text: str,
    router_llm: ChatOllama | None = None,
    *,
    propagate_errors: bool = False,
    show_raw_llm: bool = False,
) -> RoutingDecision:
    text = (user_text or "").strip()

    if not text:
        return RoutingDecision(route="clarify")

    if router_llm is None:
        return RoutingDecision(route="conversation")

    decision = _llm_route_decision(
        user_text=text,
        llm=router_llm,
        propagate_errors=propagate_errors,
        show_raw_llm=show_raw_llm,
    )

    if decision is None:
        print("ROUTER FALLBACK -> conversation")
        return RoutingDecision(route="conversation")


    return decision or RoutingDecision(route="conversation")
