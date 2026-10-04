from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal

from core.planner import PlannerRoute
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


class RouterDecisionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route: Literal["conversation", "info", "action", "clarify"]


class LangChainPlannerRouter:
    """Strict execution-mode classification; no planning or fallback routing."""

    def __init__(self, *, router_llm, show_raw_llm: bool = False):
        self.router_llm = router_llm
        self.show_raw_llm = show_raw_llm

    def route(self, user_request: str) -> PlannerRoute:
        status = current_live_status()
        if status is not None:
            status.update("planner", "routing")
        structured = self.router_llm.with_structured_output(
            RouterDecisionSchema, method="json_schema", include_raw=True,
        )
        messages = [
            SystemMessage(content=PLANNER_ROUTER_PROMPT),
            HumanMessage(content=user_request),
        ]
        exchange = structured.invoke(messages)
        add_response_usage(exchange, worker="planner")
        log_llm_exchange(
            worker="planner", operation="route", messages=messages,
            response=exchange.get("raw"), execution_id=None,
            enabled=self.show_raw_llm,
        )
        parsed = exchange.get("parsed")
        if parsed is None:
            error = exchange.get("parsing_error")
            if error is not None:
                raise error
            raise ValueError("Router structured output contained no route")
        decision = RouterDecisionSchema.model_validate(parsed)
        return PlannerRoute(route=decision.route)
