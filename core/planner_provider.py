"""LangChain transport for the current Planner and its existing intent router."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException

from core.graph_intents import planner_routing_decision
from core.planner import PlannerMessage, PlannerRouting
from core.planner_contract import PlannerInvalidOutputError, PlannerProposal
from pydantic import ValidationError


class LangChainPlannerProvider:
    def __init__(self, *, planner_llm, router_llm=None, show_raw_llm: bool = False):
        self.show_raw_llm = show_raw_llm
        self.planner_llm = planner_llm
        self.router_llm = router_llm

    def route(self, user_request: str) -> PlannerRouting:
        # Invocation exceptions must reach FAILED.
        decision = planner_routing_decision(
            user_request, router_llm=self.router_llm, propagate_errors=True, show_raw_llm=self.show_raw_llm,
        )
        return PlannerRouting(decision.route)

    def generate(self, messages: tuple[PlannerMessage, ...]) -> PlannerProposal:
        provider_messages = [
            HumanMessage(content=message.content) if message.role == "human"
            else SystemMessage(content=message.content)
            for message in messages
        ]
        structured = self.planner_llm.with_structured_output(
            PlannerProposal, method="json_schema"
        )
        try:
            value = structured.invoke(provider_messages)
            return PlannerProposal.model_validate(value)
        except (ValidationError, OutputParserException) as exc:
            raise PlannerInvalidOutputError("Planner output failed schema validation") from exc
