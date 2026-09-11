"""LangChain transport for the current Planner and its existing intent router."""

from langchain_core.messages import HumanMessage, SystemMessage

from core.graph_intents import planner_routing_decision
from core.planner import PlannerMessage, PlannerRouting


class LangChainPlannerProvider:
    def __init__(self, *, planner_llm, router_llm=None):
        self.planner_llm = planner_llm
        self.router_llm = router_llm

    def route(self, user_request: str) -> PlannerRouting:
        # Keep confidence arbitration; invocation exceptions must reach FAILED.
        decision = planner_routing_decision(
            user_request, router_llm=self.router_llm, propagate_errors=True,
        )
        return PlannerRouting(decision.route, decision.domain, decision.confidence, decision.reason)

    def generate(self, messages: tuple[PlannerMessage, ...]) -> object:
        provider_messages = [
            HumanMessage(content=message.content) if message.role == "human"
            else SystemMessage(content=message.content)
            for message in messages
        ]
        # Exceptions cross the port to the service's existing FAILED result path.
        # No corrective calls or implicit retries.
        return self.planner_llm.invoke(provider_messages).content
