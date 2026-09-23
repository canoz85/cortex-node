"""LangChain transport for Planner proposal generation."""

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from core.planner import PlannerMessage
from core.planner_contract import PlannerInvalidOutputError, PlannerProposal


class LangChainPlannerProvider:
    def __init__(self, *, planner_llm):
        self.planner_llm = planner_llm

    def generate(
        self,
        messages: tuple[PlannerMessage, ...],
    ) -> PlannerProposal:
        provider_messages = [
            (
                HumanMessage(content=message.content)
                if message.role == "human"
                else SystemMessage(content=message.content)
            )
            for message in messages
        ]

        try:

            structured = self.planner_llm.with_structured_output(
                PlannerProposal,
                method="json_schema",
            )
            value = structured.invoke(provider_messages)
            return PlannerProposal.model_validate(value)

        except (ValidationError, OutputParserException) as exc:
            raise PlannerInvalidOutputError(
                "Planner output failed schema validation"
            ) from exc