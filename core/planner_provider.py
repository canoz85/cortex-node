"""LangChain transport for Planner proposal generation."""

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from core.planner import PlannerMessage
from core.planner_contract import PlannerInvalidOutputError, PlannerProposal
from core.debug import log_llm_exchange


class LangChainPlannerProvider:
    def __init__(self, *, planner_llm, show_raw_llm: bool = False):
        self.planner_llm = planner_llm
        self.show_raw_llm = show_raw_llm

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

            try:
                structured = self.planner_llm.with_structured_output(
                    PlannerProposal, method="json_schema", include_raw=True,
                )
            except TypeError:
                structured = self.planner_llm.with_structured_output(
                    PlannerProposal, method="json_schema",
                )
            exchange = structured.invoke(provider_messages)
            is_envelope = (
                isinstance(exchange, dict)
                and "raw" in exchange
                and "parsed" in exchange
            )
            raw = exchange.get("raw") if is_envelope else exchange
            log_llm_exchange(
                worker="planner",
                operation="plan",
                messages=provider_messages,
                response=raw,
                execution_id=messages[0].execution_id if messages else None,
                enabled=self.show_raw_llm,
            )
            if is_envelope:
                parsing_error = exchange.get("parsing_error")
                if parsing_error is not None:
                    raise parsing_error
                value = exchange.get("parsed")
            else:
                value = exchange
            return PlannerProposal.model_validate(value)

        except (ValidationError, OutputParserException) as exc:
            raise PlannerInvalidOutputError(
                "Planner output failed schema validation"
            ) from exc
