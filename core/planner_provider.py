"""LangChain transport for Planner proposal generation."""

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from core.planner import PlannerMessage
from core.planner_contract import PlannerInvalidOutputError, PlannerProposal
from core.debug import log_llm_exchange
from core.logging.live_status import add_response_usage
from core.logging.live_status import current_live_status


def _extract_planner_proposal(exchange) -> PlannerProposal:
    is_envelope = (
        isinstance(exchange, dict)
        and "raw" in exchange
        and "parsed" in exchange
    )

    if not is_envelope:
        return PlannerProposal.model_validate(exchange)

    parsed = exchange.get("parsed")

    if parsed is not None:
        if isinstance(parsed, PlannerProposal):
            return parsed

        return PlannerProposal.model_validate(parsed)

    raw = exchange.get("raw")
    content = (raw.content or "").strip() if raw is not None else ""

    if content:
        try:
            print("PLANNER DEBUG: raw JSON fallback used")
            return PlannerProposal.model_validate_json(content)
        except ValidationError:
            pass

    parsing_error = exchange.get("parsing_error")

    if parsing_error is not None:
        raise parsing_error

    raise PlannerInvalidOutputError(
        "Planner structured output contained no valid proposal"
    )

class LangChainPlannerProvider:
    def __init__(self, *, planner_llm, show_raw_llm: bool = False):
        self.planner_llm = planner_llm
        self.show_raw_llm = show_raw_llm

    def generate(
        self,
        messages: tuple[PlannerMessage, ...],
    ) -> PlannerProposal:
        status = current_live_status()
        if status is not None:
            status.update("planner", "planning")
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
                    PlannerProposal, method="json_schema", include_raw=True
                )
            exchange = structured.invoke(provider_messages)
            add_response_usage(exchange, worker="planner")

            raw = exchange.get("raw") if isinstance(exchange, dict) else exchange
            log_llm_exchange(
                worker="planner",
                operation="plan",
                messages=provider_messages,
                response=raw,
                execution_id=messages[0].execution_id if messages else None,
                enabled=self.show_raw_llm,
            )
                
            return _extract_planner_proposal(exchange)

        except PlannerInvalidOutputError:
            raise
        except (ValidationError, OutputParserException) as exc:
            raise PlannerInvalidOutputError(
                f"Planner output failed schema validation: {exc}"
            ) from exc
