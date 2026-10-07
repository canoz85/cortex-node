"""LangChain transport for Planner proposal generation."""

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from core.planner import PlannerMessage
from core.planner_contract import PlannerInvalidOutputError, PlannerProposal, authorized_planner_schema
from core.planner_normalization import MAX_PROPOSED_STEPS
from core.debug import log_llm_exchange
from core.logging.live_status import add_response_usage
from core.logging.live_status import current_live_status


def _schema_diagnostic(error: Exception) -> str:
    # LangChain wraps Pydantic errors; inspect structured errors, never their text.
    if isinstance(error, OutputParserException) and error.__cause__ is not None:
        error = error.__cause__
    if isinstance(error, ValidationError):
        for issue in error.errors(include_input=False, include_url=False):
            location = issue["loc"]
            if location == ("steps",) and issue["type"] == "too_long":
                if issue.get("ctx", {}).get("max_length") == 0:
                    return "No executable steps are allowed without authorized tools"
                return f"PLAN_PROPOSED exceeds the {MAX_PROPOSED_STEPS}-step limit"
            if location and location[-1] == "primary_tool" and issue["type"] == "literal_error":
                return "primary_tool must be one of the currently authorized tools"
            if issue["type"] == "missing":
                return "Planner proposal is missing required fields"
            if issue["type"] in {"string_too_short", "string_type"}:
                return "Required Planner semantic fields must be non-empty strings"
            if issue["type"] == "extra_forbidden":
                return "Planner proposal contains unsupported fields"
    return "Planner output did not match the required proposal schema."


def _extract_planner_proposal(exchange, proposal_schema=PlannerProposal) -> PlannerProposal:
    is_envelope = (
        isinstance(exchange, dict)
        and "raw" in exchange
        and "parsed" in exchange
    )

    if not is_envelope:
        return proposal_schema.model_validate(exchange)

    parsed = exchange.get("parsed")

    if parsed is not None:
        if isinstance(parsed, PlannerProposal):
            parsed = parsed.model_dump()

        return proposal_schema.model_validate(parsed)

    parsing_error = exchange.get("parsing_error")

    if parsing_error is not None:
        raise PlannerInvalidOutputError(
            _schema_diagnostic(parsing_error)
        ) from parsing_error

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

            proposal_schema = authorized_planner_schema(
                messages[0].available_tools if messages else (),
                max_steps=MAX_PROPOSED_STEPS,
            )
            structured = self.planner_llm.with_structured_output(
                proposal_schema, method="json_schema", include_raw=True,
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
                
            return _extract_planner_proposal(exchange, proposal_schema)

        except PlannerInvalidOutputError:
            raise
        except (ValidationError, OutputParserException) as exc:
            raise PlannerInvalidOutputError(
                _schema_diagnostic(exc)
            ) from exc
