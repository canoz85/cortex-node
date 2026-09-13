import pytest
from pydantic import ValidationError

from core.graph_intents import RouterDecisionSchema, _llm_route_decision


class DummyRouterLLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def with_structured_output(self, schema, method="json_schema"):
        if self.error:
            raise self.error
        return type("Runnable", (), {"invoke": lambda _, messages: self.result})()


@pytest.mark.parametrize("route", ["conversation", "info", "action", "clarify"])
def test_route_only_contract_accepts_supported_routes(route):
    decision = _llm_route_decision("request", DummyRouterLLM(RouterDecisionSchema(route=route)))
    assert decision.route == route
    assert vars(decision) == {"route": route}


def test_route_contract_rejects_removed_fields():
    with pytest.raises(ValidationError, match="extra"):
        RouterDecisionSchema(route="info", confidence=.9, enforced=False,
                             reason="read only", domain="workspace")


def test_llm_route_decision_returns_none_on_structured_output_exception():
    assert _llm_route_decision("hello", DummyRouterLLM(error=RuntimeError("failed"))) is None
