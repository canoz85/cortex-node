import pytest

from core.graph_intents import RouterDecisionSchema, _llm_route_decision


class DummyStructuredRunnable:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.invocations: list[list] = []

    def invoke(self, messages):
        self.invocations.append(messages)
        if self.error is not None:
            raise self.error
        return self.result


class DummyRouterLLM:
    def __init__(self, result=None, error: Exception | None = None, schema_error: Exception | None = None):
        self.result = result
        self.error = error
        self.schema_error = schema_error
        self.requested_schemas: list[tuple[type, str]] = []

    def with_structured_output(self, schema, method="json_schema"):
        self.requested_schemas.append((schema, method))
        if self.error is not None:
            raise self.error
        return DummyStructuredRunnable(result=self.result, error=self.schema_error)


def test_llm_route_decision_accepts_valid_info_response():
    llm = DummyRouterLLM(
        result=RouterDecisionSchema(
            route="info",
            domain="general",
            confidence=0.9,
            enforced=False,
            needs_clarification=False,
            reason="read-only request",
        )
    )

    decision = _llm_route_decision("read workspace files", llm)

    assert decision is not None
    assert decision.route == "info"
    assert decision.domain == "general"
    assert decision.source == "llm_router"


def test_llm_route_decision_accepts_valid_action_response():
    llm = DummyRouterLLM(
        result=RouterDecisionSchema(
            route="action",
            domain="python",
            confidence=0.85,
            enforced=False,
            needs_clarification=False,
            reason="tool execution needed",
        )
    )

    decision = _llm_route_decision("create a file", llm)

    assert decision is not None
    assert decision.route == "action"
    assert decision.domain == "python"
    assert decision.source == "llm_router"


def test_llm_route_decision_rejects_invalid_route():
    llm = DummyRouterLLM(
        result=RouterDecisionSchema(
            route="casual",
            domain="general",
            confidence=0.9,
            enforced=False,
            needs_clarification=False,
            reason="invalid route",
        )
    )

    assert _llm_route_decision("hello", llm) is None


def test_llm_route_decision_rejects_invalid_domain():
    llm = DummyRouterLLM(
        result=RouterDecisionSchema(
            route="info",
            domain="finance",
            confidence=0.9,
            enforced=False,
            needs_clarification=False,
            reason="invalid domain",
        )
    )

    assert _llm_route_decision("hello", llm) is None


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_llm_route_decision_rejects_out_of_range_confidence(confidence):
    llm = DummyRouterLLM(
        result=RouterDecisionSchema(
            route="info",
            domain="general",
            confidence=confidence,
            enforced=False,
            needs_clarification=False,
            reason="bad confidence",
        )
    )

    assert _llm_route_decision("hello", llm) is None


def test_llm_route_decision_returns_none_on_structured_output_exception():
    llm = DummyRouterLLM(error=RuntimeError("structured output failed"))

    assert _llm_route_decision("hello", llm) is None