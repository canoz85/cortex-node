from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from core.planner_routing import (
    RouterDecisionSchema,
    planner_routing_decision,
)


class RouterLLM:
    def __init__(self, *, route=None, error=None, parsing_error=None):
        self.route = route
        self.error = error
        self.parsing_error = parsing_error

    def with_structured_output(
        self,
        schema,
        method="json_schema",
        include_raw=False,
    ):
        assert schema is RouterDecisionSchema
        assert method == "json_schema"

        def invoke(messages):
            if self.error is not None:
                raise self.error

            parsed = (
                RouterDecisionSchema(route=self.route)
                if self.route is not None and self.parsing_error is None
                else None
            )

            if include_raw:
                return {
                    "raw": SimpleNamespace(content="router output"),
                    "parsed": parsed,
                    "parsing_error": self.parsing_error,
                }

            return parsed

        return SimpleNamespace(invoke=invoke)


@pytest.mark.parametrize(
    ("user_text", "route"),
    [
        ("explain Python decorators", "conversation"),
        ("list files", "info"),
        ("read config and change timeout to 30", "action"),
        (
            "inspect the config, modify the timeout, and verify the change",
            "action",
        ),
        ("do the thing", "clarify"),
        (
            "change the timeout, using the applicable config file",
            "action",
        ),
    ],
)
def test_execution_mode_routes(user_text, route):
    decision = planner_routing_decision(
        user_text,
        RouterLLM(route=route),
    )

    assert decision.route == route


def test_empty_request_routes_to_clarify_without_llm():
    decision = planner_routing_decision("")

    assert decision.route == "clarify"


@pytest.mark.parametrize(
    "route",
    [
        "conversation",
        "info",
        "action",
        "clarify",
    ],
)
def test_router_schema_accepts_supported_routes(route):
    decision = RouterDecisionSchema(route=route)

    assert decision.route == route


def test_router_schema_rejects_removed_fields():
    with pytest.raises(ValidationError, match="extra"):
        RouterDecisionSchema(
            route="info",
            confidence=0.9,
            enforced=False,
            reason="read only",
            domain="workspace",
        )


def test_router_failure_falls_back_to_conversation():
    decision = planner_routing_decision(
        "hello",
        RouterLLM(error=RuntimeError("failed")),
    )

    assert decision.route == "conversation"


def test_router_parsing_failure_falls_back_to_conversation():
    decision = planner_routing_decision(
        "hello",
        RouterLLM(
            parsing_error=ValueError("invalid structured output"),
        ),
    )

    assert decision.route == "conversation"


def test_router_failure_is_propagated_when_requested():
    with pytest.raises(RuntimeError, match="failed"):
        planner_routing_decision(
            "hello",
            RouterLLM(error=RuntimeError("failed")),
            propagate_errors=True,
        )


def test_router_parsing_failure_is_propagated_when_requested():
    with pytest.raises(ValueError, match="invalid structured output"):
        planner_routing_decision(
            "hello",
            RouterLLM(
                parsing_error=ValueError(
                    "invalid structured output"
                ),
            ),
            propagate_errors=True,
        )