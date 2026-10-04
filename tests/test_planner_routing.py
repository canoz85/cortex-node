from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from core.planner_routing import (
    RouterDecisionSchema,
    LangChainPlannerRouter,
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
        assert include_raw

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
        ("hello", "conversation"),
        ("list files", "info"),
        ("write a file", "action"),
        ("which target", "clarify"),
    ],
)
def test_supported_route_transport_with_scripted_provider(user_text, route):
    decision = LangChainPlannerRouter(router_llm=RouterLLM(route=route)).route(user_text)

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


@pytest.mark.parametrize("error", [RuntimeError("provider failed"), ValueError("invalid structured output")])
def test_router_failures_propagate_without_fallback_route(error):
    router = LangChainPlannerRouter(router_llm=RouterLLM(error=error))
    with pytest.raises(type(error), match=str(error)):
        router.route("hello")


def test_router_does_not_reparse_raw_enum_or_json():
    for raw in ("info", '{"route":"info"}'):
        class Model:
            def with_structured_output(self, *_args, **_kwargs):
                return SimpleNamespace(invoke=lambda _: {"raw": SimpleNamespace(content=raw), "parsed": None})
        with pytest.raises(ValueError, match="contained no route"):
            LangChainPlannerRouter(router_llm=Model()).route("hello")
