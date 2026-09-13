from types import SimpleNamespace

import pytest

from core.graph_intents import RouterDecisionSchema, planner_routing_decision


class RouterLLM:
    def __init__(self, route):
        self.result = RouterDecisionSchema(route=route)

    def with_structured_output(self, schema, method="json_schema"):
        return SimpleNamespace(invoke=lambda messages: self.result)


@pytest.mark.parametrize(("user_text", "route"), [
    ("explain Python decorators", "conversation"),
    ("list files", "info"),
    ("read config and change timeout to 30", "action"),
    ("inspect the config, modify the timeout, and verify the change", "action"),
    ("do the thing", "clarify"),
    ("change the timeout, using the applicable config file", "action"),
])
def test_execution_mode_routes(user_text, route):
    assert planner_routing_decision(user_text, RouterLLM(route)).route == route


def test_empty_intent_is_ambiguous():
    assert planner_routing_decision("").route == "clarify"


def test_valid_model_route_is_not_overridden_by_confidence_arbitration():
    decision = planner_routing_decision("list files", RouterLLM("info"))
    assert decision.route == "info"
