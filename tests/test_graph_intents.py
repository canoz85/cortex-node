from core.graph_intents import (
    is_file_generation_request,
    planner_routing_decision,
    preferred_info_tool,
    requires_action,
)


def test_preferred_info_tool_detection():
    assert preferred_info_tool("show token usage") == "token_usage"
    assert preferred_info_tool("what time is it now") == "current_time"
    assert preferred_info_tool("show model runtime configuration") == "agent_info"
    assert preferred_info_tool("write a script") is None


def test_requires_action_detection():
    assert requires_action("create a script") is True
    assert requires_action("hello there") is False


def test_file_generation_intent_detection():
    assert is_file_generation_request("create a python script file for cli") is True
    assert is_file_generation_request("explain python classes") is False


def test_planner_route_for_casual_chat():
    decision = planner_routing_decision("hi there")
    assert decision.route == "casual"
    assert decision.domain == "general"


def test_planner_route_for_sap_action_with_explicit_override():
    decision = planner_routing_decision("[domain:sap] create a report for mara")
    assert decision.route == "action:sap"
    assert decision.domain == "sap"
    assert decision.enforced is True


def test_planner_route_for_mixed_action_requires_clarification():
    decision = planner_routing_decision("create python pandas script to analyze SAP MARA exports")
    assert decision.route == "clarify_domain"
    assert decision.needs_clarification is True


def test_planner_route_for_coding_discussion_question():
    decision = planner_routing_decision("how should I refactor this python function?")
    assert decision.route == "coding_discussion"
