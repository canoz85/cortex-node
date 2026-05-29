from core.graph_intents import (
    is_file_fact_extraction_request,
    is_read_only_file_request,
    is_read_audit_request,
    is_file_generation_request,
    planner_routing_decision,
    preferred_file_tool,
    preferred_info_tool,
    requires_action,
)


def test_preferred_info_tool_detection():
    assert preferred_info_tool("show token usage") == "token_usage"
    assert preferred_info_tool("what time is it now") == "current_time"
    assert preferred_info_tool("1 meter is 150 cm. what is 10 meter") == "solve_math"
    assert preferred_info_tool("what is 12 / (3 + 1)") == "solve_math"
    assert preferred_info_tool("how much is 1 meter") == "solve_math"
    assert preferred_info_tool("what is polymorphism in OOP") is None
    assert preferred_info_tool("show model runtime configuration") == "agent_info"
    assert preferred_info_tool("write a script") is None
    assert preferred_info_tool("what worked? what failed? what should be avoided next time?") is None


def test_requires_action_detection():
    assert requires_action("create a script") is True
    assert requires_action("read workspace/is_prime.py and explain the bug") is True
    assert requires_action("list workspace") is True
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


def test_planner_route_for_workspace_file_access_request():
    decision = planner_routing_decision("analyze workspace/is_prime.py")
    assert decision.route == "action"


def test_read_only_file_request_detection():
    assert is_read_only_file_request("read workspace/is_prime.py and explain issues") is True
    assert is_read_only_file_request("edit workspace/is_prime.py") is False


def test_preferred_file_tool_for_workspace_listing():
    assert preferred_file_tool("list workspace") == "list_files"
    assert preferred_file_tool("show files in workspace") == "list_files"
    assert preferred_file_tool("hello") is None


def test_read_audit_intent_detection():
    assert is_read_audit_request("which file did you read") is True
    assert is_read_audit_request("what files did you analyze") is True
    assert is_read_audit_request("hello there") is False


def test_file_fact_extraction_intent_detection():
    assert is_file_fact_extraction_request("what is the pressure value?") is True
    assert is_file_fact_extraction_request("tell me the device_id from latest data") is True
    assert is_file_fact_extraction_request("which file did you read") is False
    assert is_file_fact_extraction_request("create a script that validates pressure and status") is False


def test_planner_route_for_file_fact_extraction_request():
    decision = planner_routing_decision("what is the pressure value")
    assert decision.route == "action"


def test_golden_prompt_route_list_workspace():
    decision = planner_routing_decision("list workspace")
    assert decision.route == "action"


def test_golden_prompt_route_read_audit_query():
    decision = planner_routing_decision("which file did you read")
    assert decision.route == "action"


def test_reflection_prompt_routes_to_conversation_not_info():
    decision = planner_routing_decision("what worked? what failed? what should be avoided next time?")
    assert decision.route == "conversation"
