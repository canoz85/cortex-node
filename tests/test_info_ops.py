from tools import info_ops
from tools.info_ops import get_info_tools, update_token_usage

from conftest import get_tool, parse_result


def setup_function():
    info_ops._runtime.clear()


def test_agent_info_includes_runtime_configuration():
    tools = get_info_tools(model="qwen", workspace_dir="workspace")
    agent_info = get_tool(tools, "agent_info")

    result = parse_result(agent_info.invoke({}))
    assert result["success"] is True
    assert result["data"]["model"] == "qwen"
    assert result["data"]["workspace"] == "workspace"


def test_token_usage_reports_empty_before_updates():
    tools = get_info_tools(model="qwen", workspace_dir="workspace")
    token_usage = get_tool(tools, "token_usage")

    result = parse_result(token_usage.invoke({}))
    assert result["success"] is False
    assert "No token usage" in result["message"]


def test_update_token_usage_accumulates_values():
    get_info_tools(model="qwen", workspace_dir="workspace")
    update_token_usage({"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13})
    update_token_usage({"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})

    tools = get_info_tools(model="qwen", workspace_dir="workspace")
    token_usage = get_tool(tools, "token_usage")
    result = parse_result(token_usage.invoke({}))

    assert result["success"] is True
    assert result["data"]["prompt_tokens"] == 15
    assert result["data"]["completion_tokens"] == 5
    assert result["data"]["total_tokens"] == 20


def test_update_token_usage_handles_string_values():
    get_info_tools(model="qwen", workspace_dir="workspace")
    update_token_usage({"prompt_tokens": "10", "completion_tokens": "3", "total_tokens": "13"})
    update_token_usage({"prompt_tokens": "5", "completion_tokens": "2", "total_tokens": "7"})

    tools = get_info_tools(model="qwen", workspace_dir="workspace")
    token_usage = get_tool(tools, "token_usage")
    result = parse_result(token_usage.invoke({}))

    assert result["success"] is True
    assert result["data"]["prompt_tokens"] == 15
    assert result["data"]["completion_tokens"] == 5
    assert result["data"]["total_tokens"] == 20


def test_update_token_usage_ignores_invalid_or_negative_values():
    get_info_tools(model="qwen", workspace_dir="workspace")
    update_token_usage({"prompt_tokens": "bad", "completion_tokens": -2, "total_tokens": None})

    tools = get_info_tools(model="qwen", workspace_dir="workspace")
    token_usage = get_tool(tools, "token_usage")
    result = parse_result(token_usage.invoke({}))

    assert result["success"] is True
    assert result["data"]["prompt_tokens"] == 0
    assert result["data"]["completion_tokens"] == 0
    assert result["data"]["total_tokens"] == 0


def test_solve_math_handles_proportional_question_deterministically():
    tools = get_info_tools(model="qwen", workspace_dir="workspace")
    solve_math = get_tool(tools, "solve_math")

    result = parse_result(solve_math.invoke({"question": "1 meter is 150 cm. what is 10 meter"}))

    assert result["success"] is True
    assert result["data"]["method"] == "proportional"
    assert result["data"]["result"] == 1500
    assert "Based on your stated relation (assumption)" in result["display"]
    assert "Used only the relationship provided in the question." in result["display"]
    assert result["data"]["warning"] is None


def test_solve_math_handles_arithmetic_expression():
    tools = get_info_tools(model="qwen", workspace_dir="workspace")
    solve_math = get_tool(tools, "solve_math")

    result = parse_result(solve_math.invoke({"question": "what is 12 / (3 + 1)"}))

    assert result["success"] is True
    assert result["data"]["method"] == "arithmetic"
    assert result["data"]["result"] == 3
    assert result["display"] == "Result: 3"


def test_solve_math_proportional_without_canonical_uses_assumption_note_only():
    tools = get_info_tools(model="qwen", workspace_dir="workspace")
    solve_math = get_tool(tools, "solve_math")

    result = parse_result(solve_math.invoke({"question": "1 widget is 3 sprockets. what is 5 widget"}))

    assert result["success"] is True
    assert result["data"]["method"] == "proportional"
    assert result["data"]["warning"] is None
    assert "Based on your stated relation (assumption)" in result["display"]
    assert "Used only the relationship provided in the question." in result["display"]


def test_solve_math_reuses_previously_stated_relation_in_followup_question():
    tools = get_info_tools(model="qwen", workspace_dir="workspace")
    solve_math = get_tool(tools, "solve_math")

    parse_result(solve_math.invoke({"question": "1 meter is 150 cm. what is 10 meter"}))
    follow_up = parse_result(solve_math.invoke({"question": "how much is 1 meter"}))

    assert follow_up["success"] is True
    assert follow_up["data"]["method"] == "proportional_from_context"
    assert follow_up["data"]["result"] == 150
    assert "Used the previously stated relationship from this session." in follow_up["display"]
