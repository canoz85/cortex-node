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
