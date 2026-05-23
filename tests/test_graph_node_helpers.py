from core.graph_node_helpers import detect_missing_dependency, planner_execution_brief
from core.models import ToolResult


def test_detect_missing_dependency_from_modulenotfounderror():
    payload = ToolResult(
        success=False,
        message="Python execution completed",
        data={
            "exit_code": 1,
            "stdout": "<empty>",
            "stderr": "Traceback... ModuleNotFoundError: No module named 'pendulum'",
        },
    ).model_dump()

    assert detect_missing_dependency(payload) == "pendulum"


def test_detect_missing_dependency_returns_none_when_no_import_error():
    payload = ToolResult(
        success=False,
        message="Python execution completed",
        data={
            "exit_code": 1,
            "stdout": "<empty>",
            "stderr": "ValueError: bad input",
        },
    ).model_dump()

    assert detect_missing_dependency(payload) is None


def test_planner_execution_brief_truncates_long_plan_text():
    long_text = "x" * 2000
    brief = planner_execution_brief("action", "llm", long_text)

    assert "Route: action" in brief
    assert "Plan source: llm" in brief
    assert len(brief) < 1700
    assert brief.endswith("...")
