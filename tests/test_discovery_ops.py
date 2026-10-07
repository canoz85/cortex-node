from pathlib import Path

import pytest

from conftest import get_tool, parse_result
from core.protocol.models import ToolRequest
from core.tool_output import normalize_tool_output
from tools.discovery_ops import get_discovery_tools


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "z.py").write_text("needle\nexact_collection = 1\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "core" / "b.py").write_text("other\nexact_collection = 2\nneedle\n", encoding="utf-8")
    (tmp_path / "core" / "a.txt").write_text("exact_collection = 3\n", encoding="utf-8")
    return tmp_path, get_discovery_tools(str(tmp_path))


def test_find_files_recursive_and_nonrecursive_deterministic(workspace):
    _, tools = workspace
    find = get_tool(tools, "find_files")
    result = parse_result(find.invoke({"pattern": "*.py"}))
    assert result["success"] is True
    assert result["data"]["files"] == ["a.py", "core/b.py", "z.py"]
    assert result["data"]["count"] == 3
    assert result["data"]["truncated"] is False
    assert parse_result(find.invoke({"pattern": "*.py", "recursive": False}))["data"]["files"] == ["a.py", "z.py"]
    assert parse_result(find.invoke({"pattern": "*.py", "path": "core"}))["data"]["files"] == ["core/b.py"]


def test_find_files_pagination_survives_runtime_normalization(workspace):
    _, tools = workspace
    find = get_tool(tools, "find_files")
    raw = find.invoke({"pattern": "*.py", "limit": 2})
    result = normalize_tool_output(raw_content=raw, request=ToolRequest(request_id="find", tool_name="find_files"))
    assert result.data["files"] == ["a.py", "core/b.py"]
    assert result.integrity.is_truncated
    assert result.pagination.has_more
    assert result.pagination.total_items == 3
    final = parse_result(find.invoke({"pattern": "*.py", "offset": 2, "limit": 2}))
    assert final["data"]["files"] == ["z.py"]
    assert final["has_more"] is False
    assert final["is_truncated"] is False


@pytest.mark.parametrize("name,arguments", [
    ("find_files", {}), ("search_text", {"query": "needle"}),
])
def test_discovery_rejects_escape_and_invalid_bounds(workspace, name, arguments):
    _, tools = workspace
    action = get_tool(tools, name)
    for updates in ({"path": ".."}, {"offset": -1}, {"limit": 101}, {"path": "missing"}):
        assert parse_result(action.invoke({**arguments, **updates}))["success"] is False


def test_search_literal_pattern_and_regex(workspace):
    _, tools = workspace
    search = get_tool(tools, "search_text")
    literal = parse_result(search.invoke({"query": "exact_collection", "file_pattern": "*.py"}))
    assert literal["success"] is True
    assert [(item["path"], item["line_number"], item["text"]) for item in literal["data"]["matches"]] == [
        ("core/b.py", 2, "exact_collection = 2"), ("z.py", 2, "exact_collection = 1"),
    ]
    regex = parse_result(search.invoke({"query": "exact_.* = [23]", "regex": True, "path": "core"}))
    assert [item["path"] for item in regex["data"]["matches"]] == ["core/a.txt", "core/b.py"]
    assert parse_result(search.invoke({"query": "exact_.*", "regex": False}))["data"]["count"] == 0
    assert parse_result(search.invoke({"query": "[", "regex": True}))["success"] is False


def test_search_bounds_and_deterministic_continuation(workspace):
    _, tools = workspace
    search = get_tool(tools, "search_text")
    raw = search.invoke({"query": "needle", "limit": 1})
    result = normalize_tool_output(raw_content=raw, request=ToolRequest(request_id="search", tool_name="search_text"))
    assert result.data["matches"][0]["path"] == "a.py"
    assert result.data["count"] == 3
    assert result.data["truncated"] and result.pagination.has_more
    final = parse_result(search.invoke({"query": "needle", "offset": 1, "limit": 2}))
    assert [item["path"] for item in final["data"]["matches"]] == ["core/b.py", "z.py"]
    assert final["has_more"] is False


def test_search_long_line_is_explicitly_bounded(tmp_path):
    (tmp_path / "long.txt").write_text("x" * 3000 + "needle\n", encoding="utf-8")
    result = parse_result(get_tool(get_discovery_tools(str(tmp_path)), "search_text").invoke({"query": "needle"}))
    match = result["data"]["matches"][0]
    assert len(match["text"]) == 2000
    assert "needle" in match["text"]
    assert match["text_start_column"] > 1
    assert match["text_truncated"] is True


def test_discovery_does_not_follow_outside_links(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("needle", encoding="utf-8")
    try:
        (root / "outside").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this system")
    tools = get_discovery_tools(str(root))
    assert parse_result(get_tool(tools, "find_files").invoke({}))["data"]["files"] == []
    assert parse_result(get_tool(tools, "search_text").invoke({"query": "needle"}))["data"]["matches"] == []
    assert parse_result(get_tool(tools, "find_files").invoke({"path": "outside"}))["success"] is False
