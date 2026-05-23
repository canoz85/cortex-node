import shutil
import subprocess
from pathlib import Path

import pytest

from tools.git_ops import get_git_tools

from conftest import get_tool, parse_result


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _run(cmd: list[str], cwd: Path):
    subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True)


def _init_repo(path: Path):
    _run(["git", "init"], cwd=path)
    _run(["git", "config", "user.name", "Test User"], cwd=path)
    _run(["git", "config", "user.email", "test@example.com"], cwd=path)


def test_git_status_succeeds_in_initialized_repo(tmp_path: Path):
    _init_repo(tmp_path)
    tools = get_git_tools(str(tmp_path))
    git_status = get_tool(tools, "git_status")

    result = parse_result(git_status.invoke({}))
    assert result["success"] is True
    assert result["data"]["exit_code"] == 0


def test_git_diff_reports_modified_file(tmp_path: Path):
    _init_repo(tmp_path)
    file_path = tmp_path / "demo.txt"
    file_path.write_text("v1\n", encoding="utf-8")
    _run(["git", "add", "demo.txt"], cwd=tmp_path)
    _run(["git", "commit", "-m", "init"], cwd=tmp_path)
    file_path.write_text("v2\n", encoding="utf-8")

    tools = get_git_tools(str(tmp_path))
    git_diff = get_tool(tools, "git_diff")
    result = parse_result(git_diff.invoke({"path": "demo.txt"}))

    assert result["success"] is True
    assert "demo.txt" in result["data"]["stdout"]


def test_git_log_limit_is_clamped_to_50(tmp_path: Path):
    _init_repo(tmp_path)
    file_path = tmp_path / "demo.txt"
    file_path.write_text("v1\n", encoding="utf-8")
    _run(["git", "add", "demo.txt"], cwd=tmp_path)
    _run(["git", "commit", "-m", "init"], cwd=tmp_path)

    tools = get_git_tools(str(tmp_path))
    git_log = get_tool(tools, "git_log")
    result = parse_result(git_log.invoke({"limit": 999}))

    assert result["success"] is True
    assert "-n50" in result["data"]["args"]
