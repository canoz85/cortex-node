from pathlib import Path
import shutil

import pytest

from conftest import get_tool, parse_result
from core.brain_normalization import normalize_brain_output
from core.protocol.controller import CortexController
from core.protocol.models import ToolRequest, ToolExecutionRecord
from core.runtime.tool_result_integration import SerializedToolRuntimePort
from test_git_ops import _init_repo, _run
from test_brain_outcomes import brain_input, controller_input, native_action
from tools.git_ops import get_git_tools


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def changed(path, **arguments):
    return parse_result(get_tool(get_git_tools(str(path)), "git_changed_files").invoke(arguments))


def commit_initial(path):
    _init_repo(path)
    for name in ("modified.txt", "deleted.txt", "renamed.txt"):
        (path / name).write_text(f"initial {name}\n", encoding="utf-8")
    _run(["git", "add", "."], path)
    _run(["git", "commit", "-m", "initial"], path)


def test_changed_files_states_are_structured_and_read_only(tmp_path):
    commit_initial(tmp_path)
    (tmp_path / "modified.txt").write_text("staged change\n", encoding="utf-8")
    (tmp_path / "added.txt").write_text("new\n", encoding="utf-8")
    _run(["git", "add", "modified.txt", "added.txt"], tmp_path)
    (tmp_path / "modified.txt").write_text("unstaged change\n", encoding="utf-8")
    (tmp_path / "deleted.txt").unlink()
    (tmp_path / "untracked space.txt").write_text("untracked\n", encoding="utf-8")
    _run(["git", "mv", "renamed.txt", "new name.txt"], tmp_path)
    before = {p.relative_to(tmp_path).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in tmp_path.rglob("*") if p.is_file()}
    result = changed(tmp_path)
    assert result["success"] is True
    data = result["data"]
    assert data["branch"] and data["upstream"] is None
    files = {item["path"]: item for item in data["changed_files"]}
    assert list(files) == sorted(files)
    assert files["modified.txt"]["status"] == "modified"
    assert files["modified.txt"]["staged"] is True
    assert files["modified.txt"]["index_status"] == "M"
    assert files["modified.txt"]["worktree_status"] == "M"
    assert files["added.txt"]["status"] == "added" and files["added.txt"]["staged"]
    assert files["deleted.txt"]["status"] == "deleted" and not files["deleted.txt"]["staged"]
    assert files["untracked space.txt"]["status"] == "untracked"
    assert files["new name.txt"]["original_path"] == "renamed.txt"
    assert files["new name.txt"]["status"] == "renamed"
    after = {p.relative_to(tmp_path).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
             for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_clean_repository_and_upstream_metadata(tmp_path):
    commit_initial(tmp_path)
    _run(["git", "branch", "baseline"], tmp_path)
    _run(["git", "branch", "--set-upstream-to=baseline"], tmp_path)
    data = changed(tmp_path)["data"]
    assert data["changed_files"] == [] and data["count"] == 0
    assert data["upstream"] == "baseline"
    assert data["ahead"] == data["behind"] == 0


def test_non_repository_returns_failure(tmp_path):
    result = changed(tmp_path)
    assert result["success"] is False
    assert result["error_code"] == "GIT_COMMAND_FAILED"


def test_workspace_subdirectory_is_scoped_and_paths_are_relative(tmp_path):
    commit_initial(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "inside.txt").write_text("inside\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside\n", encoding="utf-8")
    assert [item["path"] for item in changed(workspace)["data"]["changed_files"]] == ["inside.txt"]


def test_changed_files_paginate_as_typed_execution_evidence(tmp_path):
    _init_repo(tmp_path)
    for name in ("c.txt", "a.txt", "b.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    runtime = SerializedToolRuntimePort(get_git_tools(str(tmp_path)))
    result = runtime.execute(ToolRequest(request_id="changes", tool_name="git_changed_files", arguments={"limit": 2}))
    assert result.success
    assert [item["path"] for item in result.data["changed_files"]] == ["a.txt", "b.txt"]
    assert result.pagination.has_more and result.pagination.total_items == 3
    assert changed(tmp_path, offset=2, limit=2)["data"]["changed_files"][0]["path"] == "c.txt"


def test_changed_files_bind_through_existing_exact_collection_provenance(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "new.txt").write_text("new", encoding="utf-8")
    runtime = SerializedToolRuntimePort(get_git_tools(str(tmp_path)))
    result = runtime.execute(ToolRequest(request_id="changes", tool_name="git_changed_files"))
    context = brain_input()
    outcome = normalize_brain_output(native_action("brain_step_completed", {
        "message": "One untracked file: new.txt.", "exact_collection_source_record_index": 0,
        "exact_collection_data_path": ["changed_files"],
    }), context, {"git_changed_files"})
    record = ToolExecutionRecord(execution_id=context.identity.execution_id, plan_id=context.active_plan.plan_id,
                                 plan_revision=context.active_plan.revision, step_id=context.active_step.step_id,
                                 tool_name="git_changed_files", result=result)
    decision = CortexController(24).decide(controller_input(context, outcome).model_copy(update={"tool_execution_history": (record,)}))
    exact = decision.completion_evidence.exact_collection
    assert exact.items[0]["path"] == "new.txt"
    assert exact.source_request_id == "changes"
    assert exact.data_path == ("changed_files",)
