"""Structured, read-only Git changed-file evidence."""

from pathlib import Path, PurePosixPath
import subprocess

from langchain_core.tools import tool

from core.error_codes import GIT_COMMAND_FAILED, GIT_NOT_INSTALLED, GIT_RUNTIME_ERROR, GIT_TIMEOUT
from core.models import ToolOutputEnvelope
from tools.discovery_ops import page_bounds, paged_result


_STATUS_NAMES = {"M": "modified", "A": "added", "D": "deleted", "R": "renamed",
                 "C": "copied", "T": "type_changed", "U": "unmerged", "?": "untracked"}


def _parse_status(raw: bytes, prefix: str) -> tuple[dict, list[dict]]:
    records = iter(raw.decode("utf-8", errors="surrogateescape").split("\0"))
    headers = {}
    files = []

    def workspace_path(path):
        if not path.startswith(prefix):
            return None
        relative = path[len(prefix):]
        if not relative or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise ValueError("Git returned a path outside the workspace")
        return relative

    for record in records:
        if not record:
            continue
        if record.startswith("# "):
            key, _, value = record[2:].partition(" ")
            headers[key] = value
            continue
        kind = record[0]
        original = None
        if kind == "?":
            path, xy, submodule = record[2:], ".?", "N..."
        elif kind in ("1", "2", "u"):
            splits = {"1": 8, "2": 9, "u": 10}[kind]
            fields = record.split(" ", splits)
            if len(fields) != splits + 1 or len(fields[1]) != 2:
                raise ValueError("Malformed Git status record")
            xy, submodule, path = fields[1], fields[2], fields[-1]
            if kind == "2":
                original = workspace_path(next(records))
        else:
            raise ValueError("Unexpected Git status record")
        path = workspace_path(path)
        if path is None:
            continue
        index, worktree = xy
        status = "unmerged" if kind == "u" else _STATUS_NAMES.get(worktree if worktree != "." else index)
        if status is None:
            raise ValueError("Unknown Git file status")
        item = {"path": path, "status": status, "staged": index != "." and kind != "u",
                "index_status": index, "worktree_status": worktree,
                "worktree_changed": worktree != ".", "submodule_state": submodule}
        if kind == "2":
            item["original_path"] = original
        files.append(item)
    return headers, sorted(files, key=lambda item: item["path"])


def get_git_changed_files_tool(workspace_root: Path):
    @tool
    def git_changed_files(offset: int = 0, limit: int = 100) -> str:
        """Return structured Git changes within the workspace, sorted by path.

        Paths are workspace-relative with forward slashes. Include branch/upstream,
        staged and working-tree status codes ('.' means unchanged), and untracked
        files; ignored files are omitted. offset and limit (1..100) paginate records.
        Read-only: Git optional index refresh locks are disabled.
        """
        try:
            offset, limit = page_bounds(offset, limit)

            def read_git(args):
                return subprocess.run(["git", "--no-optional-locks", *args], cwd=str(workspace_root),
                                      capture_output=True, timeout=20, check=False)

            prefix_result = read_git(["rev-parse", "--show-prefix"])
            result = prefix_result
            if prefix_result.returncode == 0:
                result = read_git(["status", "--porcelain=v2", "--branch", "-z", "--untracked-files=all", "--", "."])
            if result.returncode != 0:
                return ToolOutputEnvelope(success=False, message="Git changed-file discovery failed.",
                    error_code=GIT_COMMAND_FAILED, data={"exit_code": result.returncode,
                    "stderr": result.stderr.decode("utf-8", errors="replace")[:2000]}).to_tool_output()
            prefix = prefix_result.stdout.decode("utf-8", errors="surrogateescape").rstrip("\r\n")
            headers, files = _parse_status(result.stdout, prefix)
            metadata = {"branch": headers.get("branch.head"), "upstream": headers.get("branch.upstream"),
                        "head": headers.get("branch.oid")}
            if "branch.ab" in headers:
                ahead, behind = headers["branch.ab"].split()
                metadata.update(ahead=int(ahead[1:]), behind=int(behind[1:]))
            return paged_result("changed_files", files[offset:offset + limit], len(files), offset, limit, **metadata)
        except FileNotFoundError:
            return ToolOutputEnvelope(success=False, message="Git is not installed or the workspace is missing.",
                                      error_code=GIT_NOT_INSTALLED).to_tool_output()
        except subprocess.TimeoutExpired:
            return ToolOutputEnvelope(success=False, message="Git changed-file discovery timed out.",
                                      error_code=GIT_TIMEOUT).to_tool_output()
        except (OSError, ValueError, StopIteration) as exc:
            return ToolOutputEnvelope(success=False, message=f"Git changed-file discovery failed: {exc}",
                                      error_code=GIT_RUNTIME_ERROR).to_tool_output()

    return git_changed_files
