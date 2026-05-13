import subprocess
from pathlib import Path

from langchain_core.tools import tool

from core.models import ToolResult


def _resolve_workspace(workspace_dir: str) -> Path:
    root = Path(workspace_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_git(workspace_root: Path, args: list[str], timeout_seconds: int = 20) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout = result.stdout.strip() or "<empty>"
        stderr = result.stderr.strip() or "<empty>"
        return ToolResult(
            success=(result.returncode == 0),
            message="Git command completed",
            data={
                "exit_code": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "args": args,
            },
        ).to_tool_output()
    except FileNotFoundError:
        return ToolResult(
            success=False,
            message="Error: git is not installed or not available in PATH",
        ).to_tool_output()
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            message=f"Error: git command timed out after {timeout_seconds} seconds",
        ).to_tool_output()
    except Exception as exc:
        return ToolResult(success=False, message=f"Error running git command: {exc}").to_tool_output()


def get_git_tools(workspace_dir: str):
    workspace_root = _resolve_workspace(workspace_dir)

    @tool
    def git_status() -> str:
        """Show git status for the sandbox workspace repository."""
        return _run_git(workspace_root, ["status", "--short", "--branch"])

    @tool
    def git_diff(path: str = "") -> str:
        """Show unstaged git diff, optionally limited to a file path."""
        args = ["diff"]
        if path:
            args.extend(["--", path])
        return _run_git(workspace_root, args)

    @tool
    def git_log(limit: int = 5) -> str:
        """Show recent commit history."""
        safe_limit = max(1, min(limit, 50))
        return _run_git(
            workspace_root,
            [
                "log",
                f"-n{safe_limit}",
                "--pretty=format:%h %ad %an %s",
                "--date=short",
            ],
        )

    @tool
    def git_show(revision: str = "HEAD") -> str:
        """Show details of a revision, defaulting to HEAD."""
        return _run_git(workspace_root, ["show", "--stat", "--oneline", revision])

    return [git_status, git_diff, git_log, git_show]