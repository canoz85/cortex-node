import shlex
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool

from core.models import ToolResult


def _resolve_workspace(workspace_dir: str) -> Path:
    root = Path(workspace_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_safe_path(workspace_root: Path, target: str) -> Path:
    candidate = (workspace_root / target).resolve()
    if candidate != workspace_root and workspace_root not in candidate.parents:
        raise ValueError(f"Path '{target}' is outside sandbox workspace")
    return candidate


def get_exec_tools(workspace_dir: str):
    workspace_root = _resolve_workspace(workspace_dir)

    @tool
    def run_python(
        path: str,
        args: str = "",
        timeout_seconds: int = 20,
        **extra_kwargs: object,
    ) -> str:
        """Run a Python file from the sandbox workspace and return output."""
        try:
            script = _resolve_safe_path(workspace_root, path)
            if not script.exists() or script.suffix != ".py":
                return ToolResult(
                    success=False,
                    message=f"Error: Python file does not exist: {path}",
                ).to_tool_output()

            arg_list = shlex.split(args, posix=False) if args else []
            raw_extra_args = extra_kwargs.get("v__args")
            if isinstance(raw_extra_args, list):
                arg_list.extend(str(item) for item in raw_extra_args)
            cmd = [sys.executable, str(script), *arg_list]
            result = subprocess.run(
                cmd,
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
                message="Python execution completed",
                data={
                    "exit_code": result.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                },
            ).to_tool_output()
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                message=f"Error: execution timed out after {timeout_seconds} seconds",
            ).to_tool_output()
        except Exception as exc:
            return ToolResult(success=False, message=f"Error running Python: {exc}").to_tool_output()

    return [run_python]
