import shlex
import subprocess
import sys

from langchain_core.tools import tool

from core.error_codes import EXEC_EXIT_NONZERO, EXEC_RUNTIME_ERROR, EXEC_SCRIPT_NOT_FOUND, EXEC_TIMEOUT
from core.models import ToolResult
from tools.sandbox_paths import resolve_safe_path, resolve_workspace


def _append_cli_args(arg_list: list[str], raw_args: object) -> None:
    if isinstance(raw_args, list):
        arg_list.extend(str(item) for item in raw_args)
        return
    if isinstance(raw_args, str) and raw_args.strip():
        arg_list.extend(shlex.split(raw_args, posix=False))


def get_exec_tools(workspace_dir: str):
    workspace_root = resolve_workspace(workspace_dir)

    @tool
    def run_python(
        path: str,
        args: str = "",
        timeout_seconds: int = 20,
        **extra_kwargs: object,
    ) -> str:
        """Run a Python file from the sandbox workspace and return output."""
        try:
            script = resolve_safe_path(workspace_root, path)
            if not script.exists() or script.suffix != ".py":
                return ToolResult(
                    success=False,
                    message=f"Error: Python file does not exist: {path}",
                    error_code=EXEC_SCRIPT_NOT_FOUND,
                    error_details={"path": path},
                ).to_tool_output()

            arg_list: list[str] = []
            _append_cli_args(arg_list, args)

            # Accept multiple argument shapes produced by different model/tool-calling patterns.
            _append_cli_args(arg_list, extra_kwargs.get("v__args"))
            _append_cli_args(arg_list, extra_kwargs.get("args"))

            nested_extra = extra_kwargs.get("extra_kwargs")
            if isinstance(nested_extra, dict):
                _append_cli_args(arg_list, nested_extra.get("v__args"))
                _append_cli_args(arg_list, nested_extra.get("args"))

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
                error_code=(None if result.returncode == 0 else EXEC_EXIT_NONZERO),
                error_details=(
                    None
                    if result.returncode == 0
                    else {
                        "path": path,
                        "exit_code": result.returncode,
                        "args": arg_list,
                    }
                ),
            ).to_tool_output()
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                message=f"Error: execution timed out after {timeout_seconds} seconds",
                error_code=EXEC_TIMEOUT,
                error_details={
                    "path": path,
                    "timeout_seconds": timeout_seconds,
                },
            ).to_tool_output()
        except Exception as exc:
            return ToolResult(
                success=False,
                message=f"Error running Python: {exc}",
                error_code=EXEC_RUNTIME_ERROR,
                error_details={
                    "path": path,
                    "exception_type": type(exc).__name__,
                },
            ).to_tool_output()

    return [run_python]
