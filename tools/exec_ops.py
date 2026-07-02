from pathlib import Path
import shlex
import subprocess
import sys

from langchain_core.tools import tool

from core.error_codes import (
    EXEC_EXIT_NONZERO,
    EXEC_PACKAGE_INSTALL_FAILED,
    EXEC_RUNTIME_ERROR,
    EXEC_SCRIPT_NOT_FOUND,
    EXEC_TIMEOUT,
)
from core.models import ToolResult
from tools.sandbox_paths import resolve_safe_path, resolve_workspace

def _resolve_pip_command(workspace_root: Path) -> tuple[list[str] | None, list[str]]:
    attempted: list[str] = []

    # 1) Most reliable: same interpreter that runs the agent.
    current_python = Path(sys.executable)
    attempted.append(str(current_python))
    if current_python.exists():
        return [str(current_python), "-m", "pip"], attempted

    # 2) Workspace-local venv.
    ws_python = workspace_root / ".venv" / "Scripts" / "python.exe"
    attempted.append(str(ws_python))
    if ws_python.exists():
        return [str(ws_python), "-m", "pip"], attempted

    # 3) Repo-level venv when workspace is a subfolder (common in this project).
    parent_python = workspace_root.parent / ".venv" / "Scripts" / "python.exe"
    attempted.append(str(parent_python))
    if parent_python.exists():
        return [str(parent_python), "-m", "pip"], attempted

    return None, attempted



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

    @tool
    def install_package(package_name: str, timeout_seconds: int = 60) -> str:
        """Install package via pip in the active Python environment."""
        try:
            pip_cmd, attempted = _resolve_pip_command(workspace_root)
            if not pip_cmd:
                return ToolResult(
                    success=False,
                    message="No usable Python interpreter found for pip installation.",
                    data={},
                    error_code=EXEC_PACKAGE_INSTALL_FAILED,
                    error_details={"attempted_interpreters": attempted},
                ).to_tool_output()

            result = subprocess.run(
                [*pip_cmd, "install", package_name],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                cwd=str(workspace_root),
            )

            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()

            if result.returncode == 0:
                return ToolResult(
                    success=True,
                    message=f"Successfully installed {package_name}",
                    data={
                        "package": package_name,
                        "command": " ".join([*pip_cmd, "install", package_name]),
                        "stdout": stdout[-500:] or "<empty>",
                    },
                ).to_tool_output()

            return ToolResult(
                success=False,
                message=f"Failed to install {package_name}: {(stderr[-300:] if stderr else 'unknown pip error')}",
                data={},
                error_code=EXEC_PACKAGE_INSTALL_FAILED,
                error_details={
                    "command": [*pip_cmd, "install", package_name],
                    "returncode": result.returncode,
                    "stderr": stderr[-500:] or "<empty>",
                },
            ).to_tool_output()

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                message=f"Package install timeout after {timeout_seconds}s",
                data={},
                error_code=EXEC_TIMEOUT,
                error_details={"package": package_name, "timeout_seconds": timeout_seconds},
            ).to_tool_output()
        except FileNotFoundError as exc:
            return ToolResult(
                success=False,
                message=f"Package install failed: interpreter or pip not found ({exc})",
                data={},
                error_code=EXEC_PACKAGE_INSTALL_FAILED,
                error_details={"package": package_name},
            ).to_tool_output()
        except Exception as e:
            return ToolResult(
                success=False,
                message=str(e),
                data={},
                error_code=EXEC_PACKAGE_INSTALL_FAILED,
                error_details={"package": package_name, "exception_type": type(e).__name__},
            ).to_tool_output()
    
    return [run_python, install_package]
