from datetime import datetime

from langchain_core.tools import tool

from core.models import ToolResult

_runtime: dict = {}


def get_info_tools(model: str, workspace_dir: str):
    _runtime["model"] = model
    _runtime["workspace_dir"] = workspace_dir
    _runtime["context_window"] = "~128k tokens"
    _runtime["max_steps"] = 12

    @tool
    def agent_info() -> str:
        """Return current runtime configuration and the last known token usage."""
        usage = _runtime.get("token_usage")
        return ToolResult(
            success=True,
            message="CortexNode runtime info",
            data={
                "model": _runtime.get("model", "unknown"),
                "context_window": _runtime.get("context_window", "unknown"),
                "workspace": _runtime.get("workspace_dir", "unknown"),
                "max_steps": _runtime.get("max_steps", "unknown"),
                "token_usage": usage,
            },
        ).to_tool_output()

    @tool
    def token_usage() -> str:
        """Return token counts from the most recent brain node response."""
        usage = _runtime.get("token_usage")
        if not usage:
            return ToolResult(
                success=False,
                message="No token usage recorded yet.",
            ).to_tool_output()
        return ToolResult(
            success=True,
            message="Most recent token usage",
            data=usage,
        ).to_tool_output()

    @tool
    def current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Return the current local system time using an optional strftime format."""
        try:
            now = datetime.now()
            formatted = now.strftime(format)
            return ToolResult(
                success=True,
                message="Current local system time",
                data={
                    "iso": now.isoformat(),
                    "formatted": formatted,
                    "format": format,
                },
            ).to_tool_output()
        except Exception as exc:
            return ToolResult(
                success=False,
                message=f"Error formatting current time: {exc}",
            ).to_tool_output()

    return [agent_info, token_usage, current_time]


def update_token_usage(usage: dict) -> None:
    """Called by brain_node to keep the info tools up to date."""
    _runtime["token_usage"] = usage
