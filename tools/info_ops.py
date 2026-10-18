from datetime import datetime, timezone
import re

from langchain_core.tools import tool

from core.error_codes import INFO_TIME_FORMAT_ERROR, INFO_TIME_REFERENCE_INVALID, INFO_TOKEN_USAGE_UNAVAILABLE
from core.graph_constants import MAX_REASONING_STEPS
from core.models import CurrentTimeRequest, ToolOutputEnvelope

_runtime: dict = {}


# Calendar datetimes with minute/second precision and explicit ISO offsets.
# Guard fromisoformat's permissive separators and out-of-range offset minutes.
_REFERENCE_TIME_PATTERN = re.compile(
    r"(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{8})T"
    r"(?:[0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:[.,][0-9]{1,6})?)?"
    r"|[0-9]{4}(?:[0-9]{2}(?:[.,][0-9]{1,6})?)?)"
    r"(?:Z|[+-](?:[01][0-9]|2[0-3])(?::?[0-5][0-9])?)"
)


def _to_non_negative_int(value: object) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def get_info_tools(model: str, workspace_dir: str):
    _runtime["model"] = model
    _runtime["workspace_dir"] = workspace_dir
    _runtime["context_window"] = "~128k tokens"
    _runtime["max_steps"] = MAX_REASONING_STEPS

    @tool
    def agent_info() -> str:
        """Return current runtime configuration and the last known token usage."""
        usage = _runtime.get("token_usage")
        return ToolOutputEnvelope(
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
            return ToolOutputEnvelope(
                success=False,
                message="No token usage recorded yet.",
                error_code=INFO_TOKEN_USAGE_UNAVAILABLE,
            ).to_tool_output()
        return ToolOutputEnvelope(
            success=True,
            message="Most recent token usage",
            data=usage,
        ).to_tool_output()

    @tool(args_schema=CurrentTimeRequest)
    def current_time(format: str = "%Y-%m-%d %H:%M:%S", reference_time: str | None = None) -> str:
        """Read offset-aware local/UTC time using an optional strftime display format.

        For elapsed-time requests, pass reference_time as an ISO-8601 calendar
        datetime with an explicit offset or Z. This tool calculates elapsed_seconds
        in Python; never infer a timezone or subtract timestamps in the model.
        """
        reference = None
        if reference_time is not None:
            try:
                if _REFERENCE_TIME_PATTERN.fullmatch(reference_time) is None:
                    raise ValueError("Expected an ISO-8601 datetime with an explicit UTC offset.")
                reference = datetime.fromisoformat(reference_time)
                if reference.tzinfo is None or reference.utcoffset() is None:
                    raise ValueError("An explicit UTC offset is required.")
            except ValueError:
                return ToolOutputEnvelope(
                    success=False,
                    message="reference_time must be a valid ISO-8601 datetime with an explicit UTC offset (or Z); never infer a timezone.",
                    error_code=INFO_TIME_REFERENCE_INVALID,
                    error_details={"field": "reference_time"},
                ).to_tool_output()
        try:
            now = datetime.now().astimezone()
            utc_now = now.astimezone(timezone.utc)
            local_iso = now.isoformat()
            # isoformat preserves even second-resolution system UTC offsets.
            utc_offset = local_iso[len(now.replace(tzinfo=None).isoformat()):]
            formatted = now.strftime(format)
            data = {
                "iso": local_iso,
                "formatted": formatted,
                "format": format,
                "utc_iso": utc_now.isoformat(),
                "utc_offset": utc_offset,
            }
            if reference is not None:
                data.update(
                    reference_time=reference.isoformat(),
                    elapsed_seconds=(utc_now - reference.astimezone(timezone.utc)).total_seconds(),
                )
            return ToolOutputEnvelope(
                success=True,
                message="Current local system time",
                data=data,
            ).to_tool_output()
        except Exception as exc:
            return ToolOutputEnvelope(
                success=False,
                message=f"Error formatting current time: {exc}",
                error_code=INFO_TIME_FORMAT_ERROR,
                error_details={
                    "format": format,
                    "exception_type": type(exc).__name__,
                },
            ).to_tool_output()

    
    return [agent_info, token_usage, current_time]


def update_token_usage(usage: dict) -> None:
    """Accumulate token usage across turns; only update non-zero values."""
    if not usage:
        return

    prompt_tokens = _to_non_negative_int(usage.get("prompt_tokens", 0))
    completion_tokens = _to_non_negative_int(usage.get("completion_tokens", 0))
    total_tokens = _to_non_negative_int(usage.get("total_tokens", 0))

    existing = _runtime.get("token_usage", {}) or {}

    existing_prompt = _to_non_negative_int(existing.get("prompt_tokens", 0))
    existing_completion = _to_non_negative_int(existing.get("completion_tokens", 0))
    existing_total = _to_non_negative_int(existing.get("total_tokens", 0))

    _runtime["token_usage"] = {
        "prompt_tokens": existing_prompt + prompt_tokens,
        "completion_tokens": existing_completion + completion_tokens,
        "total_tokens": existing_total + total_tokens,
    }
