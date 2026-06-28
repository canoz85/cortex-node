from typing import Any

import json

from pydantic import BaseModel, ConfigDict, Field


TOOL_RESULT_MARKER = "<tool_result_json>"


class ToolSerializableModel(BaseModel):
    """Base model for tool payloads that need summary + JSON output."""

    display: str | None = None
    error_code: str | None = None
    error_details: dict[str, Any] | None = None

    @staticmethod
    def _default_display_from_payload(payload: dict[str, Any], summary: str) -> str:
        data = payload.get("data")
        data_dict = data if isinstance(data, dict) else {}

        entries = payload.get("entries")
        if not isinstance(entries, list):
            entries = data_dict.get("entries")
        if isinstance(entries, list):
            path = str(payload.get("path", "") or data_dict.get("path", "") or ".")
            if not entries:
                return f"No files found under {path}."
            lines = "\n".join(f"- {entry}" for entry in entries)
            return f"Files under {path}:\n{lines}"

        content = payload.get("content")
        if not isinstance(content, str):
            content = data_dict.get("content")
        if isinstance(content, str):
            path_label = str(payload.get("path", "") or data_dict.get("path", "") or "file")
            max_chars = 4000
            if len(content) > max_chars:
                return f"Contents of {path_label}:\n{content[:max_chars]}\n\n...[truncated]"
            return f"Contents of {path_label}:\n{content}"

        if {"prompt_tokens", "completion_tokens", "total_tokens"}.issubset(data_dict.keys()):
            prompt = data_dict.get("prompt_tokens", 0)
            completion = data_dict.get("completion_tokens", 0)
            total = data_dict.get("total_tokens", 0)
            return f"Token usage: {prompt} prompt tokens, {completion} completion tokens ({total} total)"

        if "formatted" in data_dict:
            return f"The current time is: {data_dict.get('formatted', '')}"

        if {"model", "context_window", "workspace"}.issubset(data_dict.keys()):
            model = data_dict.get("model", "unknown")
            context = data_dict.get("context_window", "unknown")
            workspace = data_dict.get("workspace", "unknown")
            return f"Agent info: Model={model}, Context window={context}, Workspace={workspace}"

        if summary and data is not None:
            if isinstance(data, str):
                return f"{summary}\nData: {data}"
            if isinstance(data, (dict, list)):
                return f"{summary}\nData:\n{json.dumps(data, indent=2, ensure_ascii=True)}"
        return summary

    def to_tool_output(self) -> str:
        summary = getattr(self, "message", self.__class__.__name__)
        payload = self.model_dump()
        if not isinstance(payload.get("display"), str) or not str(payload.get("display") or "").strip():
            payload["display"] = self._default_display_from_payload(payload, str(summary))
        return f"{summary}\n{TOOL_RESULT_MARKER}\n{json.dumps(payload, ensure_ascii=True)}"


class TokenUsage(BaseModel):
    """Validated token usage values from an LLM response."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @classmethod
    def from_response_metadata(cls, metadata: dict | None) -> "TokenUsage":
        metadata = metadata or {}
        prompt_tokens = int(metadata.get("prompt_eval_count") or metadata.get("prompt_tokens") or 0)
        completion_tokens = int(metadata.get("eval_count") or metadata.get("completion_tokens") or 0)
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


class WriteFileRequest(BaseModel):
    """Validated arguments for the write_file tool."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    content: str
    overwrite: bool = True


class WriteFileResult(ToolSerializableModel):
    """Structured write_file tool output."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    message: str
    path: str
    characters_written: int = Field(default=0, ge=0)


class ToolResult(ToolSerializableModel):
    """Generic structured response envelope for tools."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    message: str
    data: dict[str, Any] | list[Any] | str | None = None

    @staticmethod
    def split_tool_output(raw: str) -> tuple[str | None, str]:
        """Split summary + payload format, falling back to raw payload only."""
        if TOOL_RESULT_MARKER in raw:
            summary, payload = raw.split(TOOL_RESULT_MARKER, 1)
            return summary.strip() or None, payload.strip()
        return None, raw

    @classmethod
    def try_parse(cls, raw: Any) -> "ToolResult | None":
        """Best-effort parse for tool outputs that may or may not be structured JSON."""
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            try:
                return cls.model_validate(raw)
            except Exception:
                return None
        if isinstance(raw, str):
            _, candidate = cls.split_tool_output(raw)
            try:
                return cls.model_validate_json(candidate)
            except Exception:
                return None
        return None

    @staticmethod
    def unwrap_tool_output(raw: Any) -> dict[str, Any] | list[Any] | str | None:
        """Return a parsed Python payload from tool output when possible."""
        if isinstance(raw, (dict, list)):
            return raw
        if isinstance(raw, ToolResult):
            return raw.model_dump()
        if isinstance(raw, str):
            summary, candidate = ToolResult.split_tool_output(raw)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                return {
                    "success": True,
                    "message": summary or "Tool output",
                    "data": parsed,
                }
            except Exception:
                if summary:
                    return {"success": True, "message": summary, "data": None}
                return raw
        return None

    def to_pretty_text(self) -> str:
        """Human-readable rendering for console output."""
        lines = [f"Success: {self.success}", f"Message: {self.message}"]
        if self.error_code:
            lines.append(f"Error code: {self.error_code}")
        if self.data is not None:
            if isinstance(self.data, str):
                lines.append(f"Data: {self.data}")
            else:
                lines.append("Data:")
                lines.append(json.dumps(self.data, indent=2, ensure_ascii=True))
        if self.error_details is not None:
            lines.append("Error details:")
            lines.append(json.dumps(self.error_details, indent=2, ensure_ascii=True))
        return "\n".join(lines)


class ListFilesRequest(BaseModel):
    """Validated arguments for the list_files tool."""

    model_config = ConfigDict(extra="forbid")

    path: str = "."


class ListFilesResult(ToolSerializableModel):
    """Structured list_files tool output."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    message: str
    path: str
    entries: list[str] = Field(default_factory=list)
    is_file: bool = False


class ReadFileRequest(BaseModel):
    """Validated arguments for the read_file tool."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)


class ReadFileResult(ToolSerializableModel):
    """Structured read_file tool output."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    message: str
    path: str
    content: str = ""


class MakeDirectoryRequest(BaseModel):
    """Validated arguments for the make_directory tool."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)


class MakeDirectoryResult(ToolSerializableModel):
    """Structured make_directory tool output."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    message: str
    path: str