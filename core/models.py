from typing import Any

import json

from pydantic import BaseModel, ConfigDict, Field


TOOL_RESULT_MARKER = "<tool_result_json>"


class ToolSerializableModel(BaseModel):
    """Base model for tool payloads that need summary + JSON output."""

    def to_tool_output(self) -> str:
        summary = getattr(self, "message", self.__class__.__name__)
        return f"{summary}\n{TOOL_RESULT_MARKER}\n{self.model_dump_json()}"


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
        if self.data is not None:
            if isinstance(self.data, str):
                lines.append(f"Data: {self.data}")
            else:
                lines.append("Data:")
                lines.append(json.dumps(self.data, indent=2, ensure_ascii=True))
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