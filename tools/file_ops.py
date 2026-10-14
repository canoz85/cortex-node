import hashlib
import os
from pathlib import Path
import stat
import tempfile

from langchain_core.tools import tool

from core.error_codes import (
    FILE_ALREADY_EXISTS,
    FILE_FILE_NOT_FOUND,
    FILE_KNOWLEDGE_PATH_OUTSIDE_ROOT,
    FILE_LIST_FAILED,
    FILE_MAKE_DIR_FAILED,
    FILE_PATH_NOT_FOUND,
    FILE_PATCH_FAILED,
    FILE_PATCH_MATCH_COUNT,
    FILE_PATCH_NOT_TEXT,
    FILE_PATCH_TARGET_CHANGED,
    FILE_READ_FAILED,
    FILE_WRITE_FAILED,
)
from core.models import (
    ApplyPatchRequest,
    ListFilesRequest,
    ListFilesResult,
    MakeDirectoryRequest,
    MakeDirectoryResult,
    ReadFileRequest,
    ReadFileResult,
    ToolArtifact,
    ToolOutputEnvelope,
    WriteFileRequest,
    WriteFileResult,
)
from tools.sandbox_paths import resolve_safe_path, resolve_workspace
from tools.registry import get_tool_argument_schema


def _has_binary_controls(text: str) -> bool:
    return any((ord(char) < 32 and char not in "\t\n\r\f") or char == "\x7f" for char in text)


def get_file_tools(workspace_dir: str, knowledge_dir: str | None = None):
    workspace_root = resolve_workspace(workspace_dir)
    knowledge_root = Path(knowledge_dir).resolve() if knowledge_dir else None

    @tool
    def list_files(path: str = ".") -> str:
        """List files and folders inside the sandbox workspace."""
        try:
            request = ListFilesRequest(path=path)
            target = resolve_safe_path(workspace_root, request.path)
            if not target.exists():
                result = ListFilesResult(
                    success=False,
                    message=f"Error: path does not exist: {request.path}",
                    path=request.path,
                    error_code=FILE_PATH_NOT_FOUND,
                    error_details={"path": request.path},
                )
                return result.to_tool_output()

            if target.is_file():
                rel = str(target.relative_to(workspace_root))
                result = ListFilesResult(
                    success=True,
                    message=f"File: {rel}",
                    path=request.path,
                    entries=[rel],
                    is_file=True,
                )
                return result.to_tool_output()

            children = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            rel_base = target.relative_to(workspace_root)
            header = "." if str(rel_base) == "." else str(rel_base)
            entries: list[str] = []
            for child in children:
                name = child.name + ("/" if child.is_dir() else "")
                entries.append(name)

            result = ListFilesResult(
                success=True,
                message=f"Listing for {header}",
                path=request.path,
                entries=entries,
                is_file=False,
            )
            return result.to_tool_output()
        except Exception as exc:
            safe_path = path if isinstance(path, str) else "."
            result = ListFilesResult(
                success=False,
                message=f"Error listing files: {exc}",
                path=safe_path,
                error_code=FILE_LIST_FAILED,
                error_details={
                    "path": safe_path,
                    "exception_type": type(exc).__name__,
                },
            )
            return result.to_tool_output()


    @tool("read_file", args_schema=get_tool_argument_schema("read_file"))
    def read_file(path: str, offset: int = 0, limit: int = 10000) -> str:
        """Read a UTF-8 text file from inside the sandbox workspace with offset and limit support."""
        try:
            request = ReadFileRequest(path=path, offset=offset, limit=limit)
            target = resolve_safe_path(workspace_root, request.path)

            if not target.exists() or not target.is_file():
                result = ReadFileResult(
                    success=False,
                    message=f"Error: file does not exist: {request.path}",
                    path=request.path,
                    error_code=FILE_FILE_NOT_FOUND,
                    error_details={"path": request.path},
                )
                return result.to_tool_output()

            full_content = target.read_text(encoding="utf-8")
            total_len = len(full_content)
            
            # Safe bounds calculation
            safe_offset = max(0, request.offset)
            safe_limit = max(1, request.limit)
            
            chunk = full_content[safe_offset : safe_offset + safe_limit]
            is_truncated = (safe_offset + len(chunk)) < total_len

            # Format message & content for LLM visibility
            if is_truncated:
                formatted_content = f"{chunk}\n\n--- [TRUNCATED] ---"
                msg = f"Read file {request.path} (characters {safe_offset}-{safe_offset + len(chunk)} of {total_len}). File is truncated."
            else:
                formatted_content = chunk
                msg = f"Read file: {request.path} ({total_len} total characters)"

            result = ReadFileResult(
                success=True,
                message=msg,
                path=request.path,
                content=formatted_content,
                total_chars=total_len,
                offset=safe_offset,
                read_chars=len(chunk),
                is_truncated=is_truncated,
            )
            return result.to_tool_output()

        except Exception as exc:
            safe_path = path if isinstance(path, str) and path else ""
            result = ReadFileResult(
                success=False,
                message=f"Error reading file: {exc}",
                path=safe_path,
                error_code=FILE_READ_FAILED,
                error_details={
                    "path": safe_path,
                    "exception_type": type(exc).__name__,
                },
            )
            return result.to_tool_output()

    @tool
    def write_file(path: str, content: str, overwrite: bool = True) -> str:
        """Write text content to a file in the sandbox workspace."""
        try:
            request = WriteFileRequest(path=path, content=content, overwrite=overwrite)
            target = resolve_safe_path(workspace_root, request.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not request.overwrite:
                result = WriteFileResult(
                    success=False,
                    message=f"Error: file already exists: {request.path}",
                    path=request.path,
                    error_code=FILE_ALREADY_EXISTS,
                    error_details={"path": request.path},
                )
                return result.to_tool_output()

            existed_before = target.exists()

            target.write_text(request.content, encoding="utf-8")
            result = WriteFileResult(
                success=True,
                message=f"Wrote {len(request.content)} characters to {request.path}",
                path=request.path,
                characters_written=len(request.content),
                artifacts=(
                    ToolArtifact(
                        path=request.path,
                        action="modified" if existed_before else "created",
                    ),
                ),
            )
            return result.to_tool_output()
        except Exception as exc:
            safe_path = path if isinstance(path, str) and path else ""
            result = WriteFileResult(
                success=False,
                message=f"Error writing file: {exc}",
                path=safe_path,
                error_code=FILE_WRITE_FAILED,
                error_details={
                    "path": safe_path,
                    "exception_type": type(exc).__name__,
                },
            )
            return result.to_tool_output()

    @tool("apply_patch", args_schema=get_tool_argument_schema("apply_patch"))
    def apply_patch(path: str, old_text: str, new_text: str, expected_occurrences: int = 1) -> str:
        """Replace exact text in an existing workspace UTF-8 file only when its match count agrees."""
        temporary_path = None
        safe_path = path if isinstance(path, str) else ""
        try:
            request = ApplyPatchRequest(
                path=path, old_text=old_text, new_text=new_text, expected_occurrences=expected_occurrences,
            )
            target = resolve_safe_path(workspace_root, request.path)
            if not target.is_file():
                return ToolOutputEnvelope(
                    success=False, message="Patch requires an existing regular file.",
                    error_code=FILE_FILE_NOT_FOUND, error_details={"path": request.path},
                ).to_tool_output()

            original_mode = stat.S_IMODE(target.stat().st_mode)
            before = target.read_bytes()
            try:
                text = before.decode("utf-8")
            except UnicodeDecodeError:
                text = None
            if text is None or _has_binary_controls(text) or _has_binary_controls(request.new_text):
                return ToolOutputEnvelope(
                    success=False, message="Patch requires UTF-8 text without binary control characters.",
                    error_code=FILE_PATCH_NOT_TEXT, error_details={"path": request.path},
                ).to_tool_output()

            actual = text.count(request.old_text)
            if actual != request.expected_occurrences:
                return ToolOutputEnvelope(
                    success=False,
                    message=f"Patch match count mismatch: expected {request.expected_occurrences}, actual {actual}.",
                    error_code=FILE_PATCH_MATCH_COUNT,
                    error_details={"path": request.path, "expected_occurrences": request.expected_occurrences,
                                   "actual_occurrences": actual},
                ).to_tool_output()

            updated = text.replace(request.old_text, request.new_text, request.expected_occurrences)
            after = updated.encode("utf-8")
            relative_path = target.relative_to(workspace_root).as_posix()
            # Prepare the receipt before committing; never return the whole file.
            result = ToolOutputEnvelope(
                success=True, message=f"Applied {actual} exact replacement(s) to {relative_path}.",
                data={"path": relative_path, "replacements": actual,
                      "old_text_chars": len(request.old_text), "new_text_chars": len(request.new_text),
                      "before_chars": len(text), "after_chars": len(updated),
                      "before_sha256": hashlib.sha256(before).hexdigest(),
                      "after_sha256": hashlib.sha256(after).hexdigest()},
                artifacts=(ToolArtifact(path=relative_path, action="modified"),),
            ).to_tool_output()
            # Same-directory replacement is atomic; binary I/O preserves BOMs and
            # line endings outside the literal patch. No shared atomic helper exists.
            with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=".apply-patch-", delete=False) as stream:
                temporary_path = Path(stream.name)
                stream.write(after)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_path, original_mode)
            # Fail closed on an observed edit/removal/path retargeting during preparation.
            if (resolve_safe_path(workspace_root, request.path) != target or not target.is_file()
                    or target.read_bytes() != before):
                return ToolOutputEnvelope(
                    success=False, message="Patch target changed before replacement; no patch applied.",
                    error_code=FILE_PATCH_TARGET_CHANGED, error_details={"path": request.path},
                ).to_tool_output()
            os.replace(temporary_path, target)
            return result
        except Exception as exc:
            return ToolOutputEnvelope(
                success=False, message=f"Patch failed ({type(exc).__name__}); no patch applied.",
                error_code=FILE_PATCH_FAILED,
                error_details={"path": safe_path, "exception_type": type(exc).__name__},
            ).to_tool_output()
        finally:
            if temporary_path is not None:
                try:
                    # A failed replacement may leave a read-only temporary file.
                    os.chmod(temporary_path, stat.S_IRUSR | stat.S_IWUSR)
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    @tool
    def make_directory(path: str) -> str:
        """Create a directory in the sandbox workspace."""
        try:
            request = MakeDirectoryRequest(path=path)
            target = resolve_safe_path(workspace_root, request.path)
            target.mkdir(parents=True, exist_ok=True)
            result = MakeDirectoryResult(
                success=True,
                message=f"Directory ready: {request.path}",
                path=request.path,
            )
            return result.to_tool_output()
        except Exception as exc:
            safe_path = path if isinstance(path, str) and path else ""
            result = MakeDirectoryResult(
                success=False,
                message=f"Error creating directory: {exc}",
                path=safe_path,
                error_code=FILE_MAKE_DIR_FAILED,
                error_details={
                    "path": safe_path,
                    "exception_type": type(exc).__name__,
                },
            )
            return result.to_tool_output()

    tools = [list_files, read_file, write_file, make_directory, apply_patch]

    if knowledge_root is not None:
        kr = knowledge_root

        @tool
        def read_knowledge_file(path: str) -> str:
            """Read a file from the knowledge folder (read-only). Use this to access knowledge base documents, examples, and rules."""
            try:
                request = ReadFileRequest(path=path)
                candidate = (kr / request.path).resolve()
                if candidate != kr and kr not in candidate.parents:
                    result = ReadFileResult(
                        success=False,
                        message=f"Error: path '{request.path}' is outside the knowledge folder",
                        path=request.path,
                        error_code=FILE_KNOWLEDGE_PATH_OUTSIDE_ROOT,
                        error_details={"path": request.path},
                    )
                    return result.to_tool_output()
                if not candidate.exists() or not candidate.is_file():
                    result = ReadFileResult(
                        success=False,
                        message=f"Error: file does not exist: {request.path}",
                        path=request.path,
                        error_code=FILE_FILE_NOT_FOUND,
                        error_details={"path": request.path},
                    )
                    return result.to_tool_output()
                content = candidate.read_text(encoding="utf-8")
                result = ReadFileResult(
                    success=True,
                    message=f"Read knowledge file: {request.path}",
                    path=request.path,
                    content=content,
                )
                return result.to_tool_output()
            except Exception as exc:
                safe_path = path if isinstance(path, str) and path else ""
                result = ReadFileResult(
                    success=False,
                    message=f"Error reading knowledge file: {exc}",
                    path=safe_path,
                    error_code=FILE_READ_FAILED,
                    error_details={
                        "path": safe_path,
                        "exception_type": type(exc).__name__,
                    },
                )
                return result.to_tool_output()

        tools.append(read_knowledge_file)

    return tools
