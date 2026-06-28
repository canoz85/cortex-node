from pathlib import Path

from langchain_core.tools import tool

from core.error_codes import (
    FILE_ALREADY_EXISTS,
    FILE_FILE_NOT_FOUND,
    FILE_KNOWLEDGE_PATH_OUTSIDE_ROOT,
    FILE_LIST_FAILED,
    FILE_MAKE_DIR_FAILED,
    FILE_PATH_NOT_FOUND,
    FILE_READ_FAILED,
    FILE_WRITE_FAILED,
)
from core.models import (
    ListFilesRequest,
    ListFilesResult,
    MakeDirectoryRequest,
    MakeDirectoryResult,
    ReadFileRequest,
    ReadFileResult,
    WriteFileRequest,
    WriteFileResult,
)
from tools.sandbox_paths import resolve_safe_path, resolve_workspace


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

    @tool
    def read_file(path: str) -> str:
        """Read a UTF-8 text file from inside the sandbox workspace."""
        try:
            request = ReadFileRequest(path=path)
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

            content = target.read_text(encoding="utf-8")
            result = ReadFileResult(
                success=True,
                message=f"Read file: {request.path}",
                path=request.path,
                content=content,
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

            target.write_text(request.content, encoding="utf-8")
            result = WriteFileResult(
                success=True,
                message=f"Wrote {len(request.content)} characters to {request.path}",
                path=request.path,
                characters_written=len(request.content),
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

    tools = [list_files, read_file, write_file, make_directory]

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
