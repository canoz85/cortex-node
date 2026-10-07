"""Deterministic, paginated workspace evidence without shell execution."""

import fnmatch
import os
from pathlib import Path
import re

from langchain_core.tools import tool

from core.error_codes import FILE_LIST_FAILED, FILE_READ_FAILED
from core.models import ToolOutputEnvelope
from tools.sandbox_paths import resolve_safe_path


class PagedEvidenceResult(ToolOutputEnvelope):
    offset: int
    limit: int
    total_items: int
    returned_items: int
    has_more: bool
    is_truncated: bool


def paged_result(collection: str, items: list, count: int, offset: int, limit: int, **metadata) -> str:
    has_more = offset + len(items) < count
    message = f"Returned {len(items)} of {count} {collection}."
    return PagedEvidenceResult(
        success=True, message=message, display=message,
        data={collection: items, "count": count, "truncated": has_more,
              "offset": offset, "limit": limit, "has_more": has_more, **metadata},
        offset=offset, limit=limit, total_items=count, returned_items=len(items),
        has_more=has_more, is_truncated=has_more,
    ).to_tool_output()


def page_bounds(offset: int, limit: int) -> tuple[int, int]:
    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError("offset must be nonnegative and limit must be between 1 and 100")
    return offset, limit


def _files(root: Path, path: str, pattern: str, recursive: bool) -> list[Path]:
    target = resolve_safe_path(root, path)
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if target.is_file():
        return [target] if fnmatch.fnmatchcase(target.name, pattern) else []
    if not target.is_dir():
        raise ValueError("Path must be a file or directory")

    found = []
    for directory, dirs, filenames in os.walk(target, followlinks=False, onerror=_raise_walk_error):
        # Junctions can otherwise be traversed by os.walk on Windows. Links are
        # excluded from discovery, so neither outside targets nor cycles are read.
        dirs[:] = sorted(name for name in dirs if not _is_link(Path(directory) / name))
        for name in sorted(filenames):
            candidate = Path(directory) / name
            if _is_link(candidate) or not fnmatch.fnmatchcase(name, pattern):
                continue
            candidate = resolve_safe_path(root, str(candidate))
            if candidate.is_file():
                found.append(candidate)
        if not recursive:
            break
    return sorted(found, key=lambda item: item.relative_to(root).as_posix())


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _raise_walk_error(error):
    raise error


def get_discovery_tools(workspace_dir: str) -> list:
    root = Path(workspace_dir).resolve()

    @tool
    def find_files(path: str = ".", pattern: str = "*", recursive: bool = True,
                   offset: int = 0, limit: int = 100) -> str:
        """Find workspace files using a case-sensitive basename glob (e.g. *.py).

        Return sorted workspace-relative paths with forward slashes. No hidden-file
        or relevance filtering; directory links are not traversed and file links
        are excluded. offset and limit (1..100) paginate the complete sorted result.
        """
        try:
            offset, limit = page_bounds(offset, limit)
            files = _files(root, path, pattern, recursive)
            page = [item.relative_to(root).as_posix() for item in files[offset:offset + limit]]
            return paged_result("files", page, len(files), offset, limit)
        except (OSError, ValueError) as exc:
            return ToolOutputEnvelope(success=False, message=f"File discovery failed: {exc}",
                                      error_code=FILE_LIST_FAILED).to_tool_output()

    @tool
    def search_text(query: str, path: str = ".", file_pattern: str = "*", regex: bool = False,
                    recursive: bool = True, offset: int = 0, limit: int = 100) -> str:
        """Search workspace files for case-sensitive literal text or a Python regex.

        Search UTF-8 lines (invalid bytes replaced), one match per matching line;
        line numbers start at 1. Basename glob, links and path ordering follow
        find_files. offset and limit (1..100) paginate matches by path then line.
        Matching-line snippets are capped at 2000 characters and include the first
        match; text_start_column (1-based) and text_truncated describe the snippet.
        """
        try:
            offset, limit = page_bounds(offset, limit)
            if not query:
                raise ValueError("query must be nonempty")
            expression = re.compile(query) if regex else None
            matches = []
            count = 0
            for candidate in _files(root, path, file_pattern, recursive):
                # Recheck containment immediately before reading the selected file.
                candidate = resolve_safe_path(root, str(candidate))
                with candidate.open("r", encoding="utf-8", errors="replace") as stream:
                    for number, line in enumerate(stream, 1):
                        text = line.rstrip("\r\n")
                        matched = expression.search(text) if expression else None
                        match_start = (matched.start() if matched else -1) if expression else text.find(query)
                        if match_start >= 0:
                            if offset <= count < offset + limit:
                                snippet_start = min(match_start, max(0, len(text) - 2000))
                                snippet = text[snippet_start:snippet_start + 2000]
                                matches.append({"path": candidate.relative_to(root).as_posix(),
                                                "line_number": number, "text": snippet,
                                                "text_start_column": snippet_start + 1,
                                                "text_truncated": len(text) > 2000})
                            count += 1
            return paged_result("matches", matches, count, offset, limit)
        except (OSError, ValueError, re.error) as exc:
            return ToolOutputEnvelope(success=False, message=f"Text search failed: {exc}",
                                      error_code=FILE_READ_FAILED).to_tool_output()

    return [find_files, search_text]
