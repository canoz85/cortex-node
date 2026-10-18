"""Explicit baseline writes, comparisons and saved metadata reads."""

import difflib
from pathlib import Path

from langchain_core.tools import tool

from core.error_codes import (
    SOURCE_BASELINE_NOT_FOUND, SOURCE_DIFF_FAILED, SOURCE_DIFF_INCOMPLETE,
    SOURCE_SNAPSHOT_CORRUPT, SOURCE_SNAPSHOT_FAILED, SOURCE_SNAPSHOT_INCOMPLETE,
    SOURCE_STATUS_FAILED,
)
from core.models import MAX_WEB_FETCH_CHARS, SourceUrlRequest, ToolOutputEnvelope
from core.source_snapshots import (
    SnapshotCorrupt, SnapshotIncomplete, SourceSnapshot, SourceSnapshotStore, content_hash,
)
from tools import web_ops


MAX_DIFF_CHARS = 8000
MAX_DIFF_LINES = 200


class _FetchFailed(ValueError):
    def __init__(self, result):
        self.code = result.error_code
        data = result.data if isinstance(result.data, dict) else {}
        self.details = {key: data.get(key) for key in ("final_url", "content_type", "status_code")}


def _failure(url, code, message, **details):
    return ToolOutputEnvelope(success=False, message=message, display=message,
                              error_code=code, data={"url": url, **details})


def _validate_web_snapshot(snapshot: SourceSnapshot) -> None:
    """Keep web identity, metadata and extraction policy out of persistence."""
    if snapshot.source_id != web_ops.normalize_web_url(snapshot.requested_url):
        raise ValueError("Snapshot URL identity mismatch.")
    web_ops.normalize_web_url(snapshot.final_url)  # Syntax only; never re-fetch historical metadata.
    if snapshot.content_type not in {"text/html", "application/xhtml+xml"}:
        raise ValueError("Snapshot must contain HTML text.")
    if snapshot.total_chars > MAX_WEB_FETCH_CHARS:
        raise ValueError("Snapshot exceeds the web extraction limit.")


def _load_web_snapshot(store: SourceSnapshotStore, source_id: str) -> SourceSnapshot | None:
    baseline = store.load(source_id)
    if baseline is not None:
        try:
            _validate_web_snapshot(baseline)
        except ValueError as exc:
            raise SnapshotCorrupt("Snapshot web metadata validation failed.") from exc
    return baseline


def _fetch_snapshot(url: str, source_id: str) -> SourceSnapshot:
    result = web_ops.fetch_web(url, max_chars=MAX_WEB_FETCH_CHARS)
    if not result.success:
        raise _FetchFailed(result)
    data = result.data
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise ValueError("Fetch did not supply text evidence.")
    if (data.get("truncated") is not False or type(data.get("total_chars")) is not int
            or data["total_chars"] != len(data["text"])):
        raise SnapshotIncomplete("Complete source text is required.")
    snapshot = SourceSnapshot(
        schema_version=1, source_id=source_id, requested_url=url,
        final_url=data["final_url"], title=data["title"], content_type=data["content_type"],
        retrieved_at=data["retrieved_at"], text=data["text"], total_chars=data["total_chars"],
        content_hash=content_hash(data["text"]), truncated=False,
    )
    _validate_web_snapshot(snapshot)
    return snapshot


def snapshot_source(url: str, store: SourceSnapshotStore) -> ToolOutputEnvelope:
    """Explicitly establish or replace one complete baseline."""
    try:
        request = SourceUrlRequest(url=url)
        canonical_url = web_ops.normalize_web_url(request.url)
        snapshot = _fetch_snapshot(request.url, canonical_url)
        replaced = store.save(snapshot)
        status = "replaced" if replaced else "created"
        message = f"Source baseline {status}: {snapshot.title[:160] or '(untitled)'} ({snapshot.total_chars} characters)."
        return ToolOutputEnvelope(success=True, message=message, display=message, data={
            "url": request.url, "final_url": snapshot.final_url, "title": snapshot.title,
            "content_type": snapshot.content_type, "retrieved_at": snapshot.retrieved_at,
            "total_chars": snapshot.total_chars, "content_hash": snapshot.content_hash,
            "baseline_created": not replaced, "baseline_replaced": replaced, "baseline_status": status,
        })
    except SnapshotIncomplete:
        return _failure(url, SOURCE_SNAPSHOT_INCOMPLETE, "Source baseline not saved: complete extracted text is required.")
    except _FetchFailed as exc:
        return _failure(url, SOURCE_SNAPSHOT_FAILED, "Source baseline not saved: page fetch failed.",
                        cause_code=exc.code, **exc.details)
    except SnapshotCorrupt:
        return _failure(url, SOURCE_SNAPSHOT_CORRUPT, "Source baseline not saved: internal snapshot storage is invalid.")
    except Exception:
        return _failure(url, SOURCE_SNAPSHOT_FAILED, "Source baseline not saved: invalid URL/data or unavailable storage.")


def _bounded_diff(before: str, after: str) -> dict:
    retained = []
    total_chars = total_lines = total_bytes = returned_chars = returned_lines = 0
    if before != after:
        for line in difflib.unified_diff(before.splitlines(), after.splitlines(),
                                         fromfile="baseline", tofile="current", n=3, lineterm=""):
            piece = line + "\n"
            total_chars += len(piece)
            total_lines += 1
            total_bytes += len(piece.encode("utf-8"))
            if returned_chars < MAX_DIFF_CHARS and returned_lines < MAX_DIFF_LINES:
                chunk = piece[:MAX_DIFF_CHARS - returned_chars]
                retained.append(chunk)
                returned_chars += len(chunk)
                returned_lines += 1
    return {
        "diff": "".join(retained), "diff_truncated": returned_chars < total_chars,
        "total_diff_chars": total_chars, "total_diff_lines": total_lines,
        "returned_diff_chars": returned_chars, "returned_diff_lines": returned_lines,
        "total_diff_bytes": total_bytes,
    }


def diff_source(url: str, store: SourceSnapshotStore) -> ToolOutputEnvelope:
    """Compare against a saved baseline; never create or advance one."""
    try:
        request = SourceUrlRequest(url=url)
        canonical_url = web_ops.normalize_web_url(request.url)
        baseline = _load_web_snapshot(store, canonical_url)
        if baseline is None:
            return _failure(request.url, SOURCE_BASELINE_NOT_FOUND, "No saved baseline exists for this source; save one explicitly first.")
        current = _fetch_snapshot(request.url, canonical_url)
        changed = baseline.text != current.text
        difference = _bounded_diff(baseline.text, current.text)
        message = (
            f"Source text changed: {current.title[:160] or '(untitled)'}. "
            f"Diff: {difference['returned_diff_chars']} of {difference['total_diff_chars']} characters."
            + (" Diff evidence is truncated." if difference["diff_truncated"] else "")
            if changed else f"No text change detected in a complete comparison: {current.title[:160] or '(untitled)'}."
        ) + " Baseline retained."
        return ToolOutputEnvelope(success=True, message=message, display=message, data={
            "url": request.url, "final_url": current.final_url,
            "baseline_final_url": baseline.final_url, "final_url_changed": baseline.final_url != current.final_url,
            "baseline_title": baseline.title, "current_title": current.title,
            "title_changed": baseline.title != current.title,
            "baseline_retrieved_at": baseline.retrieved_at, "current_retrieved_at": current.retrieved_at,
            "baseline_hash": baseline.content_hash, "current_hash": current.content_hash,
            "changed": changed, "baseline_chars": baseline.total_chars, "current_chars": current.total_chars,
            **difference,
        })
    except SnapshotIncomplete:
        return _failure(url, SOURCE_DIFF_INCOMPLETE, "Source comparison incomplete: baseline and current extraction must both be complete.")
    except _FetchFailed as exc:
        return _failure(url, SOURCE_DIFF_FAILED, "Source comparison unavailable: page fetch failed. Baseline retained.",
                        cause_code=exc.code, **exc.details)
    except SnapshotCorrupt:
        return _failure(url, SOURCE_SNAPSHOT_CORRUPT, "Source comparison unavailable: baseline is corrupt or has an unsupported schema.")
    except Exception:
        return _failure(url, SOURCE_DIFF_FAILED, "Source comparison unavailable: invalid URL/data or unavailable storage.")


def status_source(url: str, store: SourceSnapshotStore) -> ToolOutputEnvelope:
    """Read authoritative saved metadata without fetching or modifying a baseline."""
    try:
        request = SourceUrlRequest(url=url)
        canonical_url = web_ops.normalize_web_url(request.url)
        baseline = _load_web_snapshot(store, canonical_url)
        if baseline is None:
            return _failure(request.url, SOURCE_BASELINE_NOT_FOUND, "No saved baseline exists for this source.")
        message = f"Saved baseline captured at {baseline.retrieved_at} ({baseline.total_chars} characters)."
        return ToolOutputEnvelope(success=True, message=message, display=message, data={
            "url": request.url, "canonical_url": baseline.source_id,
            "final_url": baseline.final_url, "title": baseline.title, "content_type": baseline.content_type,
            "retrieved_at": baseline.retrieved_at, "total_chars": baseline.total_chars,
            "content_hash": baseline.content_hash, "baseline_exists": True,
        })
    except (SnapshotCorrupt, SnapshotIncomplete):
        return _failure(url, SOURCE_SNAPSHOT_CORRUPT,
                        "Saved baseline metadata unavailable: baseline is incomplete, corrupt or has an unsupported schema.")
    except Exception:
        return _failure(url, SOURCE_STATUS_FAILED, "Saved baseline metadata unavailable: invalid URL or unavailable storage.")


def get_source_tools(workspace_dir: str, knowledge_dir: str | None = None, *, store: SourceSnapshotStore | None = None):
    snapshot_store = store if store is not None else SourceSnapshotStore()
    for directory in (workspace_dir, knowledge_dir):
        if directory is not None:
            root = Path(directory).resolve()
            if snapshot_store.root == root or root in snapshot_store.root.parents:
                raise ValueError("Internal source storage must be outside workspace and knowledge roots.")

    @tool("source_snapshot", args_schema=SourceUrlRequest)
    def source_snapshot(url: str) -> str:
        """Save or explicitly replace the complete current text of a known public HTML page as its baseline."""
        return snapshot_source(url, snapshot_store).to_tool_output()

    @tool("source_diff", args_schema=SourceUrlRequest)
    def source_diff(url: str) -> str:
        """Compare complete current public HTML text with its saved baseline; never update the baseline."""
        return diff_source(url, snapshot_store).to_tool_output()

    @tool("source_status", args_schema=SourceUrlRequest)
    def source_status(url: str) -> str:
        """Read saved baseline metadata from persistent storage without any network request or baseline change.

        Use this runtime evidence for baseline timestamps, hashes, titles, sizes
        and saved URL metadata; conversation history or memory may be stale after
        replacement. For baseline age, pass retrieved_at to current_time(reference_time=...).
        """
        return status_source(url, snapshot_store).to_tool_output()

    return [source_snapshot, source_diff, source_status]
