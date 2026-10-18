"""Offline complete-baseline, immutable comparison and atomic storage tests."""

import difflib
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from core import error_codes as codes
from core.models import MAX_WEB_FETCH_CHARS, SourceUrlRequest, ToolOutputEnvelope, WebFetchRequest
from core.source_snapshots import MAX_SNAPSHOT_BYTES, SourceSnapshotStore, content_hash
from tools import source_ops as source, web_ops as web
from test_web_ops import dns_records, html_response


URL = "https://example.com/page"
BEFORE = "2026-10-17T08:00:00+00:00"
AFTER = "2026-10-17T09:00:00+00:00"


def page(text="Original text.", *, url=URL, final_url=None, title="Tracked page", at=BEFORE,
         truncated=False, total_chars=None):
    return ToolOutputEnvelope(success=True, message="Fetched.", display="Fetched.", data={
        "url": url, "final_url": final_url or url, "title": title, "content_type": "text/html",
        "retrieved_at": at, "text": text, "total_chars": len(text) if total_chars is None else total_chars,
        "truncated": truncated,
    })


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(web.socket, "getaddrinfo", Mock(return_value=dns_records("93.184.216.34")))
    monkeypatch.setattr(web, "create_connection", Mock(side_effect=AssertionError("Unexpected network")))


@pytest.fixture
def store(tmp_path):
    return SourceSnapshotStore(tmp_path / "internal")


@pytest.fixture
def fetch(monkeypatch):
    fetch = Mock(return_value=page())
    monkeypatch.setattr(web, "fetch_web", fetch)
    return fetch


def baseline_path(store, url=URL):
    return store._path(web.normalize_web_url(url))


def test_first_snapshot_persists_versioned_complete_evidence_and_concise_receipt(store, fetch):
    result = source.snapshot_source(URL, store)
    assert result.success and result.data["baseline_created"] is True
    assert result.data["baseline_replaced"] is False and result.data["baseline_status"] == "created"
    snapshot = store.load(web.normalize_web_url(URL))
    assert snapshot.requested_url == snapshot.final_url == URL
    assert snapshot.text == "Original text." and snapshot.title == "Tracked page"
    assert snapshot.content_hash == hashlib.sha256(snapshot.text.encode()).hexdigest()
    assert snapshot.total_chars == len(snapshot.text) and snapshot.retrieved_at == BEFORE
    assert snapshot.schema_version == 1 and snapshot.truncated is False
    payload = json.loads(baseline_path(store).read_bytes())
    assert payload["schema_version"] == 1 and payload["content_type"] == "text/html"
    assert fetch.call_args == ((URL,), {"max_chars": 200_000})
    assert WebFetchRequest.model_json_schema()["properties"]["max_chars"]["maximum"] == MAX_WEB_FETCH_CHARS == 200_000
    assert "Original text." not in result.display and "text" not in result.data
    assert str(store.root) not in result.to_tool_output() and result.display == result.message


def test_explicit_snapshot_replaces_baseline_and_resets_comparison(store, fetch):
    assert source.snapshot_source(URL, store).success
    fetch.return_value = page("New complete text.", at=AFTER)
    replacement = source.snapshot_source(URL, store)
    assert replacement.success and replacement.data["baseline_replaced"] is True
    assert replacement.data["baseline_created"] is False and replacement.data["baseline_status"] == "replaced"
    result = source.diff_source(URL, SourceSnapshotStore(store.root))
    assert result.success and result.data["changed"] is False
    assert result.data["baseline_retrieved_at"] == AFTER
    assert len(list(store.root.glob("*.json"))) == 1


def test_baseline_survives_new_tool_and_store_instances(tmp_path, store, fetch):
    first = {tool.name: tool for tool in source.get_source_tools(str(tmp_path / "workspace"), store=store)}
    assert ToolOutputEnvelope.try_parse(first["source_snapshot"].invoke({"url": URL})).success
    second = {tool.name: tool for tool in source.get_source_tools(
        str(tmp_path / "workspace"), store=SourceSnapshotStore(store.root),
    )}
    result = ToolOutputEnvelope.try_parse(second["source_diff"].invoke({"url": URL}))
    assert result.success and result.data["changed"] is False


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("partial", [page("partial", truncated=True, total_chars=200001),
                                    page("partial", total_chars=100)])
def test_incomplete_snapshot_never_creates_or_replaces_baseline(existing, partial, store, fetch):
    if existing:
        assert source.snapshot_source(URL, store).success
    before = baseline_path(store).read_bytes() if existing else None
    fetch.return_value = partial
    result = source.snapshot_source(URL, store)
    assert not result.success and result.error_code == codes.SOURCE_SNAPSHOT_INCOMPLETE
    if existing:
        assert baseline_path(store).read_bytes() == before
    else:
        assert not store.root.exists()


def test_missing_baseline_fails_without_fetch_or_directory_creation(store, fetch):
    result = source.diff_source(URL, store)
    assert not result.success and result.error_code == codes.SOURCE_BASELINE_NOT_FOUND
    assert "changed" not in result.data and not store.root.exists()
    fetch.assert_not_called()


def test_unchanged_complete_text_returns_empty_diff_and_full_comparison_metadata(store, fetch):
    assert source.snapshot_source(URL, store).success
    before = baseline_path(store).read_bytes()
    fetch.return_value = page(at=AFTER)
    result = source.diff_source(URL, store)
    assert result.success and result.data["changed"] is False
    assert result.data["diff"] == "" and result.data["diff_truncated"] is False
    assert result.data["total_diff_chars"] == result.data["total_diff_lines"] == 0
    assert result.data["baseline_hash"] == result.data["current_hash"] == content_hash("Original text.")
    assert result.data["baseline_retrieved_at"] == BEFORE and result.data["current_retrieved_at"] == AFTER
    assert baseline_path(store).read_bytes() == before


@pytest.mark.parametrize("before,after", [
    ("one\ntwo\nthree", "one\nchanged\nthree"),
    ("one\nthree", "one\ntwo\nthree"),
    ("one\ntwo\nthree", "one\nthree"),
    ("", "Added text"), ("Removed text", ""),
    ("İstanbul\nΩ café ☕", "İstanbul\nΩ çay ☕"),
])
def test_changed_added_removed_and_unicode_text_have_deterministic_diff(before, after, store, fetch):
    fetch.return_value = page(before)
    assert source.snapshot_source(URL, store).success
    fetch.return_value = page(after, at=AFTER)
    result = source.diff_source(URL, store)
    expected = "".join(line + "\n" for line in difflib.unified_diff(
        before.splitlines(), after.splitlines(), fromfile="baseline", tofile="current", n=3, lineterm="",
    ))
    assert result.success and result.data["changed"] is True
    assert result.data["diff"] == expected and result.data["diff_truncated"] is False
    assert result.data["total_diff_chars"] == len(expected)
    assert result.data["total_diff_lines"] == len(expected.splitlines())
    assert result.data["total_diff_bytes"] == len(expected.encode())
    assert result.data["baseline_chars"] == len(before) and result.data["current_chars"] == len(after)
    assert result.data["baseline_hash"] == content_hash(before) and result.data["current_hash"] == content_hash(after)
    assert before not in result.display or not before
    assert store.load(web.normalize_web_url(URL)).text == before


def test_title_and_redirect_metadata_changes_are_separate_from_text_changes(store, fetch):
    assert source.snapshot_source(URL, store).success
    fetch.return_value = page(title="New title", final_url="https://other.example/latest", at=AFTER)
    result = source.diff_source(URL, store)
    assert result.success and result.data["changed"] is False and result.data["diff"] == ""
    assert result.data["title_changed"] is True and result.data["final_url_changed"] is True
    assert result.data["baseline_title"] == "Tracked page" and result.data["current_title"] == "New title"
    assert result.data["baseline_final_url"] == URL and result.data["final_url"] == "https://other.example/latest"
    assert "No text change" in result.message


def test_truncated_current_text_never_claims_unchanged(store, fetch):
    assert source.snapshot_source(URL, store).success
    before = baseline_path(store).read_bytes()
    fetch.return_value = page("Original text.", truncated=True, total_chars=200001)
    result = source.diff_source(URL, store)
    assert result.error_code == codes.SOURCE_DIFF_INCOMPLETE and not result.success
    assert "changed" not in result.data and baseline_path(store).read_bytes() == before


@pytest.mark.parametrize("operation,expected", [
    (source.snapshot_source, codes.SOURCE_SNAPSHOT_FAILED), (source.diff_source, codes.SOURCE_DIFF_FAILED),
])
@pytest.mark.parametrize("cause", [codes.WEB_FETCH_TIMEOUT, codes.WEB_FETCH_REJECTED,
                                  codes.WEB_FETCH_UNSUPPORTED_CONTENT_TYPE, codes.WEB_FETCH_RESPONSE_TOO_LARGE])
def test_fetch_failures_preserve_baseline_and_report_cause(operation, expected, cause, store, fetch):
    assert source.snapshot_source(URL, store).success
    before = baseline_path(store).read_bytes()
    fetch.return_value = ToolOutputEnvelope(success=False, message="Fetch unavailable.", error_code=cause,
                                           data={"final_url": URL, "content_type": "application/pdf", "status_code": 200})
    result = operation(URL, store)
    assert not result.success and result.error_code == expected and result.data["cause_code"] == cause
    assert "changed" not in result.data and baseline_path(store).read_bytes() == before


@pytest.mark.parametrize("mutation,expected", [
    (lambda data: data.update(schema_version=2), codes.SOURCE_SNAPSHOT_CORRUPT),
    (lambda data: data.pop("schema_version"), codes.SOURCE_SNAPSHOT_CORRUPT),
    (lambda data: data.update(schema_version=True), codes.SOURCE_SNAPSHOT_CORRUPT),
    (lambda data: data.update(content_hash="0" * 64), codes.SOURCE_SNAPSHOT_CORRUPT),
    (lambda data: data.update(text="tampered"), codes.SOURCE_DIFF_INCOMPLETE),
    (lambda data: data.update(total_chars=1), codes.SOURCE_SNAPSHOT_CORRUPT),
    (lambda data: data.update(truncated=True), codes.SOURCE_DIFF_INCOMPLETE),
    (lambda data: data.pop("truncated"), codes.SOURCE_SNAPSHOT_CORRUPT),
    (lambda data: data.update(retrieved_at="invalid timestamp"), codes.SOURCE_SNAPSHOT_CORRUPT),
    (lambda data: data.update(canonical_url="https://other.example/"), codes.SOURCE_SNAPSHOT_CORRUPT),
    (lambda data: data.update(requested_url="https://other.example/"), codes.SOURCE_SNAPSHOT_CORRUPT),
    (lambda data: data.update(requested_url="file:///invalid"), codes.SOURCE_SNAPSHOT_CORRUPT),
    (lambda data: data.update(final_url="file:///invalid"), codes.SOURCE_SNAPSHOT_CORRUPT),
    (lambda data: data.update(content_type="application/json"), codes.SOURCE_SNAPSHOT_CORRUPT),
    (lambda data: data.update(text="x" * 200001, total_chars=200001, content_hash=content_hash("x" * 200001)),
     codes.SOURCE_SNAPSHOT_CORRUPT),
    (lambda data: data.update(unexpected=True), codes.SOURCE_SNAPSHOT_CORRUPT),
])
def test_corrupt_unsupported_or_incomplete_baselines_fail_closed_without_fetch(mutation, expected, store, fetch):
    assert source.snapshot_source(URL, store).success
    path = baseline_path(store)
    data = json.loads(path.read_bytes())
    mutation(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    before = path.read_bytes()
    fetch.reset_mock()
    result = source.diff_source(URL, store)
    assert not result.success and result.error_code == expected and "changed" not in result.data
    fetch.assert_not_called()
    assert path.read_bytes() == before


@pytest.mark.parametrize("payload", [b"{bad", b"\xff", b"x" * (MAX_SNAPSHOT_BYTES + 1)],
                         ids=["invalid-json", "invalid-encoding", "oversized"])
def test_malformed_or_oversized_snapshot_files_fail_closed(payload, store, fetch):
    assert source.snapshot_source(URL, store).success
    baseline_path(store).write_bytes(payload)
    fetch.reset_mock()
    result = source.diff_source(URL, store)
    assert result.error_code == codes.SOURCE_SNAPSHOT_CORRUPT and "changed" not in result.data
    fetch.assert_not_called()


def test_explicit_snapshot_can_repair_corrupt_record(store, fetch):
    assert source.snapshot_source(URL, store).success
    baseline_path(store).write_bytes(b"{bad")
    assert source.snapshot_source(URL, store).data["baseline_replaced"] is True
    assert source.diff_source(URL, store).data["changed"] is False


@pytest.mark.parametrize("stage", ["fsync", "replace"])
def test_failed_atomic_write_preserves_old_baseline_and_removes_temporary_file(stage, store, fetch, monkeypatch):
    assert source.snapshot_source(URL, store).success
    before = baseline_path(store).read_bytes()
    fetch.return_value = page("Changed text.")
    monkeypatch.setattr(f"core.source_snapshots.os.{stage}", Mock(side_effect=OSError("private filesystem path")))
    result = source.snapshot_source(URL, store)
    assert result.error_code == codes.SOURCE_SNAPSHOT_FAILED
    assert baseline_path(store).read_bytes() == before
    assert list(store.root.glob(".source-*")) == []
    assert "private filesystem path" not in result.to_tool_output() and str(store.root) not in result.to_tool_output()


def test_atomic_replacement_occurs_after_fsync_and_old_record_remains_until_commit(store, fetch, monkeypatch):
    import core.source_snapshots as storage
    assert source.snapshot_source(URL, store).success
    before = baseline_path(store).read_bytes()
    events = []
    fsync, replace = storage.os.fsync, storage.os.replace

    def flush(fd):
        events.append("fsync")
        fsync(fd)

    def commit(temporary, target):
        assert events == ["fsync"] and target.read_bytes() == before
        assert Path(temporary).parent == target.parent == store.root
        events.append("replace")
        replace(temporary, target)

    monkeypatch.setattr(storage.os, "fsync", flush)
    monkeypatch.setattr(storage.os, "replace", commit)
    fetch.return_value = page("New text.")
    assert source.snapshot_source(URL, store).success and events == ["fsync", "replace"]


def test_redirects_keep_requested_identity_and_do_not_create_final_url_alias(store, fetch):
    final_url = "https://other.example/final"
    fetch.return_value = page(final_url=final_url)
    assert source.snapshot_source(URL, store).success
    assert store.load(web.normalize_web_url(URL)).final_url == final_url
    fetch.reset_mock()
    result = source.diff_source(final_url, store)
    assert result.error_code == codes.SOURCE_BASELINE_NOT_FOUND
    fetch.assert_not_called()


@pytest.mark.parametrize("first,second", [
    ("HTTPS://EXAMPLE.COM", "https://example.com/"),
    (URL + "#first", URL + "#second"),
    ("https://example.com/%7euser", "https://example.com/~user"),
])
def test_normalized_requested_urls_share_one_baseline(first, second, store, fetch):
    fetch.return_value = page(url=first)
    assert source.snapshot_source(first, store).success
    fetch.return_value = page(url=second)
    assert source.diff_source(second, store).data["changed"] is False
    assert len(list(store.root.glob("*.json"))) == 1


def test_url_normalization_does_not_resolve_dns_and_keeps_query_path_identity():
    web.socket.getaddrinfo.reset_mock()
    assert web.normalize_web_url("https://EXAMPLE.com/a?q=1&x=2#section") == "https://example.com/a?q=1&x=2"
    assert web.normalize_web_url("https://example.com:443/") != web.normalize_web_url("https://example.com/")
    assert web.normalize_web_url("https://example.com/A") != web.normalize_web_url("https://example.com/a")
    web.socket.getaddrinfo.assert_not_called()


@pytest.mark.parametrize("url", ["../../outside.json", "file:///tmp/file", "ftp://example.com/a", "https://user:secret@example.com/"])
def test_invalid_urls_cannot_control_storage_paths(url, store, fetch):
    result = source.snapshot_source(url, store)
    assert result.error_code == codes.SOURCE_SNAPSHOT_FAILED and not store.root.exists()
    fetch.assert_not_called()


def test_url_path_segments_only_select_a_hash_named_record(store, fetch):
    url = "https://example.com/a/../../outside.json?path=another.json"
    fetch.return_value = page(url=url)
    assert source.snapshot_source(url, store).success
    path, = store.root.glob("*.json")
    assert path.name == hashlib.sha256(web.normalize_web_url(url).encode()).hexdigest() + ".json"
    assert path.parent == store.root


def test_workspace_file_tools_cannot_read_or_edit_internal_baseline(tmp_path, store, fetch):
    from tools.file_ops import get_file_tools
    assert source.snapshot_source(URL, store).success
    path = baseline_path(store)
    before = path.read_bytes()
    tools = {tool.name: tool for tool in get_file_tools(str(tmp_path / "workspace"))}
    read = ToolOutputEnvelope.unwrap_tool_output(tools["read_file"].invoke({"path": str(path)}))
    write = ToolOutputEnvelope.unwrap_tool_output(tools["write_file"].invoke({"path": str(path), "content": "model edit"}))
    assert read["success"] is False and write["success"] is False
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["path", "snapshot_id", "max_chars", "store_root", "allow_private"])
def test_tool_schema_forbids_storage_transport_or_text_arguments(field, tmp_path, store, fetch):
    tools = source.get_source_tools(str(tmp_path / "workspace"), store=store)
    for tool in tools:
        with pytest.raises(ValidationError):
            tool.invoke({"url": URL, field: "model-supplied"})
    fetch.assert_not_called()
    assert not store.root.exists()


@pytest.mark.parametrize("directory", ["workspace", "knowledge"])
def test_store_cannot_be_inside_model_visible_roots(directory, tmp_path):
    workspace, knowledge = tmp_path / "workspace", tmp_path / "knowledge"
    store = SourceSnapshotStore((workspace if directory == "workspace" else knowledge) / "internal")
    with pytest.raises(ValueError, match="outside workspace and knowledge"):
        source.get_source_tools(str(workspace), str(knowledge), store=store)
    assert not store.root.exists()


@pytest.mark.parametrize("lines,long_line", [(1000, False), (1, True)])
def test_diff_evidence_is_bounded_with_correct_full_counts(lines, long_line, store, fetch):
    before = "a" * 20000 if long_line else "\n".join(f"old line {i}" for i in range(lines))
    after = "b" * 20000 if long_line else "\n".join(f"new line {i}" for i in range(lines))
    fetch.return_value = page(before)
    assert source.snapshot_source(URL, store).success
    fetch.return_value = page(after)
    result = source.diff_source(URL, store)
    data = result.data
    full = "".join(line + "\n" for line in difflib.unified_diff(
        before.splitlines(), after.splitlines(), fromfile="baseline", tofile="current", n=3, lineterm="",
    ))
    assert result.success and data["changed"] and data["diff_truncated"]
    assert len(data["diff"]) <= source.MAX_DIFF_CHARS and len(data["diff"].splitlines()) <= source.MAX_DIFF_LINES
    assert full.startswith(data["diff"]) and data["total_diff_chars"] == len(full)
    assert data["total_diff_lines"] == len(full.splitlines()) and data["total_diff_bytes"] == len(full.encode())
    assert data["returned_diff_chars"] == len(data["diff"]) and data["returned_diff_lines"] == len(data["diff"].splitlines())
    assert before not in result.display and after not in result.display


def test_repeated_diffs_are_stable_and_never_advance_baseline(store, fetch):
    assert source.snapshot_source(URL, store).success
    before = baseline_path(store).read_bytes()
    modified = baseline_path(store).stat().st_mtime_ns
    fetch.return_value = page("New content.", at=AFTER)
    first, second = source.diff_source(URL, store), source.diff_source(URL, store)
    assert first.success and second.success and first.data == second.data
    assert first.data["changed"] and first.data["baseline_retrieved_at"] == BEFORE
    assert baseline_path(store).read_bytes() == before and baseline_path(store).stat().st_mtime_ns == modified


def test_source_tools_reuse_real_fetch_transport_extraction_and_redirect_policy(store, monkeypatch):
    send = Mock(side_effect=[
        html_response(status=302, headers={"Location": "https://other.example/final"}),
        html_response(b"<title>Source</title><main>Complete <b>evidence</b>.</main>"),
        html_response(status=302, headers={"Location": "http://127.0.0.1/"}),
    ])
    monkeypatch.setattr(web._PublicHTTPAdapter, "send", send)
    assert source.snapshot_source(URL, store).success
    before = baseline_path(store).read_bytes()
    assert store.load(web.normalize_web_url(URL)).text == "Complete evidence."
    result = source.diff_source(URL, store)
    assert result.error_code == codes.SOURCE_DIFF_FAILED and result.data["cause_code"] == codes.WEB_FETCH_REJECTED
    assert send.call_count == 3 and baseline_path(store).read_bytes() == before


def test_real_fetch_truncation_is_rejected_without_changing_limits(store, monkeypatch):
    send = Mock(return_value=html_response(b"<main>" + b"x" * 200001 + b"</main>"))
    monkeypatch.setattr(web._PublicHTTPAdapter, "send", send)
    assert source.snapshot_source(URL, store).error_code == codes.SOURCE_SNAPSHOT_INCOMPLETE
    assert send.call_count == 1 and not store.root.exists()


def test_legacy_version_one_baseline_still_compares_without_rewrite(store, fetch):
    text = "Original text."
    legacy = {
        "schema_version": 1, "canonical_url": URL, "requested_url": URL + "#saved",
        "final_url": URL, "title": "Tracked page", "content_type": "text/html",
        "retrieved_at": BEFORE, "text": text, "total_chars": len(text),
        "content_hash": hashlib.sha256(text.encode()).hexdigest(), "truncated": False,
    }
    store.root.mkdir()
    path = store.root / (hashlib.sha256(URL.encode()).hexdigest() + ".json")
    original = json.dumps(legacy).encode()
    path.write_bytes(original)
    fetch.return_value = page("Changed text.", at=AFTER)
    result = source.diff_source(URL + "#current", store)
    assert result.success and result.data["changed"] is True
    assert result.data["baseline_hash"] == legacy["content_hash"]
    assert result.data["diff"] == "--- baseline\n+++ current\n@@ -1 +1 @@\n-Original text.\n+Changed text.\n"
    assert path.read_bytes() == original
    assert fetch.call_args == ((URL + "#current",), {"max_chars": 200_000})


def test_url_normalization_belongs_to_tools_and_core_load_does_not_normalize(store, fetch, monkeypatch):
    normalize = Mock(wraps=web.normalize_web_url)
    monkeypatch.setattr(web, "normalize_web_url", normalize)
    url = "HTTPS://EXAMPLE.COM/page#saved"
    assert source.snapshot_source(url, store).success
    normalize.assert_any_call(url)
    normalize.reset_mock()
    assert store.load(URL).source_id == URL
    normalize.assert_not_called()
    assert source.diff_source(URL + "#current", store).data["changed"] is False
    normalize.assert_any_call(URL + "#current")
    normalize.assert_any_call(url)  # Historical metadata is checked in the tool layer.


@pytest.mark.parametrize("changes", [
    {"content_type": "application/json"}, {"final_url": "file:///invalid"},
    {"text": "x" * 200001, "total_chars": 200001},
])
def test_web_metadata_and_text_limits_still_reject_failed_snapshot_without_replacement(changes, store, fetch):
    assert source.snapshot_source(URL, store).success
    original = baseline_path(store).read_bytes()
    fetched = page("New text.")
    fetched.data.update(changes)
    fetch.return_value = fetched
    result = source.snapshot_source(URL, store)
    assert not result.success and result.error_code == codes.SOURCE_SNAPSHOT_FAILED
    assert baseline_path(store).read_bytes() == original
    assert list(store.root.glob(".source-*")) == []
