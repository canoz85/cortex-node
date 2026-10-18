"""Saved baseline metadata comes only from read-only persistent runtime evidence."""

import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from conftest import get_tool, parse_result
from core import error_codes as codes
from core.source_snapshots import SourceSnapshot, SourceSnapshotStore, content_hash
from test_source_ops import URL, BEFORE, AFTER
from tools import source_ops as source, web_ops as web


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    probes = []
    for owner, name in ((web, "fetch_web"), (web, "create_connection"),
                        (web.socket, "getaddrinfo"), (web.requests.Session, "send")):
        probe = Mock(side_effect=AssertionError("source_status must not access the network"))
        monkeypatch.setattr(owner, name, probe)
        probes.append(probe)
    yield
    for probe in probes:
        probe.assert_not_called()


@pytest.fixture
def store(tmp_path):
    return SourceSnapshotStore(tmp_path / "internal")


def save_baseline(store, *, url=URL, text="Saved baseline text.", at=BEFORE):
    baseline = SourceSnapshot(
        schema_version=1, source_id=web.normalize_web_url(url), requested_url=url,
        final_url="https://other.example/saved", title="Saved title", content_type="text/html",
        retrieved_at=at, text=text, total_chars=len(text), content_hash=content_hash(text), truncated=False,
    )
    store.save(baseline)
    return baseline


def status_tool(store, tmp_path):
    return get_tool(source.get_source_tools(str(tmp_path / "workspace"), store=store), "source_status")


def test_existing_baseline_returns_exact_saved_metadata_without_fetch_write_or_text(store, tmp_path, monkeypatch):
    saved = save_baseline(store)
    path = store._path(saved.source_id)
    original, modified = path.read_bytes(), path.stat().st_mtime_ns
    save = Mock(side_effect=AssertionError("source_status must not save"))
    monkeypatch.setattr(store, "save", save)
    tool = status_tool(store, tmp_path)
    first = parse_result(tool.invoke({"url": URL}))
    assert first["success"] is True
    assert first["data"] == {
        "url": URL, "canonical_url": URL, "final_url": saved.final_url, "title": "Saved title",
        "content_type": "text/html", "retrieved_at": BEFORE, "total_chars": len(saved.text),
        "content_hash": content_hash(saved.text), "baseline_exists": True,
    }
    assert parse_result(tool.invoke({"url": URL})) == first
    assert saved.text not in json.dumps(first) and str(store.root) not in json.dumps(first)
    assert path.read_bytes() == original and path.stat().st_mtime_ns == modified
    assert list(store.root.iterdir()) == [path]
    save.assert_not_called()


def test_missing_baseline_fails_without_directory_or_baseline_creation(store, tmp_path):
    result = parse_result(status_tool(store, tmp_path).invoke({"url": URL}))
    assert result["success"] is False and result["error_code"] == codes.SOURCE_BASELINE_NOT_FOUND
    assert "No saved baseline exists" in result["message"]
    assert result["data"] == {"url": URL} and not store.root.exists()


@pytest.mark.parametrize("damage", [
    "invalid-json", "unsupported-version", "boolean-version", "missing-version", "hash", "length",
    "incomplete", "identity", "requested-url", "final-url", "content-type", "timestamp", "extra-field",
])
def test_corrupt_unsupported_or_incomplete_baseline_fails_closed_without_rewriting(damage, store, tmp_path):
    saved = save_baseline(store)
    path = store._path(saved.source_id)
    data = json.loads(path.read_bytes())
    mutations = {
        "unsupported-version": {"schema_version": 2}, "boolean-version": {"schema_version": True},
        "hash": {"content_hash": "0" * 64}, "length": {"total_chars": 1}, "incomplete": {"truncated": True},
        "identity": {"canonical_url": "https://another.example/"},
        "requested-url": {"requested_url": "https://another.example/"},
        "final-url": {"final_url": "file:///invalid"}, "content-type": {"content_type": "application/json"},
        "timestamp": {"retrieved_at": "invalid"}, "extra-field": {"unexpected": "private JSON detail"},
    }
    if damage == "missing-version":
        data.pop("schema_version")
    else:
        data.update(mutations.get(damage, {}))
    original = b"{invalid JSON private detail" if damage == "invalid-json" else json.dumps(data).encode()
    path.write_bytes(original)
    result = parse_result(status_tool(store, tmp_path).invoke({"url": URL}))
    assert result["success"] is False and result["error_code"] == codes.SOURCE_SNAPSHOT_CORRUPT
    assert result["data"] == {"url": URL}
    assert "private" not in result["message"] and str(store.root) not in json.dumps(result)
    assert path.read_bytes() == original


@pytest.mark.parametrize("arguments", [
    {}, {"url": None}, {"url": 1}, {"url": True}, {"url": []}, {"url": ""},
    *({"url": URL, field: "forbidden"} for field in (
        "path", "store_root", "snapshot_id", "max_chars", "allow_private", "offset", "limit",
    )),
])
def test_status_schema_is_strict_url_only(arguments, store, tmp_path):
    with pytest.raises(ValidationError):
        status_tool(store, tmp_path).invoke(arguments)
    assert not store.root.exists()


@pytest.mark.parametrize("first,second", [
    ("HTTPS://EXAMPLE.COM", "https://example.com/#current"),
    (URL + "#saved", URL + "#current"),
    ("https://example.com/%7euser", "https://example.com/~user"),
])
def test_status_uses_the_same_normalized_requested_identity(first, second, store, tmp_path):
    saved = save_baseline(store, url=first)
    result = parse_result(status_tool(store, tmp_path).invoke({"url": second}))
    assert result["success"] is True
    assert result["data"]["url"] == second
    assert result["data"]["canonical_url"] == saved.source_id == web.normalize_web_url(second)
    assert len(list(store.root.glob("*.json"))) == 1


@pytest.mark.parametrize("url", ["file:///invalid", "../../outside.json", "https://user:secret@example.com/"])
def test_invalid_url_fails_without_network_or_store_creation(url, store, tmp_path):
    result = parse_result(status_tool(store, tmp_path).invoke({"url": url}))
    assert result["success"] is False and result["error_code"] == codes.SOURCE_STATUS_FAILED
    assert not store.root.exists()


def test_storage_failure_does_not_expose_paths_or_exception_details(store, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "load", Mock(side_effect=OSError(f"private exception at {store.root}")))
    result = parse_result(status_tool(store, tmp_path).invoke({"url": URL}))
    assert result["success"] is False and result["error_code"] == codes.SOURCE_STATUS_FAILED
    assert "private exception" not in json.dumps(result) and str(store.root) not in json.dumps(result)
    assert not store.root.exists()


def test_status_reloads_replaced_baseline_from_disk_across_store_instances(store, tmp_path):
    save_baseline(store)
    tool = status_tool(store, tmp_path)
    old = parse_result(tool.invoke({"url": URL}))["data"]
    saved = save_baseline(SourceSnapshotStore(store.root), text="Replacement baseline text.", at=AFTER)
    path = store._path(saved.source_id)
    original = path.read_bytes()
    new = parse_result(tool.invoke({"url": URL}))["data"]
    assert old["retrieved_at"] == BEFORE and new["retrieved_at"] == AFTER
    assert new["content_hash"] == saved.content_hash != old["content_hash"]
    assert new["total_chars"] == len(saved.text)
    assert path.read_bytes() == original
