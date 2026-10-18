"""Source-neutral persistence and version-1 wire/filename compatibility."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from core.source_snapshots import (
    MAX_SNAPSHOT_BYTES, SnapshotCorrupt, SourceSnapshot, SourceSnapshotStore, content_hash,
)


SOURCE_ID = "https://ollama.com/blog"


def legacy_record(text="Saved Ollama baseline."):
    # Deliberately use the pre-refactor record layout, without the new model.
    return {
        "schema_version": 1, "canonical_url": SOURCE_ID,
        "requested_url": "HTTPS://OLLAMA.COM/blog#saved", "final_url": SOURCE_ID,
        "title": "Ollama blog", "content_type": "text/html",
        "retrieved_at": "2026-10-17T08:00:00+00:00", "text": text,
        "total_chars": len(text), "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "truncated": False,
    }


@pytest.fixture
def store(tmp_path):
    return SourceSnapshotStore(tmp_path / "internal")


def test_core_imports_without_tools_transport_or_web_limits():
    script = """
import importlib.abc
import sys

class BlockWebDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args, **kwargs):
        if (fullname == 'core.models' or fullname == 'tools' or fullname.startswith('tools.')
                or fullname in {'requests', 'bs4'}):
            raise AssertionError('Core persistence imported ' + fullname)

sys.meta_path.insert(0, BlockWebDependencies())
import core.source_snapshots
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_existing_version_one_baseline_loads_at_the_same_hashed_url_key(store):
    record = legacy_record()
    # Build the old filename independently of the store's path implementation.
    path = store.root / (hashlib.sha256(SOURCE_ID.encode("utf-8")).hexdigest() + ".json")
    store.root.mkdir()
    original = json.dumps(record).encode("utf-8")
    path.write_bytes(original)
    modified = path.stat().st_mtime_ns

    snapshot = store.load(SOURCE_ID)
    assert snapshot.source_id == SOURCE_ID
    assert snapshot.schema_version == 1 and snapshot.text == record["text"]
    assert snapshot.requested_url == record["requested_url"] and snapshot.final_url == SOURCE_ID
    assert snapshot.content_hash == record["content_hash"] and snapshot.truncated is False
    assert path == store._path(SOURCE_ID)
    assert path.read_bytes() == original and path.stat().st_mtime_ns == modified


def test_save_keeps_version_one_fields_without_migration(store):
    record = legacy_record()
    snapshot = SourceSnapshot.model_validate(record)
    assert store.save(snapshot) is False
    saved = json.loads(store._path(SOURCE_ID).read_bytes())
    assert saved == record and "source_id" not in saved
    assert store.save(store.load(SOURCE_ID)) is True
    assert json.loads(store._path(SOURCE_ID).read_bytes()) == record


def test_core_accepts_resolved_opaque_identity_and_treats_web_fields_as_metadata(store):
    record = legacy_record()
    record.pop("canonical_url")
    snapshot = SourceSnapshot(
        **{**record, "requested_url": "requested metadata", "final_url": "final metadata",
           "content_type": "application/x-test"},
        source_id="../../opaque-source-id",
    )
    assert not store.root.exists()
    assert store.load(snapshot.source_id) is None and not store.root.exists()
    assert store.save(snapshot) is False
    path, = store.root.iterdir()
    assert path.parent == store.root
    assert path.name == content_hash(snapshot.source_id) + ".json"
    assert store.load(snapshot.source_id) == snapshot


def test_core_still_checks_record_identity_against_lookup_key(store):
    snapshot = SourceSnapshot.model_validate(legacy_record())
    store.save(snapshot)
    path = store._path(SOURCE_ID)
    data = json.loads(path.read_bytes())
    data["canonical_url"] = "another-resolved-source"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SnapshotCorrupt):
        store.load(SOURCE_ID)


def test_internal_field_name_does_not_silently_change_version_one_disk_schema(store):
    record = legacy_record()
    record["source_id"] = record.pop("canonical_url")
    store.root.mkdir()
    store._path(SOURCE_ID).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SnapshotCorrupt):
        store.load(SOURCE_ID)


def test_core_text_bound_is_independent_of_web_extraction_limit(store):
    snapshot = SourceSnapshot.model_validate(legacy_record("x" * 200_001))
    assert store.save(snapshot) is False
    assert store.load(SOURCE_ID) == snapshot


def test_serialized_byte_budget_preserves_baseline_on_oversized_save(store):
    store.save(SourceSnapshot.model_validate(legacy_record()))
    path = store._path(SOURCE_ID)
    original = path.read_bytes()
    # JSON escaping exceeds the byte budget even though the character count fits.
    oversized = SourceSnapshot.model_validate(legacy_record("\0" * (MAX_SNAPSHOT_BYTES // 6)))
    with pytest.raises(ValueError, match="storage bound"):
        store.save(oversized)
    assert path.read_bytes() == original
    assert list(store.root.glob(".source-*")) == []
