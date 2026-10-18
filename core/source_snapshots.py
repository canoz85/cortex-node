"""Dedicated complete-text source baselines; no conversation or workspace state."""

from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024


def default_source_store_root() -> Path:
    return Path.home() / ".cortexnode" / "source_snapshots"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SourceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, populate_by_name=True)

    schema_version: Literal[1]
    # Preserve the version-1 wire name; core treats this resolved identity as opaque.
    source_id: str = Field(alias="canonical_url", min_length=1, max_length=32768)
    requested_url: str = Field(min_length=1, max_length=8192)
    final_url: str = Field(min_length=1, max_length=65536)
    title: str = Field(max_length=1000)
    content_type: str = Field(min_length=1)
    retrieved_at: str = Field(min_length=1, max_length=128)
    # Text cannot have more characters than the serialized record's byte budget.
    text: str = Field(max_length=MAX_SNAPSHOT_BYTES)
    total_chars: int = Field(ge=0, le=MAX_SNAPSHOT_BYTES)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    truncated: Literal[False]

    @field_validator("retrieved_at")
    @classmethod
    def require_utc_timestamp(cls, value):
        if datetime.fromisoformat(value).utcoffset() != timedelta(0):
            raise ValueError("Snapshot retrieval time must be UTC.")
        return value

    @model_validator(mode="after")
    def validate_evidence(self):
        if self.total_chars != len(self.text) or self.content_hash != content_hash(self.text):
            raise ValueError("Snapshot text integrity mismatch.")
        return self


class SnapshotCorrupt(ValueError):
    pass


class SnapshotIncomplete(ValueError):
    pass


class SourceSnapshotStore:
    def __init__(self, root: Path | str | None = None):
        # Construction and reads never create directories or baselines.
        self.root = Path(root if root is not None else default_source_store_root()).resolve()

    def _path(self, source_id: str) -> Path:
        if self.root.resolve() != self.root:
            raise OSError("Source store directory was retargeted.")
        path = self.root / f"{content_hash(source_id)}.json"
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise SnapshotCorrupt("Snapshot must be a regular internal file.")
        return path

    def load(self, source_id: str) -> SourceSnapshot | None:
        path = self._path(source_id)
        try:
            with path.open("rb") as stream:
                payload = stream.read(MAX_SNAPSHOT_BYTES + 1)
        except FileNotFoundError:
            return None
        if len(payload) > MAX_SNAPSHOT_BYTES:
            raise SnapshotCorrupt("Snapshot exceeds the storage bound.")
        try:
            data = json.loads(payload)
            if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
                raise SnapshotCorrupt("Unsupported snapshot schema.")
            if "canonical_url" not in data:
                raise SnapshotCorrupt("Snapshot source identity is missing.")
            if data.get("truncated") is True or (
                isinstance(data.get("text"), str) and type(data.get("total_chars")) is int
                and data["total_chars"] > len(data["text"])
            ):
                raise SnapshotIncomplete("Baseline does not contain complete text.")
            if data.get("truncated") is not False:
                raise SnapshotCorrupt("Snapshot completeness marker is missing.")
            snapshot = SourceSnapshot.model_validate(data)
            if snapshot.source_id != source_id:
                raise SnapshotCorrupt("Snapshot belongs to another source.")
            return snapshot
        except SnapshotIncomplete:
            raise
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise SnapshotCorrupt("Snapshot validation failed.") from exc

    def save(self, snapshot: SourceSnapshot) -> bool:
        """Atomically replace one baseline; return whether it previously existed.

        Concurrent writers are last-replace-wins; no cross-process locking or
        history is implied. Readers receive a complete old or new JSON record.
        """
        snapshot = SourceSnapshot.model_validate(snapshot.model_dump(by_alias=True))
        payload = snapshot.model_dump_json(by_alias=True).encode("utf-8")
        if len(payload) > MAX_SNAPSHOT_BYTES:
            raise ValueError("Snapshot exceeds the storage bound.")
        target = self._path(snapshot.source_id)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        existed = target.exists()
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=self.root, prefix=".source-", delete=False) as stream:
                temporary_path = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if self._path(snapshot.source_id) != target:
                raise OSError("Source store destination changed.")
            os.replace(temporary_path, target)
            return existed
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
