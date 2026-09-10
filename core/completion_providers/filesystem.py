"""Nonrecursive snapshot membership and single-record full-read coverage.

Completeness relies on the current list_files contract: successful directory
results enumerate all immediate children. There is no independent listing token.
Paths are lexical, case-sensitive identities, not resolved filesystem objects.
No symlink resolution, live filesystem checks, chunk aggregation or content
version guarantees are provided. Core treats these path identities as opaque.
"""

from collections.abc import Mapping

from core.completion import EvidenceSnapshot, Resolution
from core.protocol.models import CoverageAssessment, ResolvedCoverage


PROVIDER_ID = "filesystem.read_collection"
_TRUNCATION_SUFFIX = "\n\n--- [TRUNCATED] ---"


def _path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("expected workspace-relative path")
    value = value.replace("\\", "/")
    if value.startswith("/") or any(c in value for c in ':*?<>|"'):
        raise ValueError("unsupported path")
    if any(ord(c) < 32 for c in value):
        raise ValueError("unsupported path character")
    parts = []
    for part in value.split("/"):
        if part == "..":
            raise ValueError("traversal is unsupported")
        if part in ("", "."):
            continue
        if part.endswith((" ", ".")):
            raise ValueError("ambiguous path component")
        parts.append(part)
    return "/".join(parts) or "."


def _keys(value, expected):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("invalid specification fields")


def _specification(spec):
    _keys(spec, ("root", "discovery_step_ids", "processing_step_ids", "selection", "processing"))
    root = _path(spec["root"])
    for key in ("discovery_step_ids", "processing_step_ids"):
        ids = spec[key]
        if (not isinstance(ids, (tuple, list)) or not ids
                or any(not isinstance(i, str) or not i.strip() for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("expected nonempty unique step IDs")
    selection = spec["selection"]
    _keys(selection, ("kind", "suffix", "recursive"))
    suffix = selection["suffix"]
    if (selection["kind"] != "files" or selection["recursive"] is not False
            or spec["processing"] != "full_content_read"
            or not isinstance(suffix, str) or not suffix
            or any(c in suffix for c in '/\\:*?[]<>|"')
            or any(ord(c) < 32 for c in suffix)):
        raise ValueError("unsupported selection or processing")
    return root, suffix


def _data(record, fields):
    data = record["result"].get("data")
    if not isinstance(data, Mapping) or not set(fields).issubset(data):
        raise ValueError("missing structured evidence")
    return data


def _incomplete(result):
    integrity = result.get("integrity", {})
    pagination = result.get("pagination")
    return (any(integrity.get(key, False) for key in
                ("is_truncated", "stdout_truncated", "stderr_truncated"))
            or (pagination is not None and (
                pagination.get("has_more", False)
                or pagination.get("offset", 0) != 0
                or pagination.get("returned_items", 0) < pagination.get("total_items", 0))))


class FileReadCollectionProvider:
    """Strict specification: root, discovery_step_ids, processing_step_ids,
    selection={kind: files, suffix: literal, recursive: false}, and
    processing=full_content_read. Both step lists must be nonempty and unique.

    Malformed relevant evidence raises ValueError (service status: error).
    Absent/failed/incomplete discovery stays unresolved; valid partial reads
    leave members missing. Resolution ownership and freezing belong to core.
    """

    contract_version = "1"

    def validate(self, specification: Mapping) -> None:
        _specification(specification)

    def resolve(self, specification: Mapping, evidence: EvidenceSnapshot) -> Resolution:
        root, suffix = _specification(specification)
        for record in evidence.records:
            if (record["tool_name"] != "list_files"
                    or record["step_id"] not in specification["discovery_step_ids"]
                    or record["result"]["success"] is not True):
                continue
            # list_files alone has a documented default path argument.
            if _path(record["arguments"].get("path", ".")) != root:
                continue
            data = _data(record, ("path", "entries", "is_file"))
            if _path(data["path"]) != root:
                raise ValueError("listing request/result target mismatch")
            if type(data["is_file"]) is not bool:
                raise ValueError("invalid listing kind")
            if data["is_file"]:
                continue  # A file-target listing is not a directory collection.
            entries = data["entries"]
            if not isinstance(entries, (tuple, list)):
                raise ValueError("invalid listing entries")
            members = set()
            for entry in entries:
                if not isinstance(entry, str) or not entry:
                    raise ValueError("invalid directory entry")
                entry = entry.replace("\\", "/")
                directory = entry.endswith("/")
                name = entry[:-1] if directory else entry
                # Current directory results contain child names, not rooted paths.
                if "/" in name or _path(name) != name or name == ".":
                    raise ValueError("expected direct-child entry")
                if not directory and name.endswith(suffix):
                    members.add(name if root == "." else f"{root}/{name}")
            if _incomplete(record["result"]):
                continue
            return Resolution(tuple(sorted(members)), (record["result"]["request_id"],))
        return Resolution(reason="awaiting_complete_directory_listing")

    def assess(self, specification: Mapping, resolved: ResolvedCoverage,
               evidence: EvidenceSnapshot) -> CoverageAssessment:
        _specification(specification)
        required = set(resolved.required_item_ids)
        satisfied = set()
        for record in evidence.records:
            if (not required or record["tool_name"] != "read_file"
                    or record["step_id"] not in specification["processing_step_ids"]
                    or record["result"]["success"] is not True):
                continue
            target = _path(record["arguments"].get("path"))
            if target not in required:
                continue
            data = _data(record, (
                "path", "content", "total_chars", "offset", "read_chars", "is_truncated",
            ))
            if _path(data["path"]) != target:
                raise ValueError("read request/result target mismatch")
            if (not isinstance(data["content"], str)
                    or type(data["is_truncated"]) is not bool
                    or any(type(data[k]) is not int or data[k] < 0
                           for k in ("offset", "total_chars", "read_chars"))):
                raise ValueError("invalid read data types")
            requested_offset = record["arguments"].get("offset", 0)
            if type(requested_offset) is not int or requested_offset != data["offset"]:
                raise ValueError("read offset mismatch")
            content = data["content"]
            # The current tool appends this exact display suffix to partial reads.
            if data["is_truncated"] and content.endswith(_TRUNCATION_SUFFIX):
                content = content[:-len(_TRUNCATION_SUFFIX)]
            if len(content) != data["read_chars"]:
                raise ValueError("captured content length mismatch")
            if (data["offset"] == 0 and data["read_chars"] == data["total_chars"]
                    and not data["is_truncated"] and not _incomplete(record["result"])):
                satisfied.add(target)
        missing = required - satisfied
        return CoverageAssessment(
            scope_id=resolved.scope_id, evidence_id=evidence.evidence_id,
            status="missing" if missing else "satisfied",
            satisfied_item_ids=tuple(sorted(satisfied)), missing_item_ids=tuple(sorted(missing)),
            reason="full_reads_missing" if missing else "all_members_fully_read",
        )
