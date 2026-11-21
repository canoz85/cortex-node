"""Derive bounded, observational source context from execution evidence.

This is not a live store read, a completion verdict, or an authorization policy.
"""

from datetime import datetime
import re

from core.error_codes import SOURCE_BASELINE_NOT_FOUND, SOURCE_SNAPSHOT_CORRUPT
from tools.web_ops import normalize_web_url


MAX_SOURCES = 24
MAX_TEXT_CHARS = 10000
_TOOLS = {"source_status", "source_snapshot", "source_diff", "web_fetch"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _text(value):
    return value if isinstance(value, str) and value else None


def _timestamp(value):
    if not _text(value):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


def _hash(value):
    return value if isinstance(value, str) and _HASH.fullmatch(value) else None


def _bounded(value):
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT_CHARS else value[:MAX_TEXT_CHARS] + "\n...[truncated]"
    if isinstance(value, dict):
        return {key: _bounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_bounded(item) for item in value]
    return value


def _clear_comparison(source):
    comparison = source["comparison_for_latest_baseline"]
    if comparison.get("successful_diff_observed") is True:
        source["previous_comparison_observation"] = comparison
    source["comparison_for_latest_baseline"] = {"successful_diff_observed": False}


def derive_source_observations(records, execution_id, *, max_sources=MAX_SOURCES):
    """Fold all current-execution records, then bound the resulting presentation."""
    sources = {}
    absence_candidates = set()
    for record in records:
        if record.execution_id != execution_id or record.tool_name not in _TOOLS:
            continue
        url = record.arguments.get("url")
        if not isinstance(url, str):
            continue
        try:
            identity = normalize_web_url(url)
        except (ValueError, TypeError):
            continue
        source = sources.setdefault(identity, {
            "requested_identity": identity,
            "first_baseline_observation": {"state": "unknown"},
            "latest_baseline_observation": {"state": "unknown"},
            "comparison_for_latest_baseline": {"successful_diff_observed": False},
            "current_text_fetch": {"successful_web_fetch_observed": False},
        })
        result, name = record.result, record.tool_name
        data = result.data if isinstance(result.data, dict) else {}
        request_id = result.request_id

        # This is an event history, not a verdict on the active task. A later
        # failure replaces the prior event even if its returned URL is invalid.
        if not result.success:
            source["last_observed_failure"] = {
                "tool": name, "request_id": request_id, "error_code": result.error_code,
            }
            absence_candidates.discard(identity)

        # An inconsistent returned identity cannot establish any source facts.
        consistent = True
        for key in ("url", "canonical_url"):
            if key in data:
                try:
                    consistent = consistent and isinstance(data[key], str) and normalize_web_url(data[key]) == identity
                except (ValueError, TypeError):
                    consistent = False
        if not consistent:
            source["last_invalid_observation"] = {"tool": name, "request_id": request_id}
            if name in {"source_status", "source_snapshot", "source_diff"}:
                source["previous_baseline_observation"] = source["latest_baseline_observation"]
                source["latest_baseline_observation"] = {"state": "uncertain", "request_id": request_id}
                _clear_comparison(source)
            continue

        if name == "web_fetch":
            if result.success and isinstance(data.get("text"), str):
                source["current_text_fetch"] = {
                    "successful_web_fetch_observed": True,
                    "retrieved_at": _timestamp(data.get("retrieved_at")),
                    "truncated": data.get("truncated") if type(data.get("truncated")) is bool else None,
                    "request_id": request_id,
                }
            continue

        observation = None
        if not result.success:
            if name in {"source_status", "source_diff"} and result.error_code == SOURCE_BASELINE_NOT_FOUND:
                observation = {"state": "missing", "request_id": request_id}
                absence_candidates.add(identity)
            elif name == "source_snapshot" or result.error_code == SOURCE_SNAPSHOT_CORRUPT:
                observation = {"state": "uncertain", "request_id": request_id}
        elif name == "source_status":
            if data.get("baseline_exists") is True and _text(data.get("canonical_url")):
                observation = {"state": "present", "retrieved_at": _timestamp(data.get("retrieved_at")),
                               "content_hash": _hash(data.get("content_hash")), "request_id": request_id}
            else:
                observation = {"state": "uncertain", "request_id": request_id}
        elif name == "source_snapshot":
            created, replaced = data.get("baseline_created"), data.get("baseline_replaced")
            valid = type(created) is bool and type(replaced) is bool and created != replaced
            status = "created" if created is True else "replaced"
            valid = valid and data.get("baseline_status") == status
            observation = {"state": "present" if valid else "uncertain", "request_id": request_id}
            if valid:
                observation.update(origin=status + "_this_execution", retrieved_at=_timestamp(data.get("retrieved_at")),
                                   content_hash=_hash(data.get("content_hash")))
            # Even a malformed success may have changed the stored generation.
            _clear_comparison(source)
        elif name == "source_diff":
            baseline_hash = _hash(data.get("baseline_hash"))
            baseline_at = _timestamp(data.get("baseline_retrieved_at"))
            if baseline_hash and baseline_at:
                observation = {"state": "present", "retrieved_at": baseline_at,
                               "content_hash": baseline_hash, "request_id": request_id}
            else:
                observation = {"state": "uncertain", "request_id": request_id}

        if observation is not None:
            # Only link a valid, later observation to the prior absence. This
            # says nothing about task completion or the current external store.
            if (result.success and observation["state"] == "present"
                    and identity in absence_candidates and isinstance(data.get("url"), str)
                    and observation.get("retrieved_at") and observation.get("content_hash")):
                source["last_observed_failure"]["baseline_absence_superseded_by_request_id"] = request_id
                absence_candidates.discard(identity)
            first = source["first_baseline_observation"]
            if "request_id" not in first:
                source["first_baseline_observation"] = {"state": observation["state"], "request_id": request_id}
            old = source["latest_baseline_observation"]
            if observation["state"] == "uncertain" and old.get("state") != "unknown":
                source["previous_baseline_observation"] = old
            same_generation = (old.get("state") == observation["state"] == "present"
                               and old.get("content_hash") is not None
                               and old.get("content_hash") == observation.get("content_hash")
                               and old.get("retrieved_at") is not None
                               and old.get("retrieved_at") == observation.get("retrieved_at"))
            if name != "source_snapshot" and same_generation and "origin" in old:
                observation["origin"] = old["origin"]
                observation["origin_request_id"] = old.get("origin_request_id", old["request_id"])
            if not same_generation:
                _clear_comparison(source)
            source["latest_baseline_observation"] = observation

        if name == "source_diff" and result.success and observation and observation["state"] == "present":
            changed, truncated = data.get("changed"), data.get("diff_truncated")
            source["comparison_for_latest_baseline"] = {
                "successful_diff_observed": True,
                "baseline_hash": observation["content_hash"],
                "baseline_retrieved_at": observation["retrieved_at"],
                "current_retrieved_at": _timestamp(data.get("current_retrieved_at")),
                "changed": changed if type(changed) is bool else None,
                "truncated": truncated if type(truncated) is bool else None,
                "request_id": request_id,
            }

    entries = list(sources.values())
    limit = max(0, max_sources)
    return {"scope": "current_execution", "sources": _bounded(entries[:limit]),
            "omitted_source_count": max(0, len(entries) - limit)}
