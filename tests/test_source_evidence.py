import json

from core.brain import _build_step_progress_messages
from core.error_codes import SOURCE_BASELINE_NOT_FOUND, SOURCE_SNAPSHOT_FAILED
from core.protocol.models import ToolExecutionRecord, ToolResult
from core.source_evidence import MAX_SOURCES, derive_source_observations
from test_brain_outcomes import brain_input


URL = "https://example.com/page"
AT = "2026-11-17T10:00:00Z"
LATER = "2026-11-17T10:01:00Z"
HASH = "a" * 64
HASH2 = "b" * 64
EXECUTION = "brain-outcome-test"


def record(name, request_id, *, data=None, success=True, error=None, url=URL,
           execution=EXECUTION, revision=1, step="s1"):
    return ToolExecutionRecord(execution_id=execution, plan_id="p1", plan_revision=revision,
        step_id=step, tool_name=name, arguments={"url": url},
        result=ToolResult(request_id=request_id, success=success, message="result",
                          data=data, error_code=error))


def missing(request_id="missing", **kwargs):
    return record("source_status", request_id, success=False, error=SOURCE_BASELINE_NOT_FOUND,
                  data={"url": kwargs.get("url", URL)}, **kwargs)


def status(request_id="status", **kwargs):
    return record("source_status", request_id, data={"url": URL, "canonical_url": URL,
        "baseline_exists": True, "retrieved_at": AT, "content_hash": HASH}, **kwargs)


def snapshot(request_id="snapshot", *, replaced=False):
    return record("source_snapshot", request_id, data={"url": URL, "baseline_created": not replaced,
        "baseline_replaced": replaced, "baseline_status": "replaced" if replaced else "created",
        "retrieved_at": LATER, "content_hash": HASH2})


def diff(request_id="diff"):
    return record("source_diff", request_id, data={"url": URL, "baseline_hash": HASH,
        "baseline_retrieved_at": AT, "current_retrieved_at": LATER,
        "changed": True, "diff_truncated": True})


def one(*records):
    return derive_source_observations(records, EXECUTION)["sources"][0]


def test_missing_then_created_snapshot():
    source = one(missing(), snapshot())
    assert source["first_baseline_observation"] == {"state": "missing", "request_id": "missing"}
    assert source["latest_baseline_observation"] == {"state": "present", "origin": "created_this_execution",
        "retrieved_at": LATER, "content_hash": HASH2, "request_id": "snapshot"}
    assert not source["comparison_for_latest_baseline"]["successful_diff_observed"]
    assert not source["current_text_fetch"]["successful_web_fetch_observed"]
    assert source["last_observed_failure"] == {"tool": "source_status", "request_id": "missing",
        "error_code": SOURCE_BASELINE_NOT_FOUND, "baseline_absence_superseded_by_request_id": "snapshot"}


def test_missing_then_status_or_diff_supersedes_absence():
    for observation in (status(), diff(), snapshot(replaced=True)):
        failure = one(missing(), observation)["last_observed_failure"]
        assert failure["baseline_absence_superseded_by_request_id"] == observation.result.request_id


def test_unrelated_success_does_not_supersede_absence():
    web = record("web_fetch", "web", data={"url": URL, "text": "x", "retrieved_at": AT, "truncated": False})
    failure = one(missing(), web)["last_observed_failure"]
    assert failure == {"tool": "source_status", "request_id": "missing", "error_code": SOURCE_BASELINE_NOT_FOUND}
    other = record("source_snapshot", "other", url="https://example.com/other", data={
        "url": "https://example.com/other", "baseline_created": True, "baseline_replaced": False,
        "baseline_status": "created", "retrieved_at": AT, "content_hash": HASH})
    assert derive_source_observations([missing(), other], EXECUTION)["sources"][0]["last_observed_failure"] == failure


def test_web_fetch_failure_remains_historical_without_supersession():
    web_failure = record("web_fetch", "web-failed", success=False, error="network")
    failure = one(web_failure, snapshot())["last_observed_failure"]
    assert failure == {"tool": "web_fetch", "request_id": "web-failed", "error_code": "network"}


def test_malformed_later_observations_do_not_supersede():
    invalid = [
        record("source_snapshot", "bad-flags", data={"url": URL, "baseline_created": True,
            "baseline_replaced": True, "baseline_status": "created", "retrieved_at": AT, "content_hash": HASH}),
        record("source_snapshot", "bad-hash", data={"url": URL, "baseline_created": True,
            "baseline_replaced": False, "baseline_status": "created", "retrieved_at": AT, "content_hash": "bad"}),
        record("source_status", "bad-status", data={"url": URL, "canonical_url": "https://other.example/",
            "baseline_exists": True, "retrieved_at": AT, "content_hash": HASH}),
        record("source_diff", "bad-diff", data={"url": URL, "baseline_hash": HASH,
            "baseline_retrieved_at": "invalid"}),
    ]
    for observation in invalid:
        assert "baseline_absence_superseded_by_request_id" not in one(missing(), observation)["last_observed_failure"]


def test_last_failure_tracks_latest_failure_even_after_supersession():
    later_failure = record("web_fetch", "later-failure", success=False, error="network")
    failure = one(missing(), snapshot(), later_failure, status())["last_observed_failure"]
    assert failure == {"tool": "web_fetch", "request_id": "later-failure", "error_code": "network"}
    invalid_failure = record("source_diff", "invalid-failure", success=False, error="network",
                             data={"url": "https://other.example/"})
    assert one(missing(), invalid_failure)["last_observed_failure"]["request_id"] == "invalid-failure"


def test_missing_diff_and_normalized_identity_can_be_superseded():
    absent = record("source_diff", "absent-diff", url="HTTPS://EXAMPLE.COM/page#fragment",
                    success=False, error=SOURCE_BASELINE_NOT_FOUND)
    failure = one(absent, snapshot())["last_observed_failure"]
    assert failure["baseline_absence_superseded_by_request_id"] == "snapshot"
    foreign = snapshot().model_copy(update={"execution_id": "foreign"})
    assert "baseline_absence_superseded_by_request_id" not in one(absent, foreign)["last_observed_failure"]


def test_status_preserves_metadata():
    source = one(status())
    assert source["latest_baseline_observation"] == {"state": "present", "retrieved_at": AT,
        "content_hash": HASH, "request_id": "status"}


def test_replacement_resets_earlier_diff_even_if_content_matches():
    replacement = snapshot(replaced=True)
    replacement = replacement.model_copy(update={"result": replacement.result.model_copy(update={"data": {
        **replacement.result.data, "content_hash": HASH, "retrieved_at": AT}})})
    source = one(status(), diff(), replacement)
    assert source["latest_baseline_observation"]["origin"] == "replaced_this_execution"
    assert source["comparison_for_latest_baseline"] == {"successful_diff_observed": False}
    assert source["previous_comparison_observation"]["request_id"] == "diff"


def test_diff_has_generation_and_integrity():
    comparison = one(diff())["comparison_for_latest_baseline"]
    assert comparison == {"successful_diff_observed": True, "baseline_hash": HASH,
        "baseline_retrieved_at": AT, "current_retrieved_at": LATER, "changed": True,
        "truncated": True, "request_id": "diff"}


def test_web_fetch_separate_from_internal_fetch():
    fetch = record("web_fetch", "web", data={"url": URL, "text": "content", "retrieved_at": LATER,
                                                 "truncated": True})
    source = one(status(), diff(), fetch)
    assert source["current_text_fetch"] == {"successful_web_fetch_observed": True,
        "retrieved_at": LATER, "truncated": True, "request_id": "web"}
    assert source["comparison_for_latest_baseline"]["request_id"] == "diff"


def test_failure_preserves_prior_success_and_ambiguous_snapshot():
    failure = record("source_diff", "failed", success=False, error="network")
    source = one(status(), diff(), failure)
    assert source["comparison_for_latest_baseline"]["request_id"] == "diff"
    assert source["latest_baseline_observation"]["content_hash"] == HASH
    failed_write = record("source_snapshot", "failed-write", success=False, error=SOURCE_SNAPSHOT_FAILED)
    source = one(status(), diff(), failed_write)
    assert source["latest_baseline_observation"]["state"] == "uncertain"
    assert source["previous_baseline_observation"]["content_hash"] == HASH
    assert source["previous_comparison_observation"]["request_id"] == "diff"


def test_malformed_or_conflicting_evidence_is_conservative():
    invalid = record("source_status", "bad", data={"url": URL, "canonical_url": "https://other.example/",
        "baseline_exists": True, "content_hash": HASH})
    source = one(status(), invalid)
    assert source["latest_baseline_observation"]["state"] == "uncertain"
    malformed = record("source_snapshot", "bad-snapshot", data={"baseline_created": True,
        "baseline_replaced": True, "baseline_status": "created"})
    assert one(malformed)["latest_baseline_observation"]["state"] == "uncertain"
    bad_url = record("source_status", "bad-url", url="not a URL")
    assert derive_source_observations([bad_url], EXECUTION)["sources"] == []


def test_url_identity_normalization_and_redirects():
    first = missing(url="HTTPS://EXAMPLE.COM/page#fragment")
    second = record("web_fetch", "web", url=URL, data={"url": URL,
        "final_url": "https://other.example/page", "text": "x", "retrieved_at": AT, "truncated": False})
    output = derive_source_observations([first, second], EXECUTION)
    assert len(output["sources"]) == 1
    assert output["sources"][0]["requested_identity"] == URL
    separate = record("web_fetch", "other", url="https://other.example/page")
    assert len(derive_source_observations([first, second, separate], EXECUTION)["sources"]) == 2


def test_scope_includes_earlier_revision_excludes_foreign_and_unscoped():
    records = [missing(revision=1), status(execution="foreign"), status(execution=None),
               record("web_fetch", "web", revision=2, data={"url": URL, "text": "x",
                                                              "retrieved_at": AT, "truncated": False})]
    source = one(*records)
    assert source["latest_baseline_observation"]["state"] == "missing"
    assert source["current_text_fetch"]["request_id"] == "web"


def test_bound_after_fold_and_report_omissions():
    records = [record("web_fetch", str(i), url=f"https://example.com/{i}") for i in range(MAX_SOURCES + 3)]
    output = derive_source_observations(records, EXECUTION)
    assert len(output["sources"]) == MAX_SOURCES
    assert output["omitted_source_count"] == 3


def test_reducer_uses_records_beyond_brain_display_limit():
    unrelated = [record("unrelated", str(i)) for i in range(40)]
    context = brain_input().model_copy(update={"tool_execution_history": (missing(), snapshot(), *unrelated)})
    evidence = json.loads(_build_step_progress_messages(brain_input=context)[0].content.split("\n", 1)[1])
    assert len(evidence["current_attempts"]) == 32
    assert evidence["source_observations"]["sources"][0]["first_baseline_observation"]["state"] == "missing"


def test_repeated_matching_status_preserves_snapshot_origin_and_diff():
    created = snapshot()
    compare = record("source_diff", "new-diff", data={"url": URL, "baseline_hash": HASH2,
        "baseline_retrieved_at": LATER, "current_retrieved_at": LATER, "changed": False,
        "diff_truncated": False})
    same = record("source_status", "new-status", data={"url": URL, "canonical_url": URL,
        "baseline_exists": True, "retrieved_at": LATER, "content_hash": HASH2})
    source = one(created, compare, same)
    assert source["latest_baseline_observation"]["origin_request_id"] == "snapshot"
    assert source["comparison_for_latest_baseline"]["request_id"] == "new-diff"


def test_brain_message_has_projection_and_unchanged_raw_evidence():
    context = brain_input().model_copy(update={"tool_execution_history": (missing(), snapshot())})
    message, = _build_step_progress_messages(brain_input=context)
    header, body = message.content.split("\n", 1)
    assert "not a live source-store read or a completion verdict" in header
    evidence = json.loads(body)
    fresh_context = context.model_copy(update={"last_tool_result": context.tool_execution_history[-1].result})
    fresh_message, = _build_step_progress_messages(brain_input=fresh_context)
    assert json.loads(fresh_message.content.split("\n", 1)[1])["source_observations"] == evidence["source_observations"]
    assert [item["tool"] for item in evidence["current_attempts"]] == ["source_status", "source_snapshot"]
    assert evidence["current_attempts"][0]["error"]["code"] == SOURCE_BASELINE_NOT_FOUND
    assert evidence["current_attempts"][1]["evidence"]["baseline_created"] is True
    assert evidence["source_observations"]["sources"][0]["latest_baseline_observation"]["origin"] == "created_this_execution"
    source = evidence["source_observations"]["sources"][0]
    assert "last_failure" not in source
    assert source["last_observed_failure"]["baseline_absence_superseded_by_request_id"] == "snapshot"
