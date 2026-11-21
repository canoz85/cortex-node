"""Brain display policy exercised through the actual evidence message."""
import json

import pytest

from core.brain_evidence_policy import MAX_VALUE_CHARS, PAYLOAD_CHARS
from core.protocol.models import ContentIntegrity, PaginationMetadata, ToolExecutionRecord, ToolResult
from test_brain_outcomes import brain_input, evidence_prompt


def record(data, *, step="s1", truncated=False, arguments=None, success=True):
    return ToolExecutionRecord(
        step_id=step, tool_name="inspect", arguments=arguments or {},
        execution_id="brain-outcome-test", plan_id="p1", plan_revision=1,
        result=ToolResult(request_id=step, success=success, data=data,
                          message="observed", error_code=None if success else "failed",
                          integrity=ContentIntegrity(is_truncated=truncated)),
    )


def test_current_and_prior_display_preserve_small_facts_and_only_preview_large_fields():
    text = " leading " + "0123456789" * 1000 + " final "
    data = {"body": text, "flags": {"count": 2, "active": False, "tag": " OK "}}
    history = (record(data), record(data, step="earlier"))
    context = brain_input().model_copy(update={"tool_execution_history": history})
    before = context.model_dump_json()
    view = evidence_prompt(context)
    for attempt in (view["current_attempts"][0], view["prior_facts"][0]):
        assert attempt["plan_revision"] == 1 and attempt["request_id"] == attempt["step"]
        assert attempt["evidence"]["flags"] == data["flags"]
        preview = attempt["evidence"]["body"]
        assert preview["display_compacted"] and preview["type"] == "string"
        assert preview["captured_chars"] == len(text)
        assert preview["preview"] and text.startswith(preview["preview"])
        assert text not in json.dumps(view)
    assert context.model_dump_json() == before


def test_empty_missing_hidden_and_partial_source_have_distinct_displays():
    values = (None, [], {}, "", ["row"] * 500, "partial")
    context = brain_input().model_copy(update={"tool_execution_history": tuple(
        record(value, truncated=value == "partial") for value in values)})
    attempts = evidence_prompt(context)["current_attempts"]
    assert "evidence" not in attempts[0]
    assert [a["evidence"] for a in attempts[1:4]] == [[], {}, ""]
    assert attempts[4]["evidence"]["captured_count"] == 500
    assert attempts[4]["evidence"]["preview"] and "integrity" not in attempts[4]
    assert attempts[5]["evidence"] == "partial" and attempts[5]["integrity"]["is_truncated"]


def test_nested_error_payloads_and_arguments_share_the_display_budget():
    data = {str(i): [{"line": '"\\' * 500}] * 40 for i in range(40)}
    bad = record(data, arguments={"query": "x" * 5000}, success=False)
    context = brain_input().model_copy(update={"tool_execution_history": (bad,) * 32})
    view = evidence_prompt(context)
    assert evidence_prompt(context) == view
    total = 0
    for attempt in view["current_attempts"]:
        assert attempt["arguments_sha256"] and attempt["args"]["query"]["display_compacted"]
        details = attempt["error"]["details"]
        assert attempt["error"]["code"] == "failed" and details["type"] == "object"
        assert details["captured_count"] == len(data) and details["display_compacted"]
        for field in (details, attempt["args"], attempt["error"]["message"]):
            size = len(json.dumps(field, separators=(",", ":")))
            assert size <= min(MAX_VALUE_CHARS, PAYLOAD_CHARS // (4 * len(view["current_attempts"])))
            total += size
    assert total <= PAYLOAD_CHARS


def test_first_read_exposes_assignments_beyond_prefix_then_compacts_on_retry():
    content = ("# imports\n" * 75 + '    "model": "gemma4:26b",\n'
               + '    "model_planner": "qwen3.8:27b",\n')
    data = {"path": "main.py", "content": content, "offset": 400}
    read = record(data, arguments={"path": "main.py", "offset": 400}, truncated=True)
    read = read.model_copy(update={"tool_name": "read_file", "result": read.result.model_copy(
        update={"pagination": PaginationMetadata(has_more=True, offset=400, returned_items=len(content))})})
    context = brain_input().model_copy(update={"tool_execution_history": (read,),
                                               "last_tool_result": read.result})
    before = context.model_dump_json()
    first, = evidence_prompt(context)["current_attempts"]
    assert first["evidence"] == data
    assert first["integrity"]["is_truncated"] and first["pagination"]["has_more"]
    assert evidence_prompt(context)["current_attempts"] == [first]

    later = context.model_copy(update={"last_tool_result": None})
    replay, = evidence_prompt(later)["current_attempts"]
    preview = replay["evidence"]["content"]
    assert preview["display_compacted"] and "model" not in preview["preview"]
    assert replay["pagination"] == first["pagination"]
    assert context.model_dump_json() == before


@pytest.mark.parametrize("success", [True, False])
def test_only_delivered_observation_is_intact_for_arbitrary_tools(success):
    data = {"samples": list(range(50)), "body": "x" * 600}
    older = record(data)
    new = record(data, success=success).model_copy(update={"tool_name": "sample_stream"})
    new = new.model_copy(update={"result": new.result.model_copy(update={"request_id": "next"})})
    context = brain_input().model_copy(update={"tool_execution_history": (older, new),
                                               "last_tool_result": new.result})
    previous, latest = evidence_prompt(context)["current_attempts"]
    assert previous["evidence"]["body"]["display_compacted"]
    if success:
        assert latest["evidence"] == data
    else:
        assert latest["error"] == {"message": "observed", "code": "failed", "details": data}
    stale_delivery = context.model_copy(update={"last_tool_result": older.result})
    previous, latest = evidence_prompt(stale_delivery)["current_attempts"]
    details = latest["evidence"] if success else latest["error"]["details"]
    assert previous["evidence"]["body"]["display_compacted"] and details["body"]["display_compacted"]


@pytest.mark.parametrize("data,args,older_count", [
    ("x" * 2100, {}, 0),                 # value alone exceeds the ceiling
    ("x" * 1100, {"query": "y" * 950}, 0),  # each field fits, whole observation does not
    ("x" * 1100, {}, 23),                # aggregate budget lowers this decision's allowance
    ("\u2603" * 600, {}, 0),              # prompt serialization escapes non-ASCII characters
])
def test_newest_observation_outside_allowance_uses_normal_compaction(data, args, older_count):
    newest = record(data, arguments=args)
    history = (record("older", step="prior"),) * older_count + (newest,)
    context = brain_input().model_copy(update={"tool_execution_history": history})
    compact = evidence_prompt(context)
    fresh = evidence_prompt(context.model_copy(update={"last_tool_result": newest.result}))
    assert fresh == compact
    assert fresh["current_attempts"][0]["evidence"]["display_compacted"]
