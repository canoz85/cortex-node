"""Size-only correction keeps the native-action contract and execution limits."""

import inspect
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import core.brain_provider as provider_module
from core.brain_batch_policy import MAX_READ_FILE_BATCH
from core.protocol.enums import BrainOutcomeKind as Kind
from test_brain_cardinality_correction import CORRECTION
from test_brain_outcomes import brain_input, native_action
from test_brain_read_file_batch import native_batch, provider_output


def expected_oversized_instruction(count):
    return (
        f"The previous response contained {count} calls in an otherwise valid "
        f"homogeneous read-only batch, but the maximum permitted batch size is {MAX_READ_FILE_BATCH}. "
        f"Return one native action containing at most {MAX_READ_FILE_BATCH} of those useful calls. "
        "Use the same tool. Do not add new calls. Leave content empty."
    )


@pytest.mark.parametrize("count", [17, MAX_READ_FILE_BATCH])
def test_live_pattern_and_maximum_batch_are_accepted_without_correction(count):
    outcome, model = provider_output(native_batch(count))
    assert outcome.kind == Kind.TOOL_REQUESTED
    assert len(outcome.tool_requests) == count
    assert len(model.calls) == 1


@pytest.mark.parametrize("content", ["", "I'll batch these independent reads."])
def test_oversized_correction_accepts_model_selected_maximum_batch(content):
    original = native_batch(MAX_READ_FILE_BATCH + 1).model_copy(update={"content": content})
    snapshot = original.model_dump()
    # The model selects a non-prefix subset. Runtime must preserve its selection.
    corrected = AIMessage(content="", tool_calls=original.tool_calls[1:])
    outcome, model = provider_output(original, corrected)
    assert len(model.calls) == 2
    assert model.calls[1][-1].content == expected_oversized_instruction(len(original.tool_calls))
    assert "Return exactly one native call" not in model.calls[1][-1].content
    assert "Do not batch" not in model.calls[1][-1].content
    rejected = model.calls[1][-2]
    assert rejected.content == ""
    assert rejected.tool_calls == original.tool_calls
    assert len(rejected.tool_calls) == MAX_READ_FILE_BATCH + 1
    assert original.model_dump() == snapshot
    assert outcome.kind == Kind.TOOL_REQUESTED
    assert outcome.tool_request is None
    assert [request.arguments for request in outcome.tool_requests] == [
        call["args"] for call in corrected.tool_calls
    ]
    assert len({request.request_id for request in outcome.tool_requests}) == MAX_READ_FILE_BATCH


@pytest.mark.parametrize("corrected,kind,error", [
    (native_batch(MAX_READ_FILE_BATCH + 1), Kind.INVALID_OUTPUT, "exactly_one_tool_call_required"),
    (AIMessage(content=""), Kind.INVALID_OUTPUT, "native_tool_call_required"),
    (native_action("unknown", {}), Kind.INVALID_OUTPUT, "unknown_tool"),
    (native_action("read_file", {"path": "chosen"}), Kind.TOOL_REQUESTED, None),
])
def test_oversized_correction_output_uses_normal_validation_without_third_call(corrected, kind, error):
    outcome, model = provider_output(native_batch(MAX_READ_FILE_BATCH + 1), corrected)
    assert len(model.calls) == 2
    assert outcome.kind == kind
    assert outcome.error_code == error
    assert outcome.tool_requests is None
    if kind == Kind.TOOL_REQUESTED:
        assert outcome.tool_request.arguments == {"path": "chosen"}
    else:
        assert outcome.tool_request is None


@pytest.mark.parametrize("disqualifier", ["heterogeneous", "duplicate", "mutating", "ineligible", "lifecycle"])
def test_oversized_group_with_another_disqualifier_keeps_choose_one_correction(disqualifier):
    count = MAX_READ_FILE_BATCH + 1
    raw = native_batch(count)
    if disqualifier == "heterogeneous":
        raw.tool_calls[-1].update(name="list_files", args={"path": "."})
    elif disqualifier == "duplicate":
        # Duplicate appears beyond the maximum, after effective default/path normalization.
        raw.tool_calls[-1]["args"] = {"path": "./file-0.txt", "offset": 0, "limit": 1}
    elif disqualifier == "mutating":
        raw = native_batch(count, names=["write_file"] * count, args=[
            {"path": f"file-{i}", "content": "x"} for i in range(count)
        ])
    elif disqualifier == "ineligible":
        raw = native_batch(count, names=["list_files"] * count, args=[
            {"path": f"dir-{i}"} for i in range(count)
        ])
    else:
        raw.tool_calls[-1].update(name="brain_step_completed", args={"message": "Done"})
    outcome, model = provider_output(raw, native_action("read_file", {"path": "chosen"}))
    assert len(model.calls) == 2
    assert model.calls[1][-1].content == CORRECTION
    assert outcome.tool_request.arguments == {"path": "chosen"}
    assert outcome.tool_requests is None


@pytest.mark.parametrize("bad_call", [
    {"name": "unknown", "args": {}, "id": "bad"},
    {"name": "read_file", "args": "not an object", "id": "bad"},
    {"name": "read_file", "args": {}, "id": "bad"},
    {"name": "read_file", "args": {"path": "a", "limit": 0}, "id": "bad"},
    {"name": "read_file", "args": {"path": "a"}, "id": "bad", "extra": True},
])
def test_invalid_oversized_group_remains_strict_without_correction(bad_call):
    calls = [*native_batch(MAX_READ_FILE_BATCH).tool_calls, bad_call]
    raw = SimpleNamespace(content="Incidental text", tool_calls=calls)
    outcome, model = provider_output(raw)
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.tool_requests is None and outcome.tool_request is None
    assert len(model.calls) == 1


def test_unauthorized_oversized_group_remains_strict_without_correction():
    context = brain_input()
    context = context.model_copy(update={"active_plan": context.active_plan.model_copy(
        update={"available_tools": ("list_files",)},
    )})
    outcome, model = provider_output(native_batch(MAX_READ_FILE_BATCH + 1), context=context)
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert len(model.calls) == 1


def test_correction_consumes_the_authoritative_maximum():
    assert provider_module.MAX_READ_FILE_BATCH == MAX_READ_FILE_BATCH
    source = inspect.getsource(provider_module._ensure_native_call)
    assert "MAX_READ_FILE_BATCH" in source
    assert str(MAX_READ_FILE_BATCH) not in source
