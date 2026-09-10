import json

import pytest
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from core.completion import immutable
from core.graph_capture import create_capture_tool_output_node
from core.models import ListFilesResult, ReadFileResult
from core.models import ToolResult as TransportToolResult
from core.protocol.enums import AsyncJobStatus, ControllerDecisionType, ExecutionPhase, WorkerRole
from core.protocol.models import (
    ControllerDecision,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    ProtocolVisibleState,
    ToolRequest,
    WorkingState,
)


def _build_state(
    *,
    step_id: str,
    request_id: str,
    tool_name: str,
    arguments: dict,
    content: str,
    execution_state: ExecutionState | None = None,
):
    if execution_state is None:
        execution_state = ExecutionState(
            protocol_visible=ProtocolVisibleState(
                identity=ExecutionIdentity(
                    execution_id="run-1",
                    protocol_version="1.0",
                ),
                cursor=ExecutionCursor(
                    phase=ExecutionPhase.EXECUTING,
                ),
                active_plan=ExecutionPlan(
                    plan_id="p-1",
                    revision=1,
                    objective="demo",
                    steps=(
                        ExecutionStep(
                            step_id="s1",
                            title="step-1",
                        ),
                        ExecutionStep(
                            step_id="s2",
                            title="step-2",
                        ),
                    ),
                ),
                active_step=ExecutionStep(
                    step_id=step_id,
                    title=step_id,
                ),
            ),
            working=WorkingState(),
        )
    else:
        execution_state = execution_state.model_copy(
            update={
                "protocol_visible": execution_state.protocol_visible.model_copy(
                    update={
                        "active_step": ExecutionStep(
                            step_id=step_id,
                            title=step_id,
                        )
                    }
                )
            }
        )

    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
        reason="tool_request",
        next_worker=WorkerRole.TOOL_RUNTIME,
        pending_tool_request=ToolRequest(
            request_id=request_id,
            tool_name=tool_name,
            arguments=arguments,
        ),
    )

    return {
        "execution_state": execution_state,
        "controller_decision": decision,
        "messages": [
            ToolMessage(
                content=content,
                tool_call_id=request_id,
            )
        ],
    }


def test_capture_appends_tool_execution_record_with_expected_fields():
    node = create_capture_tool_output_node()
    payload = TransportToolResult(
        success=True,
        message="Listing for .",
        data={"entries": ["a.py", "b.py"]},
    ).to_tool_output()

    state = _build_state(
        step_id="s1",
        request_id="req-1",
        tool_name="list_files",
        arguments={"path": "."},
        content=payload,
    )

    update = node(state)
    working = update["execution_state"].working

    assert working.last_tool_result is not None
    assert len(working.tool_execution_history) == 1

    record = working.tool_execution_history[0]
    assert record.step_id == "s1"
    assert record.tool_name == "list_files"
    assert record.arguments == {"path": "."}
    assert record.result.request_id == "req-1"
    assert record.result.success is True
    assert record.result.data == {"entries": ["a.py", "b.py"]}


def test_capture_preserves_previous_records_in_same_step():
    node = create_capture_tool_output_node()

    first_payload = TransportToolResult(
        success=True,
        message="Read file: a.py",
        data={"path": "a.py", "content": "print('a')"},
    ).to_tool_output()
    state1 = _build_state(
        step_id="s2",
        request_id="req-1",
        tool_name="read_file",
        arguments={"path": "a.py"},
        content=first_payload,
    )
    update1 = node(state1)

    second_payload = TransportToolResult(
        success=True,
        message="Read file: b.py",
        data={"path": "b.py", "content": "print('b')"},
    ).to_tool_output()
    state2 = _build_state(
        step_id="s2",
        request_id="req-2",
        tool_name="read_file",
        arguments={"path": "b.py"},
        content=second_payload,
        execution_state=update1["execution_state"],
    )
    update2 = node(state2)

    history = update2["execution_state"].working.tool_execution_history
    assert len(history) == 2
    assert history[0].result.request_id == "req-1"
    assert history[1].result.request_id == "req-2"
    assert history[0].step_id == "s2"
    assert history[1].step_id == "s2"


def test_capture_keeps_cross_step_records_with_original_step_ids():
    node = create_capture_tool_output_node()

    step1_payload = TransportToolResult(
        success=True,
        message="Listing for .",
        data={"entries": ["a.py", "b.py", "c.py"]},
    ).to_tool_output()
    state1 = _build_state(
        step_id="s1",
        request_id="req-1",
        tool_name="list_files",
        arguments={"path": "."},
        content=step1_payload,
    )
    update1 = node(state1)

    step2_payload = TransportToolResult(
        success=True,
        message="Read file: a.py",
        data={"path": "a.py", "content": "print('a')"},
    ).to_tool_output()
    state2 = _build_state(
        step_id="s2",
        request_id="req-2",
        tool_name="read_file",
        arguments={"path": "a.py"},
        content=step2_payload,
        execution_state=update1["execution_state"],
    )
    update2 = node(state2)

    history = update2["execution_state"].working.tool_execution_history
    assert len(history) == 2
    assert history[0].step_id == "s1"
    assert history[1].step_id == "s2"


def test_capture_preserves_async_job_evidence_in_protocol_result():
    node = create_capture_tool_output_node()
    payload = TransportToolResult(
        success=True,
        message="ComfyUI workflow accepted.",
        is_async_job=True,
        async_job_id="prompt-1",
        async_job_status=AsyncJobStatus.SUBMITTED,
        async_terminal=False,
    ).to_tool_output()
    state = _build_state(
        step_id="s1",
        request_id="req-1",
        tool_name="run_comfy_workflow",
        arguments={"workflow_json": {"1": {}}},
        content=payload,
    )

    update = node(state)
    result = update["execution_state"].working.last_tool_result

    assert result is not None
    assert result.is_async_job is True
    assert result.async_job_id == "prompt-1"
    assert result.async_job_status == AsyncJobStatus.SUBMITTED
    assert result.async_terminal is False


def test_capture_does_not_count_nonterminal_async_observation_as_failure():
    node = create_capture_tool_output_node()
    payload = TransportToolResult(
        success=False,
        message="ComfyUI history is temporarily unavailable.",
        is_async_job=True,
        async_job_id="prompt-1",
        async_job_status=AsyncJobStatus.UNKNOWN,
        async_terminal=False,
    ).to_tool_output()
    state = _build_state(
        step_id="s1",
        request_id="req-1",
        tool_name="get_comfy_history",
        arguments={"prompt_id": "prompt-1"},
        content=payload,
    )

    update = node(state)

    assert update["execution_state"].working.repeat_fail_count == 0


def _capture_result(tool_name, content):
    state = _build_state(
        step_id="s1", request_id="req-1", tool_name=tool_name,
        arguments={"path": "."}, content=content,
    )
    captured = create_capture_tool_output_node()(state)["execution_state"]
    result = captured.working.tool_execution_history[-1].result
    restored = ExecutionState.model_validate_json(captured.model_dump_json())
    assert restored.working.tool_execution_history[-1].result == result
    assert captured.working.last_tool_result == result
    return result


@pytest.mark.parametrize("entries,is_file", [
    (["docs/", "a.py", "notes.txt"], False),
    ([], False),
    (["src/a.py"], True),
])
def test_capture_actual_list_files_transport(entries, is_file):
    payload = ListFilesResult(
        success=True, message="Listing", path="src", entries=entries, is_file=is_file,
    )
    result = _capture_result("list_files", payload.to_tool_output())
    assert result.success is True
    assert result.data == {"path": "src", "entries": entries, "is_file": is_file}
    with pytest.raises(ValidationError):
        result.data = {}
    snapshot = immutable(result.model_dump(mode="json"))
    with pytest.raises(TypeError):
        snapshot["data"]["entries"] = ()
    assert snapshot["data"]["entries"] == tuple(entries)


@pytest.mark.parametrize("content,total,offset,count,truncated", [
    ("print('hello')\n", 15, 0, 15, False),
    ("", 0, 0, 0, False),
    ("ç漢😀\n", 4, 0, 4, False),
    ("bc\n\n--- [TRUNCATED] ---", 6, 1, 2, True),
    ("ef", 6, 4, 2, False),
])
def test_capture_actual_read_file_transport(content, total, offset, count, truncated):
    payload = ReadFileResult(
        success=True, message="Read file", path="a.py", content=content,
        total_chars=total, offset=offset, read_chars=count, is_truncated=truncated,
    )
    result = _capture_result("read_file", payload.to_tool_output())
    assert result.success is True
    assert result.data == {
        "path": "a.py", "content": content, "total_chars": total,
        "offset": offset, "read_chars": count, "is_truncated": truncated,
    }


@pytest.mark.parametrize("tool_name", ["list_files", "read_file"])
def test_capture_file_fields_does_not_supply_missing_evidence(tool_name):
    result = _capture_result(tool_name, json.dumps({
        "success": True, "message": "Incomplete payload", "path": "a.py",
        "unexpected": "must not pass through",
    }))
    assert result.data == {"path": "a.py"}


@pytest.mark.parametrize("tool_name", ["list_files", "read_file", "other_tool"])
@pytest.mark.parametrize("data", [{"values": [1, "ç"]}, [], "text", 0, False, None])
def test_capture_existing_data_envelope_takes_precedence(tool_name, data):
    result = _capture_result(tool_name, json.dumps({
        "success": True, "message": "ok", "data": data,
        "path": "a.py", "entries": ["a.py"], "content": "ignored",
    }))
    assert result.data == data


def test_capture_does_not_preserve_arbitrary_top_level_payload():
    result = _capture_result("other_tool", json.dumps({
        "success": True, "message": "ok", "path": "a.py",
        "entries": ["a.py"], "content": "not a read_file result",
    }))
    assert result.data is None


@pytest.mark.parametrize("tool_name,payload", [
    ("list_files", ListFilesResult(success=False, message="failed", path="missing")),
    ("read_file", ReadFileResult(success=False, message="failed", path="missing")),
])
def test_capture_failed_file_result_remains_failed(tool_name, payload):
    result = _capture_result(tool_name, payload.to_tool_output())
    assert result.success is False
    assert result.data["path"] == "missing"
