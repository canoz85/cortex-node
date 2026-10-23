import pytest
from langchain_core.messages import AIMessage

from core.logging.node_update import extract_node_update
from core.logging.renderer import render_node_update
from core.protocol.enums import (
    BrainOutcomeKind,
    ControllerDecisionType,
    ExecutionStatus,
    StepStatus,
    WorkerRole,
)
from core.protocol.models import (
    BrainOutcome,
    ControllerDecision,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    ExecutionSummary,
    FinalizationResult,
    ProtocolVisibleState,
    StepCompletionEvidence,
    ToolExecutionRecord,
    ToolRequest,
    ToolResult,
    WorkingState,
)


IDENTITY = ExecutionIdentity(execution_id="observability", protocol_version="1")


def plan():
    return ExecutionPlan(
        plan_id="accepted-plan",
        revision=1,
        steps=(
            ExecutionStep(
                step_id="list",
                title="Retrieve file list",
                primary_tool="list_files",
                status=StepStatus.ACTIVE,
            ),
            ExecutionStep(
                step_id="explain",
                title="Explain the findings",
            ),
        ),
    )


def state_with_result(tool_name, result):
    accepted_plan = plan()
    request = ToolRequest(
        request_id=result.request_id,
        tool_name=tool_name,
        arguments={"path": "secret-path.txt", "content": "secret argument"},
    )
    return (
        ExecutionState(
            protocol_visible=ProtocolVisibleState(
                identity=IDENTITY,
                cursor=ExecutionCursor(current_worker=WorkerRole.TOOL_RUNTIME),
                active_plan=accepted_plan,
                active_step=accepted_plan.steps[0],
                pending_tool_request=request,
            ),
            working=WorkingState(
                last_tool_result=result,
                tool_execution_history=(ToolExecutionRecord(
                    execution_id=IDENTITY.execution_id,
                    plan_id=accepted_plan.plan_id,
                    plan_revision=accepted_plan.revision,
                    step_id="list",
                    tool_name=tool_name,
                    arguments=request.arguments,
                    result=result,
                ),),
            ),
        ),
        request,
    )


def render(value):
    update = extract_node_update(
        from_node="controller",
        to_node="controller",
        value=value,
    )
    render_node_update(update)


def test_default_does_not_retain_redundant_plan_and_tool_request_blocks(capsys):
    accepted_plan = plan()
    request = ToolRequest(
        request_id="read-1",
        tool_name="read_file",
        arguments={"path": "private.py", "offset": 0},
    )
    render({
        "controller_decision": ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_BRAIN,
            next_worker=WorkerRole.BRAIN,
            reason="Plan accepted.",
            accepted_plan=accepted_plan,
            next_step_id="list",
        ),
        "brain_result": BrainOutcome(
            outcome=BrainOutcomeKind.TOOL_REQUESTED,
            step_id="list",
            tool_request=request,
        ),
        "messages": [AIMessage(
            content="",
            tool_calls=[{
                "name": "read_file",
                "args": request.arguments,
                "id": request.request_id,
                "type": "tool_call",
            }],
        )],
    })

    output = capsys.readouterr().out
    assert output == ""


def test_default_renders_direct_portable_tool_result_without_large_payload(capsys):
    content = "PRIVATE FILE CONTENT\n" * 5000
    result = ToolResult(
        request_id="read-1",
        success=True,
        message="Read file: anomaly_detection.py (100000 total characters)",
        rendered_output=content,
        data={"path": "anomaly_detection.py", "content": content},
    )
    execution_state, request = state_with_result("read_file", result)

    render({
        "execution_state": execution_state,
        "controller_decision": ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
            next_worker=WorkerRole.TOOL_RUNTIME,
            pending_tool_request=request,
        ),
    })

    output = capsys.readouterr().out
    assert "[tool:read_file]" in output
    assert "OK — Read file:" in output
    assert len(output.splitlines()) == 1
    assert "Read file: anomaly_detection.py" in output
    assert "PRIVATE FILE CONTENT" not in output
    assert "secret-path.txt" not in output
    assert "secret argument" not in output


def test_default_uses_tool_owned_write_message_without_tool_specific_logger(capsys):
    result = ToolResult(
        request_id="write-1",
        success=True,
        message="Wrote 42 characters to report.txt",
        data={"path": "report.txt", "content": "private written content"},
    )
    execution_state, request = state_with_result("write_file", result)

    render({
        "execution_state": execution_state,
        "controller_decision": ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
            next_worker=WorkerRole.TOOL_RUNTIME,
            pending_tool_request=request,
        ),
    })

    output = capsys.readouterr().out
    assert "[tool:write_file]" in output
    assert "Wrote 42 characters to report.txt" in output
    assert "private written content" not in output


def test_default_tool_result_has_safe_generic_fallback(capsys):
    result = ToolResult(request_id="custom-1", success=True, message="", data={
        "private": "payload must not be logged",
    })
    execution_state, request = state_with_result("custom_tool", result)

    render({
        "execution_state": execution_state,
        "controller_decision": ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
            next_worker=WorkerRole.TOOL_RUNTIME,
            pending_tool_request=request,
        ),
    })

    output = capsys.readouterr().out
    assert "[tool:custom_tool]" in output
    assert "Completed." in output
    assert "payload must not be logged" not in output


def test_default_suppresses_controller_accepted_brain_completion(capsys):
    completion = StepCompletionEvidence(
        execution_id=IDENTITY.execution_id,
        plan_id="accepted-plan",
        plan_revision=1,
        step_id="list",
        summary="File list retrieved successfully.",
        evidence_id="evidence-1",
    )

    value = {
        "controller_decision": ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_BRAIN,
            next_worker=WorkerRole.BRAIN,
            completed_step_id="list",
            completion_evidence=completion,
        ),
    }

    render(value)

    output = capsys.readouterr().out
    assert "[brain]" not in output
    assert "File list retrieved successfully." not in output

    update = extract_node_update(
        from_node="brain", to_node="controller", value=value
    )
    assert update.accepted_completion_message == "File list retrieved successfully."

    render_node_update(update, verbose=True)
    verbose_output = capsys.readouterr().out
    assert "[controller]" in verbose_output
    assert "Step completed: list" in verbose_output
    assert "File list retrieved successfully." not in verbose_output


def test_default_preserves_finalizer_attribution(capsys):
    answer = "Concise final answer"
    result = FinalizationResult(
        execution_summary=ExecutionSummary(
            execution_id=IDENTITY.execution_id,
            status=ExecutionStatus.COMPLETED,
            summary_text="Completed.",
        ),
        final_answer=answer,
    )

    render({
        "finalization_result": result,
        "messages": [AIMessage(content=answer)],
    })

    output = capsys.readouterr().out
    assert "[finalizer]" in output
    assert answer in output


@pytest.mark.parametrize("success,message,expected", [
    (True, "Fetched page\n\tOne question found", "OK — Fetched page One question found"),
    (False, "Page unavailable\r\nTimeout", "FAIL — Page unavailable Timeout"),
    (True, "", "OK — Completed."),
    (False, "", "FAIL — Failed."),
])
def test_tool_result_is_one_explicit_status_line(capsys, success, message, expected):
    result = ToolResult(request_id="call-1", success=success, message=message)
    execution_state, request = state_with_result("read_file", result)
    render({"execution_state": execution_state, "controller_decision": ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
        next_worker=WorkerRole.TOOL_RUNTIME, pending_tool_request=request,
    )})
    output = capsys.readouterr().out
    assert expected in output
    assert "[tool:read_file] [list]" in output
    assert len(output.splitlines()) == 1


def test_tool_result_message_stays_bounded_and_payload_is_hidden(capsys):
    result = ToolResult(request_id="long", success=False, message="x" * 1000,
                        data={"private": "hidden payload"}, rendered_output="private stdout")
    execution_state, request = state_with_result("read_file", result)
    render({"execution_state": execution_state, "controller_decision": ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
        next_worker=WorkerRole.TOOL_RUNTIME, pending_tool_request=request,
    )})
    output = capsys.readouterr().out
    assert "FAIL — " + "x" * 500 + "..." in output
    assert "x" * 501 not in output
    assert "hidden payload" not in output and "private stdout" not in output
    assert len(output.splitlines()) == 1


@pytest.mark.parametrize("status", [
    ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
])
def test_terminal_console_reports_runtime_status_without_claiming_fulfillment(capsys, status):
    answer = "Question 2 is absent from the page and could not be solved."
    result = FinalizationResult(execution_summary=ExecutionSummary(
        execution_id=IDENTITY.execution_id, status=status, summary_text="Terminal facts.",
    ), final_answer=answer)
    before = result.model_dump(mode="json")
    # The authoritative answer must also render without a mirrored AIMessage.
    render({"finalization_result": result})
    output = capsys.readouterr().out
    assert f"[runtime:{status.value}] Runtime {status.value}." in output
    assert output.count(answer) == 1
    if status == ExecutionStatus.COMPLETED:
        assert "Request outcome is described in the final answer." in output
    assert "SUCCESS" not in output and "DONE" not in output
    assert result.model_dump(mode="json") == before


def test_completed_runtime_with_final_answer_rendering_error_is_explicit(capsys):
    result = FinalizationResult(execution_summary=ExecutionSummary(
        execution_id=IDENTITY.execution_id, status=ExecutionStatus.COMPLETED,
        summary_text="Completed.",
    ), final_answer="Execution finished, but the final answer could not be rendered.",
        final_answer_error="RuntimeError: private provider diagnostics")
    render({"finalization_result": result})
    output = capsys.readouterr().out
    assert "[runtime:completed]" in output
    assert "Final answer unavailable: rendering failed." in output
    assert "private provider diagnostics" not in output
    assert output.count(result.final_answer) == 1


def test_controller_transitions_are_verbose_only_and_semantic(capsys):
    accepted_plan = plan()
    request = ToolRequest(
        request_id="read-1",
        tool_name="read_file",
        arguments={"path": "private.py"},
    )
    value = {
        "controller_decision": ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_BRAIN,
            next_worker=WorkerRole.BRAIN,
            reason="Plan accepted.",
            accepted_plan=accepted_plan,
            next_step_id="list",
        ),
        "brain_result": BrainOutcome(
            outcome=BrainOutcomeKind.TOOL_REQUESTED,
            step_id="list",
            tool_request=request,
        ),
    }
    update = extract_node_update(
        from_node="planner", to_node="controller", value=value
    )

    render_node_update(update)
    normal = capsys.readouterr().out
    assert "[controller]" not in normal

    render_node_update(update, verbose=True)
    verbose = capsys.readouterr().out
    assert "[controller]" in verbose
    assert "Plan ready" in verbose
    assert "Tool requested: read_file" in verbose
    assert "ExecutionState" not in verbose
