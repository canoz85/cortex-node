import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.graph_runner import run_prompt
from core.models import ToolResult
from core.protocol.enums import BrainOutcomeKind
from core.protocol.models import (
    BrainOutcome,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionState,
    ProtocolVisibleState,
    ToolExecutionRecord,
    ToolRequest,
    ToolResult as ProtocolToolResult,
    WorkingState,
)


class FakeApp:
    def __init__(self, events: list[dict]):
        self._events = events
        self.initial_state: dict | None = None

    def stream(self, initial_state):
        self.initial_state = initial_state
        for event in self._events:
            yield event


def test_run_prompt_renders_portable_controller_tool_result_concisely(capsys):
    request = ToolRequest(
        request_id="call-1",
        tool_name="list_files",
        arguments={"path": "."},
    )
    brain_result = BrainOutcome(
        outcome=BrainOutcomeKind.TOOL_REQUESTED,
        tool_request=request,
    )
    request_message = AIMessage(
        content="",
        tool_calls=[{
            "name": "list_files", "args": {"path": "."},
            "id": "call-1", "type": "tool_call",
        }],
    )
    evidence = {"entries": [f"large-entry-{index}" for index in range(100)]}
    result = ProtocolToolResult(
        request_id="call-1",
        success=True,
        message="Listing for .",
        rendered_output="stdout that must stay private",
        data=evidence,
    )
    record = ToolExecutionRecord(
        step_id="step-1",
        tool_name="list_files",
        arguments={"path": "."},
        result=result,
    )
    execution_state = ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=ExecutionIdentity(execution_id="live-path", protocol_version="1"),
            cursor=ExecutionCursor(),
        ),
        working=WorkingState(
            last_tool_result=result,
            tool_execution_history=(record,),
        ),
    )
    raw_payload = "<tool_result_json>" + result.model_dump_json()
    app = FakeApp([
        {"controller": {
            "brain_result": brain_result,
            "messages": [request_message],
        }},
        {"controller": {
            "execution_state": execution_state,
            "messages": [ToolMessage(content=raw_payload, tool_call_id="call-1")],
        }},
    ])

    run_prompt(app, "list files")

    output = capsys.readouterr().out
    assert output.count("[brain]") == 1
    assert "Calling list_files" in output
    assert "Calling list_files with" not in output
    assert output.count("[tool:list_files]") == 1
    assert "Listing for ." in output
    assert "<tool_result_json>" not in output
    assert "large-entry-99" not in output
    assert "stdout that must stay private" not in output
    assert execution_state.working.last_tool_result is result
    assert execution_state.working.tool_execution_history[-1].result is result


def test_run_prompt_handles_tool_flow(capsys):
    planner_event = {
        "planner": {
            "steps": 1,
            "plan": "1. call list_files\n2. summarize",
        }
    }
    brain_with_tool_call = {
        "brain": {
            "steps": 2,
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "list_files", "args": {"path": "."}, "id": "call-1", "type": "tool_call"}],
                )
            ],
        }
    }
    tools_event = {
        "tools": {
            "steps": 3,
            "messages": [
                ToolMessage(
                    content=ToolResult(success=True, message="Listing for .", data={"entries": ["a.py"]}).to_tool_output(),
                    tool_call_id="call-1",
                )
            ],
        }
    }
    final_brain_event = {
        "brain": {
            "steps": 4,
            "messages": [AIMessage(content="Completed")],
        }
    }

    app = FakeApp([planner_event, brain_with_tool_call, tools_event, final_brain_event])
    history, _ = run_prompt(app, "list files", history=[HumanMessage(content="previous")], rolling_summary="old")

    assert [message.content for message in history] == ["previous", "list files"]

    output = capsys.readouterr().out
    assert "Completed" in output


def test_run_prompt_renders_pseudo_tool_text_without_legacy_stop_warning(capsys):
    pseudo_event = {
        "brain": {
            "steps": 2,
            "messages": [AIMessage(content="pseudo tool-call text detected")],
        }
    }
    app = FakeApp([pseudo_event])

    history, summary = run_prompt(app, "do task")

    assert [message.content for message in history] == ["do task"]
    assert summary == ""
    output = capsys.readouterr().out
    assert "pseudo tool-call text detected" in output


def test_run_prompt_does_not_infer_max_step_semantics_from_legacy_event_fields(capsys):
    max_step_event = {
        "brain": {
            "steps": 24,
            "messages": [AIMessage(content="done")],
        }
    }
    app = FakeApp([max_step_event])

    run_prompt(app, "do task")

    output = capsys.readouterr().out
    assert "done" in output


def test_run_prompt_logs_completion_metrics(caplog):
    planner_event = {
        "planner": {
            "steps": 1,
            "plan": "1. call list_files",
        }
    }
    brain_with_tool_call = {
        "brain": {
            "steps": 2,
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "list_files", "args": {"path": "."}, "id": "call-1", "type": "tool_call"}],
                )
            ],
        }
    }
    tools_event = {
        "tools": {
            "steps": 3,
            "messages": [
                ToolMessage(
                    content=ToolResult(success=True, message="Listing for .", data={"entries": ["a.py"]}).to_tool_output(),
                    tool_call_id="call-1",
                )
            ],
        }
    }
    final_brain_event = {
        "brain": {
            "steps": 4,
            "messages": [AIMessage(content="Completed")],
        }
    }

    app = FakeApp([planner_event, brain_with_tool_call, tools_event, final_brain_event])
    with caplog.at_level(logging.INFO):
        run_prompt(app, "list files")

    completed_records = [record for record in caplog.records if getattr(record, "event_name", "") == "prompt_completed"]
    assert len(completed_records) == 1
    completed = completed_records[0]
    # The raw ToolNode transport event is intentionally suppressed; the runner
    # counts only successfully extracted typed NodeUpdate events.
    assert completed.node_updates == 3
    assert completed.duration_ms >= 0
    assert completed.max_steps_reached is False


def test_run_prompt_attaches_execution_state_before_first_node():
    app = FakeApp([])

    run_prompt(app, "do task")

    assert app.initial_state is not None
    execution_state = app.initial_state.get("execution_state")
    assert isinstance(execution_state, ExecutionState)
    assert app.initial_state["execution_state"] is execution_state
