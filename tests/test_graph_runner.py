from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.graph_constants import MAX_REASONING_STEPS
from core.graph_runner import run_prompt
from core.models import ToolResult


class FakeApp:
    def __init__(self, events: list[dict]):
        self._events = events
        self.initial_state: dict | None = None

    def stream(self, initial_state):
        self.initial_state = initial_state
        for event in self._events:
            yield event


def test_run_prompt_handles_tool_flow_and_updates_summary(capsys):
    planner_event = {
        "planner": {
            "steps": 1,
            "planner_route": "action",
            "plan": "1. call list_files\n2. summarize",
            "rolling_summary": "summary-after-plan",
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
            "rolling_summary": "final-summary",
            "messages": [AIMessage(content="Completed")],
        }
    }

    app = FakeApp([planner_event, brain_with_tool_call, tools_event, final_brain_event])
    history, summary = run_prompt(app, "list files", history=[HumanMessage(content="previous")], rolling_summary="old")

    assert summary == "final-summary"
    assert len(history) == 5
    assert isinstance(history[-1], AIMessage)
    assert history[-1].content == "Completed"

    output = capsys.readouterr().out
    assert "[planner:action]" in output
    assert "Success: True" in output


def test_run_prompt_emits_pseudo_tool_stop_warning(capsys):
    pseudo_event = {
        "brain": {
            "steps": 2,
            "messages": [AIMessage(content="pseudo tool-call text detected")],
        }
    }
    app = FakeApp([pseudo_event])

    history, summary = run_prompt(app, "do task")

    assert len(history) == 2
    assert summary == ""
    output = capsys.readouterr().out
    assert "halted without executing those actions" in output


def test_run_prompt_emits_max_step_warning(capsys):
    max_step_event = {
        "brain": {
            "steps": MAX_REASONING_STEPS,
            "messages": [AIMessage(content="done")],
        }
    }
    app = FakeApp([max_step_event])

    run_prompt(app, "do task")

    output = capsys.readouterr().out
    assert "Max reasoning steps reached" in output
