"""Plan positions and actual Brain calls are separate console-only counters."""

import io
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from core.brain_provider import LangChainBrainProvider
from core.logging.live_status import LiveStatus
from core.logging.node_update import NodeUpdate
from core.logging.renderer import render_node_update
from core.protocol.enums import BrainOutcomeKind
from core.protocol.models import BrainOutcome, ExecutionPlan, ExecutionStep, ToolRequest, ToolResult
from test_brain_outcomes import SequenceModel, brain_input, native_action
from tools.file_ops import get_file_tools


def plan():
    return ExecutionPlan(plan_id="console-plan", steps=tuple(
        ExecutionStep(step_id=step_id, title=step_id) for step_id in ("discover", "inspect", "summarize")
    ))


def test_plan_step_display_resets_invocations_and_preserves_turn_total():
    stream = io.StringIO()
    status = LiveStatus(stream=stream, refresh_interval=60, enabled=True)
    accepted = plan()
    status.start("planner")
    try:
        status.set_brain_step_context(execution_id="execution", plan=accepted, step_id="discover")
        assert status.provider_invocations_by_worker == {}
        assert status.stage == "planner"
        assert status.begin_provider_invocation(worker="brain") == 1
        assert status.detail == "plan step 1/3 · invocation 1"
        status.update("find_files", "Snake")
        status.set_brain_step_context(execution_id="execution", plan=accepted, step_id="discover")
        assert status.begin_provider_invocation(worker="brain") == 2
        assert status.detail == "plan step 1/3 · invocation 2"
        status.set_brain_step_context(execution_id="execution", plan=accepted, step_id="inspect")
        assert status.begin_provider_invocation(worker="brain") == 3
        assert status.detail == "plan step 2/3 · invocation 1"
        assert status.provider_invocations_by_worker == {"brain": 3}
        status.set_brain_step_context(execution_id="execution", plan=accepted, step_id="discover")
        status.begin_provider_invocation(worker="brain")
        assert status.detail == "plan step 1/3 · invocation 1"
        assert status.usage_by_worker == {}
    finally:
        status.stop()
    output = stream.getvalue()
    assert "brain · plan step 1/3 · invocation 1" in output
    assert "brain · plan step 1/3 · invocation 2" in output
    assert "brain · plan step 2/3 · invocation 1" in output
    assert "attempt" not in output


@pytest.mark.parametrize("change", ["execution", "plan", "revision"])
def test_new_execution_or_plan_scope_resets_display_counter(change):
    status = LiveStatus(enabled=False)
    accepted = plan()
    status.set_brain_step_context(execution_id="execution", plan=accepted, step_id="inspect")
    status.begin_provider_invocation(worker="brain")
    status.begin_provider_invocation(worker="brain")
    if change == "plan":
        accepted = accepted.model_copy(update={"plan_id": "replacement-plan"})
    elif change == "revision":
        accepted = accepted.model_copy(update={"revision": accepted.revision + 1})
    status.set_brain_step_context(execution_id="another-execution" if change == "execution" else "execution",
                                  plan=accepted, step_id="inspect")
    assert status.begin_provider_invocation(worker="brain") == 3
    assert status.detail == "plan step 2/3 · invocation 1"


@pytest.mark.parametrize("unresolved", ["no_plan", "unknown_step", "duplicate_step", "empty_plan"])
def test_unsafe_plan_position_uses_stable_step_id(unresolved):
    accepted = plan()
    step_id = "inspect"
    if unresolved == "no_plan":
        accepted = None
    elif unresolved == "unknown_step":
        step_id = "stable-id-99"
    elif unresolved == "duplicate_step":
        accepted = accepted.model_copy(update={"steps": (accepted.steps[1], accepted.steps[1])})
    else:
        accepted = accepted.model_copy(update={"steps": ()})
    status = LiveStatus(enabled=False)
    status.set_brain_step_context(execution_id="execution", plan=accepted, step_id=step_id)
    status.begin_provider_invocation(worker="brain")
    assert status.detail == f"step_id {step_id} · invocation 1"
    assert "plan step" not in status.detail


def test_missing_step_id_does_not_invent_a_plan_position():
    status = LiveStatus(enabled=False)
    status.set_brain_step_context(execution_id="execution", plan=plan(), step_id=None)
    status.begin_provider_invocation(worker="brain")
    assert status.detail == "invocation 1"


def test_other_workers_do_not_reset_brain_invocations():
    status = LiveStatus(enabled=False)
    status.set_brain_step_context(execution_id="execution", plan=plan(), step_id="inspect")
    status.begin_provider_invocation(worker="brain")
    status.begin_provider_invocation(worker="planner")
    assert (status.stage, status.detail) == ("planner", "")
    status.begin_provider_invocation(worker="brain")
    assert (status.stage, status.detail) == ("brain", "plan step 2/3 · invocation 2")
    assert status.provider_invocations_by_worker == {"brain": 2, "planner": 1}


def test_provider_counts_correction_calls_and_same_step_retries_without_behavior_changes():
    accepted = plan()
    context = brain_input().model_copy(update={"active_plan": accepted, "active_step": accepted.steps[0]})
    accepted = accepted.model_copy(update={"available_tools": ("read_file",)})
    context = context.model_copy(update={"active_plan": accepted})
    retry_context = context.model_copy(update={
        "active_step": context.active_step.model_copy(update={"attempt": 2}),
        "retry": context.retry.model_copy(update={"retry_count": 1}),
    })
    next_context = context.model_copy(update={"active_step": accepted.steps[1]})
    contexts = (context, retry_context, next_context)
    snapshots = [value.model_dump(mode="json") for value in contexts]
    responses = (AIMessage(content="missing native call"),
                 native_action("read_file", {"path": "Snake/ui/view.py"}),
                 native_action("read_file", {"path": "Snake/ui/view.py", "offset": 100}),
                 native_action("read_file", {"path": "Snake/app/main.py"}))
    status = LiveStatus(enabled=False)

    class ObservedModel(SequenceModel):
        def invoke(self, messages):
            observed.append((status.stage, status.detail))
            return super().invoke(messages)

    observed = []
    tools = get_file_tools(str(Path.cwd()))
    model = ObservedModel(*responses)
    provider = LangChainBrainProvider(brain_llm=model, executable_tools=tools)
    status.start("planner")
    try:
        outcomes = [provider.generate(value, ()) for value in contexts]
        assert observed == [("brain", "plan step 1/3 · invocation 1"),
                            ("brain", "plan step 1/3 · invocation 2"),
                            ("brain", "plan step 1/3 · invocation 3"),
                            ("brain", "plan step 2/3 · invocation 1")]
        assert status.provider_invocations_by_worker == {"brain": 4}
        assert status.usage_by_worker["brain"]["calls"] == 4
    finally:
        status.stop()
    reference = SequenceModel(*responses)
    reference_provider = LangChainBrainProvider(brain_llm=reference, executable_tools=tools)
    assert outcomes == [reference_provider.generate(value, ()) for value in contexts]
    assert model.calls == reference.calls
    assert [value.model_dump(mode="json") for value in contexts] == snapshots


def test_tool_headers_include_existing_stable_step_context_without_arguments(capsys):
    render_node_update(NodeUpdate(
        from_node="brain", to_node="controller",
        brain_result=BrainOutcome(outcome=BrainOutcomeKind.TOOL_REQUESTED, step_id="step2",
            tool_request=ToolRequest(request_id="find-1", tool_name="find_files", arguments={"path": "private-path"})),
    ))
    render_node_update(NodeUpdate(
        from_node="tools", to_node="controller", tool_name="find_files", active_step_id="step2",
        tool_result=ToolResult(request_id="find-1", success=True, message="Found workspace files",
                               data={"private": "hidden payload"}),
    ))
    output = capsys.readouterr().out
    assert "[brain] [step2]" in output
    assert "Calling find_files" in output
    assert "[tool:find_files] [step2]" in output
    assert "private-path" not in output and "hidden payload" not in output
