import io

from langchain_core.messages import AIMessage

from core.logging.live_status import LiveStatus, format_compact_tokens, format_token_count
from core.logging.node_update import NodeUpdate
from core.logging.renderer import (
    _update_live_status, render_node_update,
    render_usage_summary,
)
from core.protocol.enums import BrainOutcomeKind, PlannerOutcome
from core.protocol.models import (
    BrainOutcome, ExecutionPlan, ExecutionStep, PlannerResult, ToolRequest,
    ToolResult,
)


def status_stream():
    stream = io.StringIO()
    return LiveStatus(stream=stream, refresh_interval=60, enabled=True), stream


def test_compact_token_formatting():
    assert format_token_count(842) == "842 tok"
    assert format_token_count(1000) == "1.0k tok"
    assert format_token_count(6949) == "6.9k tok"
    assert format_compact_tokens(842) == "842"
    assert format_compact_tokens(2300) == "2.3k"


def test_usage_accumulates_provider_counts():
    status, _ = status_stream()
    status.add_usage({"prompt_tokens": 800, "completion_tokens": 42}, worker="planner")
    status.add_response_usage(AIMessage(
        content="", usage_metadata={"input_tokens": 1000, "output_tokens": 7, "total_tokens": 1007}
    ), worker="brain")
    assert status.total_tokens == 1849


def test_worker_usage_aggregation_retries_summary_and_deduplication():
    status, _ = status_stream()
    responses = [
        ("planner", AIMessage(content="", usage_metadata={"input_tokens": 800, "output_tokens": 42, "total_tokens": 842})),
        ("planner", AIMessage(content="", usage_metadata={"input_tokens": 1400, "output_tokens": 58, "total_tokens": 1458})),
        ("brain", AIMessage(content="", usage_metadata={"input_tokens": 3000, "output_tokens": 500, "total_tokens": 3500})),
        ("brain", AIMessage(content="", usage_metadata={"input_tokens": 3200, "output_tokens": 300, "total_tokens": 3500})),
        ("finalizer", AIMessage(content="", usage_metadata={"input_tokens": 1500, "output_tokens": 400, "total_tokens": 1900})),
        ("memory", AIMessage(content="", usage_metadata={"input_tokens": 900, "output_tokens": 100, "total_tokens": 1000})),
    ]
    for worker, response in responses:
        status.add_response_usage(response, worker=worker)
    status.add_response_usage(responses[0][1], worker="planner")

    assert status.usage_by_worker["planner"] == {
        "input_tokens": 2200, "output_tokens": 100,
        "total_tokens": 2300, "calls": 2,
    }
    assert status.usage_by_worker["brain"]["calls"] == 2
    assert status.usage_by_worker["finalizer"]["calls"] == 1
    assert status.usage_by_worker["memory"]["calls"] == 1
    assert status.total_tokens == sum(
        usage["total_tokens"] for usage in status.usage_by_worker.values()
    ) == 12200
    assert status.format_usage_summary() == (
        "planner 2.3k (2) | brain 7.0k (2) | finalizer 1.9k (1) | "
        "memory 1.0k (1) | total 12.2k tok"
    )


def test_zero_call_workers_are_omitted_and_new_turn_is_reset():
    first, _ = status_stream()
    first.add_usage({"total_tokens": 1100}, worker="planner")
    assert "brain" not in first.format_usage_summary()
    second, _ = status_stream()
    assert second.total_tokens == 0
    assert second.usage_by_worker == {}
    assert second.format_usage_summary() == ""


def test_stage_and_detail_are_rendered():
    status, stream = status_stream()
    status.start("planner")
    status.update("brain", "step 1")
    status.stop()
    assert "brain · step 1" in stream.getvalue()


def test_brain_provider_invocation_count_is_turn_local_and_independent_of_usage():
    status = LiveStatus(enabled=False)
    status.start("planner")
    try:
        assert status.begin_provider_invocation(worker="brain") == 1
        assert (status.stage, status.detail) == ("brain", "step 1")
        assert status.begin_provider_invocation(worker="brain") == 2
        assert (status.stage, status.detail) == ("brain", "step 2")
        assert status.usage_by_worker == {}
    finally:
        status.stop()


def test_normal_live_progress_maps_semantic_workers_and_replaces_stale_detail():
    status = LiveStatus(enabled=False)
    status.start("planner", "routing")
    try:
        _update_live_status(NodeUpdate(
            from_node="planner", to_node="controller", planner_result=object(),
        ), verbose=False)
        assert (status.stage, status.detail) == ("planner", "planning")

        _update_live_status(NodeUpdate(
            from_node="controller", to_node="brain",
            brain_result=BrainOutcome(
                outcome=BrainOutcomeKind.TOOL_REQUESTED, step_id="1",
                tool_request=ToolRequest(
                    request_id="r1", tool_name="list_files", arguments={"path": "."},
                ),
            ),
        ), verbose=False)
        assert (status.stage, status.detail) == ("list_files", "")

        _update_live_status(NodeUpdate(
            from_node="tools", to_node="capture_tool_output",
            tool_result=object(), tool_name="list_files", active_step_id="step-1",
        ), verbose=False)
        assert (status.stage, status.detail) == ("list_files", "")

        _update_live_status(NodeUpdate(
            from_node="controller", to_node="finalizer",
            finalization_result=object(),
        ), verbose=False)
        assert (status.stage, status.detail) == ("finalizer", "")
    finally:
        status.stop()


def test_brain_tool_call_is_printed_before_tool_stage_redraw(capsys):
    status = LiveStatus(refresh_interval=60, enabled=True)
    status.start("brain", "step 1")
    capsys.readouterr()
    try:
        render_node_update(NodeUpdate(
            from_node="brain", to_node="controller",
            brain_result=BrainOutcome(
                outcome=BrainOutcomeKind.TOOL_REQUESTED, step_id="step-1",
                tool_request=ToolRequest(
                    request_id="r1", tool_name="list_files", arguments={"path": "."},
                ),
            ),
        ))
        added = capsys.readouterr().out
        assert added.index("[brain]") < added.rindex("list_files")
        assert (status.stage, status.detail) == ("list_files", "")
    finally:
        status.stop()


def test_planner_output_does_not_guess_brain_invocation(capsys):
    status = LiveStatus(refresh_interval=60, enabled=True)
    status.start("planner", "planning")
    capsys.readouterr()
    try:
        render_node_update(NodeUpdate(
            from_node="planner", to_node="controller",
            accepted_plan=ExecutionPlan(
                plan_id="p1",
                steps=(ExecutionStep(step_id="step-1", title="List files"),),
            ),
        ))
        capsys.readouterr()
        assert (status.stage, status.detail) == ("planner", "planning")
    finally:
        status.stop()


def test_planner_clarification_question_is_rendered(capsys):
    render_node_update(NodeUpdate(
        from_node="planner",
        to_node="controller",
        planner_result=PlannerResult(
            outcome=PlannerOutcome.CLARIFICATION_REQUIRED,
            request_id="request-1",
            message="Which MQTT password should I use?",
        ),
    ))

    output = capsys.readouterr().out
    assert "[planner:clarification_required]" in output
    assert "Which MQTT password should I use?" in output


def test_tool_output_does_not_guess_brain_invocation(capsys):
    status = LiveStatus(refresh_interval=60, enabled=True)
    status.start("list_files")
    capsys.readouterr()
    try:
        render_node_update(NodeUpdate(
            from_node="tools", to_node="controller",
            tool_result=ToolResult(
                request_id="r1", success=True, message="Listing for .",
            ),
            tool_name="list_files", active_step_id="step-1",
        ))
        output = capsys.readouterr().out
        assert "[tool:list_files]" in output
        assert (status.stage, status.detail) == ("list_files", "")
    finally:
        status.stop()


def test_finalizer_output_precedes_finalizer_redraw(capsys):
    status = LiveStatus(refresh_interval=60, enabled=True)
    status.start("finalizer")
    capsys.readouterr()
    try:
        render_node_update(NodeUpdate(
            from_node="controller", to_node="controller",
            finalization_result=object(), ai_message=AIMessage(content="Done."),
        ))
        output = capsys.readouterr().out
        assert output.index("[finalizer]") < output.rindex("finalizer")
        assert (status.stage, status.detail) == ("finalizer", "")
    finally:
        status.stop()


def test_controller_live_details_are_verbose_only():
    status = LiveStatus(enabled=False)
    status.start("brain", "step 1")
    update = NodeUpdate(
        from_node="brain", to_node="controller",
        controller_events=("Tool requested: list_files",),
    )
    try:
        _update_live_status(update, verbose=False)
        assert (status.stage, status.detail) == ("brain", "step 1")
        _update_live_status(update, verbose=True)
        assert (status.stage, status.detail) == ("controller", "dispatching tool")
    finally:
        status.stop()


def test_raw_graph_names_never_become_normal_live_status():
    status, stream = status_stream()
    status.start("brain", "step 1")
    try:
        _update_live_status(NodeUpdate(
            from_node="portable_orchestration", to_node="capture_tool_output",
        ), verbose=False)
        assert (status.stage, status.detail) == ("brain", "step 1")
        assert "portable_orchestration" not in stream.getvalue()
        assert "capture_tool_output" not in stream.getvalue()
    finally:
        status.stop()


def test_permanent_output_clears_then_redraws():
    status, stream = status_stream()
    status.start("planner")
    before = stream.getvalue()
    with status.permanent_output():
        stream.write("[planner]\nPlan\n")
    added = stream.getvalue()[len(before):]
    assert added.startswith("\r\033[2K[planner]\nPlan\n")
    assert "planner" in added.rsplit("\r\033[2K", 1)[-1]
    status.stop()


def test_stop_clears_status_and_stops_redraw():
    status, stream = status_stream()
    status.start("finalizer")
    status.stop()
    stopped = stream.getvalue()
    assert stopped.endswith("\r\033[2K")
    status.redraw()
    assert stream.getvalue() == stopped


def test_renderer_semantics_are_unchanged_without_live_status(capsys):
    request = ToolRequest(request_id="r1", tool_name="read_file", arguments={"path": "README.md"})
    render_node_update(NodeUpdate(
        from_node="controller", to_node="brain",
        brain_result=BrainOutcome(outcome=BrainOutcomeKind.TOOL_REQUESTED, tool_request=request),
    ))
    output = capsys.readouterr().out
    assert "[brain]" in output
    assert "Calling read_file" in output
    assert "\r\033[2K" not in output


def test_retryable_planner_failure_updates_status_without_permanent_block():
    status, stream = status_stream()
    status.start("planner")
    render_node_update(NodeUpdate(
        from_node="planner", to_node="controller",
        planner_result=PlannerResult(
            outcome=PlannerOutcome.FAILED, request_id="attempt-1",
            failure_category="PROVIDER_FAILURE",
            message="Planner router failed (OutputParserException)",
        ),
        planner_failure_retryable=True,
        planner_retry_detail="retrying router",
    ))
    status.stop()
    output = stream.getvalue()
    assert "retrying router" in output
    assert "[planner:failed]" not in output


def test_exhausted_planner_failure_still_renders_permanently(capsys):
    render_node_update(NodeUpdate(
        from_node="planner", to_node="controller",
        planner_result=PlannerResult(
            outcome=PlannerOutcome.FAILED, request_id="attempt-2",
            failure_category="PROVIDER_FAILURE", message="Planner router failed",
        ),
    ))
    assert "[planner:failed]" in capsys.readouterr().out


def test_terminal_failure_can_print_accumulated_usage(capsys):
    status = LiveStatus(enabled=False)
    status.start("planner")
    status.add_usage({"total_tokens": 1100}, worker="planner")
    status.stop()
    render_usage_summary(status)
    assert capsys.readouterr().out == "\n[usage] planner 1.1k (1) | total 1.1k tok\n"


def test_memory_status_is_temporary_and_cleared_before_session_saved():
    status, stream = status_stream()
    status.start("finalizer")
    status.update("memory", "saving session")
    status.stop()
    with status.permanent_output():
        stream.write("[system]\nSession saved.\n")
    output = stream.getvalue()
    assert "memory · saving session" in output
    assert "[memory]" not in output
    assert output.rfind("\r\033[2K") < output.index("[system]")
