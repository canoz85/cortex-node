"""Deterministic bounded Planner progress projection regressions."""

from core.planner_progress import (
    MAX_ACTION_GROUPS, MAX_ARGUMENT_CHARS, MAX_MESSAGE_CHARS,
    MAX_RESULT_SUMMARY_CHARS, MAX_SOURCE_REQUEST_IDS, build_planner_progress,
)
from core.protocol.models import ToolExecutionRecord, ToolResult


def record(index, *, signature=None, success=False, execution_id="e", revision=1,
           tool="read_file", arguments=None, message="result", data=None):
    return ToolExecutionRecord(
        execution_id=execution_id, plan_id="p", plan_revision=revision,
        step_id=f"s{revision}", tool_name=tool,
        arguments=arguments or {"index": index},
        result=ToolResult(
            request_id=f"r{index}", signature=signature or f"sig-{index}",
            success=success, error_code=None if success else "FAILED",
            message=message, data=data,
        ),
    )


def test_exact_signatures_group_but_different_signatures_do_not():
    projection = build_planner_progress("e", (
        record(1, signature="same", revision=1),
        record(2, signature="different", success=True, revision=2),
        record(3, signature="same", success=True, revision=3),
    ))
    assert len(projection.action_groups) == 2
    different, same = projection.action_groups
    assert different.signature == "different" and different.occurrence_count == 1
    assert different.latest_outcome == "operational_success"
    assert same.signature == "same" and same.occurrence_count == 2
    assert (same.failure_count, same.success_count) == (1, 1)
    assert (same.first_plan_revision, same.last_plan_revision) == (1, 3)
    assert same.source_request_ids == ("r1", "r3")
    assert same.semantic_conclusion == "unknown"


def test_successful_not_found_output_is_observation_not_fact():
    action = build_planner_progress("e", (
        record(1, success=True, tool="run_python", data={"stdout": "target not found"}),
    )).action_groups[0]
    assert action.latest_outcome == "operational_success"
    assert action.latest_result_summary == '{"stdout":"target not found"}'
    assert action.semantic_conclusion == "unknown"


def test_foreign_executions_excluded_and_legacy_unscoped_retained():
    projection = build_planner_progress("e", (
        record(1, execution_id="foreign"),
        record(2, execution_id=None),
        record(3, execution_id="e"),
    ))
    assert tuple(a.source_request_ids for a in projection.action_groups) == (("r2",), ("r3",))


def test_projection_bounds_newest_groups_and_each_payload_deterministically():
    records = tuple(record(
        index, arguments={"value": "a" * (MAX_ARGUMENT_CHARS * 2)},
        message="m" * (MAX_MESSAGE_CHARS * 2), data="d" * (MAX_RESULT_SUMMARY_CHARS * 2),
    ) for index in range(MAX_ACTION_GROUPS + 3))
    projection = build_planner_progress("e", records)
    assert len(projection.action_groups) == MAX_ACTION_GROUPS
    assert projection.total_action_group_count == MAX_ACTION_GROUPS + 3
    assert projection.omitted_action_group_count == 3
    assert projection.action_groups[0].source_request_ids == ("r3",)
    action = projection.action_groups[-1]
    assert len(action.arguments_json) == MAX_ARGUMENT_CHARS
    assert action.arguments_json.startswith('{"value":"aaa') and "...[truncated " in action.arguments_json
    assert action.latest_message.startswith("m" * 20) and "...[truncated " in action.latest_message
    assert len(action.latest_message) == MAX_MESSAGE_CHARS
    assert action.latest_result_summary.startswith('"' + "d" * 20)
    assert "...[truncated " in action.latest_result_summary
    assert len(action.latest_result_summary) == MAX_RESULT_SUMMARY_CHARS
    assert projection == build_planner_progress("e", records)


def test_source_ids_are_bounded_while_occurrence_count_and_revisions_remain_complete():
    records = tuple(record(index, signature="same", revision=index + 1)
                    for index in range(MAX_SOURCE_REQUEST_IDS + 3))
    action = build_planner_progress("e", records).action_groups[0]
    assert action.occurrence_count == MAX_SOURCE_REQUEST_IDS + 3
    assert action.source_request_ids == tuple(
        f"r{i}" for i in range(3, MAX_SOURCE_REQUEST_IDS + 3)
    )
    assert action.omitted_source_request_count == 3
    assert (action.first_plan_revision, action.last_plan_revision) == (1, MAX_SOURCE_REQUEST_IDS + 3)


def test_missing_signatures_are_request_specific_and_round_trip_is_stable():
    records = tuple(record(i).model_copy(update={
        "result": record(i).result.model_copy(update={"signature": ""}),
    }) for i in range(2))
    projection = build_planner_progress("e", records)
    assert len(projection.action_groups) == 2
    assert projection.model_validate_json(projection.model_dump_json()) == projection
