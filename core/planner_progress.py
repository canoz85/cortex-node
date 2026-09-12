"""Deterministic bounded execution progress for Planner revision decisions."""

from __future__ import annotations

import json

from core.protocol.models import (
    PlannerProgressProjection, ProjectedPlannerAction, ToolExecutionRecord,
)

MAX_ACTION_GROUPS = 24
MAX_SOURCE_REQUEST_IDS = 8
MAX_ARGUMENT_CHARS = 1000
MAX_MESSAGE_CHARS = 1000
MAX_RESULT_SUMMARY_CHARS = 1000


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text)
    while True:
        suffix = f"...[truncated {omitted} chars]"
        prefix_length = max(0, limit - len(suffix))
        exact_omitted = len(text) - prefix_length
        if exact_omitted == omitted:
            return f"{text[:prefix_length]}{suffix}"
        omitted = exact_omitted


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _result_summary(record: ToolExecutionRecord) -> str:
    result = record.result
    if result.data is not None:
        return _json(result.data)
    if result.rendered_output:
        return result.rendered_output
    return result.message


def build_planner_progress(
    execution_id: str,
    records: tuple[ToolExecutionRecord, ...],
) -> PlannerProgressProjection:
    """Project current and legacy-unscoped records using exact signatures only.

    Groups are selected by newest last occurrence. The retained groups are then
    emitted in chronological order of their last occurrence. Missing signatures
    are request-specific and never grouped. Source IDs retain the newest eight.
    """
    groups: dict[tuple[str, str], list[tuple[int, ToolExecutionRecord]]] = {}
    for index, record in enumerate(records):
        if record.execution_id not in (None, execution_id):
            continue
        key = (("signature", record.result.signature) if record.result.signature
               else ("request", record.result.request_id))
        groups.setdefault(key, []).append((index, record))

    selected = sorted(groups.values(), key=lambda items: items[-1][0], reverse=True)[:MAX_ACTION_GROUPS]
    selected.sort(key=lambda items: items[-1][0])
    actions = []
    for occurrences in selected:
        occurrence_records = [record for _, record in occurrences]
        latest = occurrence_records[-1]
        revisions = [record.plan_revision for record in occurrence_records
                     if record.plan_revision is not None]
        source_ids = tuple(record.result.request_id for record in occurrence_records)
        retained_source_ids = source_ids[-MAX_SOURCE_REQUEST_IDS:]
        actions.append(ProjectedPlannerAction(
            signature=latest.result.signature or None,
            tool_name=latest.tool_name,
            step_ids=tuple(dict.fromkeys(record.step_id for record in occurrence_records)),
            source_request_ids=retained_source_ids,
            omitted_source_request_count=len(source_ids) - len(retained_source_ids),
            occurrence_count=len(occurrence_records),
            success_count=sum(record.result.success for record in occurrence_records),
            failure_count=sum(not record.result.success for record in occurrence_records),
            latest_outcome=("operational_success" if latest.result.success
                            else "tool_or_transport_failure"),
            first_plan_revision=min(revisions) if revisions else None,
            last_plan_revision=max(revisions) if revisions else None,
            arguments_json=_bounded(_json(latest.arguments), MAX_ARGUMENT_CHARS),
            latest_error_code=latest.result.error_code,
            latest_message=_bounded(latest.result.message, MAX_MESSAGE_CHARS),
            latest_result_summary=_bounded(_result_summary(latest), MAX_RESULT_SUMMARY_CHARS),
        ))
    return PlannerProgressProjection(
        action_groups=tuple(actions),
        total_action_group_count=len(groups),
        omitted_action_group_count=max(0, len(groups) - len(actions)),
    )
