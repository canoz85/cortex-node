"""Stateless adapters between legacy dict state and protocol model contracts.

Bridge Contract (non-negotiable architectural guarantees):
- Stateless
- Pure and deterministic
- No IO
- No logging
- No persistence
- No runtime decisions
- No controller authority
- No worker invocation
- No mutation of input objects
- Translation only

This module maps legacy runtime dictionaries into immutable protocol contracts
and maps protocol contracts back into legacy-compatible dictionaries. It does
not define protocol semantics or runtime orchestration behavior.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence, Final

from pydantic import BaseModel

from core import protocol

from .converters import (
    _to_int,
    _to_optional_int,
    _to_str,
)

from .enums import (
    BrainOutcome,
    ControllerDecisionType,
    EventType,
    ExecutionPhase,
    ExecutionStatus,
    StepStatus,
    WorkerRole,
)
from .models import (
    BrainInput,
    BrainResult,
    ControllerInput,
    PlannerInput,
    PlannerResult,
    ControllerDecision,
    EventRecord,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    ExecutionSummary,
    ProtocolVisibleState,
    ReplanRequest,
    RetryMetadata,
    ToolInput,
    ToolRequest,
    ToolResult,
    WorkingState,
)


LegacyState = Mapping[str, Any]
LegacyPayload = dict[str, Any]
LegacySequence = Sequence[Any]

_DEFAULT_PROTOCOL_VERSION = "1.0"
_DEFAULT_EXECUTION_ID = "legacy-execution"
_DEFAULT_USER_REQUEST = "unspecified request"

_PLANNER_METADATA_KEYS: tuple[str, ...] = (
    "planner_domain",
    "planner_confidence",
    "planner_domain_enforced",
    "planner_route_source",
)

_DEBUG_METADATA_KEYS: tuple[str, ...] = (
    "token_usage",
    "run_id",
)

_WORKING_STATE_CONSUMED_KEYS: Final[frozenset[str]] = frozenset({
    "messages",
    "last_tool_output",
    "last_tool_result",
    "last_tool_rendered",
    "last_tool_signature",
    "last_tool_success",
    "repeat_fail_count",
    "tool_text_retry_used",
    "steps",
    "token_usage",
    "plan",
    "planner_result",
    "planner_domain",
    "planner_confidence",
    "planner_domain_enforced",
    "planner_route_source",
    "retrieval_messages",
    "run_id",
    "protocol_version",
    "correlation_id",
    "phase",
    "event_index",
    "plan_revision",
    "step_attempt",
    "current_worker",
    "execution_status",
    "execution_summary",
    "completed_step_ids",
    "accepted_event_history",
    "active_step_id",
    "active_step_title",
    "active_step_description",
    "active_step_status",
    "step_id",
    "max_retries",
    "last_error_code",
    "last_error_message",
    "routing_metadata",
    "brain_result",
})


def _state_or_empty(legacy_state: LegacyState | None) -> LegacyState:
    return legacy_state or {}


def _sequence_or_empty(value: Any) -> LegacySequence:
    """Return legacy sequence values as-is, otherwise an empty sequence."""
    return value if isinstance(value, Sequence) else []


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    """Return legacy mapping values as-is, otherwise an empty mapping."""
    return value if isinstance(value, Mapping) else {}


def _enum_or_default(enum_cls: type[Enum], value: Any, default: Enum) -> Enum:
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except Exception:
        return default


def _enum_or_none(enum_cls: type[Enum], value: Any) -> Enum | None:
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except Exception:
        return None


def _json_compatible(value: Any) -> Any:
    """Convert arbitrary values into serialization-friendly primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): _json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(v) for v in value]
    return str(value)


def _json_dict(value: Any) -> dict[str, Any]:
    """Return a JSON-compatible dictionary or an empty dictionary fallback."""
    normalized = _json_compatible(value)
    return normalized if isinstance(normalized, dict) else {}


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(part) for part in content if part is not None]
        return "\n".join(parts)
    if content is not None:
        return str(content)
    return str(message)


def _latest_user_request(messages: Sequence[Any]) -> str:
    for msg in reversed(messages):
        msg_type = _to_str(getattr(msg, "type", "")).lower()
        cls_name = msg.__class__.__name__.lower()
        if msg_type == "human" or "human" in cls_name:
            text = _message_text(msg).strip()
            if text:
                return text
    for msg in reversed(messages):
        text = _message_text(msg).strip()
        if text:
            return text
    return _DEFAULT_USER_REQUEST


def _build_planner_metadata(state: LegacyState) -> dict[str, Any]:
    """Collect planner-oriented helper metadata from legacy Working State fields."""
    metadata: dict[str, Any] = {}
    for key in _PLANNER_METADATA_KEYS:
        if key in state:
            metadata[key] = _json_compatible(state.get(key))
    return metadata


def _build_debug_metadata(state: LegacyState) -> dict[str, Any]:
    """Collect debug-oriented helper metadata from legacy Working State fields."""
    metadata: dict[str, Any] = {}
    for key in _DEBUG_METADATA_KEYS:
        if key in state:
            metadata[key] = _json_compatible(state.get(key))
    return metadata


def _build_capture_state(state: LegacyState) -> dict[str, Any]:
    """Collect tool-capture helper values used by Working State translation."""
    return {
        "last_tool_rendered": _json_compatible(state.get("last_tool_rendered")),
        "last_tool_success": _json_compatible(state.get("last_tool_success")),
        "repeat_fail_count": _json_compatible(state.get("repeat_fail_count")),
        "tool_text_retry_used": _json_compatible(state.get("tool_text_retry_used")),
    }


def _build_execution_summary(
    *,
    state: LegacyState,
    identity: ExecutionIdentity,
    status: ExecutionStatus,
) -> ExecutionSummary | None:
    """Build optional terminal summary contract from legacy summary text."""
    summary_text = _to_str(state.get("execution_summary"), default="").strip()
    if not summary_text:
        return None
    return ExecutionSummary(
        execution_id=identity.execution_id,
        status=status,
        summary_text=summary_text,
    )


def _build_retry_metadata(legacy_state: LegacyState | None) -> RetryMetadata:
    state = _state_or_empty(legacy_state)
    return RetryMetadata(
        step_id=_to_str(state.get("step_id") or state.get("active_step_id"), default="") or None,
        retry_count=_to_int(state.get("repeat_fail_count"), default=0, minimum=0),
        max_retries=_to_int(state.get("max_retries"), default=0, minimum=0),
        last_error_code=_to_str(state.get("last_error_code"), default="") or None,
        last_error_message=_to_str(state.get("last_error_message"), default="") or None,
    )


def _build_active_plan(legacy_state: LegacyState | None, identity: ExecutionIdentity) -> ExecutionPlan | None:
    state = _state_or_empty(legacy_state)
    plan_text = _to_str(state.get("plan"), default="").strip()
    if not plan_text:
        return None

    plan_id = _to_str(state.get("plan_id"), default=f"{identity.execution_id}:plan")
    revision = _to_int(state.get("plan_revision"), default=1, minimum=1)
    return ExecutionPlan(
        plan_id=plan_id,
        revision=revision,
        objective=plan_text,
        steps=tuple(),
    )


def _build_active_step(legacy_state: LegacyState | None) -> ExecutionStep | None:
    state = _state_or_empty(legacy_state)
    step_id = _to_str(state.get("active_step_id") or state.get("step_id"), default="")
    if not step_id:
        return None

    return ExecutionStep(
        step_id=step_id,
        title=_to_str(state.get("active_step_title"), default=step_id),
        description=_to_str(state.get("active_step_description"), default=""),
        status=_enum_or_default(StepStatus, state.get("active_step_status"), StepStatus.ACTIVE),
        attempt=_to_int(state.get("step_attempt"), default=0, minimum=0),
    )


def _parse_event_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        normalized = raw.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None


def _build_event_history(legacy_state: LegacyState | None) -> tuple[EventRecord, ...]:
    state = _state_or_empty(legacy_state)
    raw_history = state.get("accepted_event_history")
    if not isinstance(raw_history, Sequence):
        return tuple()

    records: list[EventRecord] = []
    for idx, raw in enumerate(raw_history):
        if not isinstance(raw, Mapping):
            continue

        event_type = _enum_or_none(EventType, raw.get("event_type"))
        producer = _enum_or_none(WorkerRole, raw.get("producer"))
        timestamp = _parse_event_timestamp(raw.get("timestamp_utc"))
        if event_type is None or producer is None or timestamp is None:
            continue

        records.append(
            EventRecord(
                index=_to_int(raw.get("index"), default=idx, minimum=0),
                event_type=event_type,
                producer=producer,
                timestamp_utc=timestamp,
                payload=_json_compatible(raw.get("payload")),
                correlation_id=_to_str(raw.get("correlation_id"), default="") or None,
                sequence_id=_to_str(raw.get("sequence_id"), default="") or None,
                producer_instance=_to_str(raw.get("producer_instance"), default="") or None,
            )
        )

    return tuple(records)


def legacy_tool_result_to_model(legacy_state: LegacyState | None) -> ToolResult | None:
    state = _state_or_empty(legacy_state)
    existing = state.get("last_tool_result")
    if isinstance(existing, ToolResult):
        return existing

    raw = state.get("last_tool_output")
    if raw in (None, ""):
        return None

    request_id = _to_str(state.get("last_tool_signature"), default="legacy-tool-request")

    if isinstance(raw, Mapping):
        success = bool(raw.get("success") is True)
        return ToolResult(
            request_id=request_id,
            success=success,
            message=_to_str(raw.get("message"), default=""),
            data=_json_compatible(raw.get("data")),
            error_code=_to_str(raw.get("error_code"), default="") or None,
        )

    success = bool(state.get("last_tool_success") is True)
    return ToolResult(
        request_id=request_id,
        success=success,
        message=_to_str(raw, default=""),
        data=None,
        error_code=None,
    )


def _legacy_tool_request_to_model(value: Any) -> ToolRequest | None:
    if not isinstance(value, Mapping):
        return None

    tool_name = _to_str(value.get("tool_name"), default="").strip()
    if not tool_name:
        return None

    return ToolRequest(
        request_id=_to_str(value.get("request_id"), default="legacy-tool-request"),
        tool_name=tool_name,
        arguments=_json_compatible(value.get("arguments")),
        requested_by=_enum_or_default(WorkerRole, value.get("requested_by"), WorkerRole.BRAIN),
    )


def _legacy_replan_request_to_model(value: Any) -> ReplanRequest | None:
    if not isinstance(value, Mapping):
        return None

    reason = _to_str(value.get("reason"), default="").strip()
    if not reason:
        return None

    raw_constraints = value.get("constraints")
    constraints = tuple(str(item) for item in raw_constraints) if isinstance(raw_constraints, Sequence) else tuple()

    return ReplanRequest(
        reason=reason,
        failed_step_id=_to_str(value.get("failed_step_id"), default="") or None,
        constraints=constraints,
        requested_by=_enum_or_default(WorkerRole, value.get("requested_by"), WorkerRole.BRAIN),
    )


def _legacy_planner_result_to_model(legacy_state: LegacyState | None, execution_plan: ExecutionPlan | None) -> PlannerResult | None:
    state = _state_or_empty(legacy_state)
    if execution_plan is None:
        return None

    planner_message = _to_str(state.get("planner_message"), default="")
    planner_rationale = _to_str(state.get("planner_rationale"), default="")
    planner_change_summary = _to_str(state.get("planner_change_summary"), default="")

    if not any((planner_message.strip(), planner_rationale.strip(), planner_change_summary.strip(), execution_plan.objective.strip())):
        return None

    return PlannerResult(
        proposed_plan=execution_plan,
        message=planner_message,
        planning_rationale=planner_rationale,
        change_summary=planner_change_summary,
    )


def _legacy_brain_result_to_model(legacy_state: LegacyState | None) -> BrainResult | None:
    state = _state_or_empty(legacy_state)

    outcome = _enum_or_none(BrainOutcome, state.get("brain_outcome"))
    tool_request = _legacy_tool_request_to_model(state.get("tool_request"))
    replan_request = _legacy_replan_request_to_model(state.get("replan_request"))
    proposed_step_status = _enum_or_none(StepStatus, state.get("proposed_step_status"))
    message = _to_str(state.get("brain_message"), default="")

    if outcome is None:
        if tool_request is not None:
            outcome = BrainOutcome.TOOL_REQUEST
        elif replan_request is not None:
            outcome = BrainOutcome.REPLAN_REQUEST

    if outcome is None:
        return None

    return BrainResult(
        outcome=outcome,
        message=message,
        tool_request=tool_request,
        replan_request=replan_request,
        proposed_step_status=proposed_step_status,
    )


def _collect_unknown_keys(source: LegacyState, consumed: set[str]) -> dict[str, Any]:
    """Preserve unknown legacy fields inside Working State orchestration metadata."""
    return {k: _json_compatible(v) for k, v in source.items() if k not in consumed}


def build_execution_identity(
    legacy_state: LegacyState | None = None,
    *,
    execution_id: str | None = None,
    protocol_version: str = _DEFAULT_PROTOCOL_VERSION,
    correlation_id: str | None = None,
) -> ExecutionIdentity:
    """Build ExecutionIdentity from legacy state with safe, migration-friendly defaults."""
    state = _state_or_empty(legacy_state)
    resolved_execution_id = execution_id or _to_str(state.get("run_id"), default=_DEFAULT_EXECUTION_ID)
    resolved_protocol_version = _to_str(state.get("protocol_version"), default=protocol_version)
    resolved_correlation_id = correlation_id or (_to_str(state.get("correlation_id"), default="") or None)

    return ExecutionIdentity(
        execution_id=resolved_execution_id,
        protocol_version=resolved_protocol_version,
        correlation_id=resolved_correlation_id,
    )


def build_execution_context(
    legacy_state: LegacyState | None = None,
    *,
    user_request: str | None = None,
    role: WorkerRole = WorkerRole.BRAIN,
) -> ExecutionContext:
    """Build ExecutionContext from legacy conversational/runtime inputs.

    This translation is role-scoped and does not alter legacy message payloads.
    """
    state = _state_or_empty(legacy_state)
    messages = _sequence_or_empty(state.get("messages"))
    retrieval = _sequence_or_empty(state.get("retrieval_messages"))

    resolved_user_request = (user_request or _latest_user_request(messages)).strip() or _DEFAULT_USER_REQUEST
    retrieval_messages = tuple(_message_text(item) for item in retrieval)
    recent_history = tuple(_message_text(item) for item in messages[-8:])

    return ExecutionContext(
        user_request=resolved_user_request,
        retrieval_messages=retrieval_messages,
        recent_history=recent_history,
        role=role,
    )


def build_execution_cursor(
    legacy_state: LegacyState | None = None,
    *,
    phase: ExecutionPhase | None = None,
) -> ExecutionCursor:
    """Build ExecutionCursor from legacy iteration and pointer fields."""
    state = _state_or_empty(legacy_state)
    resolved_phase = phase or _enum_or_default(ExecutionPhase, state.get("phase"), ExecutionPhase.INITIALIZING)

    return ExecutionCursor(
        phase=resolved_phase,
        step_id=_to_str(state.get("step_id") or state.get("active_step_id"), default="") or None,
        event_index=_to_int(state.get("event_index"), default=0, minimum=0),
        plan_revision=(
            _to_optional_int(state.get("plan_revision"), minimum=1)
            or (
                state.get("planner_result").proposed_plan.revision
                if isinstance(state.get("planner_result"), PlannerResult)
                and state.get("planner_result").proposed_plan is not None
                else None
            )
        ),
        step_attempt=_to_optional_int(state.get("step_attempt"), minimum=0),
        current_worker=_enum_or_none(WorkerRole, state.get("current_worker")),
        controller_iteration=_to_optional_int(state.get("steps"), minimum=0),
    )


def build_protocol_visible_state(
    legacy_state: LegacyState | None = None,
    *,
    identity: ExecutionIdentity | None = None,
    cursor: ExecutionCursor | None = None,
) -> ProtocolVisibleState:
    """Build ProtocolVisibleState from authoritative-compatible legacy fields only."""
    state = _state_or_empty(legacy_state)
    resolved_identity = identity or build_execution_identity(state)
    resolved_cursor = cursor or build_execution_cursor(state)
    resolved_status = _enum_or_default(ExecutionStatus, state.get("execution_status"), ExecutionStatus.NON_TERMINAL)

    active_plan = build_execution_plan(state, identity=resolved_identity,)
    active_step = build_execution_step(state)
    retry = _build_retry_metadata(state)
    accepted_event_history = _build_event_history(state)

    summary = _build_execution_summary(state=state, identity=resolved_identity, status=resolved_status)

    completed_step_ids_raw = state.get("completed_step_ids")
    completed_step_ids = tuple(str(s) for s in completed_step_ids_raw) if isinstance(completed_step_ids_raw, Sequence) else tuple()

    return ProtocolVisibleState(
        identity=resolved_identity,
        status=resolved_status,
        cursor=resolved_cursor,
        active_plan=active_plan,
        active_step=active_step,
        completed_step_ids=completed_step_ids,
        accepted_event_history=accepted_event_history,
        retry=retry,
        summary=summary,
    )


def build_working_state(legacy_state: LegacyState | None = None) -> WorkingState:
    """Build WorkingState from runtime helper metadata and non-authoritative values."""
    state = _state_or_empty(legacy_state)
    last_tool_result = legacy_tool_result_to_model(state)

    retrieval = _sequence_or_empty(state.get("retrieval_messages"))
    retrieval_context = tuple(_message_text(item) for item in retrieval)

    routing_metadata = _json_dict(_mapping_or_empty(state.get("routing_metadata")))
    planner_metadata = _build_planner_metadata(state)
    debug_metadata = _build_debug_metadata(state)
    capture_state = _build_capture_state(state)

    return WorkingState(
        retrieval_context=retrieval_context,
        last_tool_result=last_tool_result,
        routing_metadata=routing_metadata,
        planner_metadata=planner_metadata,
        debug_metadata=debug_metadata,
        tool_signature=_to_str(state.get("last_tool_signature"), default="") or None,
        capture_state=capture_state,
        orchestration_metadata=_collect_unknown_keys(state, _WORKING_STATE_CONSUMED_KEYS),
    )


def build_execution_state(
    legacy_state: LegacyState | None = None,
    *,
    identity: ExecutionIdentity | None = None,
    cursor: ExecutionCursor | None = None,
) -> ExecutionState:
    """Build ExecutionState by combining protocol-visible and working state translations."""
    state = _state_or_empty(legacy_state)

    existing = state.get("execution_state")

    if isinstance(existing, ExecutionState):
        protocol_visible = existing.protocol_visible

        if identity is not None:
            protocol_visible = protocol_visible.model_copy(
                update={
                    "identity": identity
                }
            )

        if cursor is not None:
            protocol_visible = protocol_visible.model_copy(
                update={
                    "cursor": cursor
                }
            )

        return ExecutionState(
            protocol_visible=protocol_visible,
            working=build_working_state(state),
        )

    resolved_identity = identity or build_execution_identity(state)
    resolved_cursor = cursor or build_execution_cursor(state)

    return ExecutionState(
        protocol_visible=build_protocol_visible_state(state, identity=resolved_identity, cursor=resolved_cursor),
        working=build_working_state(state),
    )

def build_execution_step(
    legacy_state: LegacyState | None = None,
) -> ExecutionStep | None:
    """Build ExecutionStep contract from legacy runtime state."""
    return _build_active_step(legacy_state)

def build_execution_plan(
    legacy_state: LegacyState | None = None,
    *,
    identity: ExecutionIdentity | None = None,
) -> ExecutionPlan | None:
    """Build ExecutionPlan contract from legacy planning state."""

    state = _state_or_empty(legacy_state)

    planner_result = state.get("planner_result")
    
    if isinstance(planner_result, PlannerResult):
        if planner_result.proposed_plan is not None:
            return planner_result.proposed_plan

    resolved_identity = (
        identity
        if identity is not None
        else build_execution_identity(state)
    )

    return _build_active_plan(state, resolved_identity)


def build_planner_input(legacy_state: LegacyState | None = None) -> PlannerInput:
    """Build PlannerInput contract from legacy runtime state."""

    execution_state = build_execution_state(legacy_state)

    return PlannerInput(
        identity=execution_state.protocol_visible.identity,
        context=build_execution_context(
            legacy_state,
            role=WorkerRole.PLANNER,
        ),
        active_plan=execution_state.protocol_visible.active_plan,
        completed_step_ids=execution_state.protocol_visible.completed_step_ids,
        retry=execution_state.protocol_visible.retry,
    )

def planner_result_to_legacy(result: PlannerResult) -> dict[str, Any]:
    """Translate PlannerResult into a legacy-friendly dictionary payload."""

    payload: LegacyPayload = {
        "planner_message": result.message,
    }

    if result.proposed_plan is not None:
        payload["plan"] = result.proposed_plan.objective
        payload["plan_id"] = result.proposed_plan.plan_id
        payload["plan_revision"] = result.proposed_plan.revision


    if result.planning_rationale:
        payload["planner_rationale"] = result.planning_rationale

    if result.change_summary:
        payload["planner_change_summary"] = result.change_summary
    
    return payload

def build_brain_input(legacy_state: LegacyState | None = None) -> BrainInput:
    """Build BrainInput contract from legacy runtime state."""

    execution_state = build_execution_state(legacy_state)

    return BrainInput(
        identity=execution_state.protocol_visible.identity,
        cursor=execution_state.protocol_visible.cursor,
        context=build_execution_context(legacy_state),
        active_plan=execution_state.protocol_visible.active_plan,
        active_step=execution_state.protocol_visible.active_step,
        last_tool_result=execution_state.working.last_tool_result,
        retry=execution_state.protocol_visible.retry,
    )


def build_controller_input(
    legacy_state: LegacyState | None = None,
) -> ControllerInput:
    """Build ControllerInput contract from legacy runtime state."""

    execution_state = build_execution_state(legacy_state)
    state = _state_or_empty(legacy_state)

    protocol = execution_state.protocol_visible

    planner_result: PlannerResult | None = None
    brain_result: BrainResult | None = None
    tool_result: ToolResult | None = None

    #
    # Consume newest event first.
    #
    if state.get("brain_result") is not None:
        brain_result = state["brain_result"]
    
    elif state.get("last_tool_result") is not None:
        tool_result = state["last_tool_result"]

    elif state.get("planner_result") is not None:
        planner_result = state["planner_result"]

    return ControllerInput(
        identity=protocol.identity,
        cursor=protocol.cursor,
        context=build_execution_context(state),
        active_plan=protocol.active_plan,
        active_step=protocol.active_step,
        planner_result=planner_result,
        brain_result=brain_result,
        tool_result=tool_result,
    )

# def build_controller_input(legacy_state: LegacyState | None = None) -> ControllerInput:
#     """Build ControllerInput contract from legacy runtime state."""

#     execution_state = build_execution_state(legacy_state)
#     state = _state_or_empty(legacy_state)


#     active_plan = execution_state.protocol_visible.active_plan
#     cursor = execution_state.protocol_visible.cursor

    
#     planner_result: PlannerResult | None = None
#     brain_result: BrainResult | None = None
#     tool_result: ToolResult | None = None

#     # if state.get("last_tool_result") is not None:
#     #     tool_result = state["last_tool_result"]

#     # elif state.get("brain_result") is not None:
#     #     brain_result = state["brain_result"]

#     # elif state.get("planner_result") is not None:
#     #     planner_result = state["planner_result"]

#     match cursor.current_worker:

#         case WorkerRole.PLANNER:
#             planner_result = state.get("planner_result")

#         case WorkerRole.BRAIN:
#             brain_result = state.get("brain_result")

#         case WorkerRole.TOOL_RUNTIME:
#             tool_result = state.get("last_tool_result")

#         case _:
#             pass
    
    return ControllerInput(
        identity=execution_state.protocol_visible.identity,
        cursor=cursor,
        context=build_execution_context(
            legacy_state,
            role=WorkerRole.CONTROLLER,
        ),
        active_plan=active_plan,
        active_step=execution_state.protocol_visible.active_step,
        planner_result=planner_result,
        brain_result=brain_result,
        tool_result=tool_result,
        retry=execution_state.protocol_visible.retry,
    )


def brain_result_to_legacy(result: BrainResult) -> dict[str, Any]:
    """Translate BrainResult into a legacy-friendly dictionary payload."""
    payload: LegacyPayload = {
        "brain_outcome": result.outcome.value,
        "brain_message": result.message,

        "tool_request": None,
        "replan_request": None,
        "proposed_step_status": None,
    }

    if result.proposed_step_status is not None:
        payload["proposed_step_status"] = result.proposed_step_status.value

    if result.tool_request is not None:
        payload["tool_request"] = {
            "request_id": result.tool_request.request_id,
            "tool_name": result.tool_request.tool_name,
            "arguments": _json_compatible(result.tool_request.arguments),
            "requested_by": result.tool_request.requested_by.value,
        }

    if result.replan_request is not None:
        payload["replan_request"] = {
            "reason": result.replan_request.reason,
            "failed_step_id": result.replan_request.failed_step_id,
            "constraints": list(result.replan_request.constraints),
            "requested_by": result.replan_request.requested_by.value,
        }

    return payload

def build_tool_input(legacy_state: LegacyState | None = None,) -> ToolInput:
    """Build ToolInput contract from legacy runtime state."""

    execution_state = build_execution_state(legacy_state)

    tool_result = execution_state.working.last_tool_result

    request_id = (
        tool_result.request_id
        if tool_result is not None
        else _to_str(
            _state_or_empty(legacy_state).get(
                "last_tool_signature"
            ),
            default="legacy-tool-request",
        )
    )

    tool_request = ToolRequest(
        request_id=request_id,
        tool_name=_to_str(
            _state_or_empty(legacy_state).get("tool_name"),
            default="",
        ),
        arguments=_mapping_or_empty(
            _state_or_empty(legacy_state).get("tool_args")
        ),
    )

    return ToolInput(
        identity=execution_state.protocol_visible.identity,
        cursor=execution_state.protocol_visible.cursor,
        context=build_execution_context(
            legacy_state,
            role=WorkerRole.TOOL,
        ),
        tool_request=tool_request,
        active_plan=execution_state.protocol_visible.active_plan,
        active_step=execution_state.protocol_visible.active_step,
        retry=execution_state.protocol_visible.retry,
    )

def tool_result_to_legacy(result: ToolResult) -> dict[str, Any]:
    """Translate ToolResult into the existing legacy structured tool payload shape."""
    payload: LegacyPayload = {
        "success": result.success,
        "message": result.message,
        "data": _json_compatible(result.data),
    }
    if result.error_code:
        payload["error_code"] = result.error_code
    payload["request_id"] = result.request_id
    return payload


def controller_decision_to_legacy(decision: ControllerDecision) -> dict[str, Any]:
    """Translate ControllerDecision into a legacy-friendly dictionary payload."""
    payload: LegacyPayload = {
        "controller_decision": decision.decision_type.value,
        "decision_reason": decision.reason,
        "terminal": decision.terminal,
        "requires_checkpoint": decision.requires_checkpoint,
        "requires_replan": decision.requires_replan,
    }

    if decision.next_worker is not None:
        payload["next_worker"] = decision.next_worker.value
    if decision.next_step_id is not None:
        payload["next_step_id"] = decision.next_step_id
    if decision.cursor is not None:
        payload["cursor"] = decision.cursor.model_dump(mode="json")

    return payload


def legacy_state_to_execution_state(legacy_state: LegacyState | None = None) -> ExecutionState:
    """Translate legacy runtime dictionary state into immutable ExecutionState."""
    return build_execution_state(legacy_state)


def execution_state_to_legacy(state: ExecutionState) -> dict[str, Any]:
    """Translate immutable ExecutionState into legacy runtime dictionary shape.

    The returned dictionary is intentionally minimal and bridge-oriented.
    Unknown orchestrator values remain in WorkingState.orchestration_metadata.
    """
    legacy: LegacyPayload = {}

    pv = state.protocol_visible
    wk = state.working

    legacy["run_id"] = pv.identity.execution_id
    legacy["protocol_version"] = pv.identity.protocol_version
    if pv.identity.correlation_id:
        legacy["correlation_id"] = pv.identity.correlation_id

    legacy["execution_status"] = pv.status.value
    legacy["phase"] = pv.cursor.phase.value
    legacy["step_id"] = pv.cursor.step_id
    legacy["event_index"] = pv.cursor.event_index
    legacy["plan_revision"] = pv.cursor.plan_revision
    legacy["step_attempt"] = pv.cursor.step_attempt
    legacy["current_worker"] = pv.cursor.current_worker.value if pv.cursor.current_worker else None
    legacy["steps"] = pv.cursor.controller_iteration or 0

    if pv.active_plan is not None:
        legacy["plan"] = pv.active_plan.objective
        legacy["plan_id"] = pv.active_plan.plan_id
        legacy["plan_revision"] = pv.active_plan.revision

    if pv.active_step is not None:
        legacy["active_step_id"] = pv.active_step.step_id
        legacy["active_step_title"] = pv.active_step.title
        legacy["active_step_description"] = pv.active_step.description
        legacy["active_step_status"] = pv.active_step.status.value

    legacy["completed_step_ids"] = list(pv.completed_step_ids)
    legacy["accepted_event_history"] = [event.model_dump(mode="json") for event in pv.accepted_event_history]

    legacy["repeat_fail_count"] = pv.retry.retry_count
    legacy["max_retries"] = pv.retry.max_retries
    legacy["last_error_code"] = pv.retry.last_error_code
    legacy["last_error_message"] = pv.retry.last_error_message

    if pv.summary is not None:
        legacy["execution_summary"] = pv.summary.summary_text

    legacy["retrieval_messages"] = list(wk.retrieval_context)
    legacy["routing_metadata"] = dict(wk.routing_metadata)
    legacy["planner_metadata"] = dict(wk.planner_metadata)
    legacy["debug_metadata"] = dict(wk.debug_metadata)
    legacy["last_tool_signature"] = wk.tool_signature or ""

    if wk.last_tool_result is not None:
        legacy["last_tool_output"] = tool_result_to_legacy(wk.last_tool_result)
        legacy["last_tool_success"] = wk.last_tool_result.success
    else:
        legacy["last_tool_output"] = ""
        legacy["last_tool_success"] = None

    for key, value in wk.capture_state.items():
        legacy[key] = _json_compatible(value)

    for key, value in wk.orchestration_metadata.items():
        if key not in legacy:
            legacy[key] = _json_compatible(value)

    return legacy

def with_cursor(
    execution_state: ExecutionState,
    **updates,
) -> ExecutionState:
    """Return a copy of ExecutionState with an updated execution cursor."""

    return execution_state.model_copy(
        update={
            "protocol_visible": execution_state.protocol_visible.model_copy(
                update={
                    "cursor": execution_state.protocol_visible.cursor.model_copy(
                        update=updates,
                    )
                }
            )
        }
    )


__all__ = [
    "build_execution_identity",
    "build_execution_context",
    "build_execution_cursor",
    "build_protocol_visible_state",
    "build_working_state",
    "build_execution_state",
    "build_brain_input",
    "build_planner_input",
    "build_controller_input",
    "build_tool_input",
    "brain_result_to_legacy",
    "tool_result_to_legacy",
    "legacy_tool_result_to_model",
    "controller_decision_to_legacy",
    "legacy_state_to_execution_state",
    "execution_state_to_legacy",
]
