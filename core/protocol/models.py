"""Immutable protocol-oriented data contracts for gradual runtime migration.

These models provide a typed contract layer aligned with CEP/CIS semantics while
remaining independent from current runtime orchestration modules.
"""

from __future__ import annotations

from typing import Any

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    BrainOutcome,
    ControllerDecisionType,
    EventType,
    ExecutionPhase,
    ExecutionStatus,
    PlannerOutcome,
    StepStatus,
    WorkerRole,
)


JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None

StepIdList = tuple[str, ...]
MessageList = tuple[str, ...]
ConstraintList = tuple[str, ...]
RetrievalContext = tuple[str, ...]

class ImmutableProtocolModel(BaseModel):
    """Base class for frozen, serialization-friendly protocol model contracts.

    Ownership: model-level immutability support for protocol/runtime contract types.
    Visibility: shared utility (applies to both Protocol-visible and Working State).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class ExecutionIdentity(ImmutableProtocolModel):
    """Stable protocol identity for one execution instance.

    Protocol purpose: identify execution and protocol version.
    Runtime purpose: correlate state, events, and checkpoint artifacts.
    Ownership: controller-governed identity continuity.
    Visibility: Protocol-visible State.
    """

    execution_id: str = Field(min_length=1)
    protocol_version: str = Field(min_length=1)
    correlation_id: str | None = None


class RetryMetadata(ImmutableProtocolModel):
    """Retry continuity metadata for legal continuation and resume checks.

    Protocol purpose: preserve retry counters/attempt continuity.
    Runtime purpose: support recovery and replay validation.
    Ownership: controller-governed retry progression.
    Visibility: Protocol-visible State (with Working State mirrors allowed).
    """

    step_id: str | None = None
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=0, ge=0)
    last_error_code: str | None = None
    last_error_message: str | None = None


class ExecutionStep(ImmutableProtocolModel):
    """Typed representation of a single step in an execution plan.

    Protocol purpose: represent step identity and status progression.
    Runtime purpose: provide structured step context to worker roles.
    Ownership: accepted step status is controller-governed.
    Visibility: Protocol-visible State.
    """

    step_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = ""
    primary_tool: str | None = None

    status: StepStatus = StepStatus.PENDING
    attempt: int = Field(default=0, ge=0)
    depends_on_step_ids: StepIdList = Field(default_factory=tuple)

ExecutionStepList = tuple[ExecutionStep, ...]


class ExecutionPlan(ImmutableProtocolModel):
    """Accepted or candidate plan realization with versioned step definitions.

    Protocol purpose: define step sequence and revision continuity.
    Runtime purpose: provide structured plan input to brain/summary flows.
    Ownership: planner proposes, controller accepts active plan revision.
    Visibility: Protocol-visible State.

    Note: runtime-only routing hints belong in Working State, not this model.
    """

    plan_id: str = Field(min_length=1)
    revision: int = Field(default=1, ge=1)
    objective: str = ""
    steps: ExecutionStepList = Field(default_factory=tuple)


class ExecutionCursor(ImmutableProtocolModel):
    """Typed cursor pointing to the next legal continuation position.

    Protocol purpose: represent legal resume/replay continuation position.
    Runtime purpose: support deterministic continuation and recovery validation.
    Ownership: controller-governed cursor progression.
    Visibility: Protocol-visible State.
    """

    phase: ExecutionPhase = ExecutionPhase.INITIALIZING
    step_id: str | None = None
    event_index: int = Field(default=0, ge=0)
    plan_revision: int | None = Field(default=None, ge=1)
    step_attempt: int | None = Field(default=None, ge=0)
    current_worker: WorkerRole | None = None
    controller_iteration: int | None = Field(default=None, ge=0)


class ToolRequest(ImmutableProtocolModel):
    """Deterministic tool invocation request produced from role-scoped outputs.

    Protocol purpose: represent a typed tool execution intent.
    Runtime purpose: bridge reasoning output to deterministic tool execution.
    Ownership: produced by role logic, accepted/executed under controller authority.
    Visibility: Protocol-visible exchange object.
    """

    request_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    requested_by: WorkerRole = WorkerRole.BRAIN


class ToolInput(ImmutableProtocolModel):
    """Controller-governed input contract for tool execution.

    Protocol purpose: define legal tool execution envelope.
    Runtime purpose: provide the requested tool invocation together with
    execution-scoped protocol context.
    Ownership: prepared by controller.
    Visibility: Protocol-visible envelope.
    """

    identity: ExecutionIdentity
    cursor: ExecutionCursor
    context: "ExecutionContext"

    tool_request: ToolRequest

    active_plan: ExecutionPlan | None = None
    active_step: ExecutionStep | None = None

    retry: RetryMetadata = Field(default_factory=RetryMetadata)


class ContentIntegrity(ImmutableProtocolModel):
    """Explicit indicators for content bounding and truncation.

    Protocol purpose: formalize content integrity and truncation metadata.
    Runtime purpose: allow Controller and Brain to detect truncated output.
    """

    is_truncated: bool = False
    original_bytes: int = Field(default=0, ge=0)
    captured_bytes: int = Field(default=0, ge=0)
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class PaginationMetadata(ImmutableProtocolModel):
    """Pagination tracking for chunked read/list operations.

    Protocol purpose: formalize chunking and pagination offset/limit metadata.
    Runtime purpose: allow Controller to make page continuation decisions.
    """

    has_more: bool = False
    total_items: int = Field(default=0, ge=0)
    returned_items: int = Field(default=0, ge=0)
    offset: int = Field(default=0, ge=0)
    limit: int | None = None


class ArtifactRecord(ImmutableProtocolModel):
    """Track resource or file mutations performed during tool execution."""

    artifact_id: str = Field(min_length=1)
    step_id: str = ""
    path: str = Field(min_length=1)
    action: str = "modified"
    hash_sha256: str | None = None


class ToolResult(ImmutableProtocolModel):
    """Deterministic tool invocation outcome consumed by continuation logic.

    Protocol purpose: represent tool execution outcome for step progression.
    Runtime purpose: support capture, validation, and next-step reasoning.
    Ownership: produced by tool runtime, integrated under controller authority.
    Visibility: Protocol-visible exchange object.
    """

    request_id: str = Field(min_length=1)
    signature: str = ""
    success: bool
    message: str
    rendered_output: str = ""
    data: JsonValue = None
    error_code: str | None = None
    integrity: ContentIntegrity = Field(default_factory=ContentIntegrity)
    pagination: PaginationMetadata | None = None


class ToolExecutionRecord(ImmutableProtocolModel):
    step_id: str

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: ToolResult
    artifacts: tuple[ArtifactRecord, ...] = Field(default_factory=tuple)


class ReplanRequest(ImmutableProtocolModel):
    """Typed replan request generated when plan continuation is insufficient.

    Protocol purpose: capture explicit replan trigger conditions.
    Runtime purpose: provide structured replan signal to controller/planner paths.
    Ownership: produced by role logic, accepted under controller authority.
    Visibility: Protocol-visible exchange object.
    """

    reason: str = Field(min_length=1)
    failed_step_id: str | None = None
    constraints: ConstraintList = Field(default_factory=tuple)
    requested_by: WorkerRole = WorkerRole.BRAIN


class BrainInput(ImmutableProtocolModel):
    """Controller-governed input contract for brain execution.

    Protocol purpose: define legal brain input envelope.
    Runtime purpose: assemble role-scoped context without direct worker coupling.
    Ownership: prepared by runtime services under controller authority.
    Visibility: Protocol-visible envelope with scoped runtime context.
    """

    identity: ExecutionIdentity
    cursor: ExecutionCursor
    context: ExecutionContext
    active_plan: ExecutionPlan | None = None
    active_step: ExecutionStep | None = None
    last_tool_result: ToolResult | None = None
    tool_execution_history: tuple[ToolExecutionRecord, ...] = Field(default_factory=tuple)
    retry: RetryMetadata = Field(default_factory=RetryMetadata)


class BrainResult(ImmutableProtocolModel):
    """Typed outcome contract emitted by brain runtime realization.

    Protocol purpose: represent step-scoped outcome categories.
    Runtime purpose: provide structured next-action candidates for controller evaluation.
    Ownership: produced by brain role, interpreted by controller role.
    Visibility: Protocol-visible exchange object.
    """

    outcome: BrainOutcome
    message: str = ""
    tool_request: ToolRequest | None = None
    replan_request: ReplanRequest | None = None
    final_answer: str | None = None
    proposed_step_status: StepStatus | None = None


class ControllerInput(ImmutableProtocolModel):
    """Controller-governed input contract for continuation decisions.

    Protocol purpose: define legal controller input envelope.
    Runtime purpose: assemble role-scoped execution context and latest worker
    outputs without coupling controller logic to legacy state shape.
    Ownership: prepared by runtime services under controller authority.
    Visibility: Protocol-visible envelope with scoped runtime context.
    """

    identity: ExecutionIdentity
    cursor: ExecutionCursor
    context: "ExecutionContext"
    active_plan: ExecutionPlan | None = None
    active_step: ExecutionStep | None = None
    pending_tool_request: ToolRequest | None = None
    planner_result: PlannerResult | None = None
    brain_result: BrainResult | None = None
    tool_result: ToolResult | None = None
    retry: RetryMetadata = Field(default_factory=RetryMetadata)
    tool_execution_history: tuple[ToolExecutionRecord, ...] = Field(default_factory=tuple)

    def get_step_records(self, step_id: str | None = None) -> tuple[ToolExecutionRecord, ...]:
        target_step_id = step_id or (self.active_step.step_id if self.active_step else None)
        if not target_step_id:
            return self.tool_execution_history
        return tuple(r for r in self.tool_execution_history if r.step_id == target_step_id)

    def get_consecutive_failures(self, signature: str | None = None) -> int:
        count = 0
        for record in reversed(self.tool_execution_history):
            if signature and record.result.signature != signature:
                continue
            if not record.result.success:
                count += 1
            else:
                break
        return count

    def has_unresolved_truncation(self, step_id: str | None = None) -> bool:
        records = self.get_step_records(step_id)
        if not records:
            return False
        latest = records[-1]
        if latest.result.integrity.is_truncated:
            return True
        if latest.result.pagination is not None and latest.result.pagination.has_more:
            return True
        return False

    def has_successful_artifact(self, path: str) -> bool:
        for record in self.tool_execution_history:
            if not record.result.success:
                continue
            for art in record.artifacts:
                if art.path == path:
                    return True
        return False


class PlannerInput(ImmutableProtocolModel):
    """Controller-governed input contract for planner execution.

    Protocol purpose: define legal planner input envelope.
    Runtime purpose: assemble role-scoped planning context without direct worker coupling.
    Ownership: prepared by runtime services under controller authority.
    Visibility: Protocol-visible envelope with scoped runtime context.
    """

    identity: ExecutionIdentity
    context: "ExecutionContext"
    active_plan: ExecutionPlan | None = None
    completed_step_ids: StepIdList = Field(default_factory=tuple)
    retry: RetryMetadata = Field(default_factory=RetryMetadata)

class PlannerResult(ImmutableProtocolModel):
    """Typed outcome contract emitted by planner runtime realization.

    Protocol purpose: represent a candidate plan revision.
    Runtime purpose: provide a structured planning result for controller evaluation.
    Ownership: produced by planner role, interpreted by controller role.
    Visibility: Protocol-visible exchange object.
    """

    outcome: PlannerOutcome
    proposed_plan: ExecutionPlan | None = None
    message: str = ""
    planning_rationale: str = ""
    change_summary: str = ""

class ExecutionSummary(ImmutableProtocolModel):
    """Terminal summary generated from accepted protocol-visible facts.

    Protocol purpose: represent terminal reporting state.
    Runtime purpose: provide stable completion summary contract.
    Ownership: summary role output under controller-governed terminal flow.
    Visibility: Protocol-visible State.
    """

    execution_id: str = Field(min_length=1)
    status: ExecutionStatus
    summary_text: str = Field(min_length=1)
    completed_step_ids: StepIdList = Field(default_factory=tuple)
    failed_step_ids: StepIdList = Field(default_factory=tuple)


class EventRecord(ImmutableProtocolModel):
    """Append-only accepted event record for replay and compliance reconstruction.

    Protocol purpose: immutable accepted history entry.
    Runtime purpose: deterministic replay/recovery alignment anchor.
    Ownership: appended under controller authority.
    Visibility: Protocol-visible State.
    """

    index: int = Field(ge=0)
    event_type: EventType
    producer: WorkerRole
    timestamp_utc: datetime = Field(min_length=1)
    payload: JsonValue = None
    correlation_id: str | None = None
    sequence_id: str | None = None
    producer_instance: str | None = None

EventHistory = tuple[EventRecord, ...]


class CheckpointState(ImmutableProtocolModel):
    """Recovery snapshot contract for legal continuation restoration.

    Protocol purpose: capture checkpoint continuity metadata.
    Runtime purpose: accelerate restore without replacing accepted history truth.
    Ownership: checkpoint manager under controller authority.
    Visibility: Protocol-visible checkpoint artifact.
    """

    checkpoint_id: str = Field(min_length=1)
    identity: ExecutionIdentity
    cursor: ExecutionCursor
    event_index: int = Field(ge=0)
    completed_step_ids: StepIdList = Field(default_factory=tuple)
    active_plan_id: str | None = None
    active_step_id: str | None = None
    retry: RetryMetadata = Field(default_factory=RetryMetadata)


class ProtocolVisibleState(ImmutableProtocolModel):
    """Protocol-visible state for legality, replay, and conformance evaluation.

    Protocol purpose: authoritative accepted state representation.
    Runtime purpose: provide typed state for controller-governed decisions.
    Ownership: controller is the authoritative accepted-state writer.
    Visibility: Protocol-visible State.
    """

    identity: ExecutionIdentity
    status: ExecutionStatus = ExecutionStatus.NON_TERMINAL
    cursor: ExecutionCursor
    active_plan: ExecutionPlan | None = None
    active_step: ExecutionStep | None = None
    pending_tool_request: ToolRequest | None = None
    completed_step_ids: StepIdList = Field(default_factory=tuple)
    accepted_event_history: EventHistory = Field(default_factory=tuple)
    retry: RetryMetadata = Field(default_factory=RetryMetadata)
    summary: ExecutionSummary | None = None


class WorkingState(ImmutableProtocolModel):
    """Working State supporting orchestration without becoming protocol truth.

    Protocol purpose: none; this model is intentionally non-authoritative.
    Runtime purpose: hold helper/runtime metadata for orchestration continuity.
    Ownership: runtime services under controller governance.
    Visibility: Working State.

    Note: generic dictionaries are reserved for temporary runtime orchestration
    hints and must not be treated as accepted protocol facts.
    """

    retrieval_context: RetrievalContext = Field(default_factory=tuple)
    last_tool_result: ToolResult | None = None
    tool_execution_history: tuple[ToolExecutionRecord, ...] = Field(default_factory=tuple)
    repeat_fail_count: int = 0
    routing_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    planner_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    debug_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    capture_state: dict[str, JsonValue] = Field(default_factory=dict)
    orchestration_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ExecutionState(ImmutableProtocolModel):
    """Top-level state contract separating protocol-visible and working concerns.

    Protocol purpose: encapsulate authoritative accepted state.
    Runtime purpose: keep non-authoritative orchestration state explicit and separate.
    Ownership: controller-governed state envelope.
    Visibility: mixed (Protocol-visible + Working State).
    """

    protocol_visible: ProtocolVisibleState
    working: WorkingState = Field(default_factory=WorkingState)


class ExecutionContext(ImmutableProtocolModel):
    """Role-scoped context assembled for worker execution.

    Protocol purpose: provide context contract for role-scoped execution inputs.
    Runtime purpose: carry retrieved and recent context slices without direct worker coupling.
    Ownership: assembled by runtime services under controller authority.
    Visibility: execution input context (controller-governed).
    """

    user_request: str = Field(min_length=1)
    retrieval_messages: MessageList = Field(default_factory=tuple)
    recent_history: MessageList = Field(default_factory=tuple)
    role: WorkerRole = WorkerRole.BRAIN


class ControllerDecision(ImmutableProtocolModel):
    """Typed controller continuation decision for legal runtime progression.

    Protocol purpose: represent legal continuation/termination decision outcomes.
    Runtime purpose: support controller implementation with explicit decision metadata.
    Ownership: produced by controller authority only.
    Visibility: Protocol-visible decision envelope.
    """

    accepted_plan: ExecutionPlan | None = None
    decision_type: ControllerDecisionType
    reason: str = ""

    next_worker: WorkerRole | None = None
    cursor: ExecutionCursor | None = None

    # Step transition
    completed_step_id: str | None = None
    next_step_id: str | None = None
    clear_active_step: bool = False

    requires_checkpoint: bool = False
    requires_replan: bool = False

    pending_tool_request: ToolRequest | None = None
    clear_pending_tool_request: bool = False

    terminal: bool = False
