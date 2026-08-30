"""Immutable protocol-oriented data contracts for gradual runtime migration.

These models provide a typed contract layer aligned with CEP/CIS semantics while
remaining independent from current runtime orchestration modules.
"""

from __future__ import annotations

from typing import Any

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    AsyncJobStatus,
    BrainOutcome,
    CancellationSource,
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


class AsyncJobPolicy(ImmutableProtocolModel):
    """Controller-owned limits for provider-managed asynchronous jobs."""

    visibility_grace_seconds: int = Field(default=15, ge=0)
    poll_interval_seconds: int = Field(default=5, ge=1)
    max_poll_interval_seconds: int = Field(default=30, ge=1)
    execution_timeout_seconds: int = Field(default=1800, ge=1)
    max_poll_failures: int = Field(default=3, ge=1)
    max_submission_attempts: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_async_policy_limits(self) -> "AsyncJobPolicy":
        if self.max_poll_interval_seconds < self.poll_interval_seconds:
            raise ValueError(
                "max_poll_interval_seconds must be greater than or equal to "
                "poll_interval_seconds"
            )
        if self.execution_timeout_seconds < self.visibility_grace_seconds:
            raise ValueError(
                "execution_timeout_seconds must be greater than or equal to "
                "visibility_grace_seconds"
            )
        return self


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
    is_async_job: bool = False
    async_job_id: str | None = None
    async_job_status: AsyncJobStatus | None = None
    async_terminal: bool = False
    async_observed_at_utc: datetime | None = None

    @model_validator(mode="after")
    def validate_async_job_fields(self) -> "ToolResult":
        if not self.is_async_job:
            if any((self.async_job_id, self.async_job_status, self.async_terminal, self.async_observed_at_utc)):
                raise ValueError("async fields require is_async_job=True")
            return self

        if not self.async_job_id:
            raise ValueError("async_job_id is required for async jobs")
        if self.async_job_status is None:
            raise ValueError("async_job_status is required for async jobs")

        is_terminal_status = self.async_job_status in {
            AsyncJobStatus.COMPLETED,
            AsyncJobStatus.FAILED,
            AsyncJobStatus.CANCELLED,
        }
        if self.async_terminal != is_terminal_status:
            raise ValueError("async_terminal must match async_job_status terminality")
        return self


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
    async_policy: AsyncJobPolicy = Field(default_factory=AsyncJobPolicy)
    cancel_requested: bool = False
    tool_execution_history: tuple[ToolExecutionRecord, ...] = Field(default_factory=tuple)

    def get_step_records(self, step_id: str | None = None) -> tuple[ToolExecutionRecord, ...]:
        target_step_id = step_id or (self.active_step.step_id if self.active_step else None)
        if not target_step_id:
            return self.tool_execution_history
        return tuple(r for r in self.tool_execution_history if r.step_id == target_step_id)

    def _latest_async_results_by_job(
        self,
        step_id: str | None = None,
    ) -> tuple[ToolResult, ...]:
        """Build a monotonic latest-result view for every provider job."""
        latest_by_job_id: dict[str, ToolResult] = {}
        terminal_job_ids: set[str] = set()
        observation_order: list[str] = []

        for record in self.get_step_records(step_id):
            result = record.result
            if not result.is_async_job or result.async_job_id is None:
                continue
            if result.async_job_id in terminal_job_ids:
                continue

            latest_by_job_id[result.async_job_id] = result
            if result.async_job_id in observation_order:
                observation_order.remove(result.async_job_id)
            observation_order.append(result.async_job_id)
            if result.async_terminal:
                terminal_job_ids.add(result.async_job_id)

        return tuple(latest_by_job_id[job_id] for job_id in observation_order)

    def get_latest_async_result(self, step_id: str | None = None) -> ToolResult | None:
        """Return the latest valid async observation for a step from immutable evidence."""
        results = self._latest_async_results_by_job(step_id)
        return results[-1] if results else None

    def get_active_async_job_ids(self, step_id: str | None = None) -> tuple[str, ...]:
        """Return every provider job whose monotonic latest state is non-terminal."""
        return tuple(
            result.async_job_id
            for result in self._latest_async_results_by_job(step_id)
            if not result.async_terminal and result.async_job_id is not None
        )

    def get_nonterminal_async_job_id(self, step_id: str | None = None) -> str | None:
        """Return the active provider job ID for a step, if its latest observation is non-terminal."""
        result = self.get_latest_async_result(step_id)
        if result is None or result.async_terminal:
            return None
        return result.async_job_id

    def has_terminal_async_result(self, job_id: str) -> bool:
        """Report whether immutable evidence contains a terminal observation for a provider job."""
        return any(
            record.result.is_async_job
            and record.result.async_job_id == job_id
            and record.result.async_terminal
            for record in self.tool_execution_history
        )

    def get_async_job_started_at_utc(self, job_id: str) -> datetime | None:
        """Return the earliest recorded observation time for one provider job."""
        observed_times = tuple(
            record.result.async_observed_at_utc
            for record in self.tool_execution_history
            if record.result.is_async_job
            and record.result.async_job_id == job_id
            and record.result.async_observed_at_utc is not None
        )
        return min(observed_times) if observed_times else None

    def get_async_observation_count(
        self,
        job_id: str,
        *,
        excluded_tool_names: frozenset[str] = frozenset(),
    ) -> int:
        """Count status observations used to derive polling backoff."""
        return sum(
            1
            for record in self.tool_execution_history
            if record.result.is_async_job
            and record.result.async_job_id == job_id
            and record.tool_name not in excluded_tool_names
            and record.result.async_job_status in {
                AsyncJobStatus.RUNNING,
                AsyncJobStatus.UNKNOWN,
            }
        )

    def get_consecutive_async_poll_failures(
        self,
        job_id: str,
        *,
        excluded_tool_names: frozenset[str] = frozenset(),
    ) -> int:
        """Count transport/status failures since the latest successful observation."""
        count = 0
        for record in reversed(self.tool_execution_history):
            result = record.result
            if not result.is_async_job or result.async_job_id != job_id:
                continue
            if record.tool_name in excluded_tool_names:
                continue
            if result.async_terminal or result.success:
                break
            count += 1
        return count

    def get_async_submission_attempt_count(
        self,
        step_id: str | None = None,
        *,
        submission_tool_names: frozenset[str] = frozenset(),
        ambiguous_error_codes: frozenset[str] = frozenset(),
    ) -> int:
        """Count immutable submission executions for a step."""
        return sum(
            1
            for record in self.get_step_records(step_id)
            if (
                record.tool_name in submission_tool_names
                and (
                    record.result.is_async_job
                    or record.result.error_code is None
                    or record.result.error_code in ambiguous_error_codes
                )
                if submission_tool_names
                else record.result.async_job_status == AsyncJobStatus.SUBMITTED
            )
        )

    def is_async_job_confirmed_absent(self, job_id: str) -> bool:
        """Read normalized provider visibility without exposing raw payload access."""
        latest = next(
            (
                result
                for result in reversed(self._latest_async_results_by_job())
                if result.async_job_id == job_id
            ),
            None,
        )
        return bool(
            latest is not None
            and isinstance(latest.data, dict)
            and latest.data.get("provider_visibility") == "absent"
        )

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
    async_policy: AsyncJobPolicy = Field(default_factory=AsyncJobPolicy)
    cancellation_source: CancellationSource | None = None
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
    cancel_requested: bool = False
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

    async_job_id: str | None = None
    resume_after_utc: datetime | None = None
    execution_deadline_utc: datetime | None = None
    reconciliation_required: bool = False
    cancellation_source: CancellationSource | None = None

    terminal: bool = False
