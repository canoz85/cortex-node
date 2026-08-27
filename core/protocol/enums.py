"""Protocol model enums for runtime-independent contract typing."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String enum base class for serialization-friendly contract fields."""


class WorkerRole(StrEnum):
    """Logical runtime worker roles recognized by protocol-oriented models."""

    CONTROLLER = "controller"
    PLANNER = "planner"
    BRAIN = "brain"
    TOOL_RUNTIME = "tool_runtime"
    SUMMARY = "summary"


class ExecutionPhase(StrEnum):
    """Execution lifecycle phase used by cursor and state models."""

    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REPLANNING = "replanning"
    TERMINATING = "terminating"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ExecutionStatus(StrEnum):
    """High-level execution status for protocol-visible summaries and state."""

    NON_TERMINAL = "non_terminal"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class StepStatus(StrEnum):
    """Status of a single execution step within a plan."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AsyncJobStatus(StrEnum):
    """Observed lifecycle state of a provider-managed asynchronous job."""

    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class PlannerOutcome(StrEnum):
    EXECUTION_PLAN = "execution_plan"
    DIRECT_RESPONSE = "direct_response"
    CLARIFICATION_REQUIRED = "clarification_required"
    FAILED = "failed"

class BrainOutcome(StrEnum):
    """Step-scoped outcomes produced by BrainResult."""

    CONTINUE = "continue"
    TOOL_REQUEST = "tool_request"
    REPLAN_REQUEST = "replan_request"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    FINAL_ANSWER = "final_answer"


class ControllerDecisionType(StrEnum):
    """Controller continuation decisions represented as typed outcomes."""

    DISPATCH_PLANNER = "dispatch_planner"
    DISPATCH_BRAIN = "dispatch_brain"
    DISPATCH_TOOL_RUNTIME = "dispatch_tool_runtime"
    DISPATCH_SUMMARY = "dispatch_summary"
    REQUEST_REPLAN = "request_replan"

    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    
    TERMINATE = "terminate"


class EventType(StrEnum):
    """Protocol event categories for append-only accepted history records."""

    EXECUTION_STARTED = "execution_started"
    PLAN_CREATED = "plan_created"
    PLAN_REVISED = "plan_revised"
    STEP_STARTED = "step_started"
    TOOL_REQUESTED = "tool_requested"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    REPLAN_REQUESTED = "replan_requested"
    EXECUTION_PAUSED = "execution_paused"
    EXECUTION_RESUMED = "execution_resumed"
    EXECUTION_CHECKPOINTED = "execution_checkpointed"
    SUMMARY_GENERATED = "summary_generated"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_CANCELLED = "execution_cancelled"
