"""Protocol data model layer for typed, immutable execution contracts."""

# Enums
from .enums import (
    BrainOutcome,
    ControllerDecisionType,
    EventType,
    ExecutionPhase,
    ExecutionStatus,
    StepStatus,
    WorkerRole,
)

# Models
from .models import (
    BrainInput,
    BrainResult,
    ControllerInput,
    PlannerInput,
    PlannerResult,
    CheckpointState,
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

from .converters import (
    _to_int,
    _to_optional_int,
    _to_str,
)

__all__ = [
    # Enums
    "BrainOutcome",
    "ControllerDecisionType",
    "EventType",
    "ExecutionPhase",
    "ExecutionStatus",
    "StepStatus",
    "WorkerRole",

    # Models
    "BrainInput",
    "BrainResult",
    "ControllerInput",
    "PlannerInput",
    "PlannerResult",
    "CheckpointState",
    "ControllerDecision",
    "EventRecord",
    "ExecutionContext",
    "ExecutionCursor",
    "ExecutionIdentity",
    "ExecutionPlan",
    "ExecutionState",
    "ExecutionStep",
    "ExecutionSummary",
    "ProtocolVisibleState",
    "ReplanRequest",
    "RetryMetadata",
    "ToolRequest",
    "ToolInput",
    "ToolResult",
    "WorkingState",

    # Converters
    "_to_int",
    "_to_optional_int",
    "_to_str",
]
