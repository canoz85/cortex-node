"""Framework-neutral worker ports used by runtime execution drivers."""

from __future__ import annotations

from typing import Protocol

from core.protocol.models import (
    BrainInput,
    BrainResult,
    FinalizationRequest,
    FinalizationResult,
    PlannerResult,
    PlanningRequest,
    ToolRequest,
    ToolResult,
)


class PlannerPort(Protocol):
    def run(self, request: PlanningRequest) -> PlannerResult: ...


class BrainPort(Protocol):
    def run(self, brain_input: BrainInput) -> BrainResult: ...


class ToolRuntimePort(Protocol):
    def execute(self, request: ToolRequest) -> ToolResult: ...


class FinalizerPort(Protocol):
    def finalize(self, request: FinalizationRequest) -> FinalizationResult: ...


__all__ = ["BrainPort", "FinalizerPort", "PlannerPort", "ToolRuntimePort"]
