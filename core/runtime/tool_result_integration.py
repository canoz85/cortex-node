"""Framework-neutral typed tool execution and evidence integration helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from typing import Any, ContextManager

from core.protocol.models import (
    ControllerDecision,
    ExecutionState,
    ToolExecutionRecord,
    ToolRequest,
    ToolResult,
)
from core.tool_output import compute_repeat_fail_count, parse_tool_output, unwrap_tool_output, build_artifact_records, normalize_tool_output
from tools.registry import ToolRegistry


class SerializedToolRuntimePort:
    """Execute registered tools and return the portable typed result contract."""

    def __init__(
        self,
        tools: Iterable[Any],
        *,
        observe: Callable[[ToolRequest], ContextManager[Any]] | None = None,
        require_structured: bool = True,
    ) -> None:
        self._tools_by_name = ToolRegistry.from_tools(tools).by_name
        self._observe = observe
        self._require_structured = require_structured

    def execute(self, request: ToolRequest) -> ToolResult:
        tool = self._tools_by_name.get(request.tool_name)
        if tool is None:
            raise RuntimeError(f"Tool is not available: {request.tool_name}.")
        scope = self._observe(request) if self._observe is not None else nullcontext()
        with scope:
            invoke = getattr(tool, "invoke", None)
            if callable(invoke):
                output = invoke(request.arguments)
            elif callable(tool):
                output = tool(**request.arguments)
            else:
                raise TypeError(f"Tool is not executable: {request.tool_name}.")

        if isinstance(output, ToolResult):
            return output
        if not isinstance(output, str):
            raise TypeError("Tool runtime returned no typed ToolResult")
        if (
            self._require_structured
            and parse_tool_output(output) is None
            and not isinstance(unwrap_tool_output(output), dict)
        ):
            raise TypeError("Tool runtime returned no typed ToolResult")
        return normalize_tool_output(raw_content=output, request=request)


def integrate_tool_result(
    execution_state: ExecutionState,
    decision: ControllerDecision,
    result: ToolResult,
) -> ExecutionState:
    """Append one authorized typed result to the existing evidence model."""
    request = decision.pending_tool_request
    if request is None:
        raise ValueError("Tool result integration requires pending_tool_request.")
    if result.request_id != request.request_id:
        raise ValueError("Tool result request identity mismatch")

    protocol = execution_state.protocol_visible
    if protocol.pending_tool_request != request:
        raise ValueError("Tool result request is not current in execution state")

    step = protocol.active_step
    if step is None:
        raise ValueError("Tool result integration requires an active step.")

    working = execution_state.working
    repeat_fail_count = compute_repeat_fail_count(
        previous=working.last_tool_result,
        previous_repeat_count=working.repeat_fail_count,
        current=result,
    )

    artifacts=build_artifact_records(
        result.artifacts,
        step_id=step.step_id,
    )

    record = ToolExecutionRecord(
        execution_id=protocol.identity.execution_id,
        plan_id=protocol.active_plan.plan_id if protocol.active_plan else None,
        plan_revision=protocol.active_plan.revision if protocol.active_plan else None,
        step_id=step.step_id,
        tool_name=request.tool_name,
        arguments=request.arguments,
        result=result,
        artifacts=artifacts,
    )
    return execution_state.model_copy(
        update={
            "working": working.model_copy(
                update={
                    "last_tool_result": result,
                    "tool_execution_history": (
                        *working.tool_execution_history,
                        record,
                    ),
                    "repeat_fail_count": repeat_fail_count,
                }
            )
        }
    )


__all__ = ["SerializedToolRuntimePort", "integrate_tool_result"]
