"""LangGraph transport adapters implementing portable worker ports."""

from __future__ import annotations

from core.protocol.models import BrainResult, PlannerResult, PlanningRequest, ToolResult
from core.runtime.execution_driver import WorkerDispatchError
from core.runtime.tool_result_integration import integrate_tool_result


def _invoke(node, state):
    if callable(node):
        return node(state)
    invoke = getattr(node, "invoke", None)
    if callable(invoke):
        return invoke(state)
    raise TypeError("graph worker transport is not executable")


class GraphWorkerRuntimePorts:
    """Execute legacy worker transports only from a driver authorization."""

    def __init__(self, *, tool_runtime=None):
        self._state = None
        self._planner = self._brain = self._tool = self._capture = None
        self._tool_runtime = tool_runtime
        self._update = {}

    def bind_nodes(self, *, planner, brain, tool=None, capture=None):
        self._planner, self._brain = planner, brain
        self._tool, self._capture = tool, capture

    def begin_turn(self, state):
        self._state = state
        self._update = {}

    def consume_update(self):
        update, self._update = self._update, {}
        return update

    def _authorized_state(self, execution_state, decision):
        if self._state is None:
            raise WorkerDispatchError("graph transport has no active portable turn")
        return {
            **self._state,
            "execution_state": execution_state,
            "controller_decision": decision,
        }

    def run(self, _value):
        raise WorkerDispatchError("Planner requires driver authorization")

    def run_authorized(self, value, execution_state, decision):
        node = self._planner if isinstance(value, PlanningRequest) else self._brain
        update = _invoke(node, self._authorized_state(execution_state, decision))
        self._update = dict(update)
        result = update.get("planner_result") if node is self._planner else update.get("brain_result")
        if not isinstance(result, (PlannerResult, BrainResult)):
            raise WorkerDispatchError("graph worker transport returned no typed result")
        return result

    def execute(self, _value):
        raise WorkerDispatchError("Tool runtime requires driver authorization")

    def execute_authorized(self, value, execution_state, decision):
        self._authorized_state(execution_state, decision)
        if self._tool_runtime is None:
            state = self._authorized_state(execution_state, decision)
            tool_update = _invoke(self._tool, state)
            transported = {**state, **tool_update}
            capture_update = _invoke(self._capture, transported)
            self._update = {**tool_update, **capture_update}
            result_state = capture_update.get("execution_state")
            result = getattr(
                getattr(result_state, "working", None), "last_tool_result", None
            )
            if not isinstance(result, ToolResult):
                raise WorkerDispatchError("tool transport returned no typed ToolResult")
            return result
        result = self._tool_runtime.execute(value)
        if not isinstance(result, ToolResult):
            raise WorkerDispatchError("direct tool runtime returned no typed ToolResult")
        integrated = integrate_tool_result(execution_state, decision, result)
        self._update = {"execution_state": integrated}
        return result
