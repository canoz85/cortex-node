from __future__ import annotations
from langchain_core.messages import AIMessage
from core.protocol.controller import CortexController
from core.finalizer import Finalizer

from core.graph_constants import MAX_REASONING_STEPS
from core.protocol.bridge import build_controller_input
from core.protocol.models import FinalizationResult, PlanningCapabilities
from core.state import AgentState
from core.completion import CompletionService
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver
from core.runtime.portable_orchestration import PortableExecutionRuntime


class _GraphDeferredWorkerPort:
    """Non-terminal workers remain physical graph nodes in Slice 3."""

    def run(self, _value):
        raise RuntimeError("graph worker dispatch must be deferred")

    def execute(self, _value):
        raise RuntimeError("graph tool dispatch must be deferred")


def create_controller_node(
    controller: CortexController | None = None,
    completion_service: CompletionService | None = None,
    finalizer: Finalizer | None = None,
    planning_capabilities: PlanningCapabilities | None = None,
    worker_ports=None,
):
    completion_service = completion_service or CompletionService()
    controller = controller or CortexController(
        max_reasoning_steps=MAX_REASONING_STEPS,
        planning_capabilities=planning_capabilities,
    )
    finalizer = finalizer or Finalizer()
    coordinator = ControllerCoordinator(controller)
    deferred = worker_ports or _GraphDeferredWorkerPort()
    runtime = PortableExecutionRuntime(
        driver=ExecutionDriver(
            coordinator=coordinator,
            planner=deferred,
            brain=deferred,
            tool_runtime=deferred,
            finalizer=finalizer,
        ),
        completion_service=completion_service,
    )

    def controller_node(state: AgentState):
        """
        LangGraph adapter for the protocol Controller.

        This function translates graph transport into and out of the portable
        runtime. Production workers execute through the injected driver ports;
        standalone legacy adapters may still defer dispatch for compatibility.
        """

        controller_input = build_controller_input(state)

        if worker_ports is not None:
            worker_ports.begin_turn(state)
        portable_turn = runtime.turn(
            state["execution_state"],
            controller_input,
            dispatch_worker=worker_ports is not None,
        )
        controller_input = portable_turn.controller_input
        turn = portable_turn.driver_turn
        decision = turn.decision

        # print("\n=== CONTROLLER DECISION ===")
        # print("decision:", decision)
        #print("before:", state["execution_state"].protocol_visible)

        execution_state = turn.execution_state

        print("\n=== AFTER APPLY ===")
        # print("cursor:", execution_state.protocol_visible.cursor)
        print("active_step:", execution_state.protocol_visible.active_step)
        print(
            "completed:",
            execution_state.protocol_visible.completed_step_ids,
        )
        print("========================\n")

        update = {
            "execution_state": execution_state,
            "controller_decision": decision,
        }
        if worker_ports is not None:
            worker_update = worker_ports.consume_update()
            transported_state = worker_update.pop("execution_state", None)
            if transported_state is not None:
                execution_state = execution_state.model_copy(
                    update={"working": transported_state.working}
                )
            update.update(worker_update)
            update["execution_state"] = execution_state
        if decision.planning_clarification is not None:
            update["clarification_request"] = decision.planning_clarification.prompt
        elif decision.clear_planning_clarification:
            update["clarification_request"] = ""

        if controller_input.brain_result is not None:
            update["brain_result"] = None

        if controller_input.planner_result is not None:
            update["planner_result"] = None

        if decision.terminal:
            result = turn.worker_result
            error = portable_turn.terminal_dispatch_error
            if isinstance(result, FinalizationResult):
                execution_state = execution_state.model_copy(update={
                    "protocol_visible": execution_state.protocol_visible.model_copy(update={
                        "summary": result.execution_summary,
                    }),
                })
                update["execution_state"] = execution_state
                update["finalization_result"] = result
                update["finalization_error"] = result.final_answer_error or ""
                update["final_answer"] = result.final_answer
                update["messages"] = [AIMessage(content=result.final_answer)]
                print(
                    "[finalizer] "
                    f"execution_id={controller_input.identity.execution_id} "
                    f"status={execution_state.protocol_visible.status.value} "
                    f"summary={result.execution_summary.model_dump(mode='json')}"
                )
            else:
                error_text = (
                    f"{type(error).__name__}: {error}"
                    if error is not None
                    else "TypeError: portable runtime returned no terminal result"
                )
                final_answer = "Execution finished, but finalization failed."
                update["finalization_error"] = error_text
                update["final_answer"] = final_answer
                update["messages"] = [AIMessage(content=final_answer)]
                print(
                    "[finalizer] "
                    f"execution_id={controller_input.identity.execution_id} error={error_text}"
                )


        # print("\n====CONTROLLER====:")
        # print("current_worker:", execution_state.protocol_visible.cursor.current_worker)
        # print("---------------")
        # if execution_state.protocol_visible.cursor.current_worker == WorkerRole.PLANNER:
        #     print("active_plan:", execution_state.protocol_visible.active_plan)
        #     print("---------------")
        # print("active_step:", execution_state.protocol_visible.active_step)
        # print("---------------")
        # if execution_state.protocol_visible.cursor.current_worker == WorkerRole.BRAIN:
        #     print("brain_result:", controller_input.brain_result)
        #     print("---------------")
        # print("====END CONTROLLER====\n")


        return update

    controller_node._portable_dispatch = worker_ports is not None
    controller_node._portable_runtime = runtime
    return controller_node
