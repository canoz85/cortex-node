from __future__ import annotations
from langchain_core.messages import AIMessage
from core.protocol.controller import CortexController
from core.finalizer import Finalizer

from core.graph_constants import MAX_REASONING_STEPS
from core.graph_state_machine import apply_controller_decision_to_state
from core.protocol.bridge import build_controller_input
from core.protocol.enums import WorkerRole, BrainOutcome, ExecutionStatus
from core.protocol.models import ControllerDecision, ControllerInput, ExecutionState, FinalizationRequest, PlanningCapabilities
from core.state import AgentState
from core.completion import CompletionService
from core.protocol.completion_identity import accepted_step
from core.planner_revision import RevisionRejection, reconcile_revision
from core.protocol.enums import PlanningOperation


def create_controller_node(
    controller: CortexController | None = None,
    completion_service: CompletionService | None = None,
    finalizer: Finalizer | None = None,
    planning_capabilities: PlanningCapabilities | None = None,
):
    completion_service = completion_service or CompletionService()
    controller = controller or CortexController(
        max_reasoning_steps=MAX_REASONING_STEPS,
        planning_capabilities=planning_capabilities,
    )
    finalizer = finalizer or Finalizer()

    def controller_node(state: AgentState):
        """
        LangGraph adapter for the protocol Controller.

        Responsibilities:
          1. Build ControllerInput from legacy state.
          2. Invoke the protocol Controller.
          3. Apply the ControllerDecision to runtime state.
          4. Store the decision for routing.
        """

        controller_input = build_controller_input(state)

        initial_state = state["execution_state"]
        protocol = initial_state.protocol_visible
        if controller_input.active_step is not None:
            accepted_step(controller_input.active_plan, controller_input.active_step, controller_input.cursor)
        assessment, frozen = completion_service.evaluate(
            controller_input.identity, controller_input.active_plan, controller_input.active_step,
            controller_input.tool_execution_history, protocol.resolved_coverages,
            previous=initial_state.working.coverage_assessment, bindings=protocol.accepted_requirements,
        )
        bindings = protocol.accepted_requirements
        validation_id, validation_error = None, None
        if controller_input.planner_result is not None and controller_input.planner_result.proposed_plan is not None:
            request = controller_input.planning_request
            if request is not None and request.operation == PlanningOperation.REVISE:
                try:
                    reconciled = reconcile_revision(
                        request, controller_input.active_plan,
                        controller_input.planner_result.proposed_plan,
                    )
                    controller_input = controller_input.model_copy(update={
                        "planner_result": controller_input.planner_result.model_copy(update={
                            "proposed_plan": reconciled,
                        }),
                    })
                except RevisionRejection:
                    pass
            validation_id, validation_error, bindings = completion_service.bind_plan(
                controller_input.identity, controller_input.planner_result.proposed_plan, bindings)
        controller_input = controller_input.model_copy(update={
            "coverage_assessment": assessment, "accepted_requirements": bindings, "completion_validation_id": validation_id,
            "completion_validation_error": validation_error,
        })

        decision = controller.decide(controller_input)

        # print("\n=== CONTROLLER DECISION ===")
        # print("decision:", decision)
        #print("before:", state["execution_state"].protocol_visible)

        execution_state = apply_controller_decision_to_state(
            initial_state,
            decision,
        )
        # Freeze membership on activation and commit it with the graph transition.
        if decision.accepted_plan is None:
            bindings = protocol.accepted_requirements
        next_protocol = execution_state.protocol_visible
        if next_protocol.active_step != protocol.active_step or next_protocol.active_plan != protocol.active_plan:
            assessment, frozen = completion_service.evaluate(
                controller_input.identity, next_protocol.active_plan, next_protocol.active_step,
                controller_input.tool_execution_history, frozen,
                previous=assessment, bindings=bindings,
            )
        execution_state = execution_state.model_copy(update={
            "protocol_visible": next_protocol.model_copy(update={"resolved_coverages": frozen, "accepted_requirements": bindings}),
            "working": execution_state.working.model_copy(update={"coverage_assessment": assessment}),
        })

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

        brain_result = controller_input.brain_result

        if controller_input.brain_result is not None:
            update["brain_result"] = None

        if controller_input.planner_result is not None:
            update["planner_result"] = None

        if decision.terminal:
            previous_decision = state.get("controller_decision")
            request = _build_finalization_request(
                execution_state=execution_state,
                decision=decision,
                controller_input=controller_input,
                previous_decision=(
                    previous_decision
                    if isinstance(previous_decision, ControllerDecision)
                    else None
                ),
                brain_result=brain_result,
            )
            try:
                result = finalizer.finalize(request)
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
                    f"execution_id={request.identity.execution_id} "
                    f"status={request.status.value} "
                    f"summary={result.execution_summary.model_dump(mode='json')}"
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                final_answer = "Execution finished, but finalization failed."
                update["finalization_error"] = error
                update["final_answer"] = final_answer
                update["messages"] = [AIMessage(content=final_answer)]
                print(
                    "[finalizer] "
                    f"execution_id={request.identity.execution_id} error={error}"
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

    return controller_node


def _build_finalization_request(
    *,
    execution_state: ExecutionState,
    decision: ControllerDecision,
    controller_input: ControllerInput,
    previous_decision: ControllerDecision | None,
    brain_result,
) -> FinalizationRequest:
    """Translate an already-authorized terminal transition into finalization facts."""

    protocol = execution_state.protocol_visible
    direct_response = bool(
        decision.execution_status == ExecutionStatus.COMPLETED
        and protocol.active_plan is None
        and brain_result is not None
        and brain_result.outcome == BrainOutcome.FINAL_ANSWER
        and previous_decision is not None
        and previous_decision.direct_response
    )
    return FinalizationRequest(
        identity=protocol.identity,
        status=protocol.status,
        context=controller_input.context,
        accepted_plan=protocol.active_plan,
        tool_execution_history=controller_input.tool_execution_history,
        completed_step_ids=protocol.completed_step_ids,
        terminal_reason=decision.failure_reason or decision.reason,
        direct_response=direct_response,
        cancellation_source=protocol.cancellation_source,
    )
