from __future__ import annotations
from core.protocol.controller import CortexController

from core.graph_constants import MAX_REASONING_STEPS
from core.graph_state_machine import apply_controller_decision_to_state
from core.protocol.bridge import build_controller_input
from core.protocol.enums import WorkerRole, BrainOutcome
from core.state import AgentState
from core.completion import CompletionService
from core.protocol.completion_identity import accepted_step


def create_controller_node(
    controller: CortexController | None = None,
    completion_service: CompletionService | None = None,
):
    completion_service = completion_service or CompletionService()
    controller = controller or CortexController(
        max_reasoning_steps=MAX_REASONING_STEPS,
    )

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
        if (
            brain_result is not None
            and brain_result.outcome == BrainOutcome.FINAL_ANSWER
        ):
            update["final_answer"] = brain_result.final_answer

        if controller_input.brain_result is not None:
            update["brain_result"] = None

        if controller_input.planner_result is not None:
            update["planner_result"] = None


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
