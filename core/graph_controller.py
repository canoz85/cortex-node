from __future__ import annotations
from core.protocol.controller import CortexController

from core.graph_constants import MAX_REASONING_STEPS
from core.graph_state_machine import apply_controller_decision_to_state
from core.protocol.bridge import build_controller_input
from core.protocol.enums import WorkerRole, BrainOutcome
from core.state import AgentState


def create_controller_node(
    controller: CortexController | None = None,
):
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

        decision = controller.decide(controller_input)

        print("\n=== CONTROLLER DECISION ===")
        print("decision:", decision)
        print("before:", state["execution_state"].protocol_visible)

        execution_state = apply_controller_decision_to_state(
            state["execution_state"],
            decision,
        )

        print("\n=== AFTER APPLY ===")
        print("cursor:", execution_state.protocol_visible.cursor)
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