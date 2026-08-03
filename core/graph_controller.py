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

        # print("\n===== CONTROLLER INPUT =====")
        # print("worker :", controller_input.cursor.current_worker)
        # print("planner:", controller_input.planner_result)
        # print("brain  :", controller_input.brain_result)
        # print("tool   :", controller_input.tool_result)
        # print("============================")

        decision = controller.decide(controller_input)

        execution_state = apply_controller_decision_to_state(
            state["execution_state"],
            decision,
        )

        update = {
            "execution_state": execution_state,
            "controller_decision": decision,
        }

        if decision.clear_last_tool_result:
            update.update(
                {
                    "last_tool_result": None,
                    "last_tool_output": "",
                    "last_tool_success": None,
                    "last_tool_signature": "",
                    "last_tool_rendered": "",
                }
            )

        brain_result = state.get("brain_result")
        if (
            brain_result is not None
            and brain_result.outcome == BrainOutcome.FINAL_ANSWER
        ):
            update["final_answer"] = brain_result.final_answer


        print("\n====CONTROLLER====:")
        print("current_worker:", execution_state.protocol_visible.cursor.current_worker)
        print("---------------")
        if execution_state.protocol_visible.cursor.current_worker == WorkerRole.PLANNER:
            print("active_plan:", execution_state.protocol_visible.active_plan)
            print("---------------")
        print("active_step:", execution_state.protocol_visible.active_step)
        print("---------------")
        if execution_state.protocol_visible.cursor.current_worker == WorkerRole.BRAIN:
            print("brain_result:", controller_input.brain_result)
            print("---------------")
        print("====END CONTROLLER====\n")


        return update

    return controller_node