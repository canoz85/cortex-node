from langgraph.graph import END

from core.graph_state_machine import apply_controller_decision_to_state, decide_after_brain, decide_controller_decision
from core.protocol.bridge import build_controller_input
from core.protocol.enums import ControllerDecisionType
from core.state import AgentState


def route_after_planner(state: AgentState):
    controller_input = build_controller_input(state)
    controller_decision = decide_controller_decision(state, controller_input=controller_input)
    apply_controller_decision_to_state(state, controller_decision)

    if controller_decision.decision_type == ControllerDecisionType.DISPATCH_BRAIN:
        return "brain"

    return END


def route_after_brain(state: AgentState):
    controller_input = build_controller_input(state)
    controller_decision = decide_controller_decision(state, controller_input=controller_input)
    apply_controller_decision_to_state(state, controller_decision)

    if controller_decision.decision_type == ControllerDecisionType.DISPATCH_TOOL_RUNTIME:
        return "tools"
    if controller_decision.decision_type == ControllerDecisionType.REQUEST_REPLAN:
        return "planner"
    if controller_decision.decision_type == ControllerDecisionType.DISPATCH_SUMMARY:
        return "summarize_memory"

    decision = decide_after_brain(state, controller_input=controller_input)
    if decision.next_node == "tools":
        return "tools"

    return END