from langgraph.graph import END

from core.graph_state_machine import get_controller_decision, map_controller_decision
from core.state import AgentState


def route_after_controller(state: AgentState):
        
    decision = get_controller_decision(state)

    if decision is None:
        return END

    return map_controller_decision(decision)