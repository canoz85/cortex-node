from core.graph_state_machine import decide_after_brain
from core.state import AgentState


def route_after_brain(state: AgentState):
    decision = decide_after_brain(state)
    return decision.next_node