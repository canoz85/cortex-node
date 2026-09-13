from langgraph.graph import END

from core.protocol.enums import (
    ControllerDecisionType,
)
from core.protocol.models import ControllerDecision
from core.state import AgentState


def get_controller_decision(state: AgentState) -> ControllerDecision | None:
    decision = state.get("controller_decision")

    if isinstance(decision, ControllerDecision):
        return decision

    return None

def map_controller_decision(
    decision: ControllerDecision,
) -> str:
    """Translate a protocol ControllerDecision into a LangGraph node.

    This function is the only place that knows LangGraph node names.
    The protocol layer must never depend on graph topology.
    """

    match decision.decision_type:

        case ControllerDecisionType.DISPATCH_PLANNER:
            return "planner"

        case ControllerDecisionType.DISPATCH_BRAIN:
            return "brain"

        case ControllerDecisionType.DISPATCH_TOOL_RUNTIME:
            return "tools"

        case ControllerDecisionType.DISPATCH_SUMMARY:
             return END #return "summarize_memory" todo commented for debugging, we are not using summarize_memory node

        case ControllerDecisionType.AWAIT_ASYNC_JOB:
            return END

        case ControllerDecisionType.PAUSE:
            return END

        case ControllerDecisionType.CANCEL:
            return END

        case ControllerDecisionType.TERMINATE:
            return END

    raise ValueError(
        f"Unsupported controller decision: {decision.decision_type}"
    )
