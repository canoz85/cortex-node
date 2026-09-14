"""Thin LangGraph adapter for the Planner service."""

from core.graph_constants import DOMAIN_TOOL_MAP, MUTATING_TOOLS, SYSTEM_CAPABILITIES_TEXT
from core.graph_context import retrieval_message
from core.graph_authorization import require_planner_authorization
from core.planner import PlannerService
from core.planner_provider import LangChainPlannerProvider
from core.protocol.bridge import build_planner_input
from core.state import AgentState
from core.runtime.execution_driver import WorkerDispatchError


def create_planner_node(
    *, planner_llm=None, router_llm=None, rag_service, rag_top_k: int,
    tools_set: set[str], show_raw_llm: bool = False, planner_service: PlannerService | None = None,
):
    service = planner_service or PlannerService(
        provider=LangChainPlannerProvider(planner_llm=planner_llm, router_llm=router_llm, show_raw_llm=show_raw_llm),
        tools_set=tools_set, domain_tool_map=DOMAIN_TOOL_MAP,
        show_raw_llm=show_raw_llm, mutating_tools=MUTATING_TOOLS, system_capabilities_text=SYSTEM_CAPABILITIES_TEXT,
    )

    def planner_node(state: AgentState):
        # This reads/validates a durable authorization; it does not create one.
        authorized_request = require_planner_authorization(state)
        planning_request = build_planner_input(state)
        if planning_request != authorized_request:
            raise WorkerDispatchError("Planner input does not match Controller authorization")
        retrieval_messages = []

        def retrieve(user_request: str) -> tuple[str, ...]:
            # Per-invocation context assembly; never stored on the shared service.
            retrieval_messages.extend(retrieval_message(rag_service, user_request, rag_top_k))
            return tuple(message.content for message in retrieval_messages)

        result = service.run(planning_request, retrieve=retrieve)
        return {"planner_result": result, "retrieval_messages": retrieval_messages}

    return planner_node
