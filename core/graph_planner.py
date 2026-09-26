"""Thin LangGraph adapter for the Planner service."""

from core.graph_authorization import require_planner_authorization
from core.graph_constants import MUTATING_TOOLS
from core.graph_context import retrieval_message
from core.planner import PlannerService
from core.planner_provider import LangChainPlannerProvider
from core.planner_routing import LangChainPlannerRouter
from core.protocol.bridge import build_planner_input
from core.protocol.models import PlannerMemoryContext
from core.runtime.execution_driver import WorkerDispatchError
from core.state import AgentState


def create_planner_node(
    *,
    planner_llm=None,
    router_llm=None,
    rag_service,
    rag_top_k: int,
    tools_set: set[str],
    show_raw_llm: bool = False,
    planner_service: PlannerService | None = None,
):
    service = planner_service or PlannerService(
        provider=LangChainPlannerProvider(
            planner_llm=planner_llm,
            show_raw_llm=show_raw_llm,
        ),
        router=LangChainPlannerRouter(router_llm=router_llm, show_raw_llm=show_raw_llm),
        mutating_tools=MUTATING_TOOLS,
        show_raw_llm=show_raw_llm,
    )

    def planner_node(state: AgentState):
        authorized_request = require_planner_authorization(state)
        planner_input = build_planner_input(state)

        if planner_input != authorized_request:
            raise WorkerDispatchError(
                "Planner input does not match Controller authorization"
            )

        memory_context = state.get("planner_memory_context")

        if isinstance(memory_context, PlannerMemoryContext):
            planner_input = planner_input.model_copy(
                update={
                    "context": planner_input.context.model_copy(
                        update={
                            "planner_memory_context": memory_context,
                        }
                    ),
                }
            )

        retrieval_messages = []

        def retrieve(user_request: str) -> tuple[str, ...]:
            messages = retrieval_message(
                rag_service,
                user_request,
                rag_top_k,
            )
            retrieval_messages.extend(messages)

            return tuple(
                message.content
                for message in messages
            )

        result = service.run(
            planner_input,
            retrieve=retrieve,
        )

        return {
            "planner_result": result,
            "retrieval_messages": retrieval_messages,
        }

    return planner_node
