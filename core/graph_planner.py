"""Thin LangGraph adapter for the Planner service."""

from core.graph_authorization import require_planner_authorization
from core.graph_constants import MUTATING_TOOLS
from core.graph_context import retrieval_message
from core.planner import PlannerService
from core.planner_provider import LangChainPlannerProvider
from core.planner_routing import LangChainPlannerRouter
from core.state import AgentState
from core.logging.live_status import current_live_status


def create_planner_node(
    *,
    planner_llm=None,
    router_llm=None,
    rag_service,
    rag_top_k: int,
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
        status = current_live_status()
        if status is not None:
            prior_result = state.get("planner_result")
            router_retry = (
                authorized_request.attempt > 1
                and getattr(prior_result, "message", "").startswith(
                    "Planner router failed"
                )
            )
            status.update(
                "planner",
                (
                    f"router retry {authorized_request.attempt}/{authorized_request.max_attempts}"
                    if router_retry
                    else ""
                ),
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
            authorized_request,
            retrieve=retrieve,
        )

        return {
            "planner_result": result,
            "retrieval_messages": retrieval_messages,
        }

    return planner_node
