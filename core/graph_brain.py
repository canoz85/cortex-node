"""Thin LangGraph adapter for the typed Brain service."""

from langchain_core.messages import AIMessage

from core.brain import BrainService
from core.brain_provider import LangChainBrainProvider
from core.graph_node_helpers import response_with_usage
from core.protocol.bridge import build_brain_input, build_execution_state, with_cursor
from core.protocol.enums import WorkerRole
from core.state import AgentState


def create_brain_node(
    *, brain_llm, tool_brain_llm, agent_system_prompt: str,
    final_answer_system_prompt: str, step_completed_system_prompt: str,
    casual_system_prompt: str, tools_set: set[str], show_raw_llm: bool,
    brain_service: BrainService | None = None,
    supports_native_tool_calls: bool = True,
):
    # The checker prompt remains a construction compatibility argument. The
    # accepted runtime uses the active-step prompt to assess cumulative evidence.
    service = brain_service or BrainService(
        provider=LangChainBrainProvider(
            brain_llm=brain_llm, tool_brain_llm=tool_brain_llm,
            tools_set=tools_set, show_raw_llm=show_raw_llm,
            supports_native_tool_calls=supports_native_tool_calls,
        ),
        agent_system_prompt=agent_system_prompt,
        final_answer_system_prompt=final_answer_system_prompt,
        casual_system_prompt=casual_system_prompt,
    )

    def brain_node(state: AgentState):
        brain_input = build_brain_input(state)
        outcome = service.run(brain_input)
        execution_state = with_cursor(build_execution_state(state), current_worker=WorkerRole.BRAIN)
        if brain_input.last_tool_result is not None:
            execution_state = execution_state.model_copy(update={
                "working": execution_state.working.model_copy(update={"last_tool_result": None}),
            })
        # Recreate provider transport only outside the service boundary. ToolNode
        # sees exactly the domain request Controller will accept, with the same ID.
        request = outcome.tool_request
        response = AIMessage(
            content=outcome.final_answer or outcome.message,
            tool_calls=[{
                "name": request.tool_name, "args": request.arguments,
                "id": request.request_id, "type": "tool_call",
            }] if request is not None else [],
            response_metadata={
                "prompt_tokens": outcome.usage.prompt_tokens,
                "completion_tokens": outcome.usage.completion_tokens,
            },
        )
        return {
            "brain_result": outcome, "execution_state": execution_state,
            **response_with_usage(state, response),
        }

    return brain_node
