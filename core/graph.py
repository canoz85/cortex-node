from pathlib import Path

from typing import Any, Callable


from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.state import END
from langgraph.prebuilt import ToolNode

from core.graph_constants import CASUAL_SYSTEM_PROMPT_TEMPLATE, FINAL_ANSWER_SYSTEM_PROMPT, MAX_REASONING_STEPS, SYSTEM_PROMPT_TEMPLATE, STEP_COMPLETED_SYSTEM_PROMPT
from core.graph_nodes import create_graph_nodes
from core.graph_routing import  route_after_controller
from core.graph_runner import run_prompt
from core.rag import WorkspaceRAG
from core.runtime.async_poller import CheckpointedGraphApp, LocalAsyncPollingRuntime
from core.runtime.gpu_resources import (
    GpuResourcePolicy,
    RuntimeGpuObserver,
)
from core.runtime.state_propagation import propagate_execution_state
from core.state import AgentState
from tools.comfy_ops import get_comfy_tools
from tools.exec_ops import get_exec_tools
from tools.file_ops import get_file_tools
from tools.git_ops import get_git_tools
from tools.info_ops import get_info_tools
from tools.rag_ops import get_rag_tools
from tools.sap_ops import get_sap_tools
from tools.scada_ops import get_scada_tools
from tools.vision_ops import get_vision_tools


ToolListFactory = Callable[[str, str, WorkspaceRAG, str], list[Any]]
ChatModelFactory = Callable[[str, float], Any]
RAGFactory = Callable[[Path, str, int], WorkspaceRAG]
StateNodeCallable = Callable[[AgentState], Any]
StatePropagator = Callable[
    [AgentState, Any],
    Any,
]

_STATE_PROPAGATORS: tuple[StatePropagator, ...] = (
    propagate_execution_state,
)


def _default_rag_factory(knowledge_root: Path, embedding_model: str, rag_top_k: int) -> WorkspaceRAG:
    return WorkspaceRAG(
        knowledge_root,
        embed_model=embedding_model,
        top_k=rag_top_k,
    )


def _default_tool_list_factory(workspace_root: str, knowledge_root: str, rag_service: WorkspaceRAG, model: str) -> list[Any]:
    return [
        *get_file_tools(workspace_root, knowledge_dir=knowledge_root),
        *get_exec_tools(workspace_root),
        *get_git_tools(workspace_root),
        *get_info_tools(model=model, workspace_dir=workspace_root),
        *get_rag_tools(rag_service),
        *get_sap_tools(workspace_root),
        *get_scada_tools(workspace_root),
        *get_vision_tools(workspace_root),
        *get_comfy_tools(workspace_root),
    ]


def _default_chat_model_factory(model: str, temperature: float) -> Any:
    return ChatOllama(model=model, temperature=temperature)


def _load_sap_system_prompt(project_root: Path) -> str | None:
    """Load SAP-specific system prompt from prompts folder if present."""
    sap_prompt_path = project_root / "prompts" / "systemprompts_sap.md"
    if not sap_prompt_path.exists():
        return None
    content = sap_prompt_path.read_text(encoding="utf-8").strip()
    return content or None


def _apply_state_propagators(state, node_update):

    propagated_update = node_update

    for propagator in _STATE_PROPAGATORS:
        propagated_update = propagator(state, propagated_update)

    return propagated_update


def _invoke_state_node(node: Any, state: AgentState) -> Any:
    if callable(node):
        return node(state)

    invoke = getattr(node, "invoke", None)
    if callable(invoke):
        return invoke(state)

    raise TypeError(f"State node '{getattr(node, 'name', type(node).__name__)}' is not executable")


def _runtime_observation_fields(state: AgentState) -> dict[str, object]:
    fields: dict[str, object] = {}

    run_id = state.get("run_id")
    if isinstance(run_id, str) and run_id:
        fields["run_id"] = run_id

    execution_state = state.get("execution_state")
    protocol_visible = getattr(execution_state, "protocol_visible", None)
    identity = getattr(protocol_visible, "identity", None)
    cursor = getattr(protocol_visible, "cursor", None)
    pending_tool_request = getattr(protocol_visible, "pending_tool_request", None)

    execution_id = getattr(identity, "execution_id", None)
    if isinstance(execution_id, str) and execution_id:
        fields["execution_id"] = execution_id

    step_id = getattr(cursor, "step_id", None)
    if isinstance(step_id, str) and step_id:
        fields["step_id"] = step_id

    tool_name = getattr(pending_tool_request, "tool_name", None)
    if isinstance(tool_name, str) and tool_name:
        fields["tool_name"] = tool_name

    controller_decision = state.get("controller_decision")
    async_job_id = getattr(controller_decision, "async_job_id", None)
    if isinstance(async_job_id, str) and async_job_id:
        fields["async_job_id"] = async_job_id

    return fields


def _register_state_node(
    workflow: StateGraph,
    name: str,
    node: StateNodeCallable | Any,
    *,
    resource_observer: RuntimeGpuObserver | None = None,
) -> None:
    def _state_node(state: AgentState) -> Any:
        if (
            resource_observer is not None
            and resource_observer.should_observe_graph_node(name)
        ):
            with resource_observer.observe_operation(
                component="graph_node",
                operation=name,
                fields=_runtime_observation_fields(state),
            ):
                result = _invoke_state_node(node, state)
        else:
            result = _invoke_state_node(node, state)
        return _apply_state_propagators(state, result)

    workflow.add_node(name, _state_node)


def _build_tool_transport_state(state: AgentState) -> AgentState | None:
    execution_state = state.get("execution_state")
    protocol_visible = getattr(execution_state, "protocol_visible", None)
    pending_tool_request = getattr(protocol_visible, "pending_tool_request", None)
    pending_request_id = getattr(pending_tool_request, "request_id", None)

    if not isinstance(pending_request_id, str) or not pending_request_id:
        return None

    messages = state.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return None

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return None

    tool_calls = getattr(last_message, "tool_calls", None)
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    first_call = tool_calls[0]
    if not isinstance(first_call, dict):
        return None

    tool_calls_copy = [dict(call) if isinstance(call, dict) else call for call in tool_calls]

    tool_calls_copy[0]["id"] = pending_request_id
    tool_calls_copy[0]["name"] = pending_tool_request.tool_name
    tool_calls_copy[0]["args"] = dict(pending_tool_request.arguments)

    copied_last_message = last_message.model_copy(
        update={"tool_calls": tool_calls_copy}
    )
    messages_copy = [*messages[:-1], copied_last_message]

    return {
        **state,
        "messages": messages_copy,
    }


def _wrap_tool_node_for_protocol_request_id(tool_node: Any) -> StateNodeCallable:
    def _tool_node_adapter(state: AgentState) -> Any:
        transport_state = _build_tool_transport_state(state)
        if transport_state is None:
            return _invoke_state_node(tool_node, state)

        return _invoke_state_node(tool_node, transport_state)

    return _tool_node_adapter


def build_app(
    workspace_dir: str = "workspace",
    model: str = "gpt-oss:20b",
    model_planner: str = "gpt-oss:20b",
    knowledge_dir: str = "knowledge",
    embedding_model: str = "nomic-embed-text",
    rag_top_k: int = 4,
    rag_factory: RAGFactory = _default_rag_factory,
    tool_list_factory: ToolListFactory = _default_tool_list_factory,
    chat_model_factory: ChatModelFactory = _default_chat_model_factory,
    graph_nodes_factory: Callable[..., tuple[Any, Any, Any, Any, Any]] = create_graph_nodes,
    tool_node_factory: Callable[[list[Any]], Any] = ToolNode,
    project_root: Path | None = None,
    show_raw_llm: bool = False,
    checkpointer_factory: Callable[[], Any] = InMemorySaver,
    gpu_resource_policy: GpuResourcePolicy | None = None,
) -> CheckpointedGraphApp:
    app_root = project_root or Path(__file__).resolve().parents[1]
    workspace_root = Path(workspace_dir).resolve()
    workspace_root_str = str(workspace_root)
    knowledge_root = Path(knowledge_dir).resolve()

    rag_service = rag_factory(knowledge_root, embedding_model, rag_top_k)
    sap_system_prompt = _load_sap_system_prompt(app_root)

    tools = tool_list_factory(workspace_root_str, str(knowledge_root), rag_service, model)
    tools_set = {getattr(tool, "name", "") for tool in tools}

    # Format the available tools into a clean, scannable string block
    tools_list_str = "\n".join([f"- {name}" for name in sorted(tools_set) if name])

    agent_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        model=model,
        workspace_dir=workspace_root_str,
        knowledge_dir=str(knowledge_root),
        max_steps=MAX_REASONING_STEPS,
        available_tools=tools_list_str,
    )

    casual_system_prompt = CASUAL_SYSTEM_PROMPT_TEMPLATE

    step_completed_system_prompt = STEP_COMPLETED_SYSTEM_PROMPT
    final_answer_system_prompt = FINAL_ANSWER_SYSTEM_PROMPT

    planner_llm = chat_model_factory(model_planner, 0)
    brain_llm = chat_model_factory(model, 0)
    tool_brain_llm = chat_model_factory(model, 0).bind_tools(tools)

    controller_node, planner_node, brain_node, capture_tool_output_node, summarize_memory_node = graph_nodes_factory(
        brain_llm=brain_llm,
        tool_brain_llm=tool_brain_llm,
        planner_llm=planner_llm,
        rag_service=rag_service,
        rag_top_k=rag_top_k,
        agent_system_prompt=agent_system_prompt,
        step_completed_system_prompt=step_completed_system_prompt,
        final_answer_system_prompt=final_answer_system_prompt,
        casual_system_prompt=casual_system_prompt,
        sap_system_prompt=sap_system_prompt,
        tools_set=tools_set,
        show_raw_llm=show_raw_llm,
    )

    resource_observer = (
        RuntimeGpuObserver(policy=gpu_resource_policy)
        if gpu_resource_policy is not None
        and gpu_resource_policy.telemetry_enabled
        else None
    )

    workflow = StateGraph(AgentState)
    wrapped_tool_node = _wrap_tool_node_for_protocol_request_id(tool_node_factory(tools))
    _register_state_node(
        workflow,
        "planner",
        planner_node,
        resource_observer=resource_observer,
    )
    _register_state_node(
        workflow,
        "controller",
        controller_node,
        resource_observer=resource_observer,
    )
    _register_state_node(
        workflow,
        "brain",
        brain_node,
        resource_observer=resource_observer,
    )
    _register_state_node(
        workflow,
        "tools",
        wrapped_tool_node,
        resource_observer=resource_observer,
    )
    _register_state_node(
        workflow,
        "capture_tool_output",
        capture_tool_output_node,
        resource_observer=resource_observer,
    )
    _register_state_node(
        workflow,
        "summarize_memory",
        summarize_memory_node,
        resource_observer=resource_observer,
    )

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "controller")
    workflow.add_conditional_edges("controller", route_after_controller)

    workflow.add_edge("tools", "capture_tool_output")
    workflow.add_edge("capture_tool_output", "controller")
    workflow.add_edge("brain", "controller")
    workflow.add_edge("summarize_memory", END)

    compiled_graph = workflow.compile(
        checkpointer=checkpointer_factory(),
    )
    async_runtime = LocalAsyncPollingRuntime(
        compiled_graph=compiled_graph,
        tools=tools,
        resource_observer=resource_observer,
    )
    return CheckpointedGraphApp(
        compiled_graph=compiled_graph,
        async_runtime=async_runtime,
    )
