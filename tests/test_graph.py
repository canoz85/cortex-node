from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from core.graph import _load_sap_system_prompt, build_app
from core.protocol.enums import ControllerDecisionType, ExecutionPhase, ExecutionStatus, WorkerRole
from core.protocol.models import (
    ControllerDecision,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionState,
    ProtocolVisibleState,
    ToolRequest,
    WorkingState,
)


class DummyTool:
    def __init__(self, name: str):
        self.name = name


class FakeChatModel:
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature

    def bind_tools(self, tools):
        return {
            "kind": "bound",
            "model": self.model,
            "temperature": self.temperature,
            "tools": list(tools),
        }


class InvokeOnlyNode:
    def __init__(self, handler):
        self._handler = handler

    def invoke(self, state):
        return self._handler(state)


def _sample_execution_state(execution_id: str = "run-1") -> ExecutionState:
    return ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=ExecutionIdentity(
                execution_id=execution_id,
                protocol_version="1.0",
            ),
            status=ExecutionStatus.NON_TERMINAL,
            cursor=ExecutionCursor(phase=ExecutionPhase.INITIALIZING),
        ),
        working=WorkingState(),
    )


def test_load_sap_system_prompt_returns_none_when_file_missing(tmp_path):
    assert _load_sap_system_prompt(tmp_path) is None


def test_build_app_uses_injected_factories(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "systemprompts_sap.md").write_text("custom sap prompt", encoding="utf-8")

    call_log: dict[str, object] = {}

    def rag_factory(knowledge_root: Path, embedding_model: str, rag_top_k: int):
        call_log["rag"] = (knowledge_root, embedding_model, rag_top_k)
        return {"kind": "rag-service", "knowledge_root": str(knowledge_root)}

    def tool_list_factory(workspace_root: str, knowledge_root: str, rag_service, model: str):
        call_log["tools"] = (workspace_root, knowledge_root, rag_service, model)
        return [DummyTool("list_files")]

    def chat_model_factory(model: str, temperature: float):
        return FakeChatModel(model, temperature)

    def graph_nodes_factory(**kwargs):
        call_log["graph_nodes_kwargs"] = kwargs

        def controller_node(_state):
            return {}

        def planner_node(_state):
            return {}

        def brain_node(_state):
            return {}

        def capture_tool_output_node(_state):
            return {}

        def summarize_memory_node(_state):
            return {}

        return controller_node, planner_node, brain_node, capture_tool_output_node, summarize_memory_node

    def tool_node_factory(tools):
        call_log["tool_node_tools"] = list(tools)

        def _tool_node(_state):
            return {}

        return _tool_node

    app = build_app(
        workspace_dir=str(tmp_path / "workspace"),
        knowledge_dir=str(tmp_path / "knowledge"),
        model="test-model",
        model_planner="planner-model",
        embedding_model="embed-model",
        rag_top_k=7,
        rag_factory=rag_factory,
        tool_list_factory=tool_list_factory,
        chat_model_factory=chat_model_factory,
        graph_nodes_factory=graph_nodes_factory,
        tool_node_factory=tool_node_factory,
        project_root=tmp_path,
    )

    assert app is not None

    rag_call = call_log["rag"]
    assert rag_call[1] == "embed-model"
    assert rag_call[2] == 7

    graph_nodes_kwargs = call_log["graph_nodes_kwargs"]
    assert graph_nodes_kwargs["sap_system_prompt"] == "custom sap prompt"
    assert graph_nodes_kwargs["rag_top_k"] == 7
    assert graph_nodes_kwargs["tools_set"] == {"list_files"}

    llm = graph_nodes_kwargs["tool_brain_llm"]
    brain_llm = graph_nodes_kwargs["brain_llm"]
    planner_llm = graph_nodes_kwargs["planner_llm"]
    assert llm["kind"] == "bound"
    assert llm["model"] == "test-model"
    assert brain_llm.model == "test-model"
    assert planner_llm.model == "planner-model"

    tool_node_tools = call_log["tool_node_tools"]
    assert len(tool_node_tools) == 1
    assert tool_node_tools[0].name == "list_files"


def test_build_app_propagates_same_execution_state_across_graph_nodes(tmp_path):
    observed: dict[str, list[ExecutionState]] = {
        "controller": [],
        "planner": [],
        "brain": [],
    }

    def tool_list_factory(_workspace_root: str, _knowledge_root: str, _rag_service, _model: str):
        return [DummyTool("noop")]

    def chat_model_factory(model: str, temperature: float):
        return FakeChatModel(model, temperature)

    def graph_nodes_factory(**_kwargs):
        controller_calls = 0

        def controller_node(state):
            nonlocal controller_calls
            observed["controller"].append(state["execution_state"])
            controller_calls += 1
            if controller_calls == 1:
                decision = ControllerDecision(
                    decision_type=ControllerDecisionType.DISPATCH_PLANNER,
                    next_worker=WorkerRole.PLANNER,
                )
            elif controller_calls == 2:
                decision = ControllerDecision(
                    decision_type=ControllerDecisionType.DISPATCH_BRAIN,
                    next_worker=WorkerRole.BRAIN,
                )
            else:
                decision = ControllerDecision(
                    decision_type=ControllerDecisionType.TERMINATE,
                    reason="test complete",
                    execution_status=ExecutionStatus.COMPLETED,
                    cursor=ExecutionCursor(phase=ExecutionPhase.COMPLETED),
                    terminal=True,
                )
            return {"controller_decision": decision}

        def planner_node(state):
            observed["planner"].append(state["execution_state"])
            return {}

        def brain_node(state):
            observed["brain"].append(state["execution_state"])
            return {}

        def capture_tool_output_node(_state):
            return {}

        def summarize_memory_node(_state):
            return {}

        return controller_node, planner_node, brain_node, capture_tool_output_node, summarize_memory_node

    def tool_node_factory(_tools):
        def _tool_node(state):
            observed["tools"].append(state["execution_state"])
            return {"last_tool_output": "ok"}

        return _tool_node

    app = build_app(
        workspace_dir=str(tmp_path / "workspace"),
        knowledge_dir=str(tmp_path / "knowledge"),
        chat_model_factory=chat_model_factory,
        tool_list_factory=tool_list_factory,
        graph_nodes_factory=graph_nodes_factory,
        tool_node_factory=tool_node_factory,
        project_root=tmp_path,
    )

    execution_state = _sample_execution_state()
    result = app.invoke({
        "messages": [HumanMessage(content="start")],
        "steps": 0,
        "execution_state": execution_state,
    })

    assert result["execution_state"] is execution_state
    assert observed["controller"] == [execution_state, execution_state, execution_state]
    assert observed["planner"] == [execution_state]
    assert observed["brain"] == [execution_state]


def test_build_app_supports_invoke_only_tool_nodes(tmp_path):
    observed: dict[str, list[ExecutionState]] = {
        "tools": [],
        "capture": [],
    }
    execution_state = _sample_execution_state()
    request = ToolRequest(request_id="noop-1", tool_name="noop")
    authorized_execution_state = execution_state.model_copy(update={
        "protocol_visible": execution_state.protocol_visible.model_copy(update={
            "pending_tool_request": request,
        }),
    })

    def tool_list_factory(_workspace_root: str, _knowledge_root: str, _rag_service, _model: str):
        return [DummyTool("noop")]

    def chat_model_factory(model: str, temperature: float):
        return FakeChatModel(model, temperature)

    def graph_nodes_factory(**_kwargs):
        controller_calls = 0

        def controller_node(_state):
            nonlocal controller_calls
            controller_calls += 1
            if controller_calls == 1:
                return {
                    "execution_state": authorized_execution_state,
                    "controller_decision": ControllerDecision(
                        decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
                        next_worker=WorkerRole.TOOL_RUNTIME,
                        pending_tool_request=request,
                    ),
                    "messages": [AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "noop", "args": {},
                            "id": request.request_id, "type": "tool_call",
                        }],
                    )],
                }
            return {"controller_decision": ControllerDecision(
                decision_type=ControllerDecisionType.TERMINATE,
                reason="test complete",
                execution_status=ExecutionStatus.COMPLETED,
                cursor=ExecutionCursor(phase=ExecutionPhase.COMPLETED),
                terminal=True,
            )}

        def planner_node(_state):
            return {}

        def brain_node(_state):
            return {}

        def capture_tool_output_node(state):
            observed["capture"].append(state["execution_state"])
            return {"last_tool_signature": "capture"}

        def summarize_memory_node(_state):
            return {}

        return controller_node, planner_node, brain_node, capture_tool_output_node, summarize_memory_node

    def tool_node_factory(_tools):
        return InvokeOnlyNode(lambda state: observed["tools"].append(state["execution_state"]) or {"last_tool_output": "ok"})

    app = build_app(
        workspace_dir=str(tmp_path / "workspace"),
        knowledge_dir=str(tmp_path / "knowledge"),
        chat_model_factory=chat_model_factory,
        tool_list_factory=tool_list_factory,
        graph_nodes_factory=graph_nodes_factory,
        tool_node_factory=tool_node_factory,
        project_root=tmp_path,
    )

    result = app.invoke({
        "messages": [HumanMessage(content="start")],
        "steps": 0,
        "execution_state": execution_state,
    })

    assert observed["tools"] == [authorized_execution_state]
    assert observed["capture"] == [authorized_execution_state]
    assert result["execution_state"] is authorized_execution_state
