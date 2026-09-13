"""Small integration tests using the production topology and node composition."""
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from core.graph import build_app
from core.graph_intents import RouterDecisionSchema
from core.protocol.bridge import build_execution_state
from core.protocol.enums import PlanningOperation, ExecutionStatus


class FakeModel:
    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, method):
        value = ({"route": "conversation"} if schema is RouterDecisionSchema
                 else {"result": "NO_PLAN_REQUIRED"})
        return SimpleNamespace(invoke=lambda messages: schema(**value))

    def invoke(self, messages):
        return AIMessage(content="Hello.")


def app_for(tmp_path):
    return build_app(
        project_root=tmp_path,
        workspace_dir=str(tmp_path / "workspace"), knowledge_dir=str(tmp_path / "knowledge"),
        rag_factory=lambda *args: SimpleNamespace(format_context=lambda **kwargs: ""),
        tool_list_factory=lambda *args: [], chat_model_factory=lambda *args: FakeModel(),
    )


def state_for():
    state = {"messages": [HumanMessage(content="hello")], "run_id": "p2-integration"}
    return {**state, "execution_state": build_execution_state(state)}


def test_production_graph_starts_at_controller_and_authorizes_create(tmp_path):
    app = app_for(tmp_path)
    edges = {(edge.source, edge.target) for edge in app.get_graph().edges}
    assert ("__start__", "controller") in edges
    assert ("__start__", "planner") not in edges
    events = list(app.stream(state_for(), {"configurable": {"thread_id": "p2-entry"}}))
    assert [next(iter(event)) for event in events] == ["controller", "planner", "controller"]
    request = events[0]["controller"]["execution_state"].protocol_visible.planning_request
    assert request.operation == PlanningOperation.CREATE
    assert request.context.user_request == "hello"
    assert request.capabilities.available_tools == ()
    final = events[-1]["controller"]["execution_state"].protocol_visible
    assert final.status == ExecutionStatus.COMPLETED
    assert final.planning_request is None


def test_checkpoint_before_planner_preserves_request_and_resumes(tmp_path):
    app = app_for(tmp_path)
    config = {"configurable": {"thread_id": "p2-checkpoint"}}
    list(app.stream(state_for(), config, interrupt_before=["planner"]))
    snapshot = app.get_state(config)
    request = snapshot.values["execution_state"].protocol_visible.planning_request
    assert request.operation == PlanningOperation.CREATE
    assert snapshot.next == ("planner",)
    events = list(app.stream(None, config))
    assert next(iter(events[0])) == "planner"
    assert events[-1]["controller"]["execution_state"].protocol_visible.planning_sequence == request.sequence
    assert events[-1]["controller"]["execution_state"].protocol_visible.planning_request is None
