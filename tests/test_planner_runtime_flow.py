"""Production graph integration with deterministic structured provider exchanges.

These exercise Controller, PlannerService, provider adapters, checkpoints and
run_prompt. They prove plumbing and authority, not a live model's judgment.
"""

import json
from core.graph import build_app
from core.graph_runner import run_prompt
from core.models import ToolOutputEnvelope
from core.planner_contract import PlannerProposal
from core.planner_routing import RouterDecisionSchema
from core.protocol.controller import CortexController
from core.protocol.enums import ExecutionStatus, PlanningOperation
from core.protocol.models import PlannerMemoryContext
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pathlib import Path
from types import SimpleNamespace


class StructuredModel:
    def __init__(self, proposals, route="conversation"):
        self.proposals = iter(proposals)
        self.exchanges = []
        self.route = route

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, schema, *, method, include_raw):
        assert method == "json_schema" and include_raw

        def invoke(messages):
            self.exchanges.append((schema, messages))
            value = {"route": self.route} if schema is RouterDecisionSchema else next(self.proposals)
            return {
                "parsed": schema.model_validate(value),
                "raw": AIMessage(content=json.dumps(value)),
                "parsing_error": None,
            }

        return SimpleNamespace(invoke=invoke)


def production_app(model, tools=()):
    return build_app(
        project_root=Path('.'),
        rag_factory=lambda *_args: SimpleNamespace(format_context=lambda **_kwargs: ''),
        tool_list_factory=lambda *_args: list(tools),
        chat_model_factory=lambda *_args: model,
    )


def test_executable_runtime_flow_with_scripted_provider():
    tool_calls = []

    @tool
    def read_file(path: str) -> str:
        """Read a workspace file."""
        tool_calls.append(path)
        return ToolOutputEnvelope(success=True, message="Read", data={"content": "hello"}).to_tool_output()

    class Model(StructuredModel):
        def invoke(self, messages):
            if any("Active step:" in m.content for m in messages):
                if any(isinstance(m, ToolMessage) for m in messages) or any("Execution evidence v1:" in m.content for m in messages):
                    return AIMessage(content="", tool_calls=[{"name": "brain_step_completed",
                        "args": {"message": "Read a.txt: hello"}, "id": "completed"}])
                return AIMessage(content="", tool_calls=[{"name": "read_file",
                    "args": {"path": "a.txt"}, "id": "read"}])
            return AIMessage(content="The file says hello.")

    model = Model([{"result": "PLAN_PROPOSED", "objective": "Read a.txt", "steps": [{
        "step_id": "read", "title": "Read a.txt", "description": "Read a.txt and report its content",
        "primary_tool": "read_file", "dependencies": [],
    }]}], route="info")
    app = production_app(model, [read_file])
    history, _ = run_prompt(app, "Read a.txt", run_id="runtime-executable")
    protocol = app.get_state({"configurable": {"thread_id": "runtime-executable"}}).values["execution_state"].protocol_visible
    assert protocol.status == ExecutionStatus.COMPLETED
    assert protocol.completed_step_ids == ("read",)
    assert protocol.active_plan.steps[0].primary_tool == "read_file"
    assert protocol.completion_provenance[0].tool_request_ids
    assert tool_calls == ["a.txt"]
    assert [m.content for m in history] == ["Read a.txt", "The file says hello."]
    assert sum(issubclass(schema, PlannerProposal) for schema, _ in model.exchanges) == 1


def test_create_direct_response_uses_one_human_request_and_controller_terminal_state_with_scripted_provider():
    model = StructuredModel([{"result": "NO_PLAN_REQUIRED", "message": "Hello."}])
    app = production_app(model)
    history, _ = run_prompt(app, 'hello', run_id='production-direct')
    assert [m.content for m in history] == ['hello', 'Hello.']
    protocol = app.get_state({'configurable': {'thread_id': 'production-direct'}}).values['execution_state'].protocol_visible
    assert protocol.status == ExecutionStatus.COMPLETED
    assert protocol.active_plan is None and protocol.planning_request is None
    exchange = next(messages for schema, messages in model.exchanges if issubclass(schema, PlannerProposal))
    assert [m.content for m in exchange if isinstance(m, HumanMessage)] == ['hello']
    context = json.loads(exchange[-2].content.split('\n', 1)[1])
    assert context['operation'] == 'create'
    assert 'user_request' not in context['context']


def test_checkpointed_clarification_resumes_same_authorization_without_duplicate_history_with_scripted_provider():
    model = StructuredModel([
        {'result': 'NEEDS_INPUT', 'message': 'What is your name?'},
        {'result': 'NO_PLAN_REQUIRED', 'message': 'Your name is Can.'},
    ])
    app = production_app(model)
    pending = []
    memory = PlannerMemoryContext()
    prior_history = [HumanMessage(content=f'Prior turn {i}') for i in range(32)]
    history, _ = run_prompt(app, 'What is my name?', history=prior_history, run_id='production-resume',
                            planner_memory_context=memory, clarification_sink=pending)
    paused = pending[0]
    original = paused.execution_state.protocol_visible.planning_clarification.request
    assert original.context.planner_memory_context == memory
    assert original.planner_route == 'conversation'
    history, _ = run_prompt(app, 'Can', history=history, pending_clarification=paused,
                            clarification_sink=pending)
    assert pending == []
    assert [m.content for m in history][-3:] == ['What is my name?', 'Can', 'Your name is Can.']
    state = app.get_state({'configurable': {'thread_id': paused.run_id}}).values
    user_turns = [m.content for m in state['messages'] if isinstance(m, HumanMessage)]
    assert user_turns[-2:] == ['What is my name?', 'Can']
    assert len(user_turns) == len(set(user_turns)) == 17
    protocol = state['execution_state'].protocol_visible
    assert protocol.identity == original.identity
    assert protocol.planning_sequence == original.sequence + 1
    assert protocol.original_user_request == 'What is my name?'
    assert protocol.clarification == 'Can'
    assert protocol.planning_clarification is None
    assert protocol.status == ExecutionStatus.COMPLETED
    assert sum(schema is RouterDecisionSchema for schema, _ in model.exchanges) == 1
    planner_exchanges = [messages for schema, messages in model.exchanges if issubclass(schema, PlannerProposal)]
    resumed = planner_exchanges[-1]
    assert resumed[-1].content == original.context.user_request
    context = json.loads(resumed[-2].content.split('\n', 1)[1])['context']
    assert context['clarification_question'] == 'What is your name?'
    assert context['clarification'] == 'Can'
    assert context['recent_history'] == list(original.context.recent_history)
    assert len(context['recent_history']) == len(set(context['recent_history']))


def test_runtime_checkpoint_preserves_authorization_with_scripted_provider():
    app = production_app(StructuredModel([{"result": "NO_PLAN_REQUIRED", "message": "Hello."}]))
    config = {"configurable": {"thread_id": "runtime-checkpoint"}}
    list(app.stream({"messages": [HumanMessage(content="hello")], "execution_state": CortexController.start_execution("checkpoint")}, config, interrupt_after=["controller"]))
    snapshot = app.get_state(config)
    request = snapshot.values["execution_state"].protocol_visible.planning_request
    assert request.operation == PlanningOperation.CREATE
    assert snapshot.next == ("controller",)
    assert snapshot.values["planner_result"].request_id == request.request_id
    events = list(app.stream(None, config))
    assert next(iter(events[0])) == "controller"
    assert events[-1]["controller"]["execution_state"].protocol_visible.planning_sequence == request.sequence
    assert events[-1]["controller"]["execution_state"].protocol_visible.planning_request is None
