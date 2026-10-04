"""Production graph integration with deterministic structured provider exchanges.

These exercise Controller, PlannerService, provider adapters, checkpoints and
run_prompt. They prove plumbing and authority, not a live model's judgment.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from core.graph import build_app
from core.graph_runner import run_prompt
from core.planner_contract import PlannerProposal
from core.planner_routing import RouterDecisionSchema
from core.protocol.controller import CortexController
from core.protocol.enums import ExecutionStatus
from core.protocol.models import PlannerMemoryContext


class StructuredModel:
    def __init__(self, proposals):
        self.proposals = iter(proposals)
        self.exchanges = []

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, schema, *, method, include_raw):
        assert method == "json_schema" and include_raw

        def invoke(messages):
            self.exchanges.append((schema, messages))
            value = {"route": "conversation"} if schema is RouterDecisionSchema else next(self.proposals)
            return {
                "parsed": schema.model_validate(value),
                "raw": AIMessage(content=json.dumps(value)),
                "parsing_error": None,
            }

        return SimpleNamespace(invoke=invoke)


def production_app(model):
    return build_app(
        project_root=Path('.'),
        rag_factory=lambda *_args: SimpleNamespace(format_context=lambda **_kwargs: ''),
        tool_list_factory=lambda *_args: [],
        chat_model_factory=lambda *_args: model,
    )


def test_create_direct_response_uses_one_human_request_and_controller_terminal_state():
    model = StructuredModel([{"result": "NO_PLAN_REQUIRED", "message": "Hello."}])
    app = production_app(model)
    history, _ = run_prompt(app, 'hello', run_id='production-direct')
    assert [m.content for m in history] == ['hello', 'Hello.']
    protocol = app.get_state({'configurable': {'thread_id': 'production-direct'}}).values['execution_state'].protocol_visible
    assert protocol.status == ExecutionStatus.COMPLETED
    assert protocol.active_plan is None and protocol.planning_request is None
    exchange = next(messages for schema, messages in model.exchanges if schema is PlannerProposal)
    assert [m.content for m in exchange if isinstance(m, HumanMessage)] == ['hello']
    context = json.loads(exchange[-2].content.split('\n', 1)[1])
    assert context['operation'] == 'create'
    assert 'user_request' not in context['context']


def test_checkpointed_clarification_resumes_same_authorization_without_duplicate_history():
    model = StructuredModel([
        {'result': 'NEEDS_INPUT', 'message': 'What is your name?'},
        {'result': 'NO_PLAN_REQUIRED', 'message': 'Your name is Can.'},
    ])
    app = production_app(model)
    pending = []
    memory = PlannerMemoryContext()
    history, _ = run_prompt(app, 'What is my name?', run_id='production-resume',
                            planner_memory_context=memory, clarification_sink=pending)
    paused = pending[0]
    original = paused.execution_state.protocol_visible.planning_clarification.request
    assert original.context.planner_memory_context == memory
    assert original.planner_route == 'conversation'
    history, _ = run_prompt(app, 'Can', history=history, pending_clarification=paused,
                            clarification_sink=pending)
    assert pending == []
    assert [m.content for m in history] == ['What is my name?', 'Can', 'Your name is Can.']
    state = app.get_state({'configurable': {'thread_id': paused.run_id}}).values
    assert [m.content for m in state['messages'] if isinstance(m, HumanMessage)] == ['What is my name?', 'Can']
    protocol = state['execution_state'].protocol_visible
    assert protocol.identity == original.identity
    assert protocol.planning_sequence == original.sequence + 1
    assert protocol.original_user_request == 'What is my name?'
    assert protocol.clarification == 'Can'
    assert protocol.planning_clarification is None
    assert protocol.status == ExecutionStatus.COMPLETED
    assert sum(schema is RouterDecisionSchema for schema, _ in model.exchanges) == 1
    planner_exchanges = [messages for schema, messages in model.exchanges if schema is PlannerProposal]
    resumed = planner_exchanges[-1]
    assert resumed[-1].content == original.context.user_request
    context = json.loads(resumed[-2].content.split('\n', 1)[1])['context']
    assert context['clarification_question'] == 'What is your name?'
    assert context['clarification'] == 'Can'
    assert context['recent_history'] == []


def test_clarification_resume_is_independent_of_bounded_session_message_counts():
    model = StructuredModel([
        {'result': 'NEEDS_INPUT', 'message': 'What is your name?'},
        {'result': 'NEEDS_INPUT', 'message': 'Which spelling should I use?'},
    ])
    app = production_app(model)
    pending = []
    history = [HumanMessage(content=f'Prior turn {i}') for i in range(32)]
    history, _ = run_prompt(app, 'What is my name?', history=history, run_id='bounded-resume',
                            clarification_sink=pending)
    first = pending[0]
    history, _ = run_prompt(app, 'Can', history=history, pending_clarification=first,
                            clarification_sink=pending)
    assert len(pending) == 1
    marker = pending[0].execution_state.protocol_visible.planning_clarification
    assert marker.request.sequence == first.execution_state.protocol_visible.planning_sequence + 1
    assert marker.request.context.user_request == 'What is my name?'
    assert marker.request.context.clarification == 'Can'
    assert marker.request.capabilities == first.execution_state.protocol_visible.planning_clarification.request.capabilities
    assert marker.prompt == 'Which spelling should I use?'


def test_planning_requests_and_controller_decisions_have_one_production_constructor_owner():
    creators = set()
    for path in Path('core').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'PlanningRequest', 'ControllerDecision'}:
                creators.add(path.as_posix())
    assert creators == {'core/protocol/controller.py'}


def test_controller_starts_execution_without_graph_protocol_reconstruction():
    state = CortexController.start_execution('controller-start')
    assert state.protocol_visible.identity.execution_id == 'controller-start'
    assert state.protocol_visible.status == ExecutionStatus.NON_TERMINAL
    assert state.protocol_visible.planning_request is None
    assert state.working.tool_execution_history == ()


def test_resumed_planning_failure_records_original_authorized_request():
    model = StructuredModel([
        {'result': 'NEEDS_INPUT', 'message': 'What is your name?'},
        {'result': 'PLANNING_FAILED', 'message': 'Required capability is unavailable.'},
    ])
    app = production_app(model)
    pending = []
    history, _ = run_prompt(app, 'What is my name?', run_id='failed-resume',
                            clarification_sink=pending)
    evidence = []
    run_prompt(app, 'Can', history=history, pending_clarification=pending[0],
               completed_turn_evidence=evidence)
    assert len(evidence) == 1
    assert evidence[0].user_request == 'What is my name?'
    assert evidence[0].execution_id == 'failed-resume'
