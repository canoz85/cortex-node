"""Shadow audit results never participate in Controller authorization or state."""

import json
import threading
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from core.debug import log_llm_exchange
from core.finalizer import Finalizer
from core.protocol.controller import CortexController
from core.protocol.enums import BrainOutcome, ExecutionPhase, ExecutionStatus, StepStatus
from core.protocol.models import (
    BrainResult, ControllerInput, ExecutionContext, ExecutionCursor,
    ExecutionIdentity, ExecutionPlan, ExecutionState, ExecutionStep,
    ProtocolVisibleState, ToolExecutionRecord, ToolResult, WorkingState,
    ExactCollection,
)
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver
from core.shadow_verifier import (
    ShadowEvidenceVerifier, VERIFIER_PROMPT, parse_verifier_status, select_verifier_evidence,
)


def record(request_id, **scope):
    return ToolExecutionRecord(
        execution_id='exec', plan_id='plan', plan_revision=2, step_id='s1',
        tool_name='read_file', result=ToolResult(request_id=request_id, success=True,
                                                message='Read events',
                                                data={'content': 'Event A: blue. Event B: red.'}),
    ).model_copy(update=scope)


def active_state(records=()):
    step = ExecutionStep(step_id='s1', title='Inspect events',
                         description='Report the colors of the two events.', status=StepStatus.ACTIVE)
    return ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=ExecutionIdentity(execution_id='exec', protocol_version='1'),
        cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING, step_id='s1', plan_revision=2),
        active_plan=ExecutionPlan(plan_id='plan', revision=2, steps=(step,)), active_step=step,
    ), working=WorkingState(tool_execution_history=records))


def request(state, result):
    p = state.protocol_visible
    return ControllerInput(identity=p.identity, cursor=p.cursor,
                           context=ExecutionContext(user_request='Inspect the events'),
                           active_plan=p.active_plan, active_step=p.active_step,
                           completed_step_ids=p.completed_step_ids, retry=p.retry,
                           tool_execution_history=state.working.tool_execution_history,
                           brain_result=result)


def driver(verifier=None, controller=None):
    return ExecutionDriver(coordinator=ControllerCoordinator(controller or CortexController(24)),
                           planner=None, brain=None, tool_runtime=None, finalizer=Finalizer(),
                           shadow_verifier=verifier)


def complete(state, verifier=None):
    execution = driver(verifier)
    proposal = BrainResult(outcome=BrainOutcome.STEP_COMPLETED, step_id='s1',
                           message='Event A is blue. Event B is red.')
    accepted = execution.turn(state, request(state, proposal), dispatch_worker=False)
    terminal = execution.turn(accepted.execution_state, request(accepted.execution_state,
        BrainResult(outcome=BrainOutcome.FINAL_ANSWER, message='Ready for finalization.')))
    return proposal, accepted, terminal


@pytest.mark.parametrize('text,status', [
    ('PASS', 'pass'), ('\n \nPASS\nignored prose', 'pass'),
    ('SUSPICIOUS\n- Event details were mixed.', 'suspicious'),
    ('SUSPICIOUS', 'suspicious'), ('pass', 'unavailable'),
    ('PASS because it is supported', 'unavailable'), ('', 'unavailable'),
    ('```\nPASS\n```', 'unavailable'), ('{"status":"PASS"}', 'unavailable'),
    ('Explanation\nPASS', 'unavailable'), (None, 'unavailable'), ([], 'unavailable'),
])
def test_first_nonempty_line_is_the_only_machine_status(text, status):
    assert parse_verifier_status(text) == status


@pytest.mark.parametrize('response,expected', [
    ('PASS', 'pass'), ('SUSPICIOUS\n- Unsupported factual claim.', 'suspicious'),
    ('```\nPASS\n```', 'unavailable'), (OSError('offline'), 'unavailable'),
])
def test_audit_logged_without_changing_accepted_completion_or_execution(monkeypatch, response, expected):
    logs, messages = [], []
    logged = threading.Event()

    def log(**kwargs):
        logs.append(kwargs)
        logged.set()

    def invoke(value):
        messages.extend(value)
        if isinstance(response, Exception):
            raise response
        return AIMessage(content=response, usage_metadata={
            'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12})

    monkeypatch.setattr('core.shadow_verifier.log_llm_exchange', log)
    verifier = ShadowEvidenceVerifier(lambda: SimpleNamespace(invoke=invoke),
                                      model='local-test', provider='ollama')
    state = active_state((record('r1'),))
    before = state.model_dump_json()
    baseline = complete(state)
    observed = complete(state, verifier)
    assert logged.wait(2)
    assert observed == baseline
    assert state.model_dump_json() == before
    proposal, accepted, terminal = observed
    assert accepted.execution_state.protocol_visible.completion_provenance[0].summary == proposal.message
    assert terminal.execution_state.protocol_visible.status == ExecutionStatus.COMPLETED
    assert terminal.worker_result.execution_summary.status == ExecutionStatus.COMPLETED
    assert terminal.execution_state.protocol_visible.retry == baseline[2].execution_state.protocol_visible.retry
    assert len(logs) == 1
    assert logs[0]['invocation']['verifier_status'] == expected
    assert logs[0]['execution_id'] == 'exec'
    assert logs[0]['invocation']['step_id'] == 's1'
    assert messages[0].content == VERIFIER_PROMPT
    payload = json.loads(messages[1].content)
    assert payload['accepted_completion'] == proposal.message
    assert payload['active_step']['description'] == 'Report the colors of the two events.'


def test_evidence_selection_filters_discovery_but_preserves_support_and_execution(monkeypatch):
    captured, finished = [], threading.Event()
    records = (
        record('other-execution', execution_id='foreign'), record('other-plan', plan_id='foreign'),
        record('old-revision', plan_revision=1), record('other-step', step_id='s2'),
        *tuple(record(f'r{i}') for i in range(35)),
    )
    records = (*records, record('discovery', tool_name='list_files', arguments={'path': 'unrelated'}),
        record('preparation', tool_name='install_package', arguments={'package': 'unrelated'}),
        record('failed-attempt', tool_name='find_files').model_copy(update={
        'result': ToolResult(request_id='failed-attempt', success=False, message='Not available'),
    }))

    def invoke(messages):
        captured.append(json.loads(messages[1].content))
        return AIMessage(content='PASS')

    monkeypatch.setattr('core.shadow_verifier.log_llm_exchange', lambda **k: finished.set())
    verifier = ShadowEvidenceVerifier(lambda: SimpleNamespace(invoke=invoke), model='test', provider='ollama')
    state = active_state(records)
    baseline = complete(state)
    observed = complete(state, verifier)
    assert observed == baseline
    assert finished.wait(2)
    selected = captured[0]['execution_evidence']
    assert [r['result']['request_id'] for r in selected] == [f'r{i}' for i in range(35)]
    assert all(r['result']['success'] for r in selected)
    assert 'failed-attempt' in observed[1].execution_state.protocol_visible.completion_provenance[0].tool_request_ids


def test_selection_retains_primary_evidence_even_when_it_contradicts_completion():
    records = (
        record('support', tool_name='read_file'),
        record('contradiction', tool_name='read_file').model_copy(update={
            'result': ToolResult(request_id='contradiction', success=True,
                                 message='Read another event', data={'content': 'Event A is green.'}),
        }),
        record('unrelated-discovery', tool_name='list_files'),
        record('unrelated-other-tool', tool_name='current_time'),
        record('cited-resource', tool_name='web_fetch', arguments={'url': 'https://example.test/events'}),
    )
    state = active_state(records)
    _, accepted, _ = complete(state)
    binding = accepted.execution_state.protocol_visible.completion_provenance[0].model_copy(update={
        'summary': 'Event A is blue. Source: https://example.test/events',
    })
    step = state.protocol_visible.active_step.model_copy(update={'primary_tool': 'read_file'})
    selected = select_verifier_evidence(records, binding, step)
    assert [r.result.request_id for r in selected] == ['support', 'contradiction', 'cited-resource']


def test_explicit_failure_citation_and_exact_collection_source_survive_filtering():
    records = (
        record('failed-discovery', tool_name='find_files').model_copy(update={
            'result': ToolResult(request_id='failed-discovery', success=False,
                                 message='Directory inaccessible'),
        }),
        record('unrelated-failure', tool_name='find_files').model_copy(update={
            'result': ToolResult(request_id='unrelated-failure', success=False, message='No match'),
        }),
        record('exact-discovery', tool_name='list_files'),
        record('support'),
    )
    state = active_state(records)
    _, accepted, _ = complete(state)
    binding = accepted.execution_state.protocol_visible.completion_provenance[0].model_copy(update={
        'summary': 'Event A is blue; failed-discovery could not access another directory.',
        'exact_collection': ExactCollection(source_record_index=2, source_request_id='exact-discovery', items=()),
    })
    step = state.protocol_visible.active_step.model_copy(update={'primary_tool': 'read_file'})
    selected = select_verifier_evidence(records, binding, step)
    assert [r.result.request_id for r in selected] == ['failed-discovery', 'exact-discovery', 'support']


def test_discovery_is_retained_when_it_is_the_step_deliverable():
    records = (record('listing', tool_name='list_files'),)
    state = active_state(records)
    _, accepted, _ = complete(state)
    binding = accepted.execution_state.protocol_visible.completion_provenance[0]
    step = state.protocol_visible.active_step.model_copy(update={'primary_tool': 'list_files'})
    assert select_verifier_evidence(records, binding, step) == records


def test_verifier_not_scheduled_before_acceptance_or_for_rejected_completion(monkeypatch):
    verifier = ShadowEvidenceVerifier(lambda: pytest.fail('Unexpected model call'), model='test', provider='ollama')
    queued = []
    monkeypatch.setattr(verifier, '_pending', SimpleNamespace(put=queued.append))
    verifier._started = True
    state = active_state()
    proposal = BrainResult(outcome=BrainOutcome.STEP_COMPLETED, step_id='s1', message='done')
    value = request(state, proposal)
    verifier.observe(state, state, value)
    assert queued == []

    class Rejection(CortexController):
        def _completion_permitted(self, value):
            assert queued == []
            return False

    rejected = driver(verifier, Rejection(24)).transition(state, value)
    assert rejected.decision.reason == 'completion_coverage_rejected'
    assert rejected.execution_state.protocol_visible.completion_provenance == ()
    assert queued == []
    accepted = driver(verifier).transition(state, value)
    assert len(queued) == 1
    assert queued[0][1] in accepted.execution_state.protocol_visible.completion_provenance
    verifier.observe(accepted.execution_state, accepted.execution_state, value)
    assert len(queued) == 1


def test_hung_verifier_cannot_delay_terminal_execution_or_start_more_model_calls(monkeypatch):
    started, release, logged = threading.Event(), threading.Event(), threading.Event()
    logs, calls = [], []

    def invoke(messages):
        calls.append(messages)
        started.set()
        release.wait(3)
        return AIMessage(content='PASS')

    def log(**kwargs):
        logs.append(kwargs)
        logged.set()

    monkeypatch.setattr('core.shadow_verifier.log_llm_exchange', log)
    verifier = ShadowEvidenceVerifier(lambda: SimpleNamespace(invoke=invoke),
                                      model='test', provider='ollama', timeout=.05)
    try:
        _, _, terminal = complete(active_state(), verifier)
        assert terminal.execution_state.protocol_visible.status == ExecutionStatus.COMPLETED
        assert started.wait(1) and not release.is_set()
        assert logged.wait(1)
        assert logs[0]['invocation']['reason'] == 'timeout'
        assert logs[0]['invocation']['verifier_status'] == 'unavailable'
        logged.clear()
        complete(active_state(), verifier)
        assert logged.wait(1)
        assert logs[-1]['invocation']['reason'] == 'provider_still_busy'
        assert len(calls) == 1
    finally:
        release.set()


def test_audit_scheduling_failure_cannot_change_execution():
    def fail(*args):
        raise RuntimeError('thread creation failed')
    state = active_state()
    assert complete(state, SimpleNamespace(observe=fail)) == complete(state)


def test_existing_raw_log_keeps_brain_record_and_verifier_usage(tmp_path, monkeypatch):
    path = tmp_path / 'raw.jsonl'
    monkeypatch.setenv('CORTEX_RAW_LLM_FILE', str(path))
    log_llm_exchange(worker='brain', operation='completion', messages=[],
                     response=AIMessage(content='Original Brain response'), execution_id='exec')
    original = path.read_text(encoding='utf-8')
    logged = threading.Event()

    def log(**kwargs):
        log_llm_exchange(**kwargs)
        logged.set()

    monkeypatch.setattr('core.shadow_verifier.log_llm_exchange', log)
    response = AIMessage(content='SUSPICIOUS\n- Event details mixed.',
                         response_metadata={'model': 'local-test'},
                         usage_metadata={'input_tokens': 11, 'output_tokens': 4, 'total_tokens': 15})
    verifier = ShadowEvidenceVerifier(lambda: SimpleNamespace(invoke=lambda _: response),
                                      model='local-test', provider='ollama')
    complete(active_state(), verifier)
    assert logged.wait(2)
    raw = path.read_text(encoding='utf-8')
    assert raw.startswith(original)
    audit = json.loads(raw.splitlines()[1])
    assert audit['worker'] == 'shadow_verifier'
    assert audit['response']['content'] == response.content
    assert audit['invocation']['verifier_status'] == 'suspicious'
    assert audit['invocation']['provider'] == 'ollama'
    assert audit['usage']['total_tokens'] == 15


def test_ollama_verifier_has_a_separate_plain_text_client(monkeypatch):
    import langchain_ollama

    constructed = []

    class Model:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.model = kwargs['model']
            self.base_url = kwargs.get('base_url')
            self.client_kwargs = kwargs.get('client_kwargs')

    monkeypatch.setattr(langchain_ollama, 'ChatOllama', Model)
    source = Model(model='local-model', base_url='http://localhost:11434',
                   client_kwargs={'timeout': 90})
    verifier = ShadowEvidenceVerifier.for_ollama(source)
    assert verifier.timeout == 45
    assert len(constructed) == 1  # Lazy: no provider setup on the execution path.
    client = verifier.model_factory()
    assert client is not source
    assert source.client_kwargs == {'timeout': 90}
    assert constructed[-1] == {
        'model': 'local-model', 'base_url': 'http://localhost:11434',
        'temperature': 0, 'num_predict': 512, 'client_kwargs': {'timeout': 45},
    }


def test_tool_call_output_is_unavailable_even_with_pass_text(monkeypatch):
    logs, logged = [], threading.Event()

    def log(**kwargs):
        logs.append(kwargs)
        logged.set()

    monkeypatch.setattr('core.shadow_verifier.log_llm_exchange', log)
    response = AIMessage(content='PASS', tool_calls=[{'name': 'unexpected', 'args': {}, 'id': '1'}])
    verifier = ShadowEvidenceVerifier(lambda: SimpleNamespace(invoke=lambda _: response),
                                      model='test', provider='ollama')
    _, _, terminal = complete(active_state(), verifier)
    assert logged.wait(2)
    assert logs[0]['invocation']['verifier_status'] == 'unavailable'
    assert logs[0]['response'] is response
    assert terminal.execution_state.protocol_visible.status == ExecutionStatus.COMPLETED


def test_log_failure_cannot_change_execution_or_kill_audit_consumer(monkeypatch):
    logged = threading.Event()
    calls = []

    def log(**kwargs):
        calls.append(kwargs)
        logged.set()
        raise OSError('audit disk unavailable')

    monkeypatch.setattr('core.shadow_verifier.log_llm_exchange', log)
    verifier = ShadowEvidenceVerifier(lambda: SimpleNamespace(invoke=lambda _: AIMessage(content='PASS')),
                                      model='test', provider='ollama')
    for _ in range(2):
        logged.clear()
        _, _, terminal = complete(active_state(), verifier)
        assert terminal.execution_state.protocol_visible.status == ExecutionStatus.COMPLETED
        assert logged.wait(2)
    assert len(calls) == 2


def test_evidence_binding_mismatch_is_unavailable_without_provider_call(monkeypatch):
    state = active_state((record('r1'),))
    proposal, accepted, _ = complete(state)
    binding = accepted.execution_state.protocol_visible.completion_provenance[0]
    value = request(state, proposal).model_copy(update={'tool_execution_history': ()})
    logs = []
    monkeypatch.setattr('core.shadow_verifier.log_llm_exchange', lambda **kwargs: logs.append(kwargs))
    verifier = ShadowEvidenceVerifier(lambda: pytest.fail('Must not verify a different evidence set'),
                                      model='test', provider='ollama')
    verifier._verify(value, binding)
    assert logs[0]['invocation']['verifier_status'] == 'unavailable'
    assert logs[0]['invocation']['reason'] == 'evidence_binding_mismatch'


def test_audit_does_not_inherit_execution_usage_context(monkeypatch):
    from core.logging.live_status import LiveStatus, current_live_status

    observed, logged = [], threading.Event()

    def invoke(messages):
        observed.append(current_live_status())
        return AIMessage(content='PASS')

    monkeypatch.setattr('core.shadow_verifier.log_llm_exchange', lambda **kwargs: logged.set())
    verifier = ShadowEvidenceVerifier(lambda: SimpleNamespace(invoke=invoke), model='test', provider='ollama')
    status = LiveStatus(enabled=False)
    status.start('brain')
    try:
        complete(active_state(), verifier)
        assert logged.wait(2)
        assert observed == [None]
        assert status.total_tokens == 0
        assert status.provider_invocations_by_worker == {}
    finally:
        status.stop()
