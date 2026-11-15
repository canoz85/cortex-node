"""Run isolated fixture cases through the production graph."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
import time
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from core.brain_provider import LangChainBrainProvider
from core.graph import build_app
from core.graph_runner import run_prompt
from core.logging.live_status import LiveStatus
from core.models import ToolOutputEnvelope
from core.protocol.enums import ControllerDecisionType, ExecutionPhase, StepStatus
from core.protocol.models import (BrainInput, ExecutionContext, ExecutionCursor,
    ExecutionIdentity, ExecutionPlan, ExecutionStep, RetryMetadata)
from tools.registry import get_tool_definition

ROOT = Path(__file__).resolve().parent


class FixtureError(RuntimeError):
    pass


class DiscoveryArgs(BaseModel):
    path: str = '.'
    pattern: str = '*'


class ProviderObserver(BaseCallbackHandler):
    """Observe model starts and response metadata without changing providers."""
    def __init__(self, status, role):
        self.status, self.role = status, role
        self.calls = {'router': 0, 'planner': 0, 'brain': 0, 'finalizer': 0, 'other': 0}
        self.pending = {}
        self.tokens = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        if self.role == 'planner':
            detail = self.status.detail
            worker = 'router' if detail == 'routing' else 'planner' if detail == 'planning' else 'other'
        else:
            worker = self.status.stage if self.status.stage in ('brain', 'finalizer') else 'other'
        self.calls[worker] += 1
        self.pending[str(run_id)] = worker

    def on_llm_end(self, response, *, run_id, **kwargs):
        worker = self.pending.pop(str(run_id), None)
        if worker is None:
            return
        try:
            message = response.generations[0][0].message
            usage = getattr(message, 'usage_metadata', None) or {}
            meta = getattr(message, 'response_metadata', None) or {}
            inp = usage.get('input_tokens', meta.get('prompt_eval_count'))
            out = usage.get('output_tokens', meta.get('eval_count'))
        except (AttributeError, IndexError, TypeError):
            inp = out = None
        self.tokens.setdefault(worker, []).append(None if inp is None or out is None else
            {'input_tokens': inp, 'output_tokens': out, 'total_tokens': inp + out})

    def on_llm_error(self, error, *, run_id, **kwargs):
        worker = self.pending.pop(str(run_id), None)
        if worker:
            self.tokens.setdefault(worker, []).append(None)

    def usage(self, worker):
        values = self.tokens.get(worker, [])
        if not values or any(v is None for v in values) or len(values) != self.calls[worker]:
            return None
        return {k: sum(v[k] for v in values) for k in values[0]}


def clean(value):
    if isinstance(value, dict):
        return {k: ('[redacted]' if re.search(r'password|secret|token|api.?key|authorization', k, re.I)
                    else clean(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'(?i)(bearer\s+)[^\s]+', r'\1[redacted]', value)[:2000]
    return value


class Fixtures:
    def __init__(self, case):
        self.spec = case['tools']
        self.errors = []
        self.calls = []

    def tools(self, *_args):
        result = []
        for name, spec in self.spec.items():
            definition = get_tool_definition(name)
            if definition is None:
                raise FixtureError(f'{name}: unknown registered tool')
            if spec.get('response', {}).get('is_async_job'):
                raise FixtureError(f'{name}: asynchronous fixtures are not supported')

            def invoke(_name=name, _spec=spec, **kwargs):
                self.calls.append({'name': _name, 'arguments': clean(kwargs)})
                for key, expected in _spec.get('match', {}).items():
                    if kwargs.get(key) != expected:
                        self.errors.append(f'{_name}: argument {key} did not match fixture')
                        raise FixtureError(self.errors[-1])
                if 'response' not in _spec or sum(c['name'] == _name for c in self.calls) > _spec.get('max_calls', 1):
                    self.errors.append(f'{_name}: unexpected fixture call')
                    raise FixtureError(self.errors[-1])
                return ToolOutputEnvelope.model_validate(_spec['response']).to_tool_output()

            result.append(StructuredTool.from_function(
                func=invoke, name=name, args_schema=definition.args_schema or DiscoveryArgs,
                description=definition.planning.purpose if hasattr(definition.planning, 'purpose') else str(definition.planning)))
        return result


class Observer:
    def __init__(self, app):
        self.app = app
        self.async_runtime = app.async_runtime
        self.state = self.terminal = self.final = None
        self.error = None
        self.planner_result = None
        self.dispatches = {'planner': 0, 'brain': 0}
        self.replans = []
        self.revisions = 0

    def stream(self, *args, **kwargs):
        for event in self.app.stream(*args, **kwargs):
            for update in event.values():
                if not isinstance(update, dict):
                    continue
                self.state = update.get('execution_state') or self.state
                decision = update.get('controller_decision')
                if decision:
                    if decision.decision_type == ControllerDecisionType.AWAIT_ASYNC_JOB:
                        raise FixtureError('asynchronous execution is not supported')
                    if decision.decision_type == ControllerDecisionType.DISPATCH_PLANNER:
                        self.dispatches['planner'] += 1
                        request = decision.planning_request
                        if request and str(request.operation.value) == 'revise':
                            self.revisions += 1
                    if decision.decision_type == ControllerDecisionType.DISPATCH_BRAIN:
                        self.dispatches['brain'] += 1
                    if decision.terminal:
                        self.terminal = decision
                brain = update.get('brain_result')
                if brain and brain.replan_request:
                    self.replans.append(brain.replan_request.model_dump(mode='json'))
                self.planner_result = update.get('planner_result') or self.planner_result
                self.final = update.get('finalization_result') or self.final
                self.error = update.get('finalization_error') or self.error
            yield event


def assertion(results, name, passed, detail=None):
    results.append({'id': name, 'passed': bool(passed), 'detail': detail})


def evaluate(case, outcome, records, observer, brain_invocations=None):
    checks = []
    expect = case['expect']
    names = [r['name'] for r in records]
    assertion(checks, 'outcome', outcome == expect['outcome'], outcome)
    for name in expect.get('required_tools', []):
        assertion(checks, 'required:' + name, name in names)
    for name in expect.get('forbidden_tools', []):
        assertion(checks, 'forbidden:' + name, name not in names)
    if 'max_tool_calls' in expect:
        assertion(checks, 'max_tool_calls', len(names) <= expect['max_tool_calls'], len(names))
    if 'max_brain_dispatches' in expect:
        assertion(checks, 'max_brain_dispatches', observer.dispatches['brain'] <= expect['max_brain_dispatches'])
    if 'max_brain_invocations' in expect:
        assertion(checks, 'max_brain_invocations', brain_invocations is not None and brain_invocations <= expect['max_brain_invocations'], brain_invocations)
    for before, after in expect.get('order', []):
        assertion(checks, 'order:' + before + ':' + after,
                  before in names and after in names and names.index(before) < names.index(after))
    for rule in expect.get('argument_contains', []):
        matching = [r for r in records if r['name'] == rule['tool']]
        text = str(matching[0]['arguments'].get(rule['field'], '')).lower() if matching else ''
        assertion(checks, 'argument:' + rule['tool'] + ':' + rule['value'], rule['value'].lower() in text)
    for name in expect.get('truncated_tools', []):
        assertion(checks, 'truncated:' + name, any(r['name'] == name and r['truncated'] for r in records))
    for value in expect.get('answer_contains', []):
        assertion(checks, 'answer_contains:' + value,
                  bool(observer.final) and value.lower() in observer.final.final_answer.lower())
    if 'replanning_allowed' in expect and not expect['replanning_allowed']:
        assertion(checks, 'no_replanning', not observer.replans and not observer.revisions)
    return checks


def run_e2e(case, model, planner):
    fixtures = Fixtures(case)
    status = LiveStatus(enabled=False)
    started = time.perf_counter()
    observer = None
    error = None
    provider_observers = []
    try:
        with contextlib.nullcontext(str(ROOT / 'runs' / 'workspace' / uuid.uuid4().hex)) as temp:
            Path(temp).mkdir(parents=True, exist_ok=True)
            class EmptyRag:
                def format_context(self, **kwargs): return ''
            def model_factory(name, temperature):
                role = ('planner', 'brain', 'verifier')[len(provider_observers)]
                callback = ProviderObserver(status, role)
                provider_observers.append(callback)
                return ChatOllama(model=name, temperature=temperature, callbacks=[callback])
            app = build_app(workspace_dir=temp, knowledge_dir=temp, model=model,
                model_planner=planner, model_verifier=model,
                rag_factory=lambda *_: EmptyRag(), tool_list_factory=fixtures.tools,
                chat_model_factory=model_factory)
            observer = Observer(app)
            with contextlib.redirect_stdout(sys.stderr):
                run_prompt(observer, case['prompt'], live_status=status)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        status.stop()
    snapshot = status.command_snapshot()
    state = observer.state if observer else None
    terminal = observer.terminal if observer else None
    outcome = state.protocol_visible.status.value if state and terminal else 'error'
    if (outcome == 'failed' and terminal and not state.protocol_visible.active_plan
            and observer.planner_result and observer.planner_result.failure_category
            and observer.planner_result.failure_category.value == 'UNPLANNABLE'):
        outcome = 'planning_failed'
    records = [{'name': r.tool_name, 'arguments': clean(r.arguments), 'success': r.result.success,
                'truncated': r.result.integrity.is_truncated}
               for r in state.working.tool_execution_history] if state else []
    if fixtures.errors:
        error = '; '.join(fixtures.errors)
    if observer and observer.error:
        error = observer.error
    checks = evaluate(case, outcome, records, observer, snapshot['invocations'].get('brain', 0)) if observer else []
    # Publish counts only when callback observations cover the known Brain calls.
    po = provider_observers[0] if provider_observers else None
    bo = provider_observers[1] if len(provider_observers) > 1 else None
    planner_observed = bool(po and not po.calls['other'] and po.calls['router'] + po.calls['planner'])
    brain_observed = bool(bo and bo.calls['brain'] == snapshot['invocations'].get('brain', 0))
    result = {'case_id': case['id'], 'mode': 'end_to_end', 'outcome': outcome,
        'status': 'error' if error else ('pass' if checks and all(c['passed'] for c in checks) else 'fail'),
        'error': error, 'duration_ms': round((time.perf_counter()-started)*1000),
        'terminal_reason': (terminal.failure_reason or terminal.reason) if terminal else None,
        'tool_sequence': records, 'fixture_calls': fixtures.calls,
        'dispatches': observer.dispatches if observer else None,
        'provider_invocations': {'brain': snapshot['invocations'].get('brain', 0),
            'planner': po.calls['planner'] if planner_observed else None,
            'router': po.calls['router'] if planner_observed else None},
        'usage': {'planner': po.usage('planner') if planner_observed else None,
            'router': po.usage('router') if planner_observed else None,
            'brain': bo.usage('brain') if brain_observed else None,
            'finalizer': bo.usage('finalizer') if bo else None}, 'replan_requests': observer.replans if observer else [],
        'authorized_revisions': observer.revisions if observer else 0,
        'final_answer': observer.final.final_answer if observer and observer.final else None,
        'completion_provenance': [x.model_dump(mode='json') for x in state.protocol_visible.completion_provenance] if state else [],
        'assertions': checks}
    return result


def run_boundary(case):
    class Model:
        def bind_tools(self, tools): return self
        def invoke(self, messages): return AIMessage(content=case['serialized_action'])
    step = ExecutionStep(step_id='s1', title='Recover action', status=StepStatus.ACTIVE)
    brain_input = BrainInput(identity=ExecutionIdentity(execution_id='benchmark', protocol_version='1.0'),
        cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING, step_id='s1', plan_revision=1, controller_iteration=1),
        context=ExecutionContext(user_request='Recover intended action'),
        active_plan=ExecutionPlan(plan_id='p1', steps=(step,), available_tools=()),
        active_step=step, retry=RetryMetadata())
    started = time.perf_counter()
    outcome = LangChainBrainProvider(brain_llm=Model(), executable_tools=[]).generate(brain_input, ())
    checks = []
    assertion(checks, 'lifecycle_kind', outcome.kind.value == case['expect']['kind'], outcome.kind.value)
    request = outcome.replan_request
    assertion(checks, 'preserved_reason', bool(request) and request.reason == case['expect']['reason'])
    assertion(checks, 'preserved_constraints', bool(request) and list(request.constraints) == case['expect']['constraints'])
    return {'case_id': case['id'], 'mode': 'provider_boundary', 'status': 'pass' if all(c['passed'] for c in checks) else 'fail',
            'duration_ms': round((time.perf_counter()-started)*1000), 'assertions': checks,
            'outcome': outcome.kind.value, 'error': None}


def main():
    parser = argparse.ArgumentParser(prog='python -m benchmarks')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--model', default='gpt-oss:20b')
    run.add_argument('--planner', default=None)
    run.add_argument('--case', action='append')
    run.add_argument('--output', type=Path)
    compare = sub.add_parser('compare')
    compare.add_argument('left', type=Path)
    compare.add_argument('right', type=Path)
    args = parser.parse_args()
    if args.command == 'compare':
        left, right = (json.loads(p.read_text(encoding='utf-8')) for p in (args.left, args.right))
        if left['config'] != right['config'] or left['case_hashes'] != right['case_hashes']:
            print('Warning: configuration or case hashes differ.')
        a, b = ({r['case_id']: r for r in x['results']} for x in (left, right))
        for key in sorted(a.keys() | b.keys()):
            x, y = a.get(key, {}), b.get(key, {})
            print(f"{key}: {x.get('status', 'missing')} -> {y.get('status', 'missing')}; "
                  f"tools {[t['name'] for t in x.get('tool_sequence', [])]} -> {[t['name'] for t in y.get('tool_sequence', [])]}")
            for field in ('outcome', 'dispatches', 'provider_invocations', 'authorized_revisions', 'replan_requests', 'usage', 'duration_ms', 'assertions', 'error'):
                if x.get(field) != y.get(field):
                    print(f"  {field}: {x.get(field)} -> {y.get(field)}")
        return
    paths = sorted((ROOT / 'cases').glob('*.json'))
    cases = [(p, json.loads(p.read_text(encoding='utf-8'))) for p in paths]
    if args.case:
        cases = [(p, c) for p, c in cases if c['id'] in args.case]
        missing = set(args.case) - {c['id'] for _, c in cases}
        if missing: parser.error('unknown cases: ' + ', '.join(sorted(missing)))
    results = []
    for _, case in cases:
        print('Running', case['id'], flush=True)
        try:
            result = run_boundary(case) if case['mode'] == 'provider_boundary' else run_e2e(case, args.model, args.planner or args.model)
        except Exception as exc:
            result = {'case_id': case['id'], 'mode': case['mode'], 'status': 'error', 'error': f'{type(exc).__name__}: {exc}', 'assertions': []}
        results.append(result)
        print(f"  {result['status'].upper()} {result.get('outcome', '')} {result.get('error') or ''}", flush=True)
        for check in result['assertions']:
            if not check['passed']: print('   -', check['id'], check.get('detail'))
    output = args.output or ROOT / 'runs' / (uuid.uuid4().hex[:12] + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {'schema_version': 1, 'config': {'model': args.model, 'planner': args.planner or args.model, 'temperature': 0},
        'case_hashes': {c['id']: hashlib.sha256(p.read_bytes()).hexdigest() for p, c in cases}, 'results': results}
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for mode in ('end_to_end', 'provider_boundary'):
        group = [r for r in results if r['mode'] == mode]
        print(f"{mode}: {sum(r['status'] == 'pass' for r in group)}/{len(group)} passed")
    print('Results:', output)
    if any(r['status'] != 'pass' for r in results): sys.exit(1)
