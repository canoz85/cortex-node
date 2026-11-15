import json
from types import SimpleNamespace

import pytest

from benchmarks.runner import Fixtures, FixtureError, ROOT, evaluate, run_boundary


def test_fixture_rejects_unexpected_calls():
    fixtures = Fixtures({'tools': {'sql_query': {}}})
    tool, = fixtures.tools()
    with pytest.raises(FixtureError):
        tool.invoke({'sql': 'SELECT 1'})
    assert fixtures.errors == ['sql_query: unexpected fixture call']


def test_assertions_require_partial_order_and_correct_arguments():
    case = {'expect': {'outcome': 'completed', 'order': [['current_time', 'sql_query']],
                       'argument_contains': [{'tool': 'sql_query', 'field': 'sql', 'value': '2026-05-15'}]}}
    observer = SimpleNamespace(replans=[], revisions=0)
    records = [{'name': 'sql_query', 'arguments': {'sql': 'select 1'}},
               {'name': 'current_time', 'arguments': {}}]
    checks = evaluate(case, 'completed', records, observer)
    assert [x['passed'] for x in checks] == [True, False, False]


def test_lifecycle_case_recovers_constraints():
    case = json.loads((ROOT / 'cases' / 'lifecycle_recovery.json').read_text(encoding='utf-8'))
    assert run_boundary(case)['status'] == 'pass'
