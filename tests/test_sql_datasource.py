import requests
import pytest

from core.models import ToolOutputEnvelope
from datasource.nexfoliom_http import NexFolioMHttpDataSource
from tools.sql_ops import get_sql_tools


class FakeDatasource:
    def __init__(self):
        self.sql = None

    def get_schema(self):
        return {"tables": [{"name": "sample"}]}

    def query(self, sql):
        self.sql = sql
        return {"rows": [{"value": 3}], "truncated": True}


def _output(result):
    return ToolOutputEnvelope.try_parse(result)


def test_generic_schema_tool_delegates():
    source = FakeDatasource()
    result = _output(get_sql_tools(source)[0].invoke({}))
    assert result.success is True
    assert result.data == {"tables": [{"name": "sample"}]}


def test_generic_query_delegates_unchanged_and_preserves_truncated():
    source = FakeDatasource()
    sql = "SELECT  value FROM sample;  "
    result = _output(get_sql_tools(source)[1].invoke({"sql": sql}))
    assert source.sql == sql
    assert result.data == {"rows": [{"value": 3}], "truncated": True}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_nexfoliom_adapter_schema_and_query_success(monkeypatch):
    calls = []
    payloads = [{"schema": {"tables": [{"name": "x"}]}}, {"rows": [{"x": 1}], "truncated": True}]

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return FakeResponse(payloads.pop(0))

    monkeypatch.setattr("datasource.nexfoliom_http.requests.request", request)
    source = NexFolioMHttpDataSource("http://local")
    assert source.get_schema() == {"tables": [{"name": "x"}]}
    assert source.query("SELECT x") == {"rows": [{"x": 1}], "truncated": True}
    assert calls[0][:2] == ("GET", "http://local/api/data/schema")
    assert calls[1][0:2] == ("POST", "http://local/api/data/query")
    assert calls[1][2]["json"] == {"sql": "SELECT x"}


def test_nexfoliom_adapter_rejects_invalid_response(monkeypatch):
    monkeypatch.setattr("datasource.nexfoliom_http.requests.request", lambda *a, **k: FakeResponse({"rows": "bad"}))
    with pytest.raises(ValueError):
        NexFolioMHttpDataSource("http://local").query("SELECT 1")


def test_nexfoliom_adapter_surfaces_http_and_api_failures(monkeypatch):
    def api_error(*args, **kwargs):
        return FakeResponse({"success": False, "error": "denied"})

    monkeypatch.setattr("datasource.nexfoliom_http.requests.request", api_error)
    with pytest.raises(RuntimeError, match="denied"):
        NexFolioMHttpDataSource("http://local").get_schema()

    class HttpErrorResponse(FakeResponse):
        def raise_for_status(self):
            raise requests.HTTPError("503")

    monkeypatch.setattr("datasource.nexfoliom_http.requests.request", lambda *a, **k: HttpErrorResponse({}))
    with pytest.raises(RuntimeError, match="503"):
        NexFolioMHttpDataSource("http://local").get_schema()


def test_sql_query_rejects_unexpected_arguments():
    with pytest.raises(Exception):
        get_sql_tools(FakeDatasource())[1].invoke({"sql": "SELECT 1", "extra": True})
