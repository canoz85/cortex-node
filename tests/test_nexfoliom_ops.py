from core.models import ToolOutputEnvelope
from tools.nexfoliom_ops import NEXFOLIOM_EVENTS_URL, get_nexfoliom_tools


def test_get_nexfoliom_events_passes_only_supplied_filters_and_preserves_rows(monkeypatch):
    import tools.nexfoliom_ops as module

    rows = [
        {"id": "a", "eventType": "ADD", "amount": 25},
        {"id": "b", "eventType": "SET_BALANCE", "amount": 125},
    ]
    payload = {"rows": rows, "truncated": False}
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(module.requests, "get", get)
    result = ToolOutputEnvelope.try_parse(get_nexfoliom_tools()[0].invoke({
        "accountId": 2, "from": "2026-01-01", "to": "2026-01-31",
    }))
    assert captured == {
        "url": NEXFOLIOM_EVENTS_URL,
        "params": {"accountId": 2, "from": "2026-01-01", "to": "2026-01-31"},
        "timeout": 10,
    }
    assert result.success is True
    assert result.data == payload
    assert result.data["rows"][0]["eventType"] == "ADD"
    assert result.data["rows"][1]["eventType"] == "SET_BALANCE"


def test_get_nexfoliom_events_rejects_missing_or_invalid_rows(monkeypatch):
    import tools.nexfoliom_ops as module

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"events": [{"id": "wrong-key"}], "truncated": False}

    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response())
    result = ToolOutputEnvelope.try_parse(get_nexfoliom_tools()[0].invoke({}))
    assert result.success is False
    assert "rows collection" in result.message


def test_get_nexfoliom_events_surfaces_http_failure_as_tool_failure(monkeypatch):
    import tools.nexfoliom_ops as module
    import requests

    def get(*args, **kwargs):
        raise requests.ConnectionError("local API unavailable")

    monkeypatch.setattr(module.requests, "get", get)
    result = ToolOutputEnvelope.try_parse(get_nexfoliom_tools()[0].invoke({}))
    assert result.success is False
    assert "local API unavailable" in result.message
