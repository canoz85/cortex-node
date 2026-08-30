import json

import pytest

import tools.comfy_ops as comfy_ops
from core.protocol.enums import AsyncJobStatus
from conftest import get_tool, parse_result
from tools.comfy_ops import get_comfy_tools


class FakeHttpResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _tool(tools, name: str):
    return get_tool(tools, name)


def test_run_comfy_workflow_reports_submission_evidence(monkeypatch, tmp_path):
    submitted_payloads = []

    def fake_urlopen(request, **_kwargs):
        submitted_payloads.append(json.loads(request.data.decode("utf-8")))
        return FakeHttpResponse({"prompt_id": "prompt-1"})

    monkeypatch.setattr(
        comfy_ops.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    run_workflow = _tool(get_comfy_tools(str(tmp_path)), "run_comfy_workflow")

    result = parse_result(run_workflow.invoke({"workflow_json": {"1": {}}}))

    assert result["success"] is True
    assert result["prompt_id"] == "prompt-1"
    assert result["is_async_job"] is True
    assert result["async_job_id"] == "prompt-1"
    assert result["async_job_status"] == AsyncJobStatus.SUBMITTED.value
    assert result["async_terminal"] is False
    assert submitted_payloads == [{
        "prompt": {"1": {}},
        "client_id": "cortex_node_agent",
    }]


def test_run_comfy_workflow_preserves_preallocated_id_on_ambiguous_failure(
    monkeypatch,
    tmp_path,
):
    def fail_connection(*_args, **_kwargs):
        raise comfy_ops.urllib.error.URLError("timed out")

    monkeypatch.setattr(comfy_ops.urllib.request, "urlopen", fail_connection)
    run_workflow = _tool(get_comfy_tools(str(tmp_path)), "run_comfy_workflow")

    result = parse_result(run_workflow.invoke({
        "workflow_json": {"1": {}},
        "prompt_id": "preallocated-prompt-1",
        "client_id": "run-1",
    }))

    assert result["success"] is False
    assert result["prompt_id"] == "preallocated-prompt-1"
    assert result["async_job_id"] == "preallocated-prompt-1"
    assert result["async_job_status"] == AsyncJobStatus.UNKNOWN.value
    assert result["data"] == {"submission_outcome": "ambiguous"}


def test_get_comfy_history_treats_missing_job_as_nonterminal_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(
        comfy_ops.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeHttpResponse({}),
    )
    history = _tool(get_comfy_tools(str(tmp_path)), "get_comfy_history")

    result = parse_result(history.invoke({"prompt_id": "prompt-1"}))

    assert result["success"] is True
    assert result["async_job_status"] == AsyncJobStatus.UNKNOWN.value
    assert result["async_terminal"] is False
    assert result["data"] == {"provider_visibility": "absent"}


def test_get_comfy_history_reconciles_queue_before_reporting_absence(
    monkeypatch,
    tmp_path,
):
    def fake_urlopen(request, **_kwargs):
        if request.full_url.endswith("/queue"):
            return FakeHttpResponse({
                "queue_running": [],
                "queue_pending": [[1, "prompt-1", {}, {}, []]],
            })
        return FakeHttpResponse({})

    monkeypatch.setattr(comfy_ops.urllib.request, "urlopen", fake_urlopen)
    history = _tool(get_comfy_tools(str(tmp_path)), "get_comfy_history")

    result = parse_result(history.invoke({"prompt_id": "prompt-1"}))

    assert result["success"] is True
    assert result["async_job_status"] == AsyncJobStatus.RUNNING.value
    assert result["data"] == {"provider_visibility": "queue_pending"}


@pytest.mark.parametrize(
    (
        "provider_status",
        "provider_completed",
        "expected_status",
        "expected_success",
        "expected_terminal",
    ),
    [
        ("running", False, AsyncJobStatus.RUNNING, True, False),
        ("success", True, AsyncJobStatus.COMPLETED, True, True),
        ("error", True, AsyncJobStatus.FAILED, False, True),
        ("error", False, AsyncJobStatus.FAILED, False, True),
        ("cancelled", True, AsyncJobStatus.CANCELLED, False, True),
    ],
)
def test_get_comfy_history_normalizes_provider_lifecycle(
    monkeypatch,
    tmp_path,
    provider_status,
    provider_completed,
    expected_status,
    expected_success,
    expected_terminal,
):
    monkeypatch.setattr(
        comfy_ops.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeHttpResponse(
            {
                "prompt-1": {
                    "outputs": {"9": {"images": [{"filename": "result.png"}]}},
                    "status": {
                        "completed": provider_completed,
                        "status_str": provider_status,
                    },
                }
            }
        ),
    )
    history = _tool(get_comfy_tools(str(tmp_path)), "get_comfy_history")

    result = parse_result(history.invoke({"prompt_id": "prompt-1"}))

    assert result["success"] is expected_success
    assert result["async_job_id"] == "prompt-1"
    assert result["async_job_status"] == expected_status.value
    assert result["async_terminal"] is expected_terminal
