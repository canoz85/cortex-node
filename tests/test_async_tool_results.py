from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from core.models import ToolResult as TransportToolResult
from core.protocol.enums import AsyncJobStatus
from core.protocol.models import ToolResult as ProtocolToolResult


@pytest.mark.parametrize(
    ("result_type", "extra_fields"),
    [
        (TransportToolResult, {}),
        (ProtocolToolResult, {"request_id": "request-1"}),
    ],
)
def test_sync_tool_results_remain_valid(result_type, extra_fields):
    result = result_type(
        success=True,
        message="Tool completed.",
        **extra_fields,
    )

    assert result.is_async_job is False
    assert result.async_job_id is None
    assert result.async_job_status is None
    assert result.async_terminal is False


@pytest.mark.parametrize(
    ("result_type", "extra_fields"),
    [
        (TransportToolResult, {}),
        (ProtocolToolResult, {"request_id": "request-1"}),
    ],
)
@pytest.mark.parametrize(
    ("status", "terminal"),
    [
        (AsyncJobStatus.SUBMITTED, False),
        (AsyncJobStatus.RUNNING, False),
        (AsyncJobStatus.UNKNOWN, False),
        (AsyncJobStatus.COMPLETED, True),
        (AsyncJobStatus.FAILED, True),
        (AsyncJobStatus.CANCELLED, True),
    ],
)
def test_async_tool_results_accept_valid_status_and_terminality(result_type, extra_fields, status, terminal):
    observed_at = datetime.now(timezone.utc)
    result = result_type(
        success=status not in {AsyncJobStatus.FAILED, AsyncJobStatus.CANCELLED},
        message="Async job observation.",
        is_async_job=True,
        async_job_id="comfy-prompt-1",
        async_job_status=status,
        async_terminal=terminal,
        async_observed_at_utc=observed_at,
        **extra_fields,
    )

    assert result.async_job_status == status
    assert result.async_terminal is terminal
    assert result.async_observed_at_utc == observed_at


@pytest.mark.parametrize(
    ("result_type", "extra_fields"),
    [
        (TransportToolResult, {}),
        (ProtocolToolResult, {"request_id": "request-1"}),
    ],
)
@pytest.mark.parametrize(
    "async_fields",
    [
        {"async_job_id": "comfy-prompt-1"},
        {
            "is_async_job": True,
            "async_job_status": AsyncJobStatus.RUNNING,
            "async_terminal": False,
        },
        {
            "is_async_job": True,
            "async_job_id": "comfy-prompt-1",
            "async_terminal": False,
        },
        {
            "is_async_job": True,
            "async_job_id": "comfy-prompt-1",
            "async_job_status": AsyncJobStatus.RUNNING,
            "async_terminal": True,
        },
        {
            "is_async_job": True,
            "async_job_id": "comfy-prompt-1",
            "async_job_status": AsyncJobStatus.COMPLETED,
            "async_terminal": False,
        },
    ],
)
def test_async_tool_results_reject_inconsistent_field_combinations(result_type, extra_fields, async_fields):
    with pytest.raises(ValidationError):
        result_type(
            success=True,
            message="Invalid async job observation.",
            **extra_fields,
            **async_fields,
        )