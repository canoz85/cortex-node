"""Offset-aware clock evidence and tool-owned elapsed-time arithmetic."""

from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from conftest import get_tool, parse_result
from core.brain_provider import LangChainBrainProvider
from core.error_codes import INFO_TIME_FORMAT_ERROR, INFO_TIME_REFERENCE_INVALID
from core.models import CurrentTimeRequest
from core.planner import PlannerService
from core.protocol.enums import BrainOutcomeKind, PlannerOutcome
from core.protocol.models import ExecutionContext, PlanningCapabilities, ToolRequest
from core.runtime.tool_result_integration import SerializedToolRuntimePort
from test_brain_outcomes import FakeModel, brain_input
from test_planner_service import FakeProvider, FakeRouter, planner_input
from tools import info_ops
from tools.registry import ToolRegistry, get_tool_argument_schema, planning_capability_projection


NOW = datetime.fromisoformat("2026-10-17T01:40:46.949628+03:00")
REFERENCE = "2026-10-16T22:26:07.874141+00:00"


@pytest.fixture
def current_time():
    return get_tool(info_ops.get_info_tools("test", "workspace"), "current_time")


@pytest.fixture
def clock(monkeypatch):
    clock = Mock()
    clock.fromisoformat.side_effect = datetime.fromisoformat
    # Fix the local clock independently of the test machine's timezone.
    clock.now.return_value.astimezone.return_value = NOW
    monkeypatch.setattr(info_ops, "datetime", clock)
    return clock


def test_real_no_argument_clock_is_aware_and_utc_is_the_same_instant(current_time):
    before = datetime.now(timezone.utc)
    result = parse_result(current_time.invoke({}))
    after = datetime.now(timezone.utc)
    assert result["success"] is True
    data = result["data"]
    local = datetime.fromisoformat(data["iso"])
    utc = datetime.fromisoformat(data["utc_iso"])
    assert local.utcoffset() is not None
    assert utc.utcoffset() == timedelta(0)
    assert before <= local == utc <= after
    assert data["utc_offset"].startswith(("+", "-"))
    assert data["iso"].endswith(data["utc_offset"])
    assert data["formatted"] == local.strftime("%Y-%m-%d %H:%M:%S")
    assert "reference_time" not in data and "elapsed_seconds" not in data


@pytest.mark.parametrize("local_iso,offset,utc_iso", [
    (NOW.isoformat(), "+03:00", "2026-10-16T22:40:46.949628+00:00"),
    ("2026-10-16T18:40:46.949628-04:00", "-04:00", "2026-10-16T22:40:46.949628+00:00"),
    ("2026-10-17T04:25:46.949628+05:45", "+05:45", "2026-10-16T22:40:46.949628+00:00"),
    ("2026-10-16T22:40:46.949628+00:00", "+00:00", "2026-10-16T22:40:46.949628+00:00"),
])
def test_fixed_local_clock_contract(current_time, clock, local_iso, offset, utc_iso):
    clock.now.return_value.astimezone.return_value = datetime.fromisoformat(local_iso)
    data = parse_result(current_time.invoke({}))["data"]
    assert data == {
        "iso": local_iso, "formatted": datetime.fromisoformat(local_iso).strftime("%Y-%m-%d %H:%M:%S"),
        "format": "%Y-%m-%d %H:%M:%S", "utc_iso": utc_iso, "utc_offset": offset,
    }
    clock.now.assert_called_once_with()
    clock.now.return_value.astimezone.assert_called_once_with()


@pytest.mark.parametrize("format,expected", [
    ("%d/%m/%Y %H:%M", "17/10/2026 01:40"),
    ("%Y-%m-%d %H:%M:%S %z", "2026-10-17 01:40:46 +0300"),
    ("", ""),
])
def test_existing_custom_format_keeps_structured_timezone_evidence(current_time, clock, format, expected):
    result = parse_result(current_time.invoke({"format": format}))
    assert result["success"] is True
    assert result["data"]["formatted"] == expected
    assert result["data"]["format"] == format
    assert result["data"]["iso"] == NOW.isoformat()
    assert result["data"]["utc_iso"] == NOW.astimezone(timezone.utc).isoformat()
    assert result["data"]["utc_offset"] == "+03:00"


def test_explicit_null_reference_preserves_no_argument_behavior(current_time, clock):
    absent = parse_result(current_time.invoke({}))
    assert parse_result(current_time.invoke({"reference_time": None})) == absent


@pytest.mark.parametrize("reference,expected", [
    (REFERENCE, 879.075487),  # Original three-hour regression, including microseconds.
    ("2026-10-17T03:56:07.874141+05:30", 879.075487),
    ("2026-10-16T18:26:07.874141-04:00", 879.075487),
    ("2026-10-16T22:40:46.949628+00:00", 0.0),
    ("2026-10-16T18:40:46.949628-04:00", 0.0),
    ("2026-10-17T01:40:46.949628+03:00", 0.0),
    ("2026-10-16T23:40:46.949628+03:00", 7200.0),  # Local midnight.
    ("2026-10-15T22:40:46.949628+00:00", 86400.0),  # UTC date boundary.
    ("2026-10-16T23:40:46.949628+00:00", -3600.0),  # Future references stay signed.
    ("2026-10-16T22:26:07.874141Z", 879.075487),
    ("20261016T222607.874141+0000", 879.075487),
    ("2026-10-16T22:26:07,874141+00", 879.075487),
    ("2026-10-16T22:40+00:00", 46.949628),
])
def test_elapsed_seconds_are_deterministic(current_time, clock, reference, expected):
    result = parse_result(current_time.invoke({"reference_time": reference, "format": "%Y"}))
    assert result["success"] is True
    data = result["data"]
    assert data["iso"] == NOW.isoformat()
    assert data["reference_time"] == datetime.fromisoformat(reference).isoformat()
    assert type(data["elapsed_seconds"]) is float
    assert data["elapsed_seconds"] == expected
    assert data["formatted"] == "2026"
    clock.now.assert_called_once_with()


@pytest.mark.parametrize("now,reference", [
    ("2026-11-01T00:30:00+03:00", "2026-10-31T23:30:00+03:00"),
    ("2026-01-01T00:30:00+03:00", "2025-12-31T23:30:00+03:00"),
])
def test_elapsed_crosses_month_and_year_boundaries(current_time, clock, now, reference):
    clock.now.return_value.astimezone.return_value = datetime.fromisoformat(now)
    result = parse_result(current_time.invoke({"reference_time": reference}))
    assert result["success"] is True
    assert result["data"]["elapsed_seconds"] == 3600.0


@pytest.mark.parametrize("reference", [
    "2026-10-16T22:26:07.874141", "2026-10-16", "", "yesterday", "not-a-date",
    "2026-02-30T22:26:07+00:00", "2026-13-01T22:26:07+00:00",
    "2026-10-16T25:26:07+00:00", "2026-10-16T22:26:07+24:00",
    "2026-10-16T22:26:07+03:60", "2026-10-16T22:26:07 UTC",
    "2026-10-16X22:26:07+00:00", "2026-10-16T22:26:07+00:00 trailing",
    "2026-10-16T22:26:07+00:00\n", "2026-10-16T22.5+00:00",
])
def test_naive_and_malformed_references_fail_closed(current_time, clock, reference):
    result = parse_result(current_time.invoke({"reference_time": reference}))
    assert result["success"] is False
    assert result["error_code"] == INFO_TIME_REFERENCE_INVALID
    assert "ISO-8601" in result["message"] and "explicit UTC offset" in result["message"]
    assert result["error_details"] == {"field": "reference_time"}
    assert result["data"] is None
    clock.now.assert_not_called()


@pytest.mark.parametrize("arguments", [
    {"unexpected": "ignored?"}, {"reference_time": 123}, {"reference_time": True},
    {"reference_time": []}, {"format": None}, {"format": 123},
])
def test_invalid_argument_types_and_extra_arguments_are_rejected(current_time, clock, arguments):
    with pytest.raises(ValidationError):
        current_time.invoke(arguments)
    clock.now.assert_not_called()


def test_formatting_failure_keeps_existing_canonical_error(current_time, clock):
    local = Mock(wraps=NOW)
    local.strftime.side_effect = ValueError("invalid format")
    clock.now.return_value.astimezone.return_value = local
    result = parse_result(current_time.invoke({"format": "%Y"}))
    assert result["success"] is False
    assert result["error_code"] == INFO_TIME_FORMAT_ERROR
    assert result["error_details"] == {"format": "%Y", "exception_type": "ValueError"}


def test_registry_and_native_schema_expose_optional_reference_time(current_time):
    registry = ToolRegistry.from_tools([current_time])
    assert get_tool_argument_schema("current_time") is CurrentTimeRequest
    assert get_tool_argument_schema("current_time", registry=registry) is CurrentTimeRequest
    schema = convert_to_openai_tool(current_time)["function"]["parameters"]
    assert set(schema["properties"]) == {"format", "reference_time"}
    assert not schema.get("required")
    assert schema["properties"]["reference_time"]["default"] is None
    assert schema["properties"]["reference_time"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert CurrentTimeRequest.model_json_schema()["additionalProperties"] is False
    card, = planning_capability_projection(("current_time",))
    assert "offset-aware local system clock" in card["purpose"]
    assert card["outputs"] == (
        "iso, formatted, utc_iso, utc_offset", "when reference_time is supplied: reference_time, elapsed_seconds",
    )
    assert "explicit timezone offset" in " ".join(card["limits"])
    assert "Never infer timezone for naive timestamps" in card["limits"]
    assert "tool-calculated elapsed_seconds" in " ".join(card["limits"])


def test_elapsed_request_reaches_planner_card_and_brain_native_call(current_time, clock):
    question = f"How much time has passed since {REFERENCE}?"
    provider = FakeProvider({"result": "PLAN_PROPOSED", "objective": question, "steps": [{
        "step_id": "elapsed", "title": "Measure elapsed time", "description": question,
        "deliverable_kind": "semantic", "primary_tool": "current_time", "dependencies": [],
    }]})
    request = planner_input(context=ExecutionContext(user_request=question)).model_copy(update={
        "capabilities": PlanningCapabilities(available_tools=("current_time",)),
    })
    planned = PlannerService(provider=provider, router=FakeRouter("info"), mutating_tools=set()).run(request)
    assert planned.outcome == PlannerOutcome.EXECUTION_PLAN
    card, = planning_capability_projection(("current_time",))
    assert json.dumps(card, ensure_ascii=False, separators=(",", ":")) in provider.messages[0][0].content

    context = brain_input()
    step = planned.proposed_plan.steps[0]
    context = context.model_copy(update={
        "context": request.context, "active_step": step, "active_plan": planned.proposed_plan,
    })
    model = FakeModel(AIMessage(content="", tool_calls=[{
        "name": "current_time", "args": {"reference_time": REFERENCE}, "id": "elapsed",
    }]))
    outcome = LangChainBrainProvider(brain_llm=model, executable_tools=[current_time]).generate(context, ())
    assert outcome.kind == BrainOutcomeKind.TOOL_REQUESTED
    assert outcome.tool_request.tool_name == "current_time"
    assert outcome.tool_request.arguments == {"reference_time": REFERENCE}
    result = SerializedToolRuntimePort([current_time]).execute(outcome.tool_request)
    assert result.success is True
    assert result.data["elapsed_seconds"] == 879.075487
    assert result.data["iso"] == NOW.isoformat()
    assert result.data["utc_offset"] == "+03:00"


def test_runtime_preserves_canonical_reference_error(current_time, clock):
    result = SerializedToolRuntimePort([current_time]).execute(ToolRequest(
        request_id="naive-time", tool_name="current_time", arguments={"reference_time": "2026-10-16T22:26:07"},
    ))
    assert result.success is False
    assert result.error_code == INFO_TIME_REFERENCE_INVALID
    assert result.data is None
    clock.now.assert_not_called()
