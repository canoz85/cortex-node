import json
import os

import pytest

from core.models import ToolOutputEnvelope


@pytest.fixture(autouse=True)
def isolate_raw_llm_logging(monkeypatch, request):
    """Scripted exchanges must never append to the developer's raw log.

    Live logging is opt-in through --live-raw-llm-file; logging tests explicitly
    set their own temporary destination after this fixture runs.
    """
    # The logger has a production default even when the variable is absent.
    # A null destination disables file logging without changing production code.
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", os.devnull)
    live_path = request.config.getoption("--live-raw-llm-file")
    if live_path and "live" in request.node.path.parts:
        monkeypatch.setenv("CORTEX_RAW_LLM_FILE", live_path)


def pytest_addoption(parser):
    parser.addoption("--live-raw-llm-file", default=None,
                     help="Explicit raw exchange log destination for tests/live only")


def get_tool(tools: list, name: str):
    for t in tools:
        if getattr(t, "name", "") == name:
            return t
    raise AssertionError(f"Tool not found: {name}")


def parse_result(raw: str) -> dict:
    _, payload = ToolOutputEnvelope.split_tool_output(raw)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Could not parse tool output as JSON: {raw}") from exc
    assert isinstance(parsed, dict), f"Expected dict payload, got: {type(parsed).__name__}"
    return parsed
