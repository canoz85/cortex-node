"""Final rendering has a bounded input and cannot change execution results."""
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from core.finalizer import Finalizer
from core.finalizer_provider import LangChainFinalAnswerRenderer, finalizer_facts, _bounded_value
from core.protocol.enums import ExecutionStatus
from core.protocol.models import FinalizationRequest, ToolExecutionRecord, ToolResult
from test_finalizer import IDENTITY, CONTEXT, _plan


class Model:
    def __init__(self, response):
        self.response, self.calls = response, []

    def invoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def request(records=()):
    return FinalizationRequest(identity=IDENTITY, status=ExecutionStatus.COMPLETED,
        context=CONTEXT, accepted_plan=_plan(), completed_step_ids=("step-1", "step-2"),
        terminal_reason="final_answer", tool_execution_history=records)


def record(step, data, rendered=""):
    return ToolExecutionRecord(step_id=step, tool_name="inspect", arguments={"target": step},
        result=ToolResult(request_id=step, success=True, message="Inspected", data=data,
                          rendered_output=rendered))


def render(req, response, enabled=False):
    model = Model(response)
    result = Finalizer(answer_renderer=LangChainFinalAnswerRenderer(llm=model, show_raw_llm=enabled),
                       show_raw_llm=enabled).finalize(req)
    return result, model.calls


def test_multi_step_success_renders_from_evidence_once_without_mutation():
    req = request((record("step-1", {"status": "modified"}), record("step-2", {"changes": "updated protocol"})))
    before = req.model_dump_json()
    result, calls = render(req, AIMessage(content="The repository has protocol updates."))
    assert result.final_answer == "The repository has protocol updates."
    assert result.final_answer_error is None
    assert result.execution_summary.status == ExecutionStatus.COMPLETED
    assert result.execution_summary.completed_step_ids == ("step-1", "step-2")
    assert len(calls) == 1
    assert "modified" in calls[0][-1].content and "updated protocol" in calls[0][-1].content
    assert req.model_dump_json() == before


def test_oversized_duplicate_evidence_is_bounded_and_marked_before_model_invocation():
    req = request((record("step-1", {"status": "modified"}),
                   record("step-2", {"diff": "HEAD" + "x" * 150000 + "TAIL"}, "DUPLICATE" * 30000)))
    before = req.model_dump_json()
    result, calls = render(req, AIMessage(content="Visible evidence shows modifications."))
    prompt = "\n".join(m.content for m in calls[0])
    assert len(prompt.encode("utf-8")) < 25000
    assert "DUPLICATE" not in prompt
    facts = json.loads(calls[0][-1].content.split("\n", 1)[1])
    assert facts["tool_execution_history"][1]["truncated"] is True
    assert "HEAD" in facts["tool_execution_history"][1]["excerpt"]
    assert "TAIL" in facts["tool_execution_history"][1]["excerpt"]
    assert req.model_dump_json() == before
    assert result.execution_summary == Finalizer().finalize(req).execution_summary
    assert result.final_answer_error is None


def test_many_records_have_explicit_omission_and_global_budget():
    req = request(tuple(record(f"s{i}", {"value": "x" * 10000}) for i in range(100)))
    summary = Finalizer().finalize(req).execution_summary
    facts = finalizer_facts(req, summary)
    assert facts["omitted_earlier_records"] == 76
    assert len(facts["tool_execution_history"]) == 24
    assert len(json.dumps(facts)) < 19000
    assert len(req.tool_execution_history) == 100


@pytest.mark.parametrize("value", ['"\\' * 20000, "\u4e16\u754c" * 20000, {"items": list(range(10000))}], ids=["escapes", "unicode", "array"])
def test_display_budget_accounts_for_json_escaping_without_parsing_partial_json(value):
    bounded = _bounded_value(value, 500)
    assert len(json.dumps(bounded, ensure_ascii=True)) <= 500
    assert bounded["truncated"] is True
    assert isinstance(bounded["excerpt"], str)
    assert _bounded_value({"intact": 1}, 500) == {"intact": 1}


@pytest.mark.parametrize("response", [
    AIMessage(content=""), AIMessage(content="   "),
    AIMessage(content=[{"type": "text", "text": "Do not extract this block"}]),
    AIMessage(content="Tool result", tool_calls=[{"name": "brain_step_completed", "args": {}, "id": "x"}]),
    AIMessage(content='brain_step_completed{"message":"Do not salvage me"}'),
    AIMessage(content='brain_step_failed(message="Do not salvage me")'),
    AIMessage(content='brain_replan_requested {"reason":"Do not salvage me"}'),
    SimpleNamespace(content=None),
])
def test_malformed_or_control_output_has_typed_render_error_and_preserves_summary(response):
    req = request()
    result, calls = render(req, response)
    assert result.final_answer_error.startswith("ValueError:")
    assert result.execution_summary == Finalizer().finalize(req).execution_summary
    assert result.execution_summary.status == ExecutionStatus.COMPLETED
    assert result.final_answer == Finalizer._RENDER_FAILURE_ANSWER
    assert len(calls) == 1


def test_provider_exception_is_observable_without_corrupting_status(capsys):
    result, calls = render(request(), RuntimeError("provider unavailable"), enabled=True)
    assert result.final_answer_error == "RuntimeError: provider unavailable"
    assert result.execution_summary.status == ExecutionStatus.COMPLETED
    assert "Completed steps: step-1, step-2" in result.execution_summary.summary_text
    output = capsys.readouterr().out
    assert "[finalizer:error]" in output
    assert '"stage": "provider"' in output
    assert "RuntimeError" in output and "provider unavailable" in output
    assert "[finalizer:normalized]" in output
    assert len(calls) == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_debug_flag_controls_diagnostics_without_changing_messages_or_result(capsys, enabled):
    req = request()
    expected, baseline_calls = render(req, AIMessage(content="The brain_step_completed tool was updated."))
    capsys.readouterr()
    actual, calls = render(req, AIMessage(content=expected.final_answer), enabled=enabled)
    assert actual == expected and calls == baseline_calls
    output = capsys.readouterr().out
    assert bool(output) == enabled
    if enabled:
        for marker in ("request", "prompt", "raw", "normalized"):
            assert f"[finalizer:{marker}]" in output


def test_renderer_failure_applied_by_adapter_keeps_completed_execution():
    from core.graph_controller import create_controller_node
    from core.protocol.models import BrainOutcome
    from core.protocol.enums import BrainOutcomeKind
    from test_graph_controller import _state, _completed_plan_state
    finalizer = Finalizer(answer_renderer=LangChainFinalAnswerRenderer(llm=Model(RuntimeError("offline"))))
    update = create_controller_node(finalizer=finalizer)(_state(
        execution_state=_completed_plan_state(),
        brain_result=BrainOutcome(outcome=BrainOutcomeKind.FINAL_ANSWER, message="Finalization requested."),
    ))
    assert update["execution_state"].protocol_visible.status == ExecutionStatus.COMPLETED
    assert update["execution_state"].protocol_visible.summary.completed_step_ids == ("s1",)
    assert update["finalization_error"] == "RuntimeError: offline"
    assert update["messages"][0].content == update["finalization_result"].final_answer
