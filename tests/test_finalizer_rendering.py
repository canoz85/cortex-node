"""Final rendering has a bounded input and cannot change execution results."""
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from core.finalizer import Finalizer
from core import finalizer_provider
from core.finalizer_provider import (
    FACTS_BUDGET,
    FINALIZER_SYSTEM_PROMPT,
    LangChainFinalAnswerRenderer,
    _bounded_value,
    finalizer_facts,
)
from core.protocol.enums import CancellationSource, ExecutionStatus, StepStatus
from core.protocol.models import (
    AcceptedStepResult,
    AcceptedDirectResponse,
    ExactCollection,
    FinalizationRequest,
    StepCompletionEvidence,
    ToolExecutionRecord,
    ToolResult,
)
from test_finalizer import IDENTITY, CONTEXT, _plan


class Model:
    def __init__(self, response):
        self.response, self.calls = response, []

    def invoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def request(records=(), accepted_results=()):
    return FinalizationRequest(identity=IDENTITY, status=ExecutionStatus.COMPLETED,
        context=CONTEXT, accepted_plan=_plan(), completed_step_ids=("step-1", "step-2"),
        terminal_reason="final_answer", tool_execution_history=records,
        accepted_step_results=accepted_results)


def record(step, data, rendered=""):
    return ToolExecutionRecord(step_id=step, tool_name="inspect", arguments={"target": step},
        result=ToolResult(request_id=step, success=True, message="Inspected", data=data,
                          rendered_output=rendered))


def accepted_result(
    step,
    semantic_content,
    *,
    revision=2,
    tool_request_ids=("request-1",),
):
    return AcceptedStepResult(completion_evidence=StepCompletionEvidence(
        execution_id=IDENTITY.execution_id,
        plan_id="plan-1",
        plan_revision=revision,
        step_id=step,
        summary=semantic_content,
        tool_request_ids=tool_request_ids,
        evidence_id=f"evidence-{step}-{revision}",
    ))


def exact_result(items, *, label="Files"):
    return AcceptedStepResult(completion_evidence=StepCompletionEvidence(
        execution_id=IDENTITY.execution_id,
        plan_id="plan-1",
        plan_revision=2,
        step_id="step-1",
        summary="Exact collection accepted.",
        tool_request_ids=("list-1",),
        evidence_id="evidence-exact",
        exact_collection=ExactCollection(
            source_record_index=0,
            source_request_id="list-1",
            data_path=("entries",),
            label=label,
            items=tuple(items),
        ),
    ))


def render(req, response, enabled=False):
    model = Model(response)
    result = Finalizer(answer_renderer=LangChainFinalAnswerRenderer(llm=model, show_raw_llm=enabled),
                       show_raw_llm=enabled).finalize(req)
    return result, model.calls


def representative_request():
    """Three accepted steps, a ~4 KB conclusion, and verbose duplicate evidence."""
    semantics = (
        "Found the requested files.",
        "Selected the three largest files.",
        "Accepted analysis: " + "detail " * 580 + "\nFinal accepted conclusion.",
    )
    req = request(
        tuple(record(f"raw-{i}", {"content": "TOOL-ONLY EVIDENCE " + "x" * 10000}) for i in range(24)),
        tuple(accepted_result(f"step-{i}", content) for i, content in enumerate(semantics, 1)),
    )
    third = req.accepted_plan.steps[1].model_copy(update={"step_id": "step-3", "title": "Analysis"})
    plan = req.accepted_plan.model_copy(update={
        "objective": CONTEXT.user_request,
        "available_tools": ("inspect",),
        "steps": tuple(step.model_copy(update={
            "description": "Redundant execution instructions. " * 100,
            "primary_tool": "inspect",
            "attempt": 3,
        }) for step in (*req.accepted_plan.steps, third)),
    })
    return req.model_copy(update={"accepted_plan": plan, "completed_step_ids": ("step-1", "step-2", "step-3")})


def test_success_projects_only_user_facing_facts_and_preserves_modest_semantics():
    req = representative_request()
    before = req.model_dump_json()
    result, calls = render(req, AIMessage(content="Accepted analysis rendered."))
    facts = json.loads(calls[0][-1].content.split("\n", 1)[1])

    assert facts == {
        "status": "completed",
        "terminal_reason": "final_answer",
        "accepted_step_results": [
            {"step_id": f"step-{i}", "title": title, "semantic_content": accepted.semantic_content}
            for i, (title, accepted) in enumerate(zip(
                ("First", "Second", "Analysis"), req.accepted_step_results,
            ), 1)
        ],
    }
    assert len(json.dumps(facts, ensure_ascii=True)) <= FACTS_BUDGET
    assert calls[0][1].content == req.context.user_request
    assert "TOOL-ONLY EVIDENCE" not in calls[0][-1].content
    assert req.model_dump_json() == before
    assert result.final_answer_error is None


def test_semantic_budget_reuses_space_and_never_falls_back_to_raw_evidence():
    modest = representative_request().accepted_step_results[-1].semantic_content
    req = request(
        tuple(record(f"raw-{i}", {"content": "RAW " * 10000}) for i in range(100)),
        (accepted_result("step-1", "HEAD " + "long detail " * 10000 + " TAIL"),
         accepted_result("step-2", modest), accepted_result("step-3", "Short conclusion.")),
    )
    result, calls = render(req, AIMessage(content="Rendered accepted results."))
    facts = json.loads(calls[0][-1].content.split("\n", 1)[1])
    results = facts["accepted_step_results"]

    assert [item["step_id"] for item in results] == ["step-1", "step-2", "step-3"]
    assert results[1]["semantic_content"] == modest
    assert results[2]["semantic_content"] == "Short conclusion."
    excerpt = results[0]["semantic_content"]
    assert excerpt["truncated"] is True
    assert excerpt["original_chars"] == len(json.dumps(req.accepted_step_results[0].semantic_content))
    assert "HEAD" in excerpt["excerpt"] and "TAIL" in excerpt["excerpt"]
    assert "tool_execution_history" not in facts
    assert len(json.dumps(facts, ensure_ascii=True)) <= FACTS_BUDGET
    assert result.final_answer_error is None


@pytest.mark.parametrize("content", ['"\\' * 30000, "世界" * 30000], ids=["escapes", "unicode"])
def test_semantic_projection_budget_counts_serialized_metadata_and_escaping(content):
    req = request(accepted_results=(accepted_result("step-1", content), accepted_result("step-2", "Intact")))
    summary = Finalizer().finalize(req).execution_summary
    facts = finalizer_facts(req, summary)

    assert facts["accepted_step_results"][0]["semantic_content"]["truncated"] is True
    assert facts["accepted_step_results"][1]["semantic_content"] == "Intact"
    assert len(json.dumps(facts, ensure_ascii=True)) <= FACTS_BUDGET


def test_projection_is_deterministic_and_independent_of_redundant_history():
    req = representative_request()
    summary = Finalizer().finalize(req).execution_summary
    expected = json.dumps(finalizer_facts(req, summary), ensure_ascii=True)

    assert json.dumps(finalizer_facts(req, summary), ensure_ascii=True) == expected
    without_history = req.model_copy(update={"tool_execution_history": ()})
    assert json.dumps(finalizer_facts(without_history, summary), ensure_ascii=True) == expected
    _, first_calls = render(req, AIMessage(content="Same answer."))
    _, second_calls = render(req, AIMessage(content="Same answer."))
    assert first_calls == second_calls


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


def test_accepted_semantics_are_authoritative_over_misleading_raw_evidence():
    semantic_x = "### Analysis of the Error\n\n### Summary of Findings"
    misleading_z = "Imports\nThe main() function"
    accepted = accepted_result("step-1", semantic_x)
    req = request(
        (record("step-1", {"content": misleading_z}),),
        (accepted,),
    )
    summary = Finalizer().finalize(req).execution_summary

    facts = finalizer_facts(req, summary)

    assert facts["accepted_step_results"] == [{
        "step_id": "step-1",
        "title": "First",
        "semantic_content": semantic_x,
    }]
    assert "tool_execution_history" not in facts

    result, calls = render(
        req,
        AIMessage(content="Analysis of the Error; Summary of Findings"),
    )
    model_facts = json.loads(calls[0][-1].content.split("\n", 1)[1])
    assert model_facts["accepted_step_results"][0]["semantic_content"] == semantic_x
    assert "tool_execution_history" not in model_facts
    assert misleading_z not in calls[0][-1].content
    assert "Controller-accepted step results are the authoritative factual basis" in calls[0][0].content
    assert "supporting evidence and details only" in calls[0][0].content
    assert result.final_answer == "Analysis of the Error; Summary of Findings"


def test_multiple_accepted_results_preserve_acceptance_order_and_empty_provenance():
    first = accepted_result("step-2", "First accepted conclusion")
    second = accepted_result(
        "step-1",
        "Second reasoning-only conclusion",
        revision=3,
        tool_request_ids=(),
    )
    req = request(accepted_results=(first, second))
    summary = Finalizer().finalize(req).execution_summary

    facts = finalizer_facts(req, summary)

    assert [item["semantic_content"] for item in facts["accepted_step_results"]] == [
        "First accepted conclusion",
        "Second reasoning-only conclusion",
    ]
    assert [
        (item["step_id"], item["title"])
        for item in facts["accepted_step_results"]
    ] == [("step-2", "Second"), ("step-1", "First")]
    assert all("plan_revision" not in item for item in facts["accepted_step_results"])
    assert second.completion_evidence.tool_request_ids == ()


def test_no_accepted_results_preserves_existing_model_projection_and_rendering():
    req = FinalizationRequest(
        identity=IDENTITY,
        status=ExecutionStatus.COMPLETED,
        context=CONTEXT,
        direct_response=True,
        terminal_reason="direct_response",
    )
    result, calls = render(req, AIMessage(content="Existing fallback response"))
    facts = json.loads(calls[0][-1].content.split("\n", 1)[1])

    assert facts["accepted_step_results"] == []
    assert "tool_execution_history" not in facts
    assert result.final_answer == "Existing fallback response"
    assert result.execution_summary.completed_step_ids == ()
    assert FINALIZER_SYSTEM_PROMPT == calls[0][0].content


def test_system_prompt_preserves_exact_accepted_collections():
    prompt = " ".join(FINALIZER_SYSTEM_PROMPT.split())

    assert "exact factual collection" in prompt
    assert "preserve those items exactly" in prompt
    assert "Do not add, remove, rename, substitute, infer" in prompt
    assert "without changing its factual members" in prompt


def test_system_prompt_prohibits_internal_correction_commentary():
    prompt = " ".join(FINALIZER_SYSTEM_PROMPT.split())

    assert "Do not include internal reasoning, self-corrections" in prompt
    assert '"wait", "checking", or "actually"' in prompt
    assert "Return only the polished user-facing answer" in prompt


def test_exact_filename_collection_bypasses_model_and_preserves_members():
    model = Model(AIMessage(content="invented.py and final_raw.json"))
    req = request(accepted_results=(exact_result((
        "finalizer-messages.json",
        "finalizer-raw.json",
    )),))
    summary = Finalizer().finalize(req).execution_summary

    answer = LangChainFinalAnswerRenderer(llm=model).render(req, summary)

    assert answer == (
        "Files:\n"
        "- finalizer-messages.json\n"
        "- finalizer-raw.json"
    )
    assert "final_raw.json" not in answer
    assert "invented.py" not in answer
    assert model.calls == []


def test_exact_identifier_and_path_members_preserve_case_and_punctuation():
    model = Model(AIMessage(content="must not run"))
    members = ("Build-ID:AbC-007", "src/API.v2/Finalizer-RAW.json")
    req = request(accepted_results=(exact_result(members, label="Identifiers"),))
    summary = Finalizer().finalize(req).execution_summary

    answer = LangChainFinalAnswerRenderer(llm=model).render(req, summary)

    assert answer == (
        "Identifiers:\n"
        "- Build-ID:AbC-007\n"
        "- src/API.v2/Finalizer-RAW.json"
    )
    assert model.calls == []


def test_exact_collections_and_other_semantics_bypass_projection_budget_without_raw_history(monkeypatch):
    monkeypatch.setattr(finalizer_provider, "FACTS_BUDGET", 128)
    members = ("src/Z.py", "src/İ.py", "src/Z.py") * 1000
    semantic = ("Accepted analysis. " * 2000).rstrip()
    req = request(accepted_results=(exact_result(members), accepted_result("step-2", semantic)))
    before = req.model_dump_json()
    result, calls = render(req, AIMessage(content="must not run"))

    assert result.final_answer == "Files:\n" + "\n".join(f"- {item}" for item in members) + "\n\n" + semantic
    assert result.final_answer_error is None
    assert req.tool_execution_history == ()
    assert req.model_dump_json() == before
    assert calls == []


def test_normal_prose_still_uses_model_renderer():
    result, calls = render(
        request(accepted_results=(accepted_result("step-1", "Accepted prose."),)),
        AIMessage(content="Polished prose."),
    )

    assert result.final_answer == "Polished prose."
    assert len(calls) == 1


@pytest.mark.parametrize("status,source", [
    (ExecutionStatus.FAILED, None),
    (ExecutionStatus.CANCELLED, CancellationSource.LOCAL),
    (ExecutionStatus.CANCELLED, CancellationSource.PROVIDER),
])
def test_terminal_failure_projection_preserves_reporting_facts_and_deterministic_rendering(status, source):
    plan = _plan(failed=True)
    if status == ExecutionStatus.CANCELLED:
        plan = plan.model_copy(update={"steps": (
            plan.steps[0], plan.steps[1].model_copy(update={"status": StepStatus.ACTIVE}),
        )})
    req = request(
        (record("step-1", {"irrelevant": "raw output"}),),
        (accepted_result("step-1", "First step completed."),),
    ).model_copy(update={
        "status": status, "accepted_plan": plan, "completed_step_ids": ("step-1",),
        "terminal_reason": "Retry budget exhausted." if source is None else "Execution cancelled by request.",
        "cancellation_source": source,
    })
    before = req.model_dump_json()
    result, calls = render(req, AIMessage(content="must not run"))
    facts = finalizer_facts(req, result.execution_summary)

    assert facts["status"] == status.value
    assert facts["terminal_reason"] == req.terminal_reason
    assert facts["completed_step_ids"] == ("step-1",)
    assert facts["failed_step_ids"] == (("step-2",) if source is None else ())
    if source is not None:
        assert facts["cancellation_source"] == source.value
        assert f"Cancellation source: {source.value}." in result.execution_summary.summary_text
    assert facts["accepted_step_results"][0]["semantic_content"] == "First step completed."
    assert "accepted_plan" not in facts and "tool_execution_history" not in facts
    assert req.terminal_reason in result.final_answer
    assert "First step completed." in result.final_answer
    assert result.execution_summary.status == status
    assert req.model_dump_json() == before
    assert calls == []


def test_failed_execution_is_rendered_deterministically_without_recovery_inventions():
    malicious = AIMessage(content=(
        "Retry manually with port 1883.\n"
        "```python\nclient.connect('broker.example.com', 1883)\n```"
    ))
    model = Model(malicious)
    plan = _plan(failed=True).model_copy(update={
        "objective": "Connect using password secret123",
        "steps": (
            _plan(failed=True).steps[0],
            _plan(failed=True).steps[1].model_copy(update={
                "description": "Use password secret123 and port 1883",
                "status": StepStatus.FAILED,
            }),
        ),
    })
    req = FinalizationRequest(
        identity=IDENTITY,
        status=ExecutionStatus.FAILED,
        context=CONTEXT.model_copy(update={
            "user_request": "Connect with password secret123",
        }),
        accepted_plan=plan,
        completed_step_ids=("step-1",),
        accepted_step_results=(accepted_result(
            "step-1", "The required client library is available."
        ),),
        terminal_reason="Brain returned invalid output: exactly_one_tool_call_required.",
    )

    result = Finalizer(
        answer_renderer=LangChainFinalAnswerRenderer(llm=model)
    ).finalize(req)

    assert model.calls == []
    assert result.final_answer_error is None
    assert "The required client library is available." in result.final_answer
    assert "step-2" in result.final_answer
    assert "exactly_one_tool_call_required" in result.final_answer
    for unsupported in ("```", "client.connect", "Retry", "manually", "1883", "secret123"):
        assert unsupported not in result.final_answer


def test_failed_execution_preserves_controller_bound_exact_collection():
    model = Model(AIMessage(content="must not run"))
    req = FinalizationRequest(
        identity=IDENTITY,
        status=ExecutionStatus.FAILED,
        context=CONTEXT,
        accepted_plan=_plan(failed=True),
        completed_step_ids=("step-1",),
        accepted_step_results=(exact_result(("A-1", "B-2"), label="IDs"),),
        terminal_reason="Later step failed.",
    )

    result = Finalizer(
        answer_renderer=LangChainFinalAnswerRenderer(llm=model)
    ).finalize(req)

    assert model.calls == []
    assert "IDs:\n- A-1\n- B-2" in result.final_answer
    assert "Later step failed." in result.final_answer


def test_accepted_direct_response_bypasses_model_renderer():
    model = Model(AIMessage(content="must not run"))
    req = FinalizationRequest(
        identity=IDENTITY,
        status=ExecutionStatus.COMPLETED,
        context=CONTEXT,
        accepted_direct_response=AcceptedDirectResponse(
            execution_id=IDENTITY.execution_id,
            request_id="direct-1",
            content="Exact direct response.",
        ),
        direct_response=True,
    )
    summary = Finalizer().finalize(req).execution_summary

    answer = LangChainFinalAnswerRenderer(llm=model).render(req, summary)

    assert answer == "Exact direct response."
    assert model.calls == []


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


def test_provider_exception_creates_no_fake_exchange_and_preserves_status(capsys):
    result, calls = render(request(), RuntimeError("provider unavailable"), enabled=True)
    assert result.final_answer_error == "RuntimeError: provider unavailable"
    assert result.execution_summary.status == ExecutionStatus.COMPLETED
    assert "Completed steps: step-1, step-2" in result.execution_summary.summary_text
    output = capsys.readouterr().out
    assert output == ""
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
        assert "[raw-llm:finalizer:render]" in output
        assert '"worker": "finalizer"' in output
        assert '"operation": "render"' in output
        assert '"messages": [' in output
        assert '"response": {' in output


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
