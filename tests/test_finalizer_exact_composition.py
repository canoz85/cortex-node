"""Successful exact answers preserve every other accepted step conclusion."""

import json

import pytest
from langchain_core.messages import AIMessage

from core.finalizer import Finalizer
from core.finalizer_provider import LangChainFinalAnswerRenderer
from core.protocol.enums import BrainOutcomeKind, ExecutionPhase, StepStatus, WorkerRole
from core.protocol.models import (
    BrainOutcome, ExecutionCursor, ExecutionState, ExecutionStep, ProtocolVisibleState,
)
from test_finalizer_rendering import Model, accepted_result, exact_result, request


@pytest.fixture(params=("deterministic", "provider"))
def renderer(request):
    model = Model(AssertionError("Exact answers must not invoke the model"))
    finalizer = (
        Finalizer(answer_renderer=LangChainFinalAnswerRenderer(llm=model))
        if request.param == "provider" else Finalizer()
    )
    return finalizer, model


def completed_request(*results):
    req = request(accepted_results=results)
    steps = tuple(ExecutionStep(
        step_id=result.completion_evidence.step_id,
        title="Accepted outcome", status=StepStatus.COMPLETED,
    ) for result in results)
    return req.model_copy(update={
        "accepted_plan": req.accepted_plan.model_copy(update={"steps": steps}),
        "completed_step_ids": tuple(step.step_id for step in steps),
    })


def finalize_without_mutation(renderer, req):
    finalizer, model = renderer
    before = req.model_dump_json()
    result = finalizer.finalize(req)
    assert result.final_answer_error is None
    assert result.execution_summary.completed_step_ids == req.completed_step_ids
    assert req.model_dump_json() == before
    assert model.calls == []
    return result.final_answer


def test_exact_collection_only_preserves_current_output(renderer):
    req = completed_request(exact_result(("b.py", "a.py"), label="changed_files"))
    assert finalize_without_mutation(renderer, req) == "changed_files:\n- b.py\n- a.py"


def test_exact_collection_plus_later_semantic_result_preserves_both(renderer):
    paths = ("create_structure_cli.py", "hello.py", "cheerful_calculator.py", "small.py")
    top_three = (
        "The three largest files by character count are create_structure_cli.py (3141), "
        "hello.py (3075), and cheerful_calculator.py (2689)."
    )
    summaries = (
        "create_structure_cli.py creates a project tree through a CLI.\n"
        "hello.py tests structured Planner responses.\n"
        "cheerful_calculator.py implements an interactive Turkish calculator."
    )
    req = completed_request(
        exact_result(paths, label="python_files"),
        accepted_result("step-2", top_three),
        accepted_result("step-3", summaries, tool_request_ids=()),
    )
    exact = "python_files:\n" + "\n".join(f"- {path}" for path in paths)
    assert finalize_without_mutation(renderer, req) == exact + "\n\n" + top_three + "\n\n" + summaries


def test_exact_collection_does_not_replace_any_later_completion_or_truncate_it(renderer):
    first = "First accepted comparison."
    later = "Long accepted conclusion: " + "supporting detail " * 600 + "Final conclusion retained."
    req = completed_request(
        exact_result(("a.py",)), accepted_result("step-2", first), accepted_result("step-3", later),
    )
    assert finalize_without_mutation(renderer, req) == "Files:\n- a.py\n\n" + first + "\n\n" + later


def test_exact_collection_values_not_rewritten_reordered_or_deduplicated(renderer):
    values = ("src/Z.py", "src/a.py", "src/Z.py", "lib/İ.py")
    req = completed_request(
        exact_result(values, label="CaseSensitivePaths"),
        accepted_result("step-2", "The selected module implements telemetry."),
    )
    answer = finalize_without_mutation(renderer, req)
    collection, semantic = answer.split("\n\n")
    assert collection == "CaseSensitivePaths:\n" + "\n".join(f"- {value}" for value in values)
    assert collection.splitlines()[1:] == [f"- {value}" for value in values]
    assert semantic == req.accepted_step_results[1].semantic_content


def test_multiple_exact_collections_and_semantic_results_are_all_preserved(renderer):
    second = exact_result(("ID-002", "ID-001"), label="Identifiers")
    second = second.model_copy(update={"completion_evidence": second.completion_evidence.model_copy(
        update={"step_id": "step-3"},
    )})
    req = completed_request(
        exact_result(("b.py", "a.py"), label="Files"),
        accepted_result("step-2", "Selected b.py."),
        second,
        accepted_result("step-4", "Both identifiers were verified."),
    )
    assert finalize_without_mutation(renderer, req) == (
        "Files:\n- b.py\n- a.py\n\nIdentifiers:\n- ID-002\n- ID-001"
        "\n\nSelected b.py.\n\nBoth identifiers were verified."
    )


def test_empty_and_structured_exact_collections_keep_existing_format(renderer):
    req = completed_request(exact_result((), label="Matches"))
    assert finalize_without_mutation(renderer, req) == "Matches:\n- (none)"
    value = {"path": "src/İ.py", "status": "modified"}
    req = completed_request(
        exact_result((value,), label="Changes"), accepted_result("step-2", "Verification passed."),
    )
    assert finalize_without_mutation(renderer, req) == (
        "Changes:\n- " + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        + "\n\nVerification passed."
    )


def test_semantic_only_finalization_keeps_existing_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("CORTEX_RAW_LLM_FILE", str(tmp_path / "semantic-only.jsonl"))
    req = completed_request(accepted_result("step-1", "The largest file is b.py."))
    before = req.model_dump_json()
    deterministic = Finalizer().finalize(req)
    assert deterministic.final_answer == deterministic.execution_summary.summary_text
    model = Model(AIMessage(content="The largest file is b.py."))
    result = Finalizer(answer_renderer=LangChainFinalAnswerRenderer(llm=model)).finalize(req)
    assert result.final_answer == "The largest file is b.py."
    assert len(model.calls) == 1
    facts = json.loads(model.calls[0][-1].content.split("\n", 1)[1])
    assert facts["accepted_step_results"][0]["semantic_content"] == req.accepted_step_results[0].semantic_content
    assert req.model_dump_json() == before


def test_terminal_graph_boundary_preserves_controller_completion_provenance(renderer):
    from core.graph_controller import create_controller_node

    req = completed_request(
        exact_result(("a.py", "b.py")),
        accepted_result("step-2", "b.py is the largest file."),
        accepted_result("step-3", "b.py processes telemetry.", tool_request_ids=()),
    )
    provenance = tuple(result.completion_evidence for result in req.accepted_step_results)
    state = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=req.identity, original_user_request=req.context.user_request,
        cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING, current_worker=WorkerRole.BRAIN,
                               plan_revision=req.accepted_plan.revision),
        active_plan=req.accepted_plan, completed_step_ids=req.completed_step_ids,
        completion_provenance=provenance,
    ))
    before = state.model_dump_json()
    finalizer, model = renderer
    update = create_controller_node(finalizer=finalizer)({
        "execution_state": state,
        "brain_result": BrainOutcome(outcome=BrainOutcomeKind.FINAL_ANSWER_READY,
                                     message="Finalization requested."),
    })
    assert update["finalization_result"].final_answer_error is None
    assert update["final_answer"] == (
        "Files:\n- a.py\n- b.py\n\nb.py is the largest file.\n\nb.py processes telemetry."
    )
    assert update["execution_state"].protocol_visible.completion_provenance == provenance
    assert state.model_dump_json() == before
    assert model.calls == []
