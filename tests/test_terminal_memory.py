"""Slice 3 terminal evidence and deterministic application memory update."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

import main as cli
from core.graph_runner import run_prompt
from core.memory import ConversationMemory, QuestionStatus, TurnStatus, merge_memory
from core.memory.terminal import AcceptedCompletion, CompletedTurnEvidence, extract_memory_update
from core.protocol.enums import ControllerDecisionType, ExecutionPhase, ExecutionStatus, WorkerRole
from core.protocol.models import (
    ControllerDecision, ExecutionCursor, ExecutionIdentity, ExecutionState,
    ExecutionSummary, FinalizationResult, ProtocolVisibleState, StepCompletionEvidence,
)


def completion(summary="Accepted three files"):
    return AcceptedCompletion(step_id="read", summary=summary, evidence_id="accepted-1",
                              plan_id="plan-1", plan_revision=1)


def evidence(*, status=TurnStatus.COMPLETED, request="Inspect files", direct=False,
             completions=None, answer="Delivered answer", reason=""):
    return CompletedTurnEvidence(
        turn_id="turn-1", turn_index=1, user_request=request, execution_id="exec-1",
        status=status, direct_response=direct,
        accepted_completions=(completion(),) if completions is None else completions,
        accepted_answer=answer, terminal_reason=reason,
    )


def test_evidence_is_immutable_and_requires_terminal_coherent_fields():
    item = evidence()
    with pytest.raises(ValidationError):
        item.user_request = "changed"
    with pytest.raises(ValidationError):
        evidence(status=TurnStatus.UNKNOWN)
    with pytest.raises(ValidationError):
        evidence(direct=True)
    with pytest.raises(ValidationError):
        CompletedTurnEvidence(turn_id="x", turn_index=1, user_request="x",
                              execution_id="x", status=TurnStatus.COMPLETED,
                              accepted_answer="")


def test_accepted_semantic_outcome_outranks_final_answer_and_raw_tool_text():
    item = evidence(answer="Conflicting final answer")
    update = extract_memory_update(item)
    assert update.continuity.outcome_note == "Accepted three files"
    assert update.facts == ()
    original = ConversationMemory()
    merged = merge_memory(original, update)
    assert original == ConversationMemory()
    assert merged.continuity.outcome_note == "Accepted three files"
    assert "Conflicting" not in merged.model_dump_json()
    # Raw ToolResults are deliberately excluded from the evidence contract.
    assert "tool_result" not in CompletedTurnEvidence.model_fields


def test_failed_and_cancelled_turns_never_claim_completion():
    failed = extract_memory_update(evidence(
        status=TurnStatus.FAILED, reason="Blocked by unavailable file", answer="Success!",
    ))
    assert failed.continuity.status == TurnStatus.FAILED
    assert failed.continuity.outcome_note == "Blocked by unavailable file"
    cancelled = extract_memory_update(evidence(
        status=TurnStatus.CANCELLED, completions=(), answer=None,
    ))
    assert cancelled.continuity.status == TurnStatus.CANCELLED
    assert cancelled.continuity.outcome_note == "Execution cancelled."


def test_direct_conversation_has_continuity_but_no_project_facts():
    item = evidence(direct=True, completions=(), request="How are you?", answer="I am ready.")
    update = extract_memory_update(item)
    assert update.continuity.outcome_note == "I am ready."
    assert update.continuity.status == TurnStatus.COMPLETED
    assert update.facts == () and update.questions == ()
    assert update.resolve_question_ids


def test_question_opens_on_failed_turn_then_resolves_on_accepted_answer():
    failed = evidence(status=TurnStatus.FAILED, request="Which file?", answer=None,
                      reason="File unavailable")
    opened = merge_memory(ConversationMemory(), extract_memory_update(failed))
    assert len(opened.questions) == 1
    assert opened.questions[0].status == QuestionStatus.OPEN
    assert opened.continuity.pending_follow_up == "Which file?"
    answered = evidence(request="Which file?", answer="a.py")
    answered = answered.model_copy(update={"turn_id": "turn-2", "turn_index": 2})
    resolved = merge_memory(opened, extract_memory_update(answered))
    assert resolved.questions[0].status == QuestionStatus.RESOLVED
    assert resolved.continuity.pending_follow_up == ""


def test_non_question_failed_turn_adds_no_speculative_question_or_fact():
    update = extract_memory_update(evidence(
        status=TurnStatus.FAILED, request="Inspect workspace", answer=None,
    ))
    assert update.questions == () and update.facts == ()
    assert update.continuity.pending_follow_up == ""


def terminal_event(*, status=ExecutionStatus.COMPLETED, accepted=True, direct=False,
                   with_answer=True, run_id="exec-1"):
    phase = {
        ExecutionStatus.COMPLETED: ExecutionPhase.COMPLETED,
        ExecutionStatus.FAILED: ExecutionPhase.FAILED,
        ExecutionStatus.CANCELLED: ExecutionPhase.CANCELLED,
    }[status]
    cursor = ExecutionCursor(phase=phase, current_worker=WorkerRole.SUMMARY)
    provenance = (StepCompletionEvidence(
        step_id="read", summary="Controller accepted result",
        execution_id=run_id, plan_id="plan-1", plan_revision=1,
        evidence_id="accepted-1",
    ),) if accepted else ()
    state = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=ExecutionIdentity(execution_id=run_id, protocol_version="1"),
        status=status, cursor=cursor, completion_provenance=provenance,
    ))
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.DISPATCH_SUMMARY,
        execution_status=status, cursor=cursor, terminal=True,
        direct_response=direct, reason="terminal reason",
    )
    update = {"controller_decision": decision, "execution_state": state}
    if with_answer:
        update["finalization_result"] = FinalizationResult(
            execution_summary=ExecutionSummary(
                execution_id=run_id, status=status, summary_text="Finalized.",
            ),
            final_answer="Delivered answer",
        )
    return {"controller": update}


class Events:
    def __init__(self, events, resumed=()):
        self.events = events
        self.resumed = resumed
        self.initial_state = None
        if resumed:
            self.async_runtime = self
            self.wake_count = 0

    def stream(self, initial_state, config=None):
        self.initial_state = initial_state
        return iter(self.events)

    def wake_and_resume(self, *, config, wake):
        self.wake_count += 1
        return iter(self.resumed)


def test_normal_terminal_is_captured_once_even_if_event_repeats():
    event = terminal_event()
    sink = []
    app = Events([event, event])
    history, _ = run_prompt(app, "Inspect files", run_id="exec-1",
                            completed_turn_evidence=sink, turn_index=3)
    assert len(sink) == 1
    assert sink[0].turn_index == 3
    assert sink[0].accepted_completions[0].summary == "Controller accepted result"
    assert sink[0].accepted_answer == "Delivered answer"
    assert "conversation_memory" not in app.initial_state
    assert history[0] == HumanMessage(content="Inspect files")


def test_nonterminal_wait_does_not_capture_memory_evidence():
    cursor = ExecutionCursor(phase=ExecutionPhase.WAITING,
                             step_id="step-1", current_worker=WorkerRole.CONTROLLER)
    waiting = ControllerDecision(
        decision_type=ControllerDecisionType.AWAIT_ASYNC_JOB,
        cursor=cursor, async_job_id="job-1", requires_checkpoint=True,
    )
    sink = []
    # Without async runtime, this is a nonterminal observation only.
    run_prompt(Events([{"controller": {"controller_decision": waiting}}]),
               "Generate", completed_turn_evidence=sink)
    assert sink == []


def test_resumed_async_captures_only_terminal_resolution():
    cursor = ExecutionCursor(phase=ExecutionPhase.WAITING,
                             step_id="step-1", current_worker=WorkerRole.CONTROLLER)
    waiting = ControllerDecision(
        decision_type=ControllerDecisionType.AWAIT_ASYNC_JOB,
        cursor=cursor, async_job_id="job-1", requires_checkpoint=True,
    )
    app = Events([{"controller": {"controller_decision": waiting}}],
                 resumed=[terminal_event(run_id="exec-1")])
    sink = []
    run_prompt(app, "Generate", run_id="exec-1", completed_turn_evidence=sink)
    assert app.wake_count == 1
    assert len(sink) == 1 and sink[0].status == TurnStatus.COMPLETED


def test_failed_terminal_capture_uses_controller_status_without_final_answer():
    sink = []
    run_prompt(Events([terminal_event(status=ExecutionStatus.FAILED,
                                      with_answer=False)]), "Inspect files",
               run_id="exec-1", completed_turn_evidence=sink)
    assert len(sink) == 1
    assert sink[0].status == TurnStatus.FAILED
    assert sink[0].accepted_answer is None
    assert extract_memory_update(sink[0]).continuity.status == TurnStatus.FAILED


def test_direct_response_is_identified_from_terminal_state_not_decision_flag():
    sink = []
    run_prompt(Events([terminal_event(accepted=False, direct=False)]),
               "How are you?", run_id="exec-1", completed_turn_evidence=sink)
    assert sink[0].direct_response is True
    assert extract_memory_update(sink[0]).continuity.outcome_note == "Delivered answer"


def test_completed_application_turn_merges_and_persists_continuity(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    settings = dict(cli.DEFAULT_SETTINGS, session_file=str(path))
    monkeypatch.setattr(cli, "parse_args", lambda: SimpleNamespace(prompt="Inspect files", resume=False))
    monkeypatch.setattr(cli, "_build_settings", lambda args: settings)
    app = Events([terminal_event()])
    monkeypatch.setattr(cli, "build_app", lambda **kwargs: app)
    cli.main()
    restored = cli.load_session(str(path))
    assert restored.completed_turn_count == 1
    assert restored.conversation_memory.continuity.outcome_note == "Controller accepted result"
    assert restored.conversation_memory.facts == ()
    assert [message.content for message in restored.recent_conversation] == [
        "Inspect files", "Delivered answer",
    ]
    assert "conversation_memory" not in app.initial_state


def test_updater_failure_preserves_memory_and_successful_turn(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    prior = ConversationMemory()
    cli.save_session(str(path), cli.ApplicationSession(conversation_memory=prior))
    settings = dict(cli.DEFAULT_SETTINGS, session_file=str(path))
    monkeypatch.setattr(cli, "parse_args", lambda: SimpleNamespace(prompt="Request", resume=True))
    monkeypatch.setattr(cli, "_build_settings", lambda args: settings)
    monkeypatch.setattr(cli, "build_app", lambda **kwargs: object())

    def run(app, prompt, **kwargs):
        kwargs["completed_turn_evidence"].append(evidence())
        return [HumanMessage(content=prompt)], ""

    monkeypatch.setattr(cli, "run_prompt", run)
    monkeypatch.setattr(cli, "extract_memory_update", lambda item: (_ for _ in ()).throw(ValueError("bad proposal")))
    cli.main()
    restored = cli.load_session(str(path))
    assert restored.conversation_memory == prior
    assert restored.completed_turn_count == 1
    assert [m.content for m in restored.recent_conversation] == ["Request"]
