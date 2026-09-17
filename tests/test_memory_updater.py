"""Slice 5 structured proposals, source binding, and safe compaction."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

import main as cli
from core.application_session import ApplicationSession, bounded_recent_conversation
from core.conversation_compaction import CompactionLimits, compact_recent_conversation
from core.conversation_memory_updater import (
    LLMMemoryUpdater, MemoryProposal, ProposedMemoryFact,
    MemoryUpdateRequest, build_memory_update_request,
)
from core.graph_messages import ACCEPTED_FINALIZER_PROVENANCE, CONVERSATION_PROVENANCE_KEY
from core.memory import (
    ConversationMemory, FactCategory, MemoryFact, MemorySource, OpenQuestion,
    MemoryUpdate, QuestionStatus, SourceKind, TurnStatus, merge_memory,
)
from core.memory.terminal import AcceptedCompletion, CompletedTurnEvidence, extract_memory_update
from core.memory_provider import LangChainMemoryProposalProvider
from core.planner_memory import project_planner_memory
from core.protocol.enums import ControllerDecisionType, ExecutionPhase, ExecutionStatus, WorkerRole
from core.protocol.models import (
    ControllerDecision, ExecutionCursor, ExecutionIdentity, ExecutionState,
    ExecutionSummary, FinalizationResult, ProtocolVisibleState,
)


def terminal(request="Benim adım Can.", *, turn=1, status=TurnStatus.COMPLETED,
             completions=(), answer="Accepted answer"):
    return CompletedTurnEvidence(
        turn_id=f"turn-{turn}", turn_index=turn, execution_id=f"exec-{turn}",
        user_request=request, status=status, direct_response=not completions,
        accepted_completions=completions, accepted_answer=answer,
    )


def proposed_user(text=None, *, quote="Benim adım Can.",
                  handle="human_current", scope="user.name"):
    return ProposedMemoryFact(
        category=FactCategory.USER_PROFILE, scope_key=scope, text=text or "The user's name is Can",
        source_handle=handle, evidence_quote=quote,
    )


def proposed_project(text, *, quote=None, handle="accepted:result-1"):
    return ProposedMemoryFact(
        category=FactCategory.PROJECT_REPOSITORY, scope_key="repo.language",
        text=text, source_handle=handle, evidence_quote=quote or text,
    )


class Provider:
    def __init__(self, *proposals):
        self.proposals = list(proposals)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return self.proposals.pop(0)


def accepted_answer(text):
    return AIMessage(content=text, additional_kwargs={
        CONVERSATION_PROVENANCE_KEY: ACCEPTED_FINALIZER_PROVENANCE,
    })


def test_request_is_immutable_curated_and_bounded():
    prior = ConversationMemory(facts=tuple(
        MemoryFact(category=FactCategory.USER_PREFERENCE, scope_key=f"preference.{i}",
                   text=f"preference {i}",
                   source=MemorySource(kind=SourceKind.HUMAN, turn_index=i + 1,
                                       turn_id=f"turn-{i + 1}"))
        for i in range(20)
    ))
    request = build_memory_update_request(terminal(), prior)
    assert len(request.existing_facts) == 12
    assert request.user_request == "Benim adım Can."
    assert "history" not in MemoryUpdateRequest.model_fields
    assert "rolling_summary" not in MemoryUpdateRequest.model_fields
    assert "tool_result" not in MemoryUpdateRequest.model_fields
    with pytest.raises(ValidationError):
        request.user_request = "changed"
    assert prior.facts[0].text == "preference 0"


def test_user_fact_is_bound_to_exact_current_human_source_and_merge_only():
    existing = ConversationMemory()
    provider = Provider(MemoryProposal(facts=(proposed_user(),)))
    update = LLMMemoryUpdater(provider).propose(build_memory_update_request(terminal(), existing))
    assert existing == ConversationMemory()
    assert update.facts[0].source.kind == SourceKind.HUMAN
    assert update.facts[0].source.turn_id == "turn-1"
    assert update.facts[0].source.execution_id is None
    merged = merge_memory(existing, update)
    assert merged.facts[0].text == "The user's name is Can"
    assert merge_memory(merged, update) == merged


@pytest.mark.parametrize("proposal", [
    proposed_user(handle="accepted:result-1"),
    proposed_user(quote="I am Can."),
    proposed_project("Repository uses Python"),
    proposed_project("Repository uses Python", handle="human_current"),
])
def test_unknown_or_upgraded_provenance_is_rejected(proposal):
    provider = Provider(MemoryProposal(facts=(proposal,)))
    with pytest.raises(ValueError):
        LLMMemoryUpdater(provider).propose(build_memory_update_request(terminal(), ConversationMemory()))


def test_accepted_semantic_span_can_support_project_fact_but_final_answer_cannot():
    accepted = AcceptedCompletion(step_id="inspect", summary="Repository uses Python",
                                  evidence_id="result-1", plan_id="plan", plan_revision=1)
    request = build_memory_update_request(terminal(
        "Inspect repository", completions=(accepted,), answer="Repository uses Rust",
    ), ConversationMemory())
    update = LLMMemoryUpdater(Provider(MemoryProposal(facts=(
        proposed_project("Repository uses Python"),
    )))).propose(request)
    assert update.facts[0].source.kind == SourceKind.ACCEPTED_RESULT
    assert update.facts[0].source.accepted_result_ref == "result-1"
    assert update.facts[0].source.execution_id == "exec-1"
    assert update.facts[0].claim.value == "observation"
    with pytest.raises(ValueError):
        LLMMemoryUpdater(Provider(MemoryProposal(facts=(
            proposed_project("Repository uses Rust"),
        )))).propose(request)
    with pytest.raises(ValueError):
        LLMMemoryUpdater(Provider(MemoryProposal(facts=(
            proposed_project("Python repository", quote="Repository uses Python"),
        )))).propose(request)


def test_a_user_question_cannot_be_recast_as_an_explicit_user_fact():
    request = build_memory_update_request(
        terminal("Benim adım ne?"), ConversationMemory(),
    )
    proposed = proposed_user(quote="Benim adım ne?")
    with pytest.raises(ValueError):
        LLMMemoryUpdater(Provider(MemoryProposal(facts=(proposed,)))).propose(request)


def test_verbatim_whole_utterance_is_rejected_as_user_fact():
    request = build_memory_update_request(terminal(), ConversationMemory())
    invented = proposed_user(text=request.user_request)
    with pytest.raises(ValueError):
        LLMMemoryUpdater(Provider(MemoryProposal(facts=(invented,)))).propose(request)


def test_correction_uses_existing_scope_and_slice_one_merge():
    first = terminal()
    updater = LLMMemoryUpdater(Provider(
        MemoryProposal(facts=(proposed_user(),)),
        MemoryProposal(facts=(proposed_user(
            quote="Benim adım artık Cem.",
            text="The user's name is Cem",
        ),)),
    ))
    original = merge_memory(ConversationMemory(), updater.propose(
        build_memory_update_request(first, ConversationMemory())))
    correction = terminal("Benim adım artık Cem.", turn=2)
    request = build_memory_update_request(correction, original)
    assert request.existing_facts[0].scope_key == "user.name"
    updated = merge_memory(original, updater.propose(request))
    assert len(updated.facts) == 1
    assert updated.facts[0].text == "The user's name is Cem"
    assert updated.facts[0].source.turn_index == 2


def test_empty_proposal_is_valid_and_does_not_change_durable_facts():
    original = ConversationMemory()
    update = LLMMemoryUpdater(Provider(MemoryProposal())).propose(
        build_memory_update_request(terminal(), original))
    assert update.facts == () and merge_memory(original, update) == original


def test_invalid_schema_and_oversized_proposal_fail_closed():
    with pytest.raises(ValidationError):
        ProposedMemoryFact(category=FactCategory.USER_PROFILE, scope_key="x",
                           text="x" * 501, source_handle="human_current", evidence_quote="x")
    with pytest.raises(ValidationError):
        MemoryProposal(facts=tuple(proposed_user(scope=f"s{i}") for i in range(9)))
    with pytest.raises(ValueError):
        LLMMemoryUpdater(Provider({"facts": []})).propose(
            build_memory_update_request(terminal(), ConversationMemory()))
    with pytest.raises(ValueError, match="duplicate"):
        LLMMemoryUpdater(Provider(MemoryProposal(facts=(
            proposed_user(), proposed_user(),
        )))).propose(build_memory_update_request(terminal(), ConversationMemory()))


def test_provider_uses_one_structured_call_with_curated_payload():
    class Structured:
        calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return MemoryProposal()

    class LLM:
        methods = []
        structured = Structured()

        def with_structured_output(self, schema, method):
            self.methods.append((schema, method))
            return self.structured

    llm = LLM()
    result = LangChainMemoryProposalProvider(llm).generate(
        build_memory_update_request(terminal(), ConversationMemory()))
    assert result.facts == ()
    assert llm.methods == [(MemoryProposal, "json_schema")]
    assert len(llm.structured.calls) == 1
    content = llm.structured.calls[0][-1].content
    assert "Benim adım Can." in content
    assert "rolling_summary" not in content and "tool_execution_history" not in content


def test_json_native_proposal_parses_and_freezes_before_strict_memory_binding():
    raw = {"facts": [{
        "category": "user_profile", "scope_key": "user.name",
        "text": "The user's name is Can", "source_handle": "human_current",
        "evidence_quote": "Benim adım Can.",
    }]}

    class Structured:
        def invoke(self, messages):
            return raw

    class LLM:
        def with_structured_output(self, schema, method):
            assert schema is MemoryProposal and method == "json_schema"
            return Structured()

    request = build_memory_update_request(terminal(), ConversationMemory())
    proposal = LangChainMemoryProposalProvider(LLM()).generate(request)
    assert isinstance(proposal.facts, tuple)
    assert proposal.facts[0].category is FactCategory.USER_PROFILE
    update = LLMMemoryUpdater(Provider(proposal)).propose(request)
    assert update.facts[0].text == "The user's name is Can"
    assert update.facts[0].source.kind is SourceKind.HUMAN


def test_live_whole_utterance_proposal_parses_but_cannot_be_accepted_as_fact():
    raw = {"facts": [{
        "category": "user_profile", "scope_key": "user_name",
        "text": "selam benim adım can senin ne",
        "source_handle": "human_current",
        "evidence_quote": "selam benim adım can senin ne",
    }]}
    proposal = MemoryProposal.model_validate(raw)
    prior = ConversationMemory()
    request = build_memory_update_request(
        terminal("selam benim adım can senin ne"), prior,
    )
    with pytest.raises(ValueError, match="normalized durable fact"):
        LLMMemoryUpdater(Provider(proposal)).propose(request)
    assert prior == ConversationMemory()


def test_json_proposal_cannot_invent_a_trusted_source_handle():
    raw = {"facts": [{
        "category": "user_profile", "scope_key": "user.name",
        "text": "The user's name is Can",
        "source_handle": "accepted:invented",
        "evidence_quote": "Benim adım Can.",
    }]}
    proposal = MemoryProposal.model_validate(raw)
    prior = ConversationMemory()
    request = build_memory_update_request(terminal(), prior)
    with pytest.raises(ValueError, match="unsupported human evidence"):
        LLMMemoryUpdater(Provider(proposal)).propose(request)
    assert prior == ConversationMemory()


def test_normalized_preference_is_distinct_from_exact_supporting_quote():
    request = build_memory_update_request(
        terminal("Please keep future answers concise."), ConversationMemory(),
    )
    proposal = MemoryProposal.model_validate({"facts": [{
        "category": "user_preference", "scope_key": "response.length",
        "text": "The user prefers concise answers",
        "source_handle": "human_current",
        "evidence_quote": "Please keep future answers concise.",
    }]})
    update = LLMMemoryUpdater(Provider(proposal)).propose(request)
    assert update.facts[0].text == "The user prefers concise answers"
    assert update.facts[0].source.kind is SourceKind.HUMAN


def _turns(count):
    return tuple(message for index in range(1, count + 1) for message in (
        HumanMessage(content=f"question {index}"), accepted_answer(f"answer {index}"),
    ))


def test_compaction_keeps_minimum_verbatim_window_and_whole_pairs():
    result = compact_recent_conversation(
        _turns(7), completed_turn_count=7,
        maintenance_start_turn=1, maintenance_end_turn=7,
        limits=CompactionLimits(target_turns=4, minimum_verbatim_turns=2),
    )
    assert [m.content for m in result] == [
        "question 4", "answer 4", "question 5", "answer 5",
        "question 6", "answer 6", "question 7", "answer 7",
    ]


def test_compaction_protects_unresolved_question_and_unmaintained_legacy_turns():
    question = OpenQuestion(question_id="q", text="question 1", source_turn_index=1,
                            source_turn_id="turn-1", status=QuestionStatus.OPEN)
    result = compact_recent_conversation(
        _turns(7), completed_turn_count=7,
        maintenance_start_turn=1, maintenance_end_turn=7,
        questions=(question,), limits=CompactionLimits(target_turns=4, minimum_verbatim_turns=2),
    )
    assert [m.content for m in result][:2] == ["question 1", "answer 1"]
    assert len(result) == 8
    legacy = compact_recent_conversation(
        _turns(7), completed_turn_count=7,
        maintenance_start_turn=5, maintenance_end_turn=7,
        limits=CompactionLimits(target_turns=4, minimum_verbatim_turns=2),
    )
    assert len(legacy) == 14
    assert len(bounded_recent_conversation(_turns(20))) == 32


def _stub_cli(monkeypatch, path, prompt, *, enabled=True):
    settings = dict(cli.DEFAULT_SETTINGS, session_file=str(path), memory_llm_enabled=enabled)
    monkeypatch.setattr(cli, "parse_args", lambda: SimpleNamespace(prompt=prompt, resume=True))
    monkeypatch.setattr(cli, "_build_settings", lambda args: settings)
    monkeypatch.setattr(cli, "build_app", lambda **kwargs: object())
    return settings


def _completed_run(app, prompt, **kwargs):
    index = kwargs["turn_index"]
    kwargs["completed_turn_evidence"].append(terminal(
        prompt, turn=index, answer="Delivered answer",
    ))
    return [*kwargs["history"], HumanMessage(content=prompt), accepted_answer("Delivered answer")], ""


def test_semantic_name_and_correction_reach_existing_planner_path(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    provider = Provider(
        MemoryProposal(facts=(proposed_user(),)), MemoryProposal(),
        MemoryProposal(facts=(proposed_user(
            quote="Benim adım artık Cem.",
            text="The user's name is Cem",
        ),)), MemoryProposal(),
    )
    updater = LLMMemoryUpdater(provider)
    monkeypatch.setattr(cli, "create_optional_memory_updater", lambda settings: updater)
    contexts = []

    def run(app, prompt, **kwargs):
        contexts.append(kwargs["planner_memory_context"])
        return _completed_run(app, prompt, **kwargs)

    monkeypatch.setattr(cli, "run_prompt", run)
    for prompt in ("Benim adım Can.", "Benim adım ne?",
                   "Benim adım artık Cem.", "Benim adım ne?"):
        _stub_cli(monkeypatch, path, prompt)
        cli.main()
    assert len(provider.requests) == 4
    assert contexts[1].user_facts[0].text == "The user's name is Can"
    assert contexts[3].user_facts[0].text == "The user's name is Cem"
    assert "Can" not in contexts[3].user_facts[0].text
    restored = cli.load_session(str(path))
    assert restored.completed_turn_count == 4
    assert restored.maintenance_start_turn == 1
    assert restored.maintenance_end_turn == 4
    assert restored.conversation_memory.facts[0].source.kind == SourceKind.HUMAN
    assert project_planner_memory(restored.conversation_memory).user_facts[0].text == "The user's name is Cem"


def test_planned_terminal_fact_persists_only_from_accepted_completion(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    _stub_cli(monkeypatch, path, "Inspect repository")
    accepted = AcceptedCompletion(
        step_id="inspect", summary="Repository uses Python",
        evidence_id="result-1", plan_id="plan", plan_revision=1,
    )
    updater = LLMMemoryUpdater(Provider(MemoryProposal(facts=(
        proposed_project("Repository uses Python"),
    ))))
    monkeypatch.setattr(cli, "create_optional_memory_updater", lambda settings: updater)

    def run(app, prompt, **kwargs):
        kwargs["completed_turn_evidence"].append(terminal(
            prompt, turn=kwargs["turn_index"], completions=(accepted,),
            answer="Repository uses Rust",
        ))
        return [HumanMessage(content=prompt), accepted_answer("Repository uses Rust")], ""

    monkeypatch.setattr(cli, "run_prompt", run)
    cli.main()
    restored = cli.load_session(str(path))
    assert restored.conversation_memory.facts[0].text == "Repository uses Python"
    assert restored.conversation_memory.facts[0].source.kind == SourceKind.ACCEPTED_RESULT
    assert "Repository uses Rust" not in restored.conversation_memory.facts[0].text


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), ValueError("invalid")])
def test_enrichment_failure_preserves_deterministic_memory_answer_and_streak(tmp_path, monkeypatch, failure):
    path = tmp_path / "session.json"
    _stub_cli(monkeypatch, path, "Which file?")

    class FailingUpdater:
        calls = 0

        def propose(self, request):
            self.calls += 1
            raise failure

    updater = FailingUpdater()
    monkeypatch.setattr(cli, "create_optional_memory_updater", lambda settings: updater)
    monkeypatch.setattr(cli, "run_prompt", _completed_run)
    cli.main()
    restored = cli.load_session(str(path))
    assert updater.calls == 1
    assert restored.last_enrichment_status == "failed_safe"
    assert restored.maintenance_end_turn == 1
    assert restored.conversation_memory.continuity.topic == "Which file?"
    assert restored.recent_conversation[-1].content == "Delivered answer"


def test_disabled_updater_and_nonterminal_turn_make_no_extra_call(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    _stub_cli(monkeypatch, path, "request", enabled=False)
    monkeypatch.setattr(cli, "create_optional_memory_updater", lambda settings: None)
    monkeypatch.setattr(cli, "run_prompt", _completed_run)
    cli.main()
    assert cli.load_session(str(path)).last_enrichment_status == "disabled"

    _stub_cli(monkeypatch, path, "waiting", enabled=True)
    class CountingUpdater:
        calls = 0

        def propose(self, request):
            self.calls += 1
            return MemoryUpdate()

    updater = CountingUpdater()
    monkeypatch.setattr(cli, "create_optional_memory_updater", lambda settings: updater)
    monkeypatch.setattr(cli, "run_prompt", lambda app, prompt, **kwargs: (
        [*kwargs["history"], HumanMessage(content=prompt)], "",
    ))
    cli.main()
    assert updater.calls == 0


def test_repeated_llm_failure_still_compacts_after_safe_deterministic_maintenance(tmp_path, monkeypatch):
    path = tmp_path / "session.json"

    class UnavailableUpdater:
        calls = 0

        def propose(self, request):
            self.calls += 1
            raise TimeoutError("unavailable")

    updater = UnavailableUpdater()
    monkeypatch.setattr(cli, "create_optional_memory_updater", lambda settings: updater)
    monkeypatch.setattr(cli, "run_prompt", _completed_run)
    for index in range(1, 15):
        _stub_cli(monkeypatch, path, f"request {index}")
        cli.main()
    restored = cli.load_session(str(path))
    assert updater.calls == 14
    assert restored.last_enrichment_status == "failed_safe"
    assert restored.maintenance_start_turn == 1
    assert restored.maintenance_end_turn == 14
    assert len(restored.recent_conversation) == 24
    assert restored.recent_conversation[0].content == "request 3"
    assert restored.recent_conversation[-1].content == "Delivered answer"


def test_async_waiting_polls_do_not_invoke_updater_before_terminal_resume(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    _stub_cli(monkeypatch, path, "Generate an image")

    class CountingUpdater:
        calls = 0

        def propose(self, request):
            self.calls += 1
            return MemoryUpdate()

    updater = CountingUpdater()
    monkeypatch.setattr(cli, "create_optional_memory_updater", lambda settings: updater)
    waiting_cursor = ExecutionCursor(
        phase=ExecutionPhase.WAITING, step_id="step-1",
        current_worker=WorkerRole.CONTROLLER,
    )
    waiting = ControllerDecision(
        decision_type=ControllerDecisionType.AWAIT_ASYNC_JOB,
        cursor=waiting_cursor, async_job_id="job-1", requires_checkpoint=True,
    )

    class AsyncApp:
        def __init__(self):
            self.async_runtime = self
            self.polls = 0
            self.execution_id = None

        def stream(self, initial_state, config=None):
            self.execution_id = initial_state["run_id"]
            return iter([{"controller": {"controller_decision": waiting}}])

        def wake_and_resume(self, *, config, wake):
            self.polls += 1
            if self.polls == 1:
                assert updater.calls == 0
                return iter([{"controller": {"controller_decision": waiting}}])
            cursor = ExecutionCursor(
                phase=ExecutionPhase.COMPLETED, current_worker=WorkerRole.SUMMARY,
            )
            decision = ControllerDecision(
                decision_type=ControllerDecisionType.DISPATCH_SUMMARY,
                execution_status=ExecutionStatus.COMPLETED,
                cursor=cursor, terminal=True,
            )
            state = ExecutionState(protocol_visible=ProtocolVisibleState(
                identity=ExecutionIdentity(execution_id=self.execution_id,
                                           protocol_version="1"),
                status=ExecutionStatus.COMPLETED, cursor=cursor,
            ))
            final = FinalizationResult(
                execution_summary=ExecutionSummary(
                    execution_id=self.execution_id,
                    status=ExecutionStatus.COMPLETED, summary_text="Completed.",
                ),
                final_answer="Image ready.",
            )
            return iter([{"controller": {
                "controller_decision": decision, "execution_state": state,
                "finalization_result": final,
            }}])

    app = AsyncApp()
    monkeypatch.setattr(cli, "build_app", lambda **kwargs: app)
    cli.main()
    assert app.polls == 2
    assert updater.calls == 1
    restored = cli.load_session(str(path))
    assert restored.completed_turn_count == 1
    assert restored.recent_conversation[-1].content == "Image ready."


def test_failed_execution_may_retain_explicit_user_fact_without_success_claim():
    failed = terminal("Cevapları kısa tut.", status=TurnStatus.FAILED, answer="Failed")
    provider = Provider(MemoryProposal(facts=(ProposedMemoryFact(
        category=FactCategory.USER_PREFERENCE, scope_key="response.length",
        text="The user prefers concise answers", source_handle="human_current",
        evidence_quote="Cevapları kısa tut.",
    ),)))
    request = build_memory_update_request(failed, ConversationMemory())
    deterministic = merge_memory(ConversationMemory(), extract_memory_update(failed))
    merged = merge_memory(deterministic, LLMMemoryUpdater(provider).propose(request))
    assert merged.continuity.status == TurnStatus.FAILED
    assert merged.facts[0].source.kind == SourceKind.HUMAN
    assert merged.facts[0].claim.value == "observation"
