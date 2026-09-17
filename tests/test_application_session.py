"""Slice 2 application session persistence and trusted recent conversation."""

import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_to_dict

import main as cli
from core.application_session import (
    ApplicationSession, RecentConversationLimits, bounded_recent_conversation,
)
from core.graph_messages import ACCEPTED_FINALIZER_PROVENANCE, CONVERSATION_PROVENANCE_KEY
from core.memory import ConversationMemory, FactCategory, MemoryFact, MemorySource, SourceKind


def accepted(text):
    return AIMessage(content=text, additional_kwargs={
        CONVERSATION_PROVENANCE_KEY: ACCEPTED_FINALIZER_PROVENANCE,
    })


def memory_with_provenance():
    return ConversationMemory(facts=(MemoryFact(
        category=FactCategory.USER_PREFERENCE,
        scope_key="editor",
        text="Prefers Vim",
        source=MemorySource(kind=SourceKind.HUMAN, turn_index=1, turn_id="turn-1"),
    ),))


def contents(messages):
    return [message.content for message in messages]


def test_empty_session_and_memory_round_trip(tmp_path):
    path = tmp_path / "session.json"
    assert ApplicationSession().conversation_memory == ConversationMemory()
    original = ApplicationSession(memory_with_provenance(), (
        HumanMessage(content="Editor?"), accepted("Vim"),
    ))
    cli.save_session(str(path), original)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["session_schema_version"] == 2
    assert payload["conversation_memory"]["schema_version"] == 1
    assert "history" not in payload
    restored = cli.load_session(str(path))
    assert restored.conversation_memory == original.conversation_memory
    assert restored.conversation_memory.facts[0].source.turn_id == "turn-1"
    assert contents(restored.recent_conversation) == ["Editor?", "Vim"]


def test_malformed_memory_falls_back_without_losing_valid_history(tmp_path, capsys):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({
        "session_schema_version": 2,
        "conversation_memory": {"schema_version": 1, "facts": "invalid"},
        "recent_conversation": messages_to_dict([HumanMessage(content="Keep me")]),
    }), encoding="utf-8")
    restored = cli.load_session(str(path))
    assert restored.conversation_memory == ConversationMemory()
    assert contents(restored.recent_conversation) == ["Keep me"]
    assert "Invalid conversation memory" in capsys.readouterr().err


def test_legacy_session_loads_without_importing_summary_as_memory(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({
        "rolling_summary": "legacy facts",
        "history": messages_to_dict([HumanMessage(content="old"), accepted("answer")]),
    }), encoding="utf-8")
    restored = cli.load_session(str(path))
    assert restored.conversation_memory == ConversationMemory()
    assert restored.legacy_rolling_summary == "legacy facts"
    assert contents(restored.recent_conversation) == ["old", "answer"]


def test_malformed_session_root_and_history_fail_safely(tmp_path):
    path = tmp_path / "session.json"
    path.write_text("{bad", encoding="utf-8")
    assert cli.load_session(str(path)) == ApplicationSession()
    path.write_text(json.dumps({
        "session_schema_version": 2, "conversation_memory": memory_with_provenance().model_dump(mode="json"),
        "recent_conversation": "wrong type",
    }), encoding="utf-8")
    restored = cli.load_session(str(path))
    assert restored.conversation_memory == memory_with_provenance()
    assert restored.recent_conversation == ()


def test_recent_turn_limits_prune_old_pairs_and_preserve_new_pairs():
    messages = [message for i in range(5) for message in (
        HumanMessage(content=f"user-{i}"), accepted(f"answer-{i}"),
    )]
    bounded = bounded_recent_conversation(messages, RecentConversationLimits(
        max_turns=2, max_messages=4, max_message_chars=100, max_total_chars=100,
    ))
    assert contents(bounded) == ["user-3", "answer-3", "user-4", "answer-4"]
    by_chars = bounded_recent_conversation(messages, RecentConversationLimits(
        max_turns=5, max_messages=10, max_message_chars=100, max_total_chars=16,
    ))
    assert contents(by_chars) == ["user-4", "answer-4"]


def test_oversized_message_drops_whole_turn_and_never_clips():
    messages = [HumanMessage(content="first"), accepted("reply"),
                HumanMessage(content="x" * 11), accepted("short")]
    bounded = bounded_recent_conversation(messages, RecentConversationLimits(
        max_turns=4, max_messages=8, max_message_chars=10, max_total_chars=40,
    ))
    assert contents(bounded) == ["first", "reply"]


def test_only_canonical_user_and_accepted_finalizer_messages_persist(tmp_path):
    path = tmp_path / "session.json"
    malicious = accepted("tool call")
    malicious.tool_calls = [{"name": "x", "args": {}, "id": "call"}]
    session = ApplicationSession(recent_conversation=(
        HumanMessage(content="user", additional_kwargs={"extra": "discard"}),
        AIMessage(content="unaccepted"),
        ToolMessage(content="tool output", tool_call_id="call"),
        malicious,
        accepted("trusted"),
    ))
    cli.save_session(str(path), session)
    restored = cli.load_session(str(path))
    assert contents(restored.recent_conversation) == ["user", "trusted"]
    assert restored.recent_conversation[0].additional_kwargs == {}
    assert restored.recent_conversation[1].additional_kwargs == {
        CONVERSATION_PROVENANCE_KEY: ACCEPTED_FINALIZER_PROVENANCE,
    }
    payload = path.read_text(encoding="utf-8")
    assert "unaccepted" not in payload and "tool output" not in payload
    assert "execution_state" not in payload and "tool_execution_history" not in payload


def _stub_cli(monkeypatch, path, *, prompt):
    settings = dict(cli.DEFAULT_SETTINGS, session_file=str(path))
    monkeypatch.setattr(cli, "parse_args", lambda: SimpleNamespace(prompt=prompt, resume=True))
    monkeypatch.setattr(cli, "_build_settings", lambda args: settings)
    monkeypatch.setattr(cli, "build_app", lambda **kwargs: object())
    return settings


def test_one_shot_persists_returned_turn_and_keeps_memory_out_of_workers(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    cli.save_session(str(path), ApplicationSession(memory_with_provenance()))
    _stub_cli(monkeypatch, path, prompt="new request")
    seen = []

    def run(app, prompt, *, history, rolling_summary, show_summary, **kwargs):
        seen.append((prompt, contents(history), rolling_summary))
        assert "conversation_memory" not in str(history)
        return [*history, HumanMessage(content=prompt), accepted("done")], rolling_summary

    monkeypatch.setattr(cli, "run_prompt", run)
    cli.main()
    restored = cli.load_session(str(path))
    assert seen == [("new request", [], "")]
    assert contents(restored.recent_conversation) == ["new request", "done"]
    assert restored.conversation_memory == memory_with_provenance()


def test_interactive_persists_after_each_completed_turn(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    _stub_cli(monkeypatch, path, prompt=None)
    prompts = iter(["first", "second", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))
    observed = []

    def run(app, prompt, *, history, rolling_summary, show_summary, **kwargs):
        if prompt == "second":
            assert contents(cli.load_session(str(path)).recent_conversation) == [
                "first", "reply first",
            ]
        observed.append(contents(history))
        return [*history, HumanMessage(content=prompt), accepted(f"reply {prompt}")], ""

    monkeypatch.setattr(cli, "run_prompt", run)
    cli.main()
    assert observed == [[], ["first", "reply first"]]
    assert contents(cli.load_session(str(path)).recent_conversation) == [
        "first", "reply first", "second", "reply second",
    ]


def test_failed_persistence_does_not_change_successful_turn(tmp_path, monkeypatch, capsys):
    path = tmp_path / "session.json"
    _stub_cli(monkeypatch, path, prompt="request")
    monkeypatch.setattr(cli, "run_prompt", lambda app, prompt, **kwargs: (
        [HumanMessage(content=prompt), accepted("success")], "",
    ))
    original_open = open

    def fail_write(file, mode="r", *args, **kwargs):
        if str(file) == str(path) and "w" in mode:
            raise OSError("disk unavailable")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fail_write)
    cli.main()
    assert "Failed to save session state" in capsys.readouterr().err


def test_memory_projection_is_separate_from_execution_state_and_messages(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    cli.save_session(str(path), ApplicationSession(
        conversation_memory=memory_with_provenance(),
        recent_conversation=(HumanMessage(content="previous"), accepted("prior answer")),
    ))
    _stub_cli(monkeypatch, path, prompt="next")

    class CapturingApp:
        initial_state = None

        def stream(self, initial_state):
            self.initial_state = initial_state
            return iter(())

    app = CapturingApp()
    monkeypatch.setattr(cli, "build_app", lambda **kwargs: app)
    cli.main()
    assert "conversation_memory" not in app.initial_state
    assert "Prefers Vim" not in str(app.initial_state["execution_state"])
    assert "Prefers Vim" in str(app.initial_state["planner_memory_context"])
    assert contents(app.initial_state["messages"]) == ["previous", "prior answer", "next"]
