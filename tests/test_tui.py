"""Native-scrollback shell: real key processing and recorded terminal output."""

import asyncio
import io
import json
import logging
import sys
import threading
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.utils import get_cwidth

import main as cli
from core.debug import log_llm_exchange
from core.graph_messages import ACCEPTED_FINALIZER_PROVENANCE, CONVERSATION_PROVENANCE_KEY
from core.logging.node_update import NodeUpdate
from core.logging.renderer import render_node_update, render_system_message
from core.protocol.enums import ExecutionStatus
from core.protocol.models import ExecutionPlan, ExecutionStep, ExecutionSummary, FinalizationResult, ToolResult
from core.tui import CortexTUI


class RecordingOutput(DummyOutput):
    def __init__(self):
        self.permanent = []
        self.paint = []
        self.alternate_screen = False
        self.mouse_enabled = False

    def write_raw(self, text):
        self.permanent.append(text)

    def write(self, text):
        self.paint.append(text)

    def enter_alternate_screen(self):
        self.alternate_screen = True

    def enable_mouse_support(self):
        self.mouse_enabled = True

    @property
    def text(self):
        return ''.join(self.permanent)


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), timeout=4)


async def stop(shell, task):
    if not task.done():
        shell.request_exit()
    await asyncio.wait_for(task, 4)


@pytest.mark.parametrize('vim_mode', [False, True])
def test_multiline_submit_and_background_output_preserve_active_draft(vim_mode):
    started, release = threading.Event(), threading.Event()
    seen = []
    output = RecordingOutput()

    def complete(prompt, status):
        seen.append((prompt, threading.get_ident(), status.enabled))
        status.start('brain')
        started.set()
        try:
            assert release.wait(4)
            render_system_message('Finished turn')
        finally:
            status.stop()

    async def scenario(pipe, shell):
        task = asyncio.create_task(shell.run_async())
        try:
            await until(lambda: shell.app.is_running)
            pipe.send_text('first line\r  second line')
            await until(lambda: shell.editor.text == 'first line\n  second line')
            assert not seen
            pipe.send_text('\x1bOQ')  # F2.
            await until(started.is_set)
            assert len(seen) == 1
            assert seen[0][0] == 'first line\n  second line'
            assert seen[0][1] != threading.get_ident() and seen[0][2] is False
            pipe.send_text('next draft\x1b[D\x1b[D')
            await until(lambda: shell.editor.text == 'next draft' and shell.editor.buffer.cursor_position == 8)
            draft = shell.editor.buffer.document
            shell.stream.write('output during editing\n')
            await until(lambda: 'output during editing' in output.text)
            assert shell.editor.buffer.document == draft
            assert shell.app.layout.has_focus(shell.editor)
            assert shell.editor.window.render_info.window_height == 5
            pipe.send_text('\x1bOQ')
            await asyncio.sleep(0.05)
            assert len(seen) == 1 and shell.editor.buffer.document == draft
            release.set()
            await until(lambda: not shell.running and 'Finished turn' in output.text)
            assert shell.editor.buffer.document == draft
            await stop(shell, task)
        finally:
            release.set()
            await stop(shell, task)

    with create_pipe_input() as pipe:
        shell = CortexTUI(complete, vim_mode=vim_mode, input=pipe, output=output)
        assert shell.app.editing_mode == (EditingMode.VI if vim_mode else EditingMode.EMACS)
        asyncio.run(scenario(pipe, shell))
    assert output.text.count('output during editing') == 1
    assert output.text.count('Finished turn') == 1
    assert not output.alternate_screen and not output.mouse_enabled


@pytest.mark.parametrize('vim_mode', [False, True])
def test_alt_enter_submission_and_multiline_history(vim_mode):
    seen = []
    async def scenario(pipe, shell):
        task = asyncio.create_task(shell.run_async())
        try:
            await until(lambda: shell.app.is_running)
            pipe.send_text('line one\rline two\x1b\r')
            await until(lambda: bool(seen) and not shell.running)
            await until(lambda: shell.editor.buffer.working_index > 0)
            pipe.send_text('\x1b[1;5A')  # Ctrl-Up, library previous history.
            await until(lambda: shell.editor.text == 'line one\nline two')
            assert shell.editor.buffer.history.get_strings() == ['line one\nline two']
            assert shell.app.layout.has_focus(shell.editor)
            await stop(shell, task)
        finally:
            await stop(shell, task)
    with create_pipe_input() as pipe:
        shell = CortexTUI(lambda prompt, status: seen.append(prompt), vim_mode=vim_mode,
                          input=pipe, output=RecordingOutput())
        asyncio.run(scenario(pipe, shell))
    assert seen == ['line one\nline two']


def test_shell_has_no_output_pane_or_application_output_navigation():
    with create_pipe_input() as pipe:
        shell = CortexTUI(lambda *args: None, input=pipe, output=RecordingOutput())
        assert not hasattr(shell, 'events')
        assert not shell.app.full_screen and not shell.app.mouse_support()
        for key in (Keys.PageUp, Keys.PageDown, Keys.ControlHome, Keys.ControlEnd,
                    Keys.ScrollUp, Keys.ScrollDown, Keys.F6):
            assert not shell.app.key_bindings.get_bindings_for_keys((key,))


def test_output_routes_renderer_stderr_logging_and_raw_records_once(tmp_path, monkeypatch):
    output = RecordingOutput()
    raw_path = tmp_path / 'raw.jsonl'
    monkeypatch.setenv('CORTEX_RAW_LLM_FILE', str(raw_path))
    original_stdout, original_stderr = sys.stdout, sys.stderr
    handler = logging.StreamHandler(original_stderr)
    file_handler = logging.FileHandler(tmp_path / 'diagnostic.log', encoding='utf-8')
    root = logging.getLogger()
    root.addHandler(handler)
    root.addHandler(file_handler)

    async def scenario(pipe, shell):
        task = asyncio.create_task(shell.run_async())
        try:
            await until(lambda: shell.app.is_running)
            shell.editor.text = 'unfinished\ndraft'
            shell.editor.buffer.cursor_position = 4
            draft = shell.editor.buffer.document
            for success in (True, False):
                render_node_update(NodeUpdate(from_node='tools', to_node='controller',
                    tool_name='read_file', active_step_id='read',
                    tool_result=ToolResult(request_id=str(success), success=success, message='Read page')))
            print('stderr warning', file=sys.stderr)
            root.warning('console warning')
            log_llm_exchange(worker='brain', operation='step', messages=[HumanMessage(content='request')],
                             response=AIMessage(content='response'), enabled=True)
            await until(lambda: '[raw-llm:brain:step]' in output.text)
            assert shell.editor.buffer.document == draft
            assert file_handler.stream is not shell.stream
            await stop(shell, task)
        finally:
            await stop(shell, task)

    try:
        with create_pipe_input() as pipe:
            asyncio.run(scenario(pipe, CortexTUI(lambda *args: None, input=pipe, output=output)))
        assert output.text.count('OK \u2014 Read page') == 1
        assert output.text.count('FAIL \u2014 Read page') == 1
        assert output.text.count('stderr warning') == 1
        assert output.text.count('console warning') == 1
        assert output.text.count('[raw-llm:brain:step]') == 1
        assert sys.stdout is original_stdout and sys.stderr is original_stderr
        assert handler.stream is original_stderr
        assert 'console warning' in (tmp_path / 'diagnostic.log').read_text(encoding='utf-8')
        records = [json.loads(line) for line in raw_path.read_text(encoding='utf-8').splitlines()]
        assert len(records) == 1 and records[0]['response']['content'] == 'response'
    finally:
        root.removeHandler(handler)
        root.removeHandler(file_handler)
        file_handler.close()


def test_partial_lines_and_final_output_are_flushed_before_exit():
    output = RecordingOutput()
    async def scenario(pipe, shell):
        task = asyncio.create_task(shell.run_async())
        try:
            await until(lambda: shell.app.is_running)
            shell.stream.write('partial')
            frames = shell.app.render_counter
            await until(lambda: shell.app.render_counter > frames + 1)
            assert not output.text
            shell.stream.write(' line\n')
            await until(lambda: output.text == 'partial line\n')
            shell.stream.write('final fragment')
            await stop(shell, task)
            assert output.text == 'partial line\nfinal fragment'
        finally:
            await stop(shell, task)
    with create_pipe_input() as pipe:
        asyncio.run(scenario(pipe, CortexTUI(lambda *args: None, input=pipe, output=output)))


def test_live_toolbar_refreshes_without_printing_status_to_scrollback(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('core.logging.live_status.time.perf_counter', lambda: clock[0])
    started, release = threading.Event(), threading.Event()
    output = RecordingOutput()
    def complete(prompt, status):
        status.start('planner')
        try:
            status.set_brain_step_context(execution_id='live', step_id='inspect',
                plan=ExecutionPlan(plan_id='p', steps=(ExecutionStep(step_id='inspect', title='Inspect logs'),)))
            status.begin_provider_invocation(worker='brain')
            status.update_tool('read_file')
            started.set()
            assert release.wait(4)
        finally:
            status.stop()
    async def scenario(pipe, shell):
        frames = []
        shell.app.after_render += lambda app: frames.append(shell.status_text())
        task = asyncio.create_task(shell.run_async())
        try:
            await until(lambda: shell.app.is_running)
            assert 'IDLE' in shell.status_text()
            pipe.send_text('inspect logs\x1bOQ')
            await until(started.is_set)
            await until(lambda: output.text == '\nYou> inspect logs\n')
            frames.clear()
            clock[0] = 118.4
            shell.live_status.add_usage({'total_tokens': 36700}, worker='brain')
            await until(lambda: any('18.4s' in text and '36.7k tok' in text for text in frames))
            assert 'RUNNING' in shell.status_text() and 'read_file' in shell.status_text()
            assert 'Step 1/1: Inspect' in shell.status_text()
            shell.live_status.begin_provider_invocation(worker='brain')
            assert 'call 2' in shell.status_text()
            assert output.text == '\nYou> inspect logs\n'
            release.set()
            await until(lambda: not shell.running)
            assert 'IDLE' in shell.status_text() and '36.7k tok' in shell.status_text()
            await stop(shell, task)
        finally:
            release.set()
            await stop(shell, task)
    with create_pipe_input() as pipe:
        asyncio.run(scenario(pipe, CortexTUI(complete, input=pipe, output=output)))


@pytest.mark.parametrize('runtime_status,answer_error', [
    (ExecutionStatus.COMPLETED, None), (ExecutionStatus.FAILED, None),
    (ExecutionStatus.CANCELLED, None), (ExecutionStatus.COMPLETED, 'Provider error'),
])
def test_idle_toolbar_preserves_runtime_outcome_and_authoritative_answer(runtime_status, answer_error):
    output = RecordingOutput()
    answer = 'The requested question was absent and could not be solved.'
    result = FinalizationResult(execution_summary=ExecutionSummary(
        execution_id='outcome', status=runtime_status, summary_text='Terminal facts'),
        final_answer=answer, final_answer_error=answer_error)
    before = result.model_dump(mode='json')
    def complete(prompt, status):
        status.start('finalizer')
        try:
            status.add_usage({'total_tokens': 12300}, worker='brain')
            render_node_update(NodeUpdate(from_node='controller', to_node='controller', finalization_result=result))
        finally:
            status.stop()
    async def scenario(pipe, shell):
        task = asyncio.create_task(shell.run_async())
        try:
            await until(lambda: shell.app.is_running)
            pipe.send_text('request\x1bOQ')
            await until(lambda: answer in output.text and not shell.running)
            text = shell.status_text()
            assert f'IDLE \u00b7 Runtime {runtime_status.value}' in text and '12.3k tok' in text
            if answer_error:
                assert 'answer unavailable' in text
            assert result.model_dump(mode='json') == before
            await stop(shell, task)
        finally:
            await stop(shell, task)
    with create_pipe_input() as pipe:
        asyncio.run(scenario(pipe, CortexTUI(complete, input=pipe, output=output)))
    assert output.text.count(answer) == 1


def test_quit_waits_for_active_turn_and_final_output():
    started, release = threading.Event(), threading.Event()
    output = RecordingOutput()
    def complete(prompt, status):
        started.set()
        assert release.wait(4)
        print('final output before exit')
    async def scenario(pipe, shell):
        task = asyncio.create_task(shell.run_async())
        try:
            await until(lambda: shell.app.is_running)
            pipe.send_text('work\x1bOQ')
            await until(started.is_set)
            pipe.send_text('\x11')
            await until(lambda: shell.exit_when_idle)
            assert not task.done()
            release.set()
            await asyncio.wait_for(task, 4)
            assert output.text.count('final output before exit') == 1
        finally:
            release.set()
            await stop(shell, task)
    with create_pipe_input() as pipe:
        asyncio.run(scenario(pipe, CortexTUI(complete, input=pipe, output=output)))


def test_turn_error_is_printed_once_and_toolbar_returns_to_idle():
    output = RecordingOutput()
    def fail(prompt, status):
        raise ValueError('test failure')
    async def scenario(pipe, shell):
        task = asyncio.create_task(shell.run_async())
        try:
            await until(lambda: shell.app.is_running)
            pipe.send_text('request\x1bOQ')
            await until(lambda: '[error] ValueError: test failure' in output.text)
            assert 'IDLE' in shell.status_text() and 'Error: ValueError' in shell.status_text()
            await stop(shell, task)
        finally:
            await stop(shell, task)
    with create_pipe_input() as pipe:
        asyncio.run(scenario(pipe, CortexTUI(fail, input=pipe, output=output)))
    assert output.text.count('[error] ValueError: test failure') == 1


@pytest.mark.parametrize("vim_mode", [False, True])
def test_long_step_title_leaves_elapsed_tokens_and_call_count_visible(vim_mode):
    from core.logging.live_status import LiveStatus

    with create_pipe_input() as pipe:
        shell = CortexTUI(lambda *args: None, vim_mode=vim_mode, input=pipe, output=DummyOutput())
        status = shell.live_status = LiveStatus(enabled=False)
        shell.running = True
        status.start("planner")
        try:
            status.set_brain_step_context(execution_id="width", step_id="step",
                plan=ExecutionPlan(plan_id="p", steps=(
                    ExecutionStep(step_id="step", title="Inspect runtime logs " * 20),
                )))
            status.begin_provider_invocation(worker="brain")
            status.add_usage({"total_tokens": 36700}, worker="brain")
            text = shell.status_text()
            assert get_cwidth(text) <= 80
            assert "Step 1/1: I" in text and "…" in text
            assert "0.0s" in text and "call 1" in text and "36.7k tok" in text
            assert ("Vi insert" if vim_mode else "Emacs") in text
        finally:
            status.stop()



def test_tui_and_vim_settings_follow_existing_precedence(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text('{"tui": true, "vim_mode": false}', encoding="utf-8")
    monkeypatch.setenv("CORTEX_TUI", "false")
    monkeypatch.setenv("CORTEX_VIM_MODE", "true")
    monkeypatch.setattr(sys, "argv", ["main.py"])
    settings = cli._build_settings(cli.parse_args())
    assert not settings["tui"] and settings["vim_mode"]
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config)])
    settings = cli._build_settings(cli.parse_args())
    assert settings["tui"] and not settings["vim_mode"]
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config), "--plain", "--vim-mode"])
    settings = cli._build_settings(cli.parse_args())
    assert not settings["tui"] and settings["vim_mode"]



@pytest.mark.parametrize("stream_name", ["stdin", "stdout", "stderr"])
def test_redirected_streams_use_plain_console(monkeypatch, stream_name):
    for name in ("stdin", "stdout", "stderr"):
        monkeypatch.setattr(sys, name, SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(sys, stream_name, io.StringIO())
    assert not cli.run_interactive_tui(lambda *args: pytest.fail("unexpected turn"), {"tui": True})



def test_plain_flag_dumb_terminal_and_missing_dependency_fall_back(monkeypatch):
    for name in ("stdin", "stdout", "stderr"):
        stream = io.StringIO()
        stream.isatty = lambda: True
        monkeypatch.setattr(sys, name, stream)
    assert not cli.run_interactive_tui(None, {"tui": False})
    monkeypatch.setenv("TERM", "dumb")
    assert not cli.run_interactive_tui(None, {"tui": True})
    monkeypatch.delenv("TERM")
    monkeypatch.setitem(sys.modules, "core.tui", None)
    assert not cli.run_interactive_tui(None, {"tui": True})
    assert "using plain console" in sys.stdout.getvalue()



def test_interactive_selection_passes_callback_and_vim_mode(monkeypatch):
    for name in ("stdin", "stdout", "stderr"):
        monkeypatch.setattr(sys, name, SimpleNamespace(isatty=lambda: True))
    monkeypatch.delenv("TERM", raising=False)
    seen = []
    callback = lambda *args: None

    class FakeShell:
        def __init__(self, complete_turn, *, vim_mode):
            seen.append((complete_turn, vim_mode))

        def run(self):
            seen.append("run")

    monkeypatch.setitem(sys.modules, "core.tui", SimpleNamespace(CortexTUI=FakeShell))
    assert cli.run_interactive_tui(callback, {"tui": True, "vim_mode": True})
    assert seen == [(callback, True), "run"]



@pytest.mark.parametrize("mode", ["tui", "plain", "one_shot", "eof"])
def test_cli_uses_existing_turn_and_session_path_in_each_mode(tmp_path, monkeypatch, mode):
    path = tmp_path / "session.json"
    settings = dict(cli.DEFAULT_SETTINGS, session_file=str(path), memory_llm_enabled=False,
                    tui=mode == "tui")
    monkeypatch.setattr(cli, "parse_args", lambda: SimpleNamespace(
        prompt="request" if mode == "one_shot" else "", resume=False,
    ))
    monkeypatch.setattr(cli, "_build_settings", lambda args: settings)
    monkeypatch.setattr(cli, "build_app", lambda **kwargs: object())
    seen, shells = [], []

    def run(app, prompt, **kwargs):
        status = kwargs["live_status"]
        seen.append((prompt, status))
        from core.logging.live_status import current_live_status
        assert current_live_status() is status
        return [HumanMessage(content=prompt), AIMessage(content="accepted answer", additional_kwargs={
            CONVERSATION_PROVENANCE_KEY: ACCEPTED_FINALIZER_PROVENANCE,
        })], ""

    def select_shell(callback, settings):
        shells.append(mode)
        if mode != "tui":
            return False
        from core.logging.live_status import LiveStatus
        assert callback("line one\n  line two", LiveStatus(enabled=False)) == "accepted answer"
        return True

    prompts = iter(["request", "exit"])

    def read(*args):
        if mode == "eof":
            raise EOFError
        return next(prompts)

    monkeypatch.setattr(cli, "run_prompt", run)
    monkeypatch.setattr(cli, "run_interactive_tui", select_shell)
    monkeypatch.setattr("builtins.input", read)
    cli.main()
    if mode == "one_shot":
        assert shells == []
    if mode == "eof":
        assert seen == []
    else:
        assert len(seen) == 1
        assert seen[0][0] == ("line one\n  line two" if mode == "tui" else "request")
        if mode == "tui":
            assert not seen[0][1].enabled
        session = cli.load_session(str(path))
        assert session.recent_conversation[-1].content == "accepted answer"
