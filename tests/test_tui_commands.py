"""Local commands exercise real submission, output suspension and terminal layout."""

import asyncio
import base64
import threading
from types import SimpleNamespace

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output

import main as cli
from core.application_session import ApplicationSession
from core.logging.live_status import LiveStatus
from core.logging.node_update import NodeUpdate
from core.logging.renderer import render_node_update
from core.protocol.enums import ExecutionStatus
from core.protocol.models import ExecutionSummary, FinalizationResult
from core.terminal_actions import clear_terminal, copy_to_clipboard
from core.terminal_style import TerminalStyle, strip_ansi
from core.tui import CortexTUI, TerminalStream
from core.tui_commands import Command


class RecordingOutput(DummyOutput):
    def __init__(self):
        self.events = []

    def get_size(self):
        return Size(30, 80)

    def get_rows_below_cursor_position(self):
        return 29

    def write_raw(self, text):
        self.events.append(text)

    def erase_screen(self):
        self.events.append('CLEAR')

    def enter_alternate_screen(self):
        pytest.fail('Must remain in the normal buffer')

    def enable_mouse_support(self):
        pytest.fail('Host must own mouse selection')

    @property
    def text(self):
        return ''.join(self.events)


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(.01)
    await asyncio.wait_for(wait(), 4)


async def finish(shell, task):
    shell.request_exit()
    await asyncio.wait_for(task, 4)


def finalization(answer, error=None, outcome=ExecutionStatus.COMPLETED):
    return NodeUpdate(from_node='controller', to_node='controller',
        finalization_result=FinalizationResult(
            execution_summary=ExecutionSummary(execution_id='test', status=outcome,
                                                summary_text='Terminal facts'),
            final_answer=answer, final_answer_error=error))


def test_commands_dispatch_locally_and_help_uses_registry(monkeypatch):
    seen = []
    output = RecordingOutput()
    monkeypatch.setattr(cli, 'save_session', lambda *a, **k: pytest.fail('Local command persisted'))

    async def scenario(pipe):
        shell = CortexTUI(lambda *a: pytest.fail('Local command invoked agent'), input=pipe, output=output)
        shell.commands.register(Command('/extra', 'A registered extension.', lambda: seen.append('extra')))
        task = asyncio.create_task(shell.run_async())
        try:
            await until(lambda: shell.app.is_running)
            for command, expected in [('/help', '/extra'), ('/unknown', 'Type /help'),
                                      ('/copy extra', 'Usage: /copy'), ('/extra', None)]:
                # Real Alt+Enter key processing, including multiline rejection.
                pipe.send_text(command + '\x1b\r')
                await until(lambda expected=expected: not shell.editor.text and (expected is None or expected in output.text))
            await until(lambda: seen == ['extra'])
            assert not shell.running and shell.worker is None and shell.live_status is None
            assert shell.editor.buffer.history.get_strings() == []
            assert 'You>' not in output.text
            pipe.send_text('draft\rsecond')
            await until(lambda: shell.editor.text == 'draft\nsecond')
            pipe.send_text('\x03')
            await until(lambda: shell.editor.text == '')
            pipe.send_text('\x11')
            await asyncio.wait_for(task, 4)
        finally:
            await finish(shell, task)

    with create_pipe_input() as pipe:
        asyncio.run(scenario(pipe))


@pytest.mark.parametrize('clipboard_error', [False, True])
@pytest.mark.parametrize('outcome', [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED])
def test_copy_uses_successful_finalizer_only_and_preserves_markdown(monkeypatch, clipboard_error, outcome):
    copied = []
    output = RecordingOutput()
    answer = '# Answer\n\n**Caf\u00e9**\n```py\nprint(1)\n```\n'

    def clipboard(text):
        copied.append(text)
        if clipboard_error:
            raise OSError('sensitive backend error details')

    monkeypatch.setattr('core.tui_commands.copy_to_clipboard', clipboard)

    def complete(prompt, status):
        status.start('brain')
        try:
            print('Planner / Brain / tool output must never be copied')
            if prompt == 'good':
                render_node_update(finalization('\x1b[94m' + answer + '\x1b[0m', outcome=outcome))
            elif prompt == 'bad':
                render_node_update(finalization('render failure placeholder', 'Provider failed'))
            return 'Clarification or intermediate callback text'
        finally:
            status.stop()

    async def scenario(pipe):
        shell = CortexTUI(complete, input=pipe, output=output)
        task = asyncio.create_task(shell.run_async())
        try:
            await until(lambda: shell.app.is_running)
            for prompt in ('clarify', 'good', 'bad', 'clarify'):
                shell.editor.text = prompt
                shell.submit()
                await until(lambda: not shell.running)
                shell.editor.text = '/copy'
                shell.submit()
                await until(lambda: not shell._local_tasks)
                await shell.flush_output()
            assert copied == [answer, answer, answer]
            assert 'No successfully rendered Finalizer answer' in output.text
            assert ('copy failed' if clipboard_error else 'Finalizer answer copied.') in output.text
            assert 'sensitive backend error' not in output.text
        finally:
            await finish(shell, task)

    with create_pipe_input() as pipe:
        asyncio.run(scenario(pipe))


def test_status_reads_existing_metrics_and_allowlisted_session_data():
    session = ApplicationSession(completed_turn_count=7, last_enrichment_status='accepted')
    settings = dict(cli.DEFAULT_SETTINGS, api_key='secret-not-for-display')
    with create_pipe_input() as pipe:
        shell = CortexTUI(lambda *a: pytest.fail('Agent invoked'), input=pipe, output=RecordingOutput(),
                         status_provider=lambda: cli.tui_session_status(settings, session))
        status = shell.live_status = LiveStatus(enabled=False)
        status.begin_provider_invocation(worker='planner')
        status.add_usage({'input_tokens': 12, 'output_tokens': 3}, worker='planner')
        status.add_usage({'input_tokens': 20, 'output_tokens': 8}, worker='brain')
        shell.running = True
        shell.last_outcome = 'Runtime completed'
        before = status.command_snapshot()
        shell.editor.text = '/status'
        shell.submit()
        text = ''.join(shell.stream.take())
        for expected in ('Ollama', settings['model'], settings['model_planner'], 'RUNNING',
                         'Runtime completed', 'Session completed turns', '7',
                         'planner tokens', '15 (in 12, out 3)', '28 (in 20, out 8)',
                         'Total tokens', '43', 'planner provider calls', 'Current turn'):
            assert expected in text
        assert 'secret-not-for-display' not in text
        assert status.command_snapshot() == before
        assert session.completed_turn_count == 7 and shell.worker is None


def test_clear_orders_output_preserves_active_turn_and_redraws():
    started, release = threading.Event(), threading.Event()
    output = RecordingOutput()
    session = ApplicationSession(completed_turn_count=3)

    def complete(prompt, status):
        status.add_usage({'total_tokens': 42}, worker='brain')
        started.set()
        assert release.wait(4)
        print('worker finished')

    async def scenario(pipe):
        shell = CortexTUI(complete, input=pipe, output=output,
                         status_provider=lambda: cli.tui_session_status(cli.DEFAULT_SETTINGS, session))
        shell.latest_final_answer = 'previous answer'
        task = asyncio.create_task(shell.run_async())
        try:
            await until(lambda: shell.app.is_running)
            shell.editor.text = 'work'
            shell.submit()
            await until(started.is_set)
            worker, status = shell.worker, shell.live_status
            history = shell.editor.buffer.history.get_strings()
            shell.stream.write('before clear partial')
            shell.editor.text = '/clear'
            shell.submit()
            shell.stream.write('after clear\n')
            shell.editor.text = 'draft\nstill editing'
            document = shell.editor.buffer.document
            await shell.flush_output()
            assert output.text.index('before clear partial') < output.text.index('CLEAR') < output.text.index('after clear')
            assert output.text.count('after clear') == 1
            assert 'scrollback clearing is not confirmed' in output.text
            assert shell.running and shell.worker is worker and not worker.done()
            assert shell.live_status is status and status.total_tokens == 42
            assert shell.latest_final_answer == 'previous answer'
            assert shell.editor.buffer.document == document
            assert shell.editor.buffer.history.get_strings() == history
            assert session.completed_turn_count == 3
            assert shell.app.layout.container.align.value == 'BOTTOM'
            screen = shell.app.renderer.last_rendered_screen
            row = ''.join(cell.char for cell in screen.data_buffer[screen.height - 1].values())
            assert 'RUNNING' in row
            assert shell.editor.window.render_info.window_height == 5
            release.set()
            await until(lambda: not shell.running)
            await shell.flush_output()
            assert output.text.count('worker finished') == 1
        finally:
            release.set()
            await finish(shell, task)

    with create_pipe_input() as pipe:
        asyncio.run(scenario(pipe))


def test_terminal_stream_actions_preserve_partial_and_later_output():
    stream = TerminalStream()
    action = lambda: None
    stream.write('partial')
    stream.enqueue_action(action)
    stream.write('later\n')
    assert stream.take() == ['partial', action, 'later\n']


def test_vt_clear_requests_scrollback_without_claiming_confirmation(monkeypatch):
    import io
    monkeypatch.delenv('TERM', raising=False)
    target = io.StringIO()
    output = Vt100_Output(target, lambda: Size(30, 80))
    message = clear_terminal(output)
    assert '\x1b[3J' in target.getvalue()
    assert 'requested' in message and 'support varies' in message
    assert '\x1b[?1049h' not in target.getvalue()


def test_style_plain_color_and_ansi_stripping(monkeypatch):
    from prompt_toolkit.output.color_depth import ColorDepth
    output = SimpleNamespace(get_default_color_depth=lambda: ColorDepth.TRUE_COLOR)
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.delenv('TERM', raising=False)
    colored = TerminalStyle(output).summary('Heading', [('Total', 42)])
    assert '\x1b[96m' in colored
    monkeypatch.setenv('NO_COLOR', '')
    plain = TerminalStyle(output).summary('Heading', [('Total', 42)])
    assert plain == strip_ansi(colored)
    assert strip_ansi('\x1b]8;;https://example.com\x1b\\**link**\x1b]8;;\x1b\\') == '**link**'


def test_windows_clipboard_uses_stdin_and_preserves_unicode(monkeypatch):
    import core.terminal_actions as actions
    calls = []
    monkeypatch.setattr(actions.sys, 'platform', 'win32')
    monkeypatch.setattr(actions.shutil, 'which', lambda name: name)
    monkeypatch.setattr(actions.subprocess, 'CREATE_NO_WINDOW', 0x08000000, raising=False)
    monkeypatch.setattr(actions.subprocess, 'run', lambda args, **kwargs: calls.append((args, kwargs)))
    text = '# \u00dc\n$(never execute)\n'
    copy_to_clipboard(text)
    args, kwargs = calls[0]
    assert text not in ' '.join(args)
    assert base64.b64decode(kwargs['input']).decode('utf-8') == text
    assert kwargs['timeout'] == 5 and kwargs['check']
    assert kwargs['creationflags'] == 0x08000000


@pytest.mark.parametrize('platform,environment,available,expected', [
    ('darwin', {}, set(), ['pbcopy']),
    ('linux', {'WAYLAND_DISPLAY': 'wayland-0'}, {'wl-copy'}, ['wl-copy']),
    ('linux', {'DISPLAY': ':0'}, {'xclip'}, ['xclip', '-selection', 'clipboard']),
    ('linux', {'DISPLAY': ':0'}, {'xsel'}, ['xsel', '--clipboard', '--input']),
])
def test_clipboard_platform_selection(monkeypatch, platform, environment, available, expected):
    import core.terminal_actions as actions
    calls = []
    monkeypatch.setattr(actions.sys, 'platform', platform)
    for name in ('DISPLAY', 'WAYLAND_DISPLAY'):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(actions.shutil, 'which', lambda name: name if name in available else None)
    monkeypatch.setattr(actions.subprocess, 'run', lambda args, **kwargs: calls.append((args, kwargs)))
    copy_to_clipboard('exact\ntext\n')
    assert calls[0][0] == expected
    assert calls[0][1]['input'] == b'exact\ntext\n'


def test_clipboard_without_backend_fails_gracefully(monkeypatch):
    import core.terminal_actions as actions
    monkeypatch.setattr(actions.sys, 'platform', 'linux')
    monkeypatch.setattr(actions.shutil, 'which', lambda name: None)
    with pytest.raises(RuntimeError, match='No clipboard backend'):
        copy_to_clipboard('answer')


def test_clear_failure_does_not_drop_later_output(monkeypatch):
    output = RecordingOutput()

    def fail(output):
        raise OSError('backend failed')

    monkeypatch.setattr('core.tui_commands.clear_terminal', fail)

    async def scenario(pipe):
        shell = CortexTUI(lambda *a: pytest.fail('Agent invoked'), input=pipe, output=output)
        task = asyncio.create_task(shell.run_async())
        try:
            await until(lambda: shell.app.is_running)
            shell.editor.text = '/clear'
            shell.submit()
            shell.stream.write('still printing\n')
            await shell.flush_output()
            assert 'Terminal clear failed' in output.text
            assert output.text.endswith('still printing\n')
        finally:
            await finish(shell, task)

    with create_pipe_input() as pipe:
        asyncio.run(scenario(pipe))
