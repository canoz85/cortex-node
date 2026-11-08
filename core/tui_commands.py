"""Local slash-command registry. Handlers never call the agent."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from core.terminal_actions import clear_terminal, copy_to_clipboard
from core.terminal_style import strip_ansi


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    handler: Callable[[], None]
    usage: str = ''


class LocalCommands:
    def __init__(self, shell):
        self.shell = shell
        self.registry: dict[str, Command] = {}
        self.register(Command('/copy', 'Copy the latest Finalizer answer.', self.copy))
        self.register(Command('/status', 'Show runtime, models and available usage.', self.status))
        self.register(Command('/help', 'List local commands and usage.', self.help))
        self.register(Command('/clear', 'Clear terminal display; keep the session.', self.clear))

    def register(self, command: Command) -> None:
        self.registry[command.name] = command

    def dispatch(self, text: str) -> bool:
        if not text.lstrip().startswith('/'):
            return False
        name, *args = text.strip().split(maxsplit=1)
        command = self.registry.get(name)
        if command is None:
            self.message('Unknown command. Type /help for available commands.', 'warning')
        elif args:
            self.message(f'Usage: {command.usage or command.name}', 'warning')
        else:
            command.handler()
        return True

    def message(self, text: str, role: str = 'muted') -> None:
        self.shell.stream.write(self.shell.command_style.paint(text, role) + '\n')

    def help(self) -> None:
        self.shell.stream.write(self.shell.command_style.summary('Local commands', [
            (command.usage or command.name, command.description)
            for command in self.registry.values()
        ]))
        self.message('Enter: newline | Alt+Enter / F2: submit | Ctrl+Q: exit after turn')

    def status(self) -> None:
        shell = self.shell
        rows = [('Runtime', 'RUNNING' if shell.running else 'IDLE'),
                ('Last outcome', shell.last_outcome)]
        if shell.exit_when_idle:
            rows.append(('Exit', 'Waiting for active turn'))
        if shell.status_provider is not None:
            try:
                rows.extend(shell.status_provider())
            except Exception:  # noqa: BLE001 - Optional display data must not break the editor.
                rows.append(('Session', 'Information temporarily unavailable'))
        else:
            rows.append(('Configuration / session', 'Unavailable'))
        if shell.live_status is not None:
            metrics = shell.live_status.command_snapshot()
            rows.append(('Usage scope', 'Current turn' if shell.running else 'Last turn'))
            if metrics['stage']:
                rows.append(('Stage', metrics['stage']))
            for worker, usage in metrics['usage'].items():
                rows.append((f'{worker} tokens',
                             f"{usage['total_tokens']} (in {usage['input_tokens']}, out {usage['output_tokens']})"))
            rows.append(('Total tokens', metrics['total_tokens']))
            for worker, count in metrics['invocations'].items():
                rows.append((f'{worker} provider calls', count))
        shell.stream.write(shell.command_style.summary('CortexNode status', rows))

    def copy(self) -> None:
        shell = self.shell
        answer = shell.latest_final_answer
        if shell.live_status is not None:
            answer = shell.live_status.command_snapshot()['final_answer'] or answer
        if not answer:
            self.message('No successfully rendered Finalizer answer available yet.', 'warning')
            return

        async def perform():
            try:
                await asyncio.to_thread(copy_to_clipboard, strip_ansi(answer))
            except Exception:  # noqa: BLE001 - Clipboard backend failures stay local to the command.
                self.message('Clipboard unavailable or copy failed. Select the answer and use terminal Copy.', 'error')
            else:
                self.message('Finalizer answer copied.', 'success')

        shell.start_local_task(perform())

    def clear(self) -> None:
        def perform():
            try:
                message = clear_terminal(self.shell.app.output)
            except Exception:  # noqa: BLE001 - Preserve the output queue on terminal backend failure.
                message = 'Terminal clear failed; session state is unchanged.'
                role = 'error'
            else:
                role = 'muted'
            self.shell._write_terminal(self.shell.command_style.paint(message, role) + '\n')

        self.shell.stream.enqueue_action(perform)
