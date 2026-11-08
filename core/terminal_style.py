"""Small, optional ANSI presentation helpers for local terminal commands."""

import os
import re
from typing import ClassVar

from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.color_depth import ColorDepth

_ESCAPES = re.compile(
    r'(?:\x1b\][^\x07\x1b]*(?:\x07|\x1b\\))'
    r'|(?:\x1b[P^_X].*?\x1b\\)'
    r'|(?:(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~])'
    r'|(?:\x1b[ -/]*[@-~])', re.DOTALL,
)


def strip_ansi(text: str) -> str:
    return _ESCAPES.sub('', text)


class TerminalStyle:
    colors: ClassVar = {'heading': '96', 'success': '92', 'warning': '93',
              'error': '91', 'muted': '90'}

    def __init__(self, output):
        self.enabled = (
            'NO_COLOR' not in os.environ and os.getenv('TERM') != 'dumb'
            and not isinstance(output, DummyOutput)
            and output.get_default_color_depth() != ColorDepth.DEPTH_1_BIT
        )

    def paint(self, text: str, role: str) -> str:
        text = strip_ansi(str(text))
        return f'\x1b[{self.colors[role]}m{text}\x1b[0m' if self.enabled else text

    def summary(self, heading: str, rows: list[tuple[str, object]]) -> str:
        width = max((len(label) for label, _ in rows), default=0)
        lines = [self.paint(heading, 'heading')]
        for label, value in rows:
            # Metadata is single-line, never terminal control input.
            value = ' '.join(strip_ansi(str(value)).split())
            lines.append(f"  {self.paint(label.ljust(width), 'muted')}  {value}")
        return '\n'.join(lines) + '\n'
