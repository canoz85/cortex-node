"""Optional terminal shell around the existing, synchronous turn callback."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import logging
from queue import Empty, SimpleQueue
import sys
import threading
import time
import math
from typing import Callable

from prompt_toolkit import Application
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import Frame, TextArea
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.styles import DynamicStyle, Style
from prompt_toolkit.layout import HSplit, VSplit, Layout, Window

from core.logging.live_status import LiveStatus, format_token_count

USE_SWEEP_BORDER = True

class TerminalStream(io.TextIOBase):
    """Batch console lines for safe printing above the editor on the UI thread."""

    def __init__(self):
        self.pending: SimpleQueue[str] = SimpleQueue()
        self._buffer = ""
        self._lock = threading.RLock()

    @property
    def encoding(self):
        return "utf-8"

    def writable(self):
        return True

    def write(self, text: str) -> int:
        with self._lock:
            self._buffer += text
            if "\n" in self._buffer:
                complete, self._buffer = self._buffer.rsplit("\n", 1)
                self.pending.put(complete + "\n")
        return len(text)

    def flush(self):
        with self._lock:
            if self._buffer:
                self.pending.put(self._buffer)
                self._buffer = ""

    def take(self) -> str:
        chunks = []
        while True:
            try:
                chunks.append(self.pending.get_nowait())
            except Empty:
                return "".join(chunks)


class CortexTUI:
    """Normal terminal scrollback with a multiline editor and live status row."""

    def __init__(self, complete_turn: Callable[[str, LiveStatus], str], *,
                 vim_mode: bool = False, input=None, output=None):
        self.complete_turn = complete_turn
        self.running = False
        self.exit_when_idle = False
        self.last_outcome = "Ready"
        self.live_status: LiveStatus | None = None
        self.worker: asyncio.Task | None = None
        self._printer: asyncio.Task | None = None
        self._printer_running = False
        self._printing = False
        self._closing = False
        self.editor = TextArea(
            multiline=True, height=Dimension.exact(5), wrap_lines=True,
            history=InMemoryHistory(), name="prompt", style="class:prompt-area",
        )
        bindings = KeyBindings()

        @bindings.add("f2", filter=has_focus(self.editor))
        @bindings.add("escape", "enter", filter=has_focus(self.editor))
        def submit(event):
            self.submit()

        @bindings.add("up", filter=has_focus(self.editor))
        def prompt_up(event):
            # Keep explicit Ctrl-Up/Down history separate from multiline movement.
            self.editor.buffer.cursor_up(count=event.arg)

        @bindings.add("down", filter=has_focus(self.editor))
        def prompt_down(event):
            self.editor.buffer.cursor_down(count=event.arg)

        @bindings.add("c-c")
        def clear_prompt(event):
            # This edits the draft only; it never cancels an authorized runtime turn.
            self.editor.buffer.reset()
            event.app.layout.focus(self.editor)

        @bindings.add("c-q")
        @bindings.add("c-d", filter=has_focus(self.editor) & Condition(lambda: not self.editor.text))
        def quit_shell(event):
            self.request_exit()

        prompt_container = (
            self._build_sweep_prompt()
            if USE_SWEEP_BORDER
            else self._build_pulse_prompt()
        )

        self.app = Application(
            layout=Layout(HSplit([
                prompt_container,
                Window(FormattedTextControl(self.status_text), height=1),
            ]), focused_element=self.editor),
            key_bindings=bindings,
            editing_mode=EditingMode.VI if vim_mode else EditingMode.EMACS,
            full_screen=False, mouse_support=False, refresh_interval=0.05,
            erase_when_done=True, enable_page_navigation_bindings=False,
            style=DynamicStyle(self.tui_style),
            input=input, output=output,
        )
        self.stream = TerminalStream()

    def _border_color(self, position: int, perimeter: int) -> str:
        if not self.running:
            return "#4b5563"

        head = int(time.monotonic() * 30) % perimeter
        distance = (position - head) % perimeter

        if distance <= 2:
            return "#dbeafe"
        if distance <= 5:
            return "#bfdbfe"
        if distance <= 9:
            return "#93c5fd"
        if distance <= 14:
            return "#60a5fa"
        if distance <= 20:
            return "#3b82f6"
        if distance <= 28:
            return "#2563eb"

        return "#4b5563"

    def _border_geometry(self):
        width = max(10, self.app.output.get_size().columns)
        height = 5

        title = " Prompt · Enter: newline · F2 / Alt-Enter: submit "
        if len(title) > width - 4:
            title = title[:width - 5] + "…"

        title_start = max(1, (width - len(title)) // 2)
        title_end = title_start + len(title)

        top_left_len = title_start
        top_right_len = width - title_end

        top_len = top_left_len + top_right_len

        right_start = top_len
        bottom_start = right_start + height
        left_start = bottom_start + width

        perimeter = left_start + height

        return {
            "width": width,
            "height": height,
            "title": title,
            "title_start": title_start,
            "title_end": title_end,
            "top_left_len": top_left_len,
            "top_right_len": top_right_len,
            "right_start": right_start,
            "bottom_start": bottom_start,
            "left_start": left_start,
            "perimeter": perimeter,
        }
        
    def _horizontal_border(self, top: bool):
        g = self._border_geometry()

        width = g["width"]
        title = g["title"]
        title_start = g["title_start"]
        title_end = g["title_end"]
        perimeter = g["perimeter"]

        fragments = []

        for x in range(width):
            in_title = top and title_start <= x < title_end

            if top:
                if x < title_start:
                    pos = x
                elif x >= title_end:
                    pos = title_start + (x - title_end)
                else:
                    pos = None
            else:
                pos = g["bottom_start"] + (width - 1 - x)

            if in_title:
                fragments.append(("fg:#9ca3af", title[x - title_start]))
                continue

            color = self._border_color(pos, perimeter)

            is_bright = color in {
                "#e0f2fe",
                "#bae6fd",
                "#7dd3fc",
            }

            if x == 0:
                char = "╔" if is_bright else ("╭" if top else "╰")
            elif x == width - 1:
                char = "╗" if is_bright else ("╮" if top else "╯")
            else:
                char = "═" if is_bright else "─"

            if is_bright:
                style = f"fg:{color} bold"
            else:
                style = f"fg:{color}"

            fragments.append((style, char))

        return fragments

    def _vertical_border_text(self, left: bool):
        g = self._border_geometry()

        fragments = []

        for row in range(g["height"]):
            if row > 0:
                fragments.append(("", "\n"))

            if left:
                pos = g["left_start"] + (g["height"] - 1 - row)
            else:
                pos = g["right_start"] + row

            color = self._border_color(pos, g["perimeter"])

            is_bright = color in {
                "#e0f2fe",
                "#bae6fd",
                "#7dd3fc",
            }

            char = "║" if is_bright else "│"

            if is_bright:
                style = f"fg:{color} bold"
            else:
                style = f"fg:{color}"

            fragments.append((style, char))

        return fragments
    
    def _build_sweep_prompt(self):
        left_border = Window(
            FormattedTextControl(
                lambda: self._vertical_border_text(left=True)
            ),
            width=1,
            height=Dimension.exact(5),
        )

        right_border = Window(
            FormattedTextControl(
                lambda: self._vertical_border_text(left=False)
            ),
            width=1,
            height=Dimension.exact(5),
        )

        top_border = Window(
            FormattedTextControl(
                lambda: self._horizontal_border(top=True)
            ),
            height=1,
        )

        bottom_border = Window(
            FormattedTextControl(
                lambda: self._horizontal_border(top=False)
            ),
            height=1,
        )

        return HSplit([
            top_border,
            VSplit([
                left_border,
                self.editor,
                right_border,
            ]),
            bottom_border,
        ])

    def _build_pulse_prompt(self):
        return Frame(
            self.editor,
            title="Prompt · Enter: newline · F2 / Alt-Enter: submit",
        )

    def tui_style(self) -> Style:
        if self.running:
            # 0.0 -> 1.0 -> 0.0 şeklinde yumuşak pulse
            pulse = (math.sin(time.monotonic() * 2.5) + 1.0) / 2.0

            dark = (37, 99, 235)
            light = (147, 197, 253)

            r = int(dark[0] + (light[0] - dark[0]) * pulse)
            g = int(dark[1] + (light[1] - dark[1]) * pulse)
            b = int(dark[2] + (light[2] - dark[2]) * pulse)

            frame_color = f"#{r:02x}{g:02x}{b:02x}"
        else:
            frame_color = "#6b7280"

        return Style.from_dict({
            "frame": f"fg:{frame_color}",
            "prompt-area": "bg:#24272e fg:#e5e7eb",
            "status-running": "fg:#22d3ee bold",
            "status-idle": "fg:#6b7280",

        })

    def _format_status_fragments(self, parts: list[str]):
        fragments = []

        for index, part in enumerate(parts):
            if index > 0:
                fragments.append(("class:status-separator", " · "))

            if index == 0:
                style = (
                    "class:status-running"
                    if self.running
                    else "class:status-idle"
                )
                fragments.append((style, part))
            else:
                fragments.append(("", part))

        return fragments

    def status_text(self) -> str:
        mode = "Emacs"
        if self.app.editing_mode == EditingMode.VI:
            mode = f"Vi {self.app.vi_state.input_mode.name.lower()}"
        parts = ["RUNNING" if self.running else "IDLE"]
        step_index = None
        snapshot = self.live_status.snapshot() if self.live_status is not None else None
        if self.running and self.live_status is not None:
            phases = {"planner": "Planning", "brain": "Executing",
                      "finalizer": "Preparing answer", "memory": "Saving session",
                      "controller": "Coordinating"}
            phase = phases.get(snapshot.stage, "Executing")
            parts.append(phase)
            if snapshot.step_label and (snapshot.active_tool or snapshot.stage in {"brain", "controller"}):
                step_index = len(parts)
                parts.append(snapshot.step_label.replace("plan step ", "Step ", 1))
            if snapshot.active_tool:
                parts.append(snapshot.active_tool)
            elif snapshot.stage:
                if snapshot.stage not in phases:
                    parts.append(snapshot.stage)
                detail = snapshot.detail.removeprefix(snapshot.step_label).strip(" ·")
                if detail and not detail.startswith("invocation ") and detail not in {"planning", "routing"}:
                    parts.append(detail)
            parts.append(f"{snapshot.elapsed_seconds:.1f}s")
            if snapshot.invocation_count:
                parts.append(f"call {snapshot.invocation_count}")
        else:
            parts.append(self.last_outcome)
        if snapshot is not None and snapshot.total_tokens:
            parts.append(format_token_count(snapshot.total_tokens))
        parts.append(mode)
        if self.exit_when_idle:
            parts.append("exiting after turn")
        # Keep live metrics on-screen on narrower terminals by shortening the title.
        width = self.app.output.get_size().columns
        if step_index is not None:
            step = parts[step_index]
            while len(step) > 12 and get_cwidth(" · ".join(parts)) > width:
                step = step[:-1]
                parts[step_index] = step + "…"
        return self._format_status_fragments(parts)

    def _write_terminal(self, text: str) -> None:
        # Called only with the editor suspended (or after it has exited).
        # Preserve renderer ANSI/text and raw-console diagnostics verbatim.
        self.app.output.enable_autowrap()
        self.app.output.write_raw(text)
        self.app.output.flush()

    async def _print_output(self) -> None:
        while self._printer_running or not self.stream.pending.empty():
            text = self.stream.take()
            if text:
                self._printing = True
                try:
                    await run_in_terminal(lambda: self._write_terminal(text))
                finally:
                    self._printing = False
            else:
                await asyncio.sleep(0.05)

    async def flush_output(self) -> None:
        """Wait for queued output before shutdown, including partial final lines."""
        self.stream.flush()
        while not self.stream.pending.empty() or self._printing:
            if self._printer is not None and self._printer.done():
                await self._printer  # Surface a terminal write failure, not a hang.
                return
            await asyncio.sleep(0.01)

    def submit(self) -> bool:
        """Submit exactly one multiline draft; drafts remain editable while busy."""
        prompt = self.editor.text
        if self.running or not prompt.strip() or self.exit_when_idle:
            return False
        if prompt.strip().lower() in {"exit", "quit"}:
            self.request_exit()
            return False
        self.editor.buffer.append_to_history()
        self.editor.buffer.reset()
        self.stream.write(f"\nYou> {prompt}\n")
        self.running = True
        self.live_status = LiveStatus(enabled=False)
        self.worker = asyncio.create_task(self._execute(prompt, self.live_status))
        return True

    async def _execute(self, prompt: str, status: LiveStatus) -> None:
        try:
            await asyncio.to_thread(self.complete_turn, prompt, status)
        except Exception as exc:
            self.last_outcome = f"Error: {type(exc).__name__}"
            self.stream.write(f"\n[error] {type(exc).__name__}: {exc}\n")
        else:
            self.last_outcome = status.snapshot().outcome or "Turn ended"
        finally:
            self.running = False
            self.app.invalidate()
            if self.exit_when_idle and self.app.is_running:
                self.request_exit()

    def request_exit(self) -> None:
        self.exit_when_idle = True
        if self.running:
            self.app.invalidate()
        elif self.app.is_running and not self._closing:
            self._closing = True
            self.app.create_background_task(self._finish_exit())

    async def _finish_exit(self) -> None:
        await self.flush_output()
        self.app.exit()

    @contextmanager
    def route_output(self):
        """Queue prints and console handlers; leave diagnostic file handlers alone."""
        originals = (sys.stdout, sys.stderr)
        handlers = []
        for handler in logging.getLogger().handlers:
            if (isinstance(handler, logging.StreamHandler)
                    and not isinstance(handler, logging.FileHandler)
                    and handler.stream in originals):
                handlers.append((handler, handler.stream))
                handler.setStream(self.stream)
        try:
            with redirect_stdout(self.stream), redirect_stderr(self.stream):
                yield
        finally:
            for handler, original in handlers:
                handler.setStream(original)

    async def run_async(self) -> None:
        # Application creates its real terminal output before stdout is redirected.
        def start_printer():
            self._printer_running = True
            self._printer = asyncio.create_task(self._print_output())

        with self.route_output():
            try:
                await self.app.run_async(pre_run=start_printer)
            finally:
                # Never abandon a synchronous turn because terminal input closed.
                if self.worker is not None:
                    await asyncio.shield(self.worker)
                self.stream.flush()
                self._printer_running = False
                if self._printer is not None:
                    await self._printer

    def run(self) -> None:
        asyncio.run(self.run_async())
