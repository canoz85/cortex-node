"""Single-owner, temporary console status for an active user turn."""

from __future__ import annotations

import contextvars
import sys
import threading
import time
from contextlib import contextmanager
from typing import TextIO


_CURRENT: contextvars.ContextVar["LiveStatus | None"] = contextvars.ContextVar(
    "cortex_live_status", default=None
)


def format_token_count(total: int) -> str:
    return f"{format_compact_tokens(total)} tok"


def format_compact_tokens(total: int) -> str:
    total = max(0, int(total or 0))
    if total < 1000:
        return str(total)
    return f"{total / 1000:.1f}k"


def current_live_status() -> "LiveStatus | None":
    return _CURRENT.get()


def add_response_usage(response: object, *, worker: str) -> None:
    status = current_live_status()
    if status is not None:
        status.add_response_usage(response, worker=worker)


def begin_provider_invocation(*, worker: str) -> int:
    """Record an actual provider call and expose its temporary live status."""
    status = current_live_status()
    if status is None:
        return 0
    return status.begin_provider_invocation(worker=worker)


class LiveStatus:
    _SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(
        self, *, stream: TextIO | None = None, refresh_interval: float = 0.1,
        enabled: bool | None = None,
    ):
        self.stream = stream or sys.stdout
        self.enabled = (
            bool(getattr(self.stream, "isatty", lambda: False)())
            if enabled is None else enabled
        )
        self.refresh_interval = refresh_interval
        self.stage = ""
        self.detail = ""
        self.total_tokens = 0
        self.usage_by_worker: dict[str, dict[str, int]] = {}
        self.provider_invocations_by_worker: dict[str, int] = {}
        self._accounted_response_ids: set[int] = set()
        self._accounted_responses: list[object] = []
        self._started_at = 0.0
        self._frame = 0
        self._running = False
        self._visible = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._token = None

    def start(self, stage: str = "planner", detail: str = "") -> None:
        with self._lock:
            if self._running:
                self.update(stage, detail)
                return
            self.stage, self.detail = stage, detail
            self._started_at = time.perf_counter()
            self._running = True
            self._stop_event.clear()
            self._token = _CURRENT.set(self)
            if self.enabled:
                self.redraw()
                self._thread = threading.Thread(
                    target=self._refresh, name="cortex-live-status", daemon=True
                )
                self._thread.start()

    def update(self, stage: str, detail: str = "") -> None:
        with self._lock:
            self.stage, self.detail = str(stage), str(detail or "")
            if self._running and self.enabled:
                self.redraw()

    def begin_provider_invocation(self, *, worker: str) -> int:
        """Increment a turn-local provider-call count before the blocking call."""
        with self._lock:
            invocation = self.provider_invocations_by_worker.get(worker, 0) + 1
            self.provider_invocations_by_worker[worker] = invocation
            detail = f"step {invocation}" if worker == "brain" else ""
            self.update(worker, detail)
            return invocation

    def add_usage(self, usage: object, *, worker: str) -> None:
        if usage is None:
            return
        getter = usage.get if isinstance(usage, dict) else lambda key, default=0: getattr(usage, key, default)
        input_tokens = getter("input_tokens", 0) or getter("prompt_tokens", 0) or 0
        output_tokens = getter("output_tokens", 0) or getter("completion_tokens", 0) or 0
        total = getter("total_tokens", 0) or 0
        if not total:
            total = input_tokens + output_tokens
        try:
            input_value = max(0, int(input_tokens))
            output_value = max(0, int(output_tokens))
            value = max(0, int(total))
        except (TypeError, ValueError):
            input_value = 0
            output_value = 0
            value = 0
        with self._lock:
            worker_usage = self.usage_by_worker.setdefault(worker, {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            })
            worker_usage["input_tokens"] += input_value
            worker_usage["output_tokens"] += output_value
            worker_usage["total_tokens"] += value
            worker_usage["calls"] += 1
            self.total_tokens += value
            if self._running and self.enabled:
                self.redraw()

    def add_response_usage(self, response: object, *, worker: str) -> None:
        if isinstance(response, dict) and "raw" in response:
            response = response.get("raw")
        if response is None:
            return
        with self._lock:
            response_id = id(response)
            if response_id in self._accounted_response_ids:
                return
            self._accounted_response_ids.add(response_id)
            self._accounted_responses.append(response)
        usage = getattr(response, "usage_metadata", None) or {}
        metadata = getattr(response, "response_metadata", None) or {}
        self.add_usage(usage or metadata, worker=worker)

    def format_usage_summary(self) -> str:
        with self._lock:
            segments = [
                f"{worker} {format_compact_tokens(usage['total_tokens'])} ({usage['calls']})"
                for worker in ("planner", "brain", "finalizer", "memory")
                if (usage := self.usage_by_worker.get(worker)) is not None
                and usage["calls"] > 0
            ]
            if not segments:
                return ""
            segments.append(f"total {format_token_count(self.total_tokens)}")
            return " | ".join(segments)

    def clear(self) -> None:
        with self._lock:
            if self._visible:
                # self.stream.write("\r\033[2K")
                self.stream.write("\n")
                self.stream.flush()
                self._visible = False

    def redraw(self) -> None:
        with self._lock:
            if not self._running or not self.enabled:
                return
            elapsed = time.perf_counter() - self._started_at
            label = self.stage + (f" · {self.detail}" if self.detail else "")
            line = (
                f"{self._SPINNER[self._frame % len(self._SPINNER)]} "
                f"{label:<36} {format_token_count(self.total_tokens)} · {elapsed:.1f}s"
            )
            self._frame += 1
            self.stream.write(f"\r\033[2K{line}")
            # self.stream.write(line + "\n")
            self.stream.flush()
            self._visible = True

    @contextmanager
    def permanent_output(self, *, redraw: bool = True):
        with self._lock:
            self.clear()
            try:
                yield
            finally:
                if redraw and self._running:
                    self.redraw()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()
            self.clear()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.2, self.refresh_interval * 2))
        if self._token is not None:
            _CURRENT.reset(self._token)
            self._token = None

    def _refresh(self) -> None:
        while not self._stop_event.wait(self.refresh_interval):
            self.redraw()

