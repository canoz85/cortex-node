"""Bounded optional browser tool. No browser state escapes an invocation."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time

from langchain_core.tools import tool
from pydantic import ValidationError

from core.models import RenderedWebFetchRequest, ToolOutputEnvelope
from core import error_codes as codes
from tools.web_ops import _parse_web_url, _RejectedURL

OVERALL_TIMEOUT = 30
MAX_IPC_LINE = 3_000_000  # Includes JSON surrogate escapes for 200k non-BMP characters.


def result_data(url):
    return dict(url=url, final_url=None, title="", text="", total_chars=0,
                total_text_bytes=0, returned_chars=0, truncated=False,
                capture_status="failed", completion_reason="render_failed", elapsed_ms=0,
                retrieved_at=None, warnings=["request_validation_not_socket_pinned", "bounded_text_settling_only"],
                status_code=None, content_type=None, settle_observed=False,
                blocked_requests=[], blocked_request_count=0)


def envelope(data, code, started):
    data = dict(data)
    data["elapsed_ms"] = round((time.monotonic() - started) * 1000)
    data["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    data["capture_status"] = "rendered" if code is None else ("partial" if data["text"] else "failed")
    message = f"Rendered page: {data['capture_status']} ({data['completion_reason']}); {data['returned_chars']} characters."
    return ToolOutputEnvelope(success=code is None, message=message, display=message,
                              data=data, error_code=code)


def _windows_job(process):
    """Own all descendants, including if the parent exits or Chromium hangs."""
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64),
                    ("Flags", wintypes.DWORD), ("MinWorkingSet", ctypes.c_size_t),
                    ("MaxWorkingSet", ctypes.c_size_t), ("ActiveProcesses", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("Priority", wintypes.DWORD),
                    ("Scheduling", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("Basic", Basic), ("Io", ctypes.c_uint64 * 6),
                    ("ProcessMemory", ctypes.c_size_t), ("JobMemory", ctypes.c_size_t),
                    ("PeakProcessMemory", ctypes.c_size_t), ("PeakJobMemory", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise OSError("Cannot create renderer job")
    limits = Extended()
    limits.Basic.Flags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
    limits.JobMemory = 1024 * 1024 * 1024
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(job, int(process._handle)):
        kernel.CloseHandle(job)
        raise OSError("Cannot constrain renderer job")
    return kernel, job


def _stop_worker(process, job=None):
    # The parent owns the deadline, including stuck JS, DNS and browser cleanup.
    if os.name == "nt":
        if job is not None:
            kernel, handle = job
            kernel.CloseHandle(handle)
        else:
            # Job setup failed before stdin released the worker to launch Chromium.
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=3)


def _run_worker(request, started):
    temporary = tempfile.TemporaryDirectory(prefix="cortex-render-")
    result = None
    try:
        result = _supervise(request, started, temporary.name)
        return result
    finally:
        # Windows can retain profile-file locks briefly after job termination.
        # Never replace useful timeout evidence with a temporary cleanup error.
        cleanup_deadline = time.monotonic() + 3
        while True:
            try:
                temporary.cleanup()
                break
            except OSError:
                if time.monotonic() >= cleanup_deadline:
                    if result is not None:
                        result.data["warnings"].append("temporary_cleanup_incomplete")
                    break
                time.sleep(0.05)


def _supervise(request, started, temporary):
    data = result_data(request.url)
    # Do not pass API keys, proxy credentials, home/profile or workspace settings.
    allowed = {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "TMPDIR", "LOCALAPPDATA",
               "PLAYWRIGHT_BROWSERS_PATH"}
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env.update(TEMP=temporary, TMP=temporary, TMPDIR=temporary)
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen([sys.executable, "-m", "tools.rendered_web_worker"],
        cwd=Path(__file__).resolve().parents[1], env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **options)
    messages = queue.Queue(maxsize=1)
    job = None

    def receive():
        while True:
            line = process.stdout.readline(MAX_IPC_LINE + 1)
            if not line:
                return
            if len(line) > MAX_IPC_LINE:
                return
            try:
                item = json.loads(line)
            except (ValueError, UnicodeError):
                return
            try:
                messages.get_nowait()
            except queue.Empty:
                pass
            messages.put_nowait(item)

    try:
        if os.name == "nt":
            try:
                job = _windows_job(process)
            except OSError:
                data["completion_reason"] = "worker_isolation_unavailable"
                return envelope(data, codes.RENDERED_WEB_UNAVAILABLE, started)
        process.stdin.write(request.model_dump_json().encode() + b"\n")
        process.stdin.close()
        reader = threading.Thread(target=receive, daemon=True)
        reader.start()
        while time.monotonic() - started < OVERALL_TIMEOUT:
            try:
                item = messages.get(timeout=0.05)
            except queue.Empty:
                if process.poll() is not None and not reader.is_alive():
                    data["completion_reason"] = "worker_exited"
                    return envelope(data, codes.RENDERED_WEB_RENDER_FAILED, started)
                continue
            data = item["data"]
            if item.get("done"):
                return envelope(data, item.get("code"), started)
        data["completion_reason"] = "overall_deadline"
        return envelope(data, codes.RENDERED_WEB_TIMEOUT, started)
    finally:
        _stop_worker(process, job)
        if not process.stdin.closed:
            process.stdin.close()
        process.stdout.close()


def fetch_rendered_web(url: str, max_chars: int = 50_000) -> ToolOutputEnvelope:
    started = time.monotonic()
    data = result_data(url if isinstance(url, str) else "")
    try:
        request = RenderedWebFetchRequest(url=url, max_chars=max_chars)
        _parse_web_url(request.url)
    except (ValidationError, _RejectedURL):
        data["completion_reason"] = "invalid_input"
        return envelope(data, codes.RENDERED_WEB_REJECTED, started)
    try:
        return _run_worker(request, started)
    except Exception:
        data["completion_reason"] = "worker_failed"
        return envelope(data, codes.RENDERED_WEB_RENDER_FAILED, started)


def get_rendered_web_tools():
    @tool("rendered_web_fetch", args_schema=RenderedWebFetchRequest)
    def rendered_web_fetch(url: str, max_chars: int = 50_000) -> str:
        """Read bounded text from a known public JavaScript-rendered page; no interactions or continuation."""
        return fetch_rendered_web(url, max_chars).to_tool_output()

    return [rendered_web_fetch]
