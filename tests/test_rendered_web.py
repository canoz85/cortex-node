"""Offline policy/runtime tests plus optional real Chromium/local HTTP fixtures."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from core import error_codes as codes
from core.brain_batch_policy import validate_read_only_batch
from core.models import RenderedWebFetchRequest
from core.tool_output import _normalize_transport_payload
from tools import rendered_web_ops as ops
from tools import rendered_web_worker as worker
from tools import web_ops
from tools.registry import build_tool_registry, default_enabled_tools, get_tool_definition, planning_capability_projection


@pytest.mark.parametrize("arguments", [
    {}, {"url": "https://example.com", "selector": "main"},
    {"url": 123}, {"url": "x", "max_chars": True},
    {"url": "x", "max_chars": "10"}, {"url": "x", "max_chars": 0},
    {"url": "x", "max_chars": 200001}, {"url": "x" * 8193},
])
def test_strict_schema(arguments):
    with pytest.raises(ValidationError):
        RenderedWebFetchRequest(**arguments)


@pytest.mark.parametrize("url", ["file:///secret", "https://user:pass@example.com", "http://a\\b", " javascript:alert(1)"])
def test_invalid_syntax_never_starts_worker(monkeypatch, url):
    start = Mock(side_effect=AssertionError("worker started"))
    monkeypatch.setattr(ops, "_run_worker", start)
    assert ops.fetch_rendered_web(url).error_code == codes.RENDERED_WEB_REJECTED
    start.assert_not_called()


@pytest.mark.parametrize("url", ["http://127.0.0.1", "http://localhost", "http://169.254.169.254", "http://[::1]", "http://10.0.0.1"])
def test_unsafe_initial_url(url):
    data, code = asyncio.run(worker.render(RenderedWebFetchRequest(url=url), lambda _: None))
    assert code == codes.RENDERED_WEB_REJECTED
    assert data["text"] == ""


def test_public_dns_and_mixed_answers(monkeypatch):
    monkeypatch.setattr(web_ops.socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in ["93.184.216.34", "10.0.0.1"]])
    policy = worker.Policy(ops.result_data("https://example.com"))
    assert not asyncio.run(policy.request_allowed("https://example.com", "GET", "Script"))
    assert policy.code is None and "unsafe_destination" in policy.data["warnings"]


@pytest.mark.parametrize("method,kind,main,reason", [
    ("POST", "Fetch", True, "method_blocked"),
    ("GET", "Document", False, "frame_blocked"),
    ("GET", "WebSocket", True, "resource_type_blocked"),
])
def test_policy(method, kind, main, reason):
    policy = worker.Policy(ops.result_data("https://example.com"), validate=lambda _: None)
    assert not asyncio.run(policy.request_allowed("https://example.com", method, kind, main))
    assert reason in policy.data["warnings"]
    assert policy.code is None


def test_blocked_main_post_and_hard_failure_priority():
    policy = worker.Policy(ops.result_data("https://example.com"), validate=lambda _: None)
    assert not asyncio.run(policy.request_allowed("https://example.com/metrics", "POST", "XHR"))
    assert policy.code is None
    assert not asyncio.run(policy.request_allowed("https://example.com/action", "POST", "Document"))
    assert policy.code == codes.RENDERED_WEB_POLICY_BLOCKED and policy.stop
    assert policy.data["blocked_requests"][-1]["is_main_navigation"]


def test_blocked_metadata_is_sanitized_bounded_and_origin_aware():
    policy = worker.Policy(ops.result_data("https://example.com/start"), validate=lambda _: None)
    for _ in range(10):
        assert not asyncio.run(policy.request_allowed(
            "https://user:password@other.example:443/metrics?token=secret#private", "POST", "XHR",
            document_url="https://other.example/page"))
    assert policy.data["blocked_request_count"] == 10
    assert len(policy.data["blocked_requests"]) == 8
    record = policy.data["blocked_requests"][0]
    assert record == dict(method="POST", url="https://other.example:443/metrics",
                          resource_type="XHR", same_origin=True, is_main_navigation=False)
    assert len(worker._request_identity("https://example.com/" + "x" * 1000)[0]) == 512
    assert worker._request_identity("data:text/plain,secret")[0] == "[unsupported URL]"
    assert worker._request_identity("https://example.com:bad/")[1] is None
    assert not asyncio.run(policy.request_allowed("https://elsewhere.example/action", "POST", "Document"))
    assert policy.code == codes.RENDERED_WEB_POLICY_BLOCKED


def test_request_limit():
    policy = worker.Policy(ops.result_data("https://example.com"), validate=lambda _: None)
    policy.requests = worker.MAX_REQUESTS
    assert not asyncio.run(policy.request_allowed("https://example.com", "GET", "Script"))
    assert policy.code == codes.RENDERED_WEB_RESOURCE_LIMIT


def test_package_unavailable(monkeypatch):
    import builtins
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "playwright.async_api":
            raise ImportError("fixture missing package")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    data, code = asyncio.run(worker.render(RenderedWebFetchRequest(url="https://example.com"),
        lambda _: None, validate=lambda _: None))
    assert code == codes.RENDERED_WEB_UNAVAILABLE and data["completion_reason"] == "playwright_missing"


def test_registry_batch_and_integrity(tmp_path):
    registry = build_tool_registry(str(tmp_path), None, SimpleNamespace(), "test")
    definition = get_tool_definition("rendered_web_fetch")
    assert not definition.mutating and definition.max_batch_calls == 2
    assert "rendered_web_fetch" in registry.names
    assert ("rendered_web_fetch" in {tool.name for tool in default_enabled_tools(registry)}) == definition.enabled_by_default
    card, = planning_capability_projection(("rendered_web_fetch",))
    assert "pagination" not in card and "JavaScript" in card["use_when"]
    calls = [("rendered_web_fetch", {"url": f"https://example.com/{i}"}) for i in range(3)]
    validate_read_only_batch(calls[:2], {"rendered_web_fetch"})
    with pytest.raises(ValueError, match="invalid_batch_size"):
        validate_read_only_batch(calls, {"rendered_web_fetch"})
    with pytest.raises(ValueError, match="not_authorized"):
        validate_read_only_batch(calls[:2], {"web_fetch"})
    data = ops.result_data("https://example.com")
    data.update(text="Ω", returned_chars=1, total_chars=2, total_text_bytes=4, truncated=True, completion_reason="text_stable")
    result = _normalize_transport_payload(ops.envelope(data, None, time.monotonic()).to_tool_output(), tool_name="rendered_web_fetch")
    assert result.integrity.is_truncated and result.integrity.original_bytes == 4
    assert result.integrity.captured_bytes == 2 and result.pagination is None


def test_parent_deadline_preserves_snapshot_and_kills_worker(monkeypatch):
    data = ops.result_data("https://example.com/#route")
    data.update(text="last usable text", returned_chars=16, total_chars=16, total_text_bytes=16)
    message = json.dumps({"data": data, "done": False})
    script = "import time; print(" + repr(message) + ", flush=True); time.sleep(60)"
    original = subprocess.Popen
    spawned = []

    def popen(command, **kwargs):
        process = original([sys.executable, "-c", script], **kwargs)
        spawned.append((process, kwargs["env"]["TEMP"]))
        return process

    monkeypatch.setattr(ops.subprocess, "Popen", popen)
    monkeypatch.setattr(ops, "OVERALL_TIMEOUT", 1)
    started = time.monotonic()
    result = ops.fetch_rendered_web(data["url"])
    assert time.monotonic() - started < 5
    assert result.error_code == codes.RENDERED_WEB_TIMEOUT
    assert result.data["capture_status"] == "partial" and result.data["text"] == "last usable text"
    assert spawned[0][0].poll() is not None
    assert not Path(spawned[0][1]).exists()


def test_real_worker_rejects_local_url_and_cleans_up():
    result = ops.fetch_rendered_web("http://127.0.0.1/secret")
    assert result.error_code == codes.RENDERED_WEB_REJECTED
    assert result.data["capture_status"] == "failed"


def test_cleanup_retries_transient_file_locks_without_losing_evidence(monkeypatch):
    directory = SimpleNamespace(name="unused-fixture", cleanup=Mock(side_effect=[PermissionError(), None]))
    data = ops.result_data("https://example.com")
    data.update(text="partial", returned_chars=7, completion_reason="overall_deadline")
    result = ops.envelope(data, codes.RENDERED_WEB_TIMEOUT, time.monotonic())
    monkeypatch.setattr(ops.tempfile, "TemporaryDirectory", lambda **_: directory)
    monkeypatch.setattr(ops, "_supervise", lambda *_: result)
    assert ops._run_worker(RenderedWebFetchRequest(url=data["url"]), time.monotonic()) is result
    assert directory.cleanup.call_count == 2
    assert result.data["capture_status"] == "partial"


@pytest.fixture(scope="module")
def chromium():
    api = pytest.importorskip("playwright.sync_api")
    with api.sync_playwright() as pw:
        if not Path(pw.chromium.executable_path).is_file():
            pytest.skip("Optional Chromium is not installed")


@pytest.fixture(scope="module")
def site():
    seen = []
    inserted = "Rendered Ω evidence"
    html = "<title>Fixture</title><main><script>setTimeout(() => document.querySelector('main').append('Rendered Ω evidence ' + location.hash), 100)</script></main>"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen.append(self.path)
            content = html
            headers = {}
            status = 200
            if self.path == "/changing":
                content = "<main>partial<script>setInterval(() => document.querySelector('main').append('x'), 100)</script></main>"
            elif self.path == "/private":
                content = "<main>safe<script>fetch('http://127.0.0.2:9999/private')</script></main>"
            elif self.path == "/post":
                content = "<main>safe<script>fetch('/posted', {method:'POST',body:'no'})</script></main>"
            elif self.path == "/empty-post":
                content = "<main></main><script>fetch('/posted',{method:'POST'}).catch(()=>{})</script>"
            elif self.path == "/changing-post":
                content = "<main>partial</main><script>fetch('/posted',{method:'POST'}).catch(()=>{});setInterval(()=>document.querySelector('main').append('x'),100)</script>"
            elif self.path == "/hydrate":
                content = "<main></main><script>fetch('/posted',{method:'POST'}).catch(()=>{}); setTimeout(() => fetch('/data').then(r=>r.json()).then(v=>document.querySelector('main').textContent=v.text), 300)</script>"
            elif self.path == "/data":
                content, headers = json.dumps({"text": inserted}), {"Content-Type": "application/json"}
            elif self.path == "/private-script":
                content = "<main>safe<script src='/redirect'></script></main>"
            elif self.path == "/hung-js":
                content = "<main>partial before hang<script>setTimeout(() => {while(true) {}}, 800)</script></main>"
            elif self.path == "/redirect":
                status, headers = 302, {"Location": "http://127.0.0.2:9999/private"}
            elif self.path == "/public-redirect":
                status, headers = 302, {"Location": "/"}
            elif self.path == "/download":
                headers = {"Content-Disposition": 'attachment; filename="fixture.html"'}
            elif self.path == "/popup":
                content = "<main>safe<script>try {window.open('/opened')} catch(e) {document.querySelector('main').append(' blocked')}</script></main>"
            elif self.path == "/state":
                content = "<main></main><script>document.querySelector('main').textContent = localStorage.getItem('prior') || document.cookie || 'fresh'; localStorage.setItem('prior','leaked'); document.cookie='prior=leaked'</script>"
            elif self.path == "/nonhtml":
                headers = {"Content-Type": "application/pdf"}
            elif self.path == "/error":
                status = 503
            elif self.path == "/auth":
                status, headers = 401, {"WWW-Authenticate": 'Basic realm="fixture"'}
            elif self.path == "/disconnect":
                self.connection.close()
                return
            body = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", headers.pop("Content-Type", "text/html; charset=utf-8"))
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            seen.append("POST_REACHED_SERVER")
            self.send_response(500)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def validate(url):
        # Fixture-only dependency injection. Production has no local-network switch.
        if url.startswith(base + "/"):
            return
        web_ops._validate_public_url(url)

    yield SimpleNamespace(base=base, validate=validate, html=html, seen=seen, inserted=inserted)
    server.shutdown()
    server.server_close()
    thread.join()


def read(site, path="/", max_chars=50000):
    return asyncio.run(worker.render(RenderedWebFetchRequest(url=site.base + path, max_chars=max_chars),
        lambda _: None, validate=site.validate))


def test_javascript_hash_and_static_extraction(chromium, site):
    data, code = read(site, "/#details")
    assert code is None, data
    assert data["text"] == site.inserted + " #details"
    assert data["final_url"].endswith("/#details")
    assert site.inserted not in web_ops._extract_html(site.html.encode(), "utf-8")[1]


def test_browser_truncation(chromium, site):
    data, code = read(site, max_chars=5)
    assert code is None and data["text"] == "Rende" and data["truncated"]
    assert data["total_chars"] == len(site.inserted) and data["returned_chars"] == 5


def test_timeout_partial(chromium, site, monkeypatch):
    monkeypatch.setattr(worker, "SETTLE_SECONDS", 1)
    data, code = read(site, "/changing")
    assert code == codes.RENDERED_WEB_TIMEOUT and data["text"].startswith("partial")
    result = ops.envelope(data, code, time.monotonic())
    assert not result.success and result.data["capture_status"] == "partial"


@pytest.mark.parametrize("path,code", [
    ("/redirect", codes.RENDERED_WEB_POLICY_BLOCKED),
    ("/download", codes.RENDERED_WEB_UNSUPPORTED),
    ("/nonhtml", codes.RENDERED_WEB_UNSUPPORTED),
    ("/error", codes.RENDERED_WEB_HTTP_ERROR),
    ("/auth", codes.RENDERED_WEB_UNSUPPORTED),
    ("/disconnect", codes.RENDERED_WEB_NAVIGATION_FAILED),
])
def test_browser_failures(chromium, site, path, code):
    data, actual = read(site, path)
    assert actual == code, data
    assert "POST_REACHED_SERVER" not in site.seen


def test_popup_is_not_opened(chromium, site):
    data, code = read(site, "/popup")
    assert code is None and "blocked" in data["text"]
    assert "/opened" not in site.seen


def test_blocked_post_does_not_prevent_permitted_hydration(chromium, site):
    data, code = read(site, "/hydrate")
    assert code is None and data["settle_observed"]
    assert data["text"] == site.inserted
    assert "method_blocked" in data["warnings"]
    blocked = next(item for item in data["blocked_requests"] if item["method"] == "POST")
    assert blocked["resource_type"] in {"Fetch", "XHR"}
    assert blocked["url"] == site.base + "/posted"
    assert blocked["same_origin"] and not blocked["is_main_navigation"]
    result = ops.envelope(data, code, time.monotonic())
    assert result.success and result.data["capture_status"] == "rendered"
    assert "POST_REACHED_SERVER" not in site.seen


@pytest.mark.parametrize("path", ["/post", "/private", "/private-script"])
def test_blocked_secondary_request_with_settled_text_is_warning(chromium, site, path):
    data, code = read(site, path)
    result = ops.envelope(data, code, time.monotonic())
    assert result.success and result.data["capture_status"] == "rendered"
    assert data["text"] == "safe" and data["settle_observed"]
    assert data["blocked_request_count"] >= 1 and data["blocked_requests"]
    assert data["completion_reason"] == "text_stable"
    assert "POST_REACHED_SERVER" not in site.seen


@pytest.mark.parametrize("path,expected,status", [
    ("/empty-post", codes.RENDERED_WEB_RENDER_FAILED, "failed"),
    ("/changing-post", codes.RENDERED_WEB_TIMEOUT, "partial"),
])
def test_secondary_warning_does_not_hide_unusable_capture(chromium, site, monkeypatch, path, expected, status):
    monkeypatch.setattr(worker, "SETTLE_SECONDS", 1)
    data, code = read(site, path)
    result = ops.envelope(data, code, time.monotonic())
    assert code == expected and not result.success and result.data["capture_status"] == status
    assert not data["settle_observed"] and "method_blocked" in data["warnings"]
    assert "POST_REACHED_SERVER" not in site.seen


def test_public_redirect_and_fresh_storage(chromium, site):
    data, code = read(site, "/public-redirect")
    assert code is None and site.inserted in data["text"]
    for _ in range(2):
        data, code = read(site, "/state")
        assert code is None and data["text"] == "fresh"


def test_browser_executable_unavailable(chromium, monkeypatch):
    monkeypatch.setattr(worker.Path, "is_file", lambda _: False)
    data, code = asyncio.run(worker.render(RenderedWebFetchRequest(url="https://example.com"),
        lambda _: None, validate=lambda _: None))
    assert code == codes.RENDERED_WEB_UNAVAILABLE and data["completion_reason"] == "chromium_missing"


@pytest.mark.parametrize("path", ["/#supervised", "/hung-js"])
def test_supervised_browser_with_private_temporary_state(chromium, site, monkeypatch, path):
    # Exercise production supervision and browser environment. Only the child
    # fixture validator is substituted; no production URL policy switch exists.
    script = (
        "import asyncio,json,sys,time; "
        "from tools.rendered_web_worker import render; "
        "from tools.web_ops import _validate_public_url; "
        "from core.models import RenderedWebFetchRequest; "
        "req=RenderedWebFetchRequest.model_validate_json(sys.stdin.buffer.readline()); "
        "emit=lambda data: print(json.dumps(dict(data=data,done=False)),flush=True); "
        "validate=lambda url: None if url.startswith(" + repr(site.base + "/") + ") else _validate_public_url(url); "
        "data,code=asyncio.run(render(req,emit,validate=validate)); "
        "print(json.dumps(dict(data=data,done=True,code=code)),flush=True); time.sleep(60)"
    )
    original = subprocess.Popen

    def popen(command, **kwargs):
        return original([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(ops.subprocess, "Popen", popen)
    if path == "/hung-js":
        monkeypatch.setattr(ops, "OVERALL_TIMEOUT", 5)
    result = ops._run_worker(RenderedWebFetchRequest(url=site.base + path), time.monotonic())
    if path == "/hung-js":
        assert result.error_code == codes.RENDERED_WEB_TIMEOUT, result
        assert result.data["text"] == "partial before hang" and result.data["capture_status"] == "partial"
    else:
        assert result.success, result
        assert result.data["text"] == site.inserted + " #supervised"
