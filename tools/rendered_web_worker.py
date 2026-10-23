"""Private disposable Chromium worker; stdout is bounded snapshot IPC only.

CDP validates each redirected request, but Chromium resolves DNS independently.
This is deliberately NOT the socket-pinned transport used by web_fetch.
"""

import asyncio
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit, urlunsplit

from core import error_codes as codes
from core.models import RenderedWebFetchRequest
from tools.rendered_web_ops import result_data
from tools.web_ops import _validate_public_url, _RejectedURL, _normalize_text

NAVIGATION_MS = 15_000
SETTLE_SECONDS = 5
MIN_SETTLE_SECONDS = 1.5
STABLE_SECONDS = 1
MAX_REQUESTS = 100
MAX_NAVIGATIONS = 6
MAX_RESOURCE_BYTES = 5 * 1024 * 1024
MAX_TRANSFER_BYTES = 20 * 1024 * 1024
MAX_BLOCKED_REQUESTS = 8


def _request_identity(url):
    """Diagnostic identity only; never used to authorize a request."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return "[unsupported URL]", None
        host = parsed.hostname.lower()
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        authority = f"[{host}]" if ":" in host else host
        if parsed.port is not None:
            authority += f":{parsed.port}"
        clean = urlunsplit((parsed.scheme, authority, parsed.path, "", ""))
        clean = "".join(c for c in clean if ord(c) >= 32 and ord(c) != 127)
        return clean[:512], (parsed.scheme, host, port)
    except ValueError:
        return "[invalid URL]", None

# An additional CSP intersects (never replaces) the site's own policy.
CSP = "sandbox allow-scripts allow-same-origin; worker-src 'none'; frame-src 'none'; object-src 'none'; form-action 'none'; connect-src http: https:"
INIT_SCRIPT = """(() => {
  for (const name of ['RTCPeerConnection', 'webkitRTCPeerConnection', 'RTCDataChannel',
                       'WebSocket', 'Worker', 'SharedWorker', 'open']) {
    Object.defineProperty(globalThis, name, {value: undefined, configurable: false, writable: false});
  }
})()"""

# Fixed application code only. Do not mutate the live DOM to remove content.
EXTRACT_SCRIPT = """() => {
 const root = document.querySelector('article') || document.querySelector('main') || document.body;
 const skip = new Set(['SCRIPT','STYLE','NOSCRIPT','SVG','CANVAS','IFRAME','FORM','TEMPLATE','OBJECT','EMBED']);
 let pieces = [], chars = 0, nodes = 0, limited = false;
 function add(s) { chars += s.length; if(chars > 2000000) {limited = true; return;} pieces.push(s); }
 function walk(n) {
   if(limited) return;
   if(++nodes > 50000) {limited = true; return;}
   if(n.nodeType === 3) {add(n.textContent); return;}
   if(n.nodeType !== 1 || skip.has(n.tagName)) return;
   const style = getComputedStyle(n);
   if(style.display === 'none' || style.visibility === 'hidden' || n.hidden) return;
   const block = !['inline','inline-block','contents'].includes(style.display);
   if(block || n.tagName === 'BR') add('\\n');
   for(const child of n.childNodes) walk(child);
   if(block) add('\\n');
 }
 if(root) walk(root);
 return {url: location.href, title: document.title.slice(0,1000), text: pieces.join(''), limited};
}"""


class Policy:
    def __init__(self, data, validate=_validate_public_url):
        self.data, self.validate = data, validate
        self.code = None
        self.stop = False
        self.requests = self.navigations = self.transferred = 0

    def fail(self, code, reason, *, fatal=True):
        self.stop = self.stop or fatal
        if fatal and self.code is None:
            self.code = code
            self.data["completion_reason"] = reason
        if reason not in self.data["warnings"] and len(self.data["warnings"]) < 12:
            self.data["warnings"].append(reason)

    async def request_allowed(self, url, method, resource_type, main_document=True, *, document_url=None):
        allowed = await self._request_allowed(url, method, resource_type, main_document)
        if not allowed:
            self.data["blocked_request_count"] += 1
            if len(self.data["blocked_requests"]) < MAX_BLOCKED_REQUESTS:
                sanitized, origin = _request_identity(url)
                _, document_origin = _request_identity(document_url or self.data["url"])
                self.data["blocked_requests"].append(dict(
                    method=method[:16], url=sanitized, resource_type=resource_type[:32],
                    same_origin=(origin == document_origin) if origin and document_origin else None,
                    is_main_navigation=resource_type == "Document" and main_document,
                ))
        return allowed

    async def _request_allowed(self, url, method, resource_type, main_document):
        document = resource_type == "Document" and main_document
        self.requests += 1
        if self.requests > MAX_REQUESTS:
            self.fail(codes.RENDERED_WEB_RESOURCE_LIMIT, "request_limit")
            return False
        try:
            await asyncio.wait_for(asyncio.to_thread(self.validate, url), 3)
        except _RejectedURL:
            self.fail(codes.RENDERED_WEB_POLICY_BLOCKED, "unsafe_destination", fatal=document)
            return False
        except (OSError, asyncio.TimeoutError):
            self.fail(codes.RENDERED_WEB_NAVIGATION_FAILED, "dns_failed", fatal=document)
            return False
        if method not in {"GET", "HEAD"}:
            self.fail(codes.RENDERED_WEB_POLICY_BLOCKED, "method_blocked", fatal=document)
            return False
        if resource_type == "Document":
            self.navigations += 1
            if not main_document:
                self.fail(codes.RENDERED_WEB_POLICY_BLOCKED, "frame_blocked", fatal=False)
                return False
            if self.navigations > MAX_NAVIGATIONS:
                self.fail(codes.RENDERED_WEB_RESOURCE_LIMIT, "navigation_limit")
                return False
        if resource_type in {"Image", "Media", "Font"}:
            if "images_media_fonts_omitted" not in self.data["warnings"]:
                self.data["warnings"].append("images_media_fonts_omitted")
            return False
        if resource_type not in {"Document", "Script", "Stylesheet", "XHR", "Fetch"}:
            self.fail(codes.RENDERED_WEB_POLICY_BLOCKED, "resource_type_blocked", fatal=False)
            return False
        return True


async def render(request, emit, *, validate=_validate_public_url):
    data = result_data(request.url)
    policy = Policy(data, validate)
    try:
        await asyncio.wait_for(asyncio.to_thread(validate, request.url), 3)
    except _RejectedURL:
        data["completion_reason"] = "unsafe_initial_url"
        return data, codes.RENDERED_WEB_REJECTED
    except (OSError, asyncio.TimeoutError):
        data["completion_reason"] = "dns_failed"
        return data, codes.RENDERED_WEB_NAVIGATION_FAILED
    try:
        from playwright.async_api import async_playwright, TimeoutError as BrowserTimeout
    except ImportError:
        data["completion_reason"] = "playwright_missing"
        return data, codes.RENDERED_WEB_UNAVAILABLE

    browser = None
    async with async_playwright() as playwright:
        if not Path(playwright.chromium.executable_path).is_file():
            data["completion_reason"] = "chromium_missing"
            return data, codes.RENDERED_WEB_UNAVAILABLE
        try:
            browser = await playwright.chromium.launch(headless=True, chromium_sandbox=True,
                timeout=8000, args=["--no-proxy-server", "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"])
        except Exception:
            data["completion_reason"] = "chromium_launch_unavailable"
            return data, codes.RENDERED_WEB_UNAVAILABLE
        try:
            context = await browser.new_context(accept_downloads=False, service_workers="block",
                permissions=[], ignore_https_errors=False, viewport={"width": 1280, "height": 720})
            await context.add_init_script(INIT_SCRIPT)
            await context.route_web_socket("**/*", lambda socket: socket.close())
            page = await context.new_page()
            cdp = await context.new_cdp_session(page)
            tree = await cdp.send("Page.getFrameTree")
            main_frame = tree["frameTree"]["frame"]["id"]
            await cdp.send("Browser.setDownloadBehavior", {"behavior": "deny"})

            async def popup(other):
                if other != page:
                    policy.fail(codes.RENDERED_WEB_POLICY_BLOCKED, "popup_blocked")
                    await other.close()

            context.on("page", popup)
            page.on("download", lambda _: policy.fail(codes.RENDERED_WEB_UNSUPPORTED, "download_blocked"))
            page.on("dialog", lambda dialog: dialog.dismiss())

            async def authentication(event):
                policy.fail(codes.RENDERED_WEB_UNSUPPORTED, "authentication_required")
                await cdp.send("Fetch.continueWithAuth", {"requestId": event["requestId"],
                    "authChallengeResponse": {"response": "CancelAuth"}})

            async def paused(event):
                identity = {"requestId": event["requestId"]}
                try:
                    if "responseStatusCode" not in event and "responseErrorReason" not in event:
                        req = event["request"]
                        allowed = await policy.request_allowed(req["url"], req["method"], event["resourceType"],
                            event.get("frameId") == main_frame,
                            document_url=page.url if page.url != "about:blank" else request.url)
                        await cdp.send("Fetch.continueRequest" if allowed else "Fetch.failRequest",
                            identity if allowed else {**identity, "errorReason": "BlockedByClient"})
                        return
                    headers = event.get("responseHeaders", [])
                    if "responseErrorReason" in event:
                        policy.fail(codes.RENDERED_WEB_NAVIGATION_FAILED, "network_response_failed",
                                    fatal=event["resourceType"] == "Document" and event.get("frameId") == main_frame)
                        await cdp.send("Fetch.failRequest", {**identity, "errorReason": "Failed"})
                        return
                    values = {h["name"].lower(): h["value"] for h in headers}
                    rejected = False
                    length = values.get("content-length", "")
                    if length.isdigit() and int(length) > MAX_RESOURCE_BYTES:
                        policy.fail(codes.RENDERED_WEB_RESOURCE_LIMIT, "declared_response_limit")
                        rejected = True
                    if "attachment" in values.get("content-disposition", "").lower():
                        policy.fail(codes.RENDERED_WEB_UNSUPPORTED, "download_blocked")
                        rejected = True
                    if event["resourceType"] == "Document" and event.get("frameId") == main_frame:
                        status = event.get("responseStatusCode", 0)
                        if not 300 <= status < 400:
                            data["status_code"] = status
                            data["content_type"] = values.get("content-type", "").split(";")[0].strip().lower()
                            if not 200 <= status < 300:
                                policy.fail(codes.RENDERED_WEB_HTTP_ERROR, "http_error")
                                rejected = True
                            elif data["content_type"] not in {"text/html", "application/xhtml+xml"}:
                                policy.fail(codes.RENDERED_WEB_UNSUPPORTED, "non_html")
                                rejected = True
                    if rejected or policy.stop:
                        await cdp.send("Fetch.failRequest", {**identity, "errorReason": "BlockedByClient"})
                    else:
                        await cdp.send("Fetch.continueResponse", {**identity,
                            "responseCode": event["responseStatusCode"],
                            "responseHeaders": [*headers, {"name": "Content-Security-Policy", "value": CSP}]})
                except Exception:
                    policy.fail(codes.RENDERED_WEB_RENDER_FAILED, "interception_failed")
                    try:
                        await cdp.send("Fetch.failRequest", {**identity, "errorReason": "BlockedByClient"})
                    except Exception:
                        pass

            resource_bytes = {}

            def received(event):
                policy.transferred += event.get("dataLength", 0)
                request_id = event["requestId"]
                resource_bytes[request_id] = resource_bytes.get(request_id, 0) + event.get("dataLength", 0)
                if policy.transferred > MAX_TRANSFER_BYTES or resource_bytes[request_id] > MAX_RESOURCE_BYTES:
                    already_stopping = policy.stop
                    policy.fail(codes.RENDERED_WEB_RESOURCE_LIMIT, "observed_transfer_limit")
                    if not already_stopping:
                        asyncio.create_task(cdp.send("Page.stopLoading"))

            cdp.on("Fetch.requestPaused", paused)
            cdp.on("Fetch.authRequired", authentication)
            cdp.on("Network.dataReceived", received)
            await cdp.send("Network.enable")
            await cdp.send("Network.setCacheDisabled", {"cacheDisabled": True})
            await cdp.send("Fetch.enable", {"handleAuthRequests": True, "patterns": [
                {"urlPattern": "*", "requestStage": "Request"},
                {"urlPattern": "*", "requestStage": "Response"}]})

            async def capture():
                snapshot = await page.evaluate(EXTRACT_SCRIPT)
                if snapshot["limited"]:
                    policy.fail(codes.RENDERED_WEB_RESOURCE_LIMIT, "extraction_limit")
                    return
                full = _normalize_text(snapshot["text"])
                data.update(final_url=snapshot["url"], title=" ".join(snapshot["title"].split()),
                    text=full[:request.max_chars], total_chars=len(full),
                    total_text_bytes=len(full.encode("utf-8")), returned_chars=len(full[:request.max_chars]),
                    truncated=len(full) > request.max_chars)
                emit(dict(data))
                return full

            try:
                await page.goto(request.url, wait_until="domcontentloaded", timeout=NAVIGATION_MS)
            except BrowserTimeout:
                policy.fail(codes.RENDERED_WEB_TIMEOUT, "navigation_timeout")
            except Exception:
                policy.fail(codes.RENDERED_WEB_NAVIGATION_FAILED, "navigation_failed")
            start = stable_since = time.monotonic()
            previous = None
            while True:
                try:
                    full = await capture()
                except Exception:
                    policy.fail(codes.RENDERED_WEB_RENDER_FAILED, "extraction_failed")
                    break
                if policy.stop:
                    break
                now = time.monotonic()
                if full != previous:
                    previous, stable_since = full, now
                if full and now - start >= MIN_SETTLE_SECONDS and now - stable_since >= STABLE_SECONDS:
                    data["settle_observed"] = True
                    data["completion_reason"] = "text_stable"
                    break
                if now - start >= SETTLE_SECONDS:
                    policy.fail(codes.RENDERED_WEB_TIMEOUT if full else codes.RENDERED_WEB_RENDER_FAILED,
                                "settle_timeout" if full else "no_usable_text")
                    break
                await asyncio.sleep(0.25)
            if data["final_url"] is None:
                data["final_url"] = page.url if page.url != "about:blank" else None
            return data, policy.code
        finally:
            await browser.close()


def main():
    request = RenderedWebFetchRequest.model_validate_json(sys.stdin.buffer.readline(65536))

    def emit(data, done=False, code=None):
        print(json.dumps(dict(data=data, done=done, code=code), ensure_ascii=True), flush=True)

    try:
        data, code = asyncio.run(render(request, emit))
    except Exception:
        data, code = result_data(request.url), codes.RENDERED_WEB_RENDER_FAILED
    emit(data, True, code)
    # Stay alive until the parent tears down the owned process tree.
    time.sleep(60)


if __name__ == "__main__":
    main()
