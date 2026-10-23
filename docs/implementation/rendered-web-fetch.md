# Optional rendered public-page reader

`rendered_web_fetch` is a separate read-only capability for known public HTML
pages that need JavaScript. `web_fetch` remains the preferred cheap static path;
neither tool automatically calls the other. Source baseline tools are unchanged.

## Deployment and authorization

The tool's deployment enablement follows its Registry definition. Enable it
only in an environment where the limitations
below are acceptable. Existing Controller capability ceilings still apply;
registration, availability and planning guidance do not grant authorization.
`mutating=False`, `max_batch_calls=2`; independent calls use the existing
sequential batch lifecycle. There is no URL-array input or browser handle.

Optional installation (never performed by tool execution):

```text
python -m pip install -r requirements-rendered-web.txt
python -m playwright install chromium
```

Chromium and Playwright are not imported at Registry construction. Missing
package/executable or failed browser launch returns `RENDERED_WEB_UNAVAILABLE`.
There is no system-browser fallback or runtime package/browser download.
Tested version information and validation results are recorded below.

## Executable contract

Strict Pydantic input; extras and type coercion are forbidden:

* `url`: required string, 1..8192 characters, public HTTP/HTTPS without credentials.
* `max_chars`: optional integer, default 50000, range 1..200000.

Fragments are preserved in navigation and evidence, including hash application
routes. No model-supplied selectors, scripts, headers, credentials or wait options.

The ordinary `ToolOutputEnvelope` contains `url`, `final_url`, `title`, `text`,
`retrieved_at` (UTC observation-completion time, before parent teardown), `total_chars`, `total_text_bytes`,
`returned_chars`, `truncated`, `capture_status`, `completion_reason`, `elapsed_ms`,
`warnings`, `status_code`, `content_type`, `settle_observed`, `blocked_requests`,
and `blocked_request_count`. At most eight blocked request records are retained;
the count includes omitted records. Each contains method (16 characters), URL
(512 characters, credentials/query/fragment removed), resource_type (32 characters),
same_origin (relative to the current document; null if unknown), and
is_main_navigation. No request bodies, headers, cookies or stacks are recorded.

Capture statuses:

* `rendered`: the main page loaded, nonempty normalized text settled and no hard
  failure occurred; `success=true`, including when secondary requests were blocked.
* `partial`: failure with a previously captured text snapshot; `success=false`.
* `failed`: failure without text; `success=false`.

`rendered` is not proof of page completeness, relevance, or absence of a login,
consent or bot barrier. Such HTTP-200 interstitials can be captured as ordinary
text; the tool does not claim reliable barrier detection. `warnings` always
identifies heuristic settling and absence of socket-pinned request enforcement.
Additional reason tokens are bounded; exception strings and response bodies are
not copied into the display. A failure's text is partial evidence, not a success.
Secondary blocks remain warnings and diagnostics; they do not imply that the
blocked requests were optional or irrelevant. There is no site-specific rule or
dependency analysis. Empty captures fail with `no_usable_text`; nonempty captures
that do not settle fail with `settle_timeout`. Hard failures remain failures even
if an earlier snapshot contains text.

Extraction uses the first article, otherwise main, otherwise body. Fixed
implementation-owned JavaScript traverses the live rendered DOM without mutating
it, omitting scripts, styles, hidden elements, forms, frames, SVG/canvas, templates,
objects and embeds. Whitespace normalization reuses the static reader helper.
Block boundaries are retained. Offscreen DOM text can be included; nothing is
scrolled or clicked. No shadow-root traversal, screenshots, raw HTML or OCR.

The full successfully captured text is counted before returning its first
`max_chars` Python characters. Unicode byte counts are UTF-8. Caller truncation
does not make the capture fail. No pagination or continuation is invented.
The normalizer records the same integrity fields as `web_fetch`; Brain's existing
independent evidence-projection limit and truncation markers are unchanged.
If extraction hits its internal cap, that snapshot is discarded and any previous
bounded snapshot is retained. Counts then describe that previous snapshot only.

## Bounds and lifecycle

Every invocation starts a separate Python worker and fresh Chromium process and
nonpersistent context. Cookies/local storage can work inside that call, but no
state is imported or reused between calls. Browser permissions are empty;
credentials, workspace settings and ambient proxy environment variables are not
passed in the worker environment. Browser default security and TLS verification
stay enabled; Chromium sandboxing is explicitly enabled.

The parent enforces a 30-second observation deadline, independent of DNS or page
JavaScript responsiveness. Process teardown allows up to three more seconds.
Temporary-directory deletion retries transient file locks for up to three further
seconds; unresolved cleanup is an explicit `temporary_cleanup_incomplete` warning
and never discards captured evidence. OS scheduling/filesystem stalls
are not a real-time guarantee. The latest complete snapshot survives an overall
timeout. A Windows kill-on-close Job Object owns the worker and descendants,
with a 1 GiB aggregate committed-memory limit; failure to establish it fails
closed. POSIX uses an owned process group terminated on return/cancellation, but
does not yet enforce an aggregate memory limit. A private per-call temporary
directory holds transient browser files and is removed after worker teardown.

Browser startup timeout: eight seconds. DNS check timeout: three seconds.
Navigation waits for DOMContentLoaded for at most 15 seconds. After navigation,
sample text every 250 ms, allow at least 1.5 seconds, and finish after one second
of stable normalized text. Stop after five seconds without settling. The parent
deadline caps all phases together. Network quiet is not a readiness requirement;
content delayed beyond the settling window may be absent. Partial timeout is
explicit, and retries remain Controller policy.

Request attempts are capped at 100; document requests (including HTTP redirects)
at six. This is not a crawl. Images/media/fonts are omitted with a warning.
Extraction stops at 50000 nodes or 2000000 JavaScript string code units; this is
separate from caller truncation. Declared resource bodies over 5 MiB are rejected.
CDP decoded-data observations stop loading when a resource exceeds 5 MiB or the
aggregate exceeds 20 MiB. These are **observational limits**, not streaming hard
byte/memory bounds: browser buffering/decompression can run ahead of events.

## Enforced request policy and deliberate limitations

The worker reuses `_validate_public_url`, `_parse_web_url` and the existing
IPv4/IPv6 public-address rules. Initial URLs and intercepted resource/navigation
requests must resolve only to public addresses. Mixed public/private answers,
localhost, private/loopback/link-local addresses and credentials are rejected.
CDP Fetch interception validates every request-stage event, including HTTP
redirect hops, rather than relying on Playwright's first-hop route callback.

Only GET/HEAD document/script/style/XHR/fetch requests are allowed. Unsafe
destinations, other methods, disallowed request types and excessive navigation
produce explicit failures. Response interception rejects non-HTML documents,
HTTP errors, attachments and excessive declared lengths before continuing.
HTTP authentication challenges are canceled. Service workers are blocked.
An unsafe or non-GET subresource is blocked without aborting other permitted
resources: the tool continues bounded rendering, then returns settled nonempty
text with a warning, provided no hard failure occurred. This allows a GET-based application to render
even if its analytics tries to POST. Unsafe main navigation and resource limits
stop the observation. The blocked request is never retried by the tool.

An additional response CSP intersects the site's policy: sandboxed documents
allow scripts and same-origin behavior, but no popups, downloads or form actions;
frames, workers and objects are prohibited. Playwright WebSocket routing closes
connections; CSP restricts connect schemes to HTTP/HTTPS. An initialization script
removes WebSocket, WebRTC, worker and window.open APIs. Chromium additionally uses
the `disable_non_proxied_udp` WebRTC policy. Downloads are denied at context and
browser levels; popups are also closed defensively. These are browser-layer
restrictions, not an OS network firewall. Unsupported schemes fail URL checks;
the tool never invokes a host protocol handler or exposes a file API.

**This renderer does not have `web_fetch`'s SSRF guarantee.** DNS is checked by
Python and subsequently resolved by Chromium. DNS rebinding/TOCTOU remains
possible. Browser background traffic, preconnects and implementation gaps are
not a proven complete mediation boundary. WebRTC restrictions are defense in
depth, not proof of zero arbitrary socket traffic. Run only where that residual
exposure is acceptable, preferably with external egress restrictions. A future
hardened renderer needs enforced network isolation or a validated transport
broker that pins sockets, plus hard streaming and cross-platform resource caps.

The browser has a new profile, no supplied credentials, no permission grants and
no model-facing host-file operation. It is not in an OS filesystem jail: Chromium
and its worker run as the application's account. Browser sandboxing does not
replace container/VM isolation against browser vulnerabilities. GET requests can
still cause server logging or badly designed server-side effects; read-only
means no intentional mutation operation is exposed.

## Canonical failures

All codes use the `RENDERED_WEB_` prefix:

| Suffix | Meaning |
| --- | --- |
| REJECTED | Invalid input or unsafe initial URL |
| POLICY_BLOCKED | Unsafe resource/navigation, method, frame or unsupported request type |
| UNSUPPORTED | Non-HTML, attachment/download or authentication challenge |
| HTTP_ERROR | Non-success terminal document response |
| NAVIGATION_FAILED | DNS/connection/TLS/navigation failure |
| TIMEOUT | Navigation, settling or parent deadline |
| RESOURCE_LIMIT | Observed request/navigation/body/extraction limit |
| RENDER_FAILED | Unexpected browser/interception/extraction/worker failure |
| UNAVAILABLE | Missing package/browser, failed launch or unavailable Windows job isolation |

The first hard failure is retained; subsequent navigation-aborted errors must
not overwrite it. Abrupt OS resource termination can appear as RENDER_FAILED
because there is no trustworthy worker result identifying its cause.

## Tests

`test_rendered_web.py` contains offline schema/policy/supervision tests and real
Chromium tests against a local HTTP server. Only the in-process test fixture
injects a validator allowing its exact server origin; there is no production
private-network flag. Real-browser cases skip when optional prerequisites are
absent. `test_rendered_web_runtime.py` exercises native selection, Controller
execution and Brain evidence through the existing production test harness.

No live public website or live model is required. The existing web/source tests
are unchanged; the Registry inventory assertion includes the new disabled tool.

Validation on 2026-10-23: Python 3.14.3, Playwright 1.64.1, bundled Chromium
156.0.8078.4 on Windows. The focused suite passed **423 tests**, including all
46 new renderer/runtime cases and real-browser fixtures, with one existing
LangChain deprecation warning. No browser cases were skipped. The suite includes
static fetch, source operations/runtime/status/storage, Registry, and metadata,
ownership and mixed batch regressions. Scoped diff whitespace checks passed;
existing unrelated `main.py` edits were not changed.

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q -p no:cacheprovider --tb=short tests/test_rendered_web.py tests/test_rendered_web_runtime.py tests/test_web_ops.py tests/test_web_fetch_runtime.py tests/test_source_ops.py tests/test_source_runtime.py tests/test_source_status.py tests/test_source_snapshots.py tests/test_tool_registry.py tests/test_brain_metadata_batch.py tests/test_brain_batch_policy_ownership.py tests/test_brain_mixed_batch.py
```

Test execution required access outside the development sandbox for pytest's
temporary directories and Chromium processes. Optional package/browser installation
was performed explicitly for development validation, never by the tool itself.

Classification follow-up (2026-10-23): the same focused suite passed **427 tests**
with no skips and one existing LangChain deprecation warning. Added coverage for
bounded/sanitized blocked-request metadata, main-navigation failure priority,
settled text with blocked secondary requests, empty and unsettled captures, and
warning propagation into Brain evidence. POSTs still never reach the fixture
server. Deployment assertions now follow the current Registry definition rather
than assuming the renderer remains disabled; no authorization code was changed.
