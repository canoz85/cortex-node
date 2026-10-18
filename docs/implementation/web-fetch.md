# Public single-page HTML fetch

`web_fetch` reads the current textual content of one known public URL. It is a
normal, enabled-by-default, read-only capability in the production manifest and
ToolRegistry. Planner info requests can select it; Brain binds its native schema
and executes it through the existing runtime. Controller and duplicate-successful
call semantics are unchanged. The existing DuckDuckGo `search_web()` implementation
is unchanged and remains unregistered.

## Arguments and evidence

The executable argument schema is `WebFetchRequest`, with strict types and extra
arguments forbidden:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["url"],
  "properties": {
    "url": {"type": "string", "minLength": 1, "maxLength": 8192},
    "max_chars": {"type": "integer", "default": 50000, "minimum": 1, "maximum": 200000}
  }
}
```

LangChain/Ollama can omit validation keywords when transporting native schemas;
the executable Pydantic schema still validates locally. There are no query,
authentication, private-network bypass, or continuation parameters.

`fetch_web()` returns a `ToolOutputEnvelope`; the registered wrapper serializes
it using the ordinary summary-plus-JSON contract. Success has this data shape:

```json
{
  "url": "https://example.com/",
  "final_url": "https://example.com/",
  "status_code": 200,
  "content_type": "text/html",
  "title": "Example",
  "text": "Bounded extracted evidence",
  "retrieved_at": "2026-10-16T09:00:00+00:00",
  "total_chars": 26,
  "truncated": false,
  "total_text_bytes": 26
}
```

`retrieved_at` is the UTC retrieval-completion time. `total_chars` counts all
normalized extracted characters before truncation. `total_text_bytes` counts
their UTF-8 bytes, allowing normalized ToolResult integrity to retain accurate
byte counts, including Unicode. `text` contains only the first `max_chars`
characters. Failure data includes requested URL, last attempted final URL if
known, status code if known, and normalized media type if known. Messages and
display never include the page body or transport exception details.

The normalizer records truncation in `ContentIntegrity`, with no pagination.
`read_file` continuation uses file-specific offset/limit semantics; the runtime
has no generic stable web character-range contract. A truncated fetch explicitly
reports its full extracted length and lack of continuation. The existing Brain
evidence projection also has its own 10,000-character string budget and explicit
truncation marker; this change does not alter that shared budget.

## URL and transport policy

Only HTTP/HTTPS URLs with a hostname and without embedded credentials are
accepted. Malformed ports, controls, backslashes and IPv6 scope identifiers are
rejected. Localhost names are rejected directly. Every resolved IPv4/IPv6 address
must be global and must not be private, loopback, link-local, multicast, reserved
or unspecified. Mixed public/private DNS answers fail closed. Shared address
space and IPv6 transition addresses that could hide local IPv4 targets are also
rejected.

The Requests adapter uses instance-local urllib3 connection classes. Immediately
before opening a socket, it re-resolves and revalidates all addresses, then
connects to a validated numeric IP. It retains the original hostname for Host,
TLS SNI and certificate verification (`verify=True`). This prevents a DNS change
between the URL check and connection from authorizing a private socket. It does
not change process-wide DNS or connection classes. It uses the first validated
address; there is no automatic address failover.

Redirects (301/302/303/307/308) are handled manually, up to five hops. Relative
locations resolve against the preceding URL. Each new URL is validated before
sending, including the final destination. No redirect body is downloaded. The
implementation calls `HTTPAdapter.send` directly: a Requests Session can consume
redirect bodies even with `allow_redirects=False`. Direct adapter use also avoids
ambient proxies, netrc, cookies and login sessions.

The User-Agent identifies CortexNode. Connect/read timeouts are 5/15 seconds,
with no retries. These are per-operation timeouts, not a total wall-clock deadline;
OS DNS resolution follows platform resolver timing. Non-2xx terminal statuses
fail, including 3xx statuses outside the supported redirect set.

The response body limit is **2 MiB (2,097,152 bytes) after HTTP content decoding**,
independent of `max_chars`. Oversized Content-Length is rejected before reading;
absent, invalid, or dishonest lengths cannot bypass the streamed bound. Requests
reads in 16 KiB decoded chunks and stops before retaining bytes beyond the limit.
`Accept-Encoding: identity` requests an uncompressed representation; if a server
still sends supported compression, the decoded-size limit applies. Headers and
transport/decompression buffers are additional to the retained body bound.

## Extraction and errors

Only `text/html` and `application/xhtml+xml` are accepted, case-insensitively and
without media-type parameters. Missing Content-Type, PDF and binary types fail
before reading their bodies. No files are written.

BeautifulSoup uses `html.parser`. Explicit HTTP charset takes precedence;
otherwise HTML meta/BOM detection applies. Title whitespace is normalized and
the title is capped at 1,000 characters (160 in display). Scripts, styles,
noscript, SVG, canvas, iframes and forms are removed. The first article is
preferred, then main, then body, then the document. Head content is excluded from
body evidence. Block/line-break boundaries are preserved, inline text remains
connected, horizontal whitespace is collapsed, and excess blank lines are
reduced to one blank line. No navigation/readability heuristics discard other
semantic content.

Canonical constants in `core/error_codes.py` cover:

- `WEB_FETCH_REJECTED`: invalid arguments, unsafe URL or unsafe resolved address.
- `WEB_FETCH_TIMEOUT`: connect/read timeout, including streaming timeout wrapping.
- `WEB_FETCH_UNSUPPORTED_CONTENT_TYPE`: missing or non-HTML media type.
- `WEB_FETCH_RESPONSE_TOO_LARGE`: declared or streamed body limit exceeded.
- `WEB_FETCH_FAILED`: HTTP, DNS, connection, parser or redirect failure.

The concise capability card describes final URL/title/text/retrieval metadata,
public HTTP/HTTPS and HTML limits, incomplete JavaScript-rendered content,
truncation without continuation, and known-URL use. It has `mutating=false`, no
pagination and no exact-collection metadata.

Search, browser interaction/JavaScript rendering, login/cookies, PDF/OCR,
crawling/link following, snapshots/diffs, publishing and continuation remain
outside this capability.

## Offline validation

`tests/test_web_ops.py` mocks DNS and sockets/HTTP transport. It covers URL/IP
rejections, mixed DNS answers, DNS rebinding, every supported redirect status,
redirect safety/limits/body avoidance, TLS hostname preservation, content types,
timeouts/status failures, extraction/fallbacks/Unicode, truncation, concise
display, and declared/streamed/decoded response-size limits.

`tests/test_web_fetch_runtime.py` covers production registry defaults and cards,
real ChatOllama native schema serialization with a mocked transport, the info
Planner/Brain/Controller production loop, normalized text/integrity evidence
(including truncated evidence without pagination), and duplicate-successful-call
rejection at native normalization and Controller authorization.

Validation on 2026-10-16 (offline, Python 3.14.3):

| Check | Result |
| --- | --- |
| New fetch/security/runtime cases | 114 passed within the focused and full runs |
| Focused web/registry/schema/Brain/driver/Controller/duplicate/file regression suite | 370 passed, 1 pre-existing failure |
| Full offline suite (`--ignore=tests/live`) | 1,468 passed, 6 pre-existing failures, 3 skipped |
| Core/tools coverage | 87.04%; required 71% passed |
| `git diff --check` | Passed |
| Existing `search_web` function AST comparison with HEAD | Unchanged |

The six failures were reproduced with the original HEAD models, output normalizer
and registry loaded in memory, without changing the working tree:

- `test_async_poller.py::test_build_app_runs_submission_await_poll_capture_and_resume_end_to_end`
  (legacy five-node factory return versus four-node unpacking).
- `test_default_observability.py::test_default_renders_accepted_plan_and_coalesced_brain_tool_call`
  (planner display expectation).
- `test_direct_tool_runtime.py::test_direct_integration_preserves_artifacts_and_repeat_failure_accounting`
  (missing artifact on a failed write; also the focused-suite failure).
- `test_live_status.py::test_permanent_output_clears_then_redraws` and
  `test_live_status.py::test_stop_clears_status_and_stops_redraw` (terminal escaping).
- `test_terminal_memory.py::test_updater_failure_preserves_memory_and_successful_turn`
  (mock turn has no accepted final answer).

The broader command was:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q -p no:cacheprovider --cov=core --cov=tools --cov-report=term --cov-fail-under=71 --tb=short --ignore=tests/live
```

Pytest needed execution outside the sandbox to access its temporary directories.
Its normal temporary directory was used for final runs so Git non-repository
tests did not accidentally inherit this repository from a workspace-local temp
directory. No live web or model calls were used by the new tests.
