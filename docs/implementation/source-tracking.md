# Explicit single-page source baselines

`source_snapshot(url)`, `source_diff(url)` and `source_status(url)` are ordinary
enabled production tools. Snapshot and diff call `tools.web_ops.fetch_web()` directly, with its
unchanged maximum of 200,000 extracted characters. Neither invokes another
registered tool. HTTP transport, SSRF/DNS/socket validation, redirects, HTML
extraction, decoded-body limits and timeouts remain owned by `web_fetch`.
`source_status` only reads the saved baseline; it never fetches or accesses the network.

There is no new Controller lifecycle, graph node, Planner branch, Brain action,
ExactCollection metadata, watcher or background work.

## Schemas and capability selection

All three executable schemas are strict `SourceUrlRequest` models:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["url"],
  "properties": {
    "url": {
      "type": "string",
      "minLength": 1,
      "maxLength": 8192,
      "description": "Known public HTTP/HTTPS page URL."
    }
  }
}
```

Storage paths/IDs, text, hashes, transport settings, private-network flags and
`max_chars` are not model inputs. The argument-correction path continues enforcing
these strict executable schemas.

The capability cards distinguish four purposes:

| Tool | Effect | Use | Avoid |
| --- | --- | --- | --- |
| `web_fetch` | Read-only | Read/summarize a known page's current content | Saving or comparing a baseline |
| `source_snapshot` | Mutating | Start tracking, save current state, establish/reset/replace a baseline | Only reading current content or comparing against a baseline |
| `source_diff` | Read-only | Determine/explain text changes since a saved baseline | Missing baseline, current-content-only requests, replacing a baseline |
| `source_status` | Read-only | Read the saved timestamp, hash, title, character count and URL metadata | Current content, baseline replacement, change comparison |

Cards include complete public HTTP/HTTPS HTML extraction requirements. Diff
evidence may truncate, and comparison never advances the baseline. Existing info
route filtering removes `source_snapshot`; action routes can select it. The tools
remain singleton capabilities (`max_batch_calls=1`) and semantic evidence.

Offline scripted routing/planning tests exercise these intended distinctions:

- "Track https://example.com from now on." → action / `source_snapshot`.
- "Save the current version ... as my baseline." → action / `source_snapshot`.
- "Has ... changed since I last saved it?" → info / `source_diff`.
- "What changed ... since my baseline?" → info / `source_diff`.
- "Read https://example.com." → info / `web_fetch`.

These verify schema/card/authority plumbing with deterministic model exchanges;
they do not claim to measure a live model's tool-selection accuracy.

## Saved baseline metadata

`source_status(url)` normalizes the URL in the source/web layer and reads the
existing `SourceSnapshotStore` record with all integrity and web-metadata checks.
Its success envelope contains exactly these data fields: `url`, `canonical_url`,
`final_url`, `title`, `content_type`, `retrieved_at`, `total_chars`, `content_hash`,
and `baseline_exists=true`. It returns neither saved page text nor current-page
metadata, and has no pagination, async state or ExactCollection metadata.

Saved-baseline questions use this runtime evidence as the source of truth.
Conversation history and Planner memory can contain older timestamps or hashes
after a baseline replacement. Baseline-age questions should first read
`source_status`, then pass its `retrieved_at` to `current_time(reference_time=...)`
for deterministic elapsed-time arithmetic. Capability cards supply this guidance;
no Planner, Brain, Controller or Finalizer special cases are added.

A missing baseline returns `SOURCE_BASELINE_NOT_FOUND` without creating a store
directory or baseline. Incomplete, corrupt or unsupported records return
`SOURCE_SNAPSHOT_CORRUPT`. Invalid URLs or unavailable storage return
`SOURCE_STATUS_FAILED`. Messages expose no internal paths or exception details.
Status reads make no fetch/network calls and never write, replace or advance the
baseline; the existing version-1 JSON layout and hashed lookup stay unchanged.

## Persistent store and identity

Investigation found application-session JSON persistence but no general internal
data directory, shared persistent-state abstraction, shared atomic-write helper,
or textual diff helper. The dedicated `SourceSnapshotStore` uses:

```text
~/.cortexnode/source_snapshots/<SHA-256 of normalized requested URL>.json
```

On Windows, `~` is the current OS user's home directory. The store is shared by
that user's CortexNode instances and persists independently of sessions,
conversation history, working directory and workspace. Construction and reads
never create directories. Trusted application/test code can inject a store root;
no model-facing configuration selects it.

The source-tool factory rejects stores inside the workspace or knowledge root.
Ordinary workspace file tools cannot read or write these records, and source-tool
outputs expose no internal filesystem paths. Internal state is not emitted as a
workspace artifact. This is application-owned state under the OS user's normal
permissions, not a cryptographic authenticity or hostile-process isolation layer.

URL normalization reuses Requests preparation plus existing web URL syntax
validation, with no DNS/network access. It normalizes scheme/host/IDNA, empty
paths and Requests percent-escaping, then removes fragments. Path case, query
order/values and explicit ports are preserved; explicit default ports can have
separate identities. No query stripping, host aliasing, URL discovery or broad
canonicalization is attempted.

The **requested URL identity** selects the record. `requested_url` preserves the
original input, while `canonical_url` stores its normalized lookup form and
`final_url` records the fetched redirect destination. Redirects never migrate or
alias a baseline to the final URL. Tracking the final URL separately creates a
separate baseline. All actual redirects retain the existing fetch safety policy.

URL normalization and HTML/extraction policy belong to `tools/source_ops.py` and
`tools/web_ops.py`. Core persistence accepts an already-resolved, opaque
`source_id`; it neither imports tools nor normalizes URLs. The internal field
retains `canonical_url` as its version-1 serialization alias, preserving existing
JSON records and hashed lookup filenames without migration. URL and content-type
fields remain metadata in core, with web-specific validation in the source tools
on both creation and baseline loading.

## Stored record and completeness

Each version-1 record includes:

- `schema_version`, `canonical_url`, `requested_url`, `final_url`.
- `title`, HTML `content_type`, UTC ISO-8601 `retrieved_at`.
- Full normalized extracted `text`, `total_chars`, `truncated=false`.
- SHA-256 `content_hash` over the saved text's UTF-8 bytes.

Snapshot/current comparison extraction must report success, `truncated=false`,
and `total_chars == len(text)`. Partial or inconsistent evidence is rejected,
even if its exposed prefix matches a saved page. `web_fetch` limits are not raised.
No PDF, binary, OCR, login or JavaScript rendering is added.

Loads are bounded to 2 MiB of JSON and validate strict schema/types, completeness,
UTC timestamp, source identity, text length and content hash. Unsupported versions,
bad JSON/encoding, missing fields and hash/identity mismatches fail closed. A
declared incomplete baseline produces an incomplete-comparison failure. No failure
returns a `changed=false` claim.

Core text bounds derive from its independent 2 MiB record budget. Source tools
retain the 200,000-character web extraction limit for fetched and saved web
evidence; storage does not depend on that transport-specific constant.

## Snapshot writes

`source_snapshot` fetches complete evidence, hashes it and explicitly creates or
replaces the single record. The receipt includes URL/final URL/title/content type,
retrieval time, total characters, hash, `baseline_created`, `baseline_replaced`
and `baseline_status` (`created`/`replaced`). It contains no page text. An explicit
snapshot can reset a corrupt old JSON record.

Writes follow the existing `apply_patch` pattern: same-directory temporary file,
write, flush, `os.fsync`, close, `os.replace`, and temporary-file cleanup. Failures
before replacement preserve the old record. No model-provided path or page text
is used as a filename. Reads/writes reject linked snapshot targets and detected
store-directory retargeting.

There is no cross-process lock subsystem. Concurrent snapshots are
last-successful-replace-wins; created/replaced flags reflect whether the target
existed before that write. Serialized runtime use is the supported receipt
interpretation. Atomic replacement prevents partial JSON records, but this does
not add database transactions, history, directory fsync, or a power-loss journal.

## Comparison semantics and bounds

`source_diff` loads and validates the baseline **before fetching**. Missing or
unusable baselines cause no network call. It then fetches complete current text,
computes its hash and compares the authoritative strings. It never saves,
replaces, creates, deletes or advances a baseline.

`changed` describes normalized extracted **text**, not every HTML/DOM/visual
change. Title-only and redirect-destination changes are exposed separately as
`title_changed` and `final_url_changed`, with baseline/current metadata. The body
text hash excludes the separately extracted title, consistent with `web_fetch`.
Script/style/navigation outside the selected article/main/body, CSS, attributes
and JavaScript-only changes may not affect the tracked text.

Changed text uses deterministic `difflib.unified_diff`, with three context lines
and stable `baseline`/`current` labels. The returned diff is its prefix bounded to
**8,000 characters and 200 lines**; it can end mid-line. Full diff character, line
and UTF-8 byte counts are computed while retaining only bounded output:

```text
url, final_url, baseline_final_url, final_url_changed
baseline_title, current_title, title_changed
baseline_retrieved_at, current_retrieved_at
baseline_hash, current_hash, changed, baseline_chars, current_chars
diff, diff_truncated, total_diff_chars, total_diff_lines, total_diff_bytes
returned_diff_chars, returned_diff_lines
```

Identical complete text yields `changed=false`, equal hashes and an empty diff
with zero counts. Retrieval timestamps can differ on repeated comparisons;
unchanged current text continues comparing against the original baseline.

The tool normalizer records bounded diff integrity with accurate UTF-8 byte counts
and no fake pagination. The 8,000-character diff fits the existing Brain per-string
evidence budget. A successful complete comparison may still have truncated diff
evidence; Brain can explain only changes supported by the returned portion.
Console/message output reports concise status/counts and never dumps the diff or
page text. Large changes, especially a very long single line, may expose only an
initial portion; full-diff continuation is deferred.

## Errors and replay

All operations return `ToolOutputEnvelope` and canonical constants:

| Code | Meaning |
| --- | --- |
| `SOURCE_BASELINE_NOT_FOUND` | No baseline; no automatic creation or fetch |
| `SOURCE_SNAPSHOT_INCOMPLETE` | Current extraction cannot establish a full baseline |
| `SOURCE_DIFF_INCOMPLETE` | Baseline/current extraction cannot support a complete comparison |
| `SOURCE_SNAPSHOT_CORRUPT` | Corrupt/unsupported stored record or invalid linked storage target |
| `SOURCE_SNAPSHOT_FAILED` | Snapshot fetch, arguments/data, or storage failed |
| `SOURCE_DIFF_FAILED` | Comparison fetch, arguments/data, or storage failed |

Wrapped fetch failures retain the existing `WEB_FETCH_*` code in `data.cause_code`
and safe known transport metadata. Raw filesystem paths and exception traces are
never returned. Errors omit `changed` rather than reporting unchanged.

The existing step-local successful-call guard applies to both tools. Native
correction can reuse prior evidence; Controller and ExecutionDriver reject
duplicate proposals/replayed successful authorizations before another mutation.
A new explicit execution can reset a baseline. Fresh diff executions compare
against the retained baseline. No mutation idempotency journal was added: a
process crash after file commit but before success/history capture retains the
runtime's existing ambiguous-side-effect limitation.

## Manual live smoke

Run these in the same CortexNode conversation:

```text
Track this page from now on:
https://docs.python.org/3.15/whatsnew/3.15.html

Has this page changed since my saved baseline?
```

Expected: action / `source_snapshot`, then info / `source_diff`; initially
`changed=false` if complete normalized page text remains identical. If extraction
is truncated or unavailable, expect a structured failure instead. No automatic
baseline update occurs. To explicitly reset it:

```text
Save the current version of https://docs.python.org/3.15/whatsnew/3.15.html as my baseline.
```

This sequence is documented for manual use; automated validation mocks all
network/model access.

## Deferred scope

Watchlists, multiple-source batching, scheduling/background monitoring, automatic
refresh/baseline advancement, search, PDF/OCR, semantic/LLM-generated diff,
ranking, browser/login/JavaScript, publishing/notifications, ETag/Last-Modified,
baseline history/version chains, continuation and cross-process locking are not
implemented.

## Offline validation

Validation on 2026-10-17 (Python 3.14.3):

| Check | Result |
| --- | --- |
| New source storage/operation/runtime cases | 82 passed |
| Focused web/source/registry/Planner/Brain/Controller/mutation/duplicate/replay suite | 536 passed |
| Broader offline suite (`--ignore=tests/live`) | 1,588 passed, 6 pre-existing failures, 3 skipped |
| Core/tools coverage | 87.37%; required 71% passed |
| `git diff --check` | Passed |

`tests/test_source_ops.py` covers create/replace, new-instance persistence,
completeness refusal, missing baselines, unchanged/changed/added/removed content,
title/redirect metadata, Unicode/hash integrity, corrupt/unsupported/incomplete
records, bounded reads/diffs and exact truncation counts, requested URL identity,
model path restrictions, workspace isolation, flush/fsync/replace ordering,
failure cleanup, unchanged baseline bytes/mtime, and real fetch-layer redirect
rejection/truncation with mocked HTTP transport.

`tests/test_source_runtime.py` covers strict native schemas and effect metadata,
all five requested route/tool distinctions, info-route mutation exclusion,
ordinary Brain/Controller execution/evidence, diff integrity without pagination,
failure routing, repeated comparisons in fresh executions, duplicate correction,
checkpoint restoration and replayed-authorization mutation rejection. Planner
selection exchanges are scripted; network access is mocked.

The six broader failures match the baseline previously reproduced against HEAD:
legacy async factory compatibility, default planner observability, failed-write
artifact accounting, two terminal escaping expectations, and a terminal-memory
mock without an accepted final answer. Exact names and reproduction are in
[the existing baseline report](web-fetch.md#offline-validation). No new broader
suite failures were introduced.

Focused command:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q -p no:cacheprovider --no-cov --tb=short tests/test_source_ops.py tests/test_source_runtime.py tests/test_web_ops.py tests/test_web_fetch_runtime.py tests/test_tool_registry.py tests/test_planner_context_hardening.py tests/test_planner_service.py tests/test_planner_routing.py tests/test_brain_outcomes.py tests/test_brain_argument_correction.py tests/test_execution_driver.py tests/test_controller_execution.py tests/test_duplicate_successful_call.py tests/test_planner_runtime_flow.py tests/test_brain_read_file_batch.py
```

Broader command:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q -p no:cacheprovider --cov=core --cov=tools --cov-report=term --cov-fail-under=71 --tb=short --ignore=tests/live
```

Tests used pytest's normal temporary directory outside the sandbox, as required
by the established temporary-directory access restriction. Tests inject isolated
store roots; they do not write the user's production source store.
