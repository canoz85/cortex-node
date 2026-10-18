# Executable argument correction at the Brain provider boundary

## Root cause

`LangChainBrainProvider.generate()` called `_ensure_native_call()` before its
final executable-schema check. `_ensure_native_call()` offered one correction
for missing native calls, disallowed otherwise-valid batches, and duplicate
successful calls. Schema-invalid arguments were absent from that classification.

For a singleton, normalization accepted the authorized native name and argument
object into a ToolRequest proposal. The provider then ran
`_are_valid_native_calls()`, swallowed schema ValidationError into `False`, and
returned `INVALID_OUTPUT / invalid_native_tool_arguments`. Correction had already
been skipped. Controller applied its existing failure policy; with no remaining
step retries, the execution failed after one provider invocation.

Invalid arguments in a batch also caused `_is_valid_native_batch()` to return
False, preventing the response from entering the existing batch correction path.

## Validation and correction boundaries

1. LangChain/Ollama parses the native provider channel into `AIMessage.tool_calls`
   (or reports parser failures through `invalid_tool_calls`). Text is never
   parsed for actions.
2. `brain_normalization._parse_call()` and `validate_native_call()` validate the
   call envelope, authorized name, argument-object shape, and strict lifecycle
   fields. `_are_valid_native_calls()` additionally validates call type/id and
   JSON-safe values. These remain prerequisites for argument correction.
3. `ToolRegistry.resolve()` checks executable inventory/selection. It does not
   validate arguments. `get_tool_argument_schema(..., registry=...)` supplies the
   registered executable's schema, without falling back to a production
   declaration for an explicit registry.
4. The provider validates executable arguments using that schema's
   `model_validate()` or legacy `parse_obj()`. Validation also occurs in existing
   candidate/post-normalization checks. The fix diagnoses actual schema failures
   before `_ensure_native_call()` makes its correction decision.
5. `_ensure_native_call()` appends the rejected native AIMessage and a bounded
   structured correction instruction to the same messages and invokes the
   already-bound model once. It creates no ToolRequest or execution record and
   invokes no executable tool. The correction is shared with all other triggers;
   triggers do not receive independent correction budgets.
6. Corrected output is checked again before normalization creates a ToolRequest.
   Persistent schema failure returns the existing concise invalid-argument
   outcome. Valid repaired output still passes normalization, batch policy,
   duplicate guard, and the provider's existing final executable-schema check.
7. Multi-call normalization and Controller authorization/resumption each run
   the shared batch validator. Single-call Controller authorization retains its
   existing capability/duplicate/lifecycle checks; executable LangChain tools
   validate their input again when invoked. This change adds no Controller or
   execution-driver branches.

## Feedback contract

The correction SystemMessage contains one compact JSON diagnostic, for example:

```json
{
  "code": "invalid_native_tool_arguments",
  "issues": [
    {
      "tool": "web_fetch",
      "call_index": 0,
      "field": ["max_chars"],
      "type": "less_than_equal",
      "reason": "Input should be less than or equal to 200000"
    }
  ],
  "issues_truncated": false
}
```

`call_index` is zero-based; `field` preserves nested field/index paths. At most
eight issues and 4,000 serialized JSON characters are included. Tool names,
field paths, diagnostic types and reasons are also individually bounded. Omitted
issues set `issues_truncated=true`. No raw argument inputs, Pydantic contexts,
documentation URLs or stack traces are included. Custom ValueError/assertion
messages and equivalent validators without structured errors receive a generic
schema-mismatch reason. Pydantic v2 and legacy v1 structured errors are supported.

Missing fields, forbidden extras, strict type errors, numeric limits, enum/Literal
choices, and nested schemas all use the executable's existing validation rules.
No tool name or field name is hardcoded into classification or repair.

## Runtime behavior

- First invalid, then valid: two provider invocations, one final ordinary proposal,
  and normal Controller-authorized execution. Rejected calls create no execution
  records or request identities. The corrected proposal's identity is the same
  identity that a valid first response with those arguments would receive.
- Persistently invalid: two provider invocations maximum, then `INVALID_OUTPUT`
  with `invalid_native_tool_arguments` and the existing message. Controller's
  configured retry/failure policy is unchanged. Correction exceptions remain
  `PROVIDER_FAILURE` and receive no new provider retry.
- Valid first call: one provider invocation, unchanged schema and proposal.
- Multi-call: every member must pass before any proposal is executable. A valid
  member of an invalid response never executes early. The corrected response is
  checked as a whole against existing eligibility, limits and duplicate rules.
  Repairing arguments but retaining an oversized/disallowed batch fails after
  the same single correction; no separate third invocation is added.
- Duplicate protection: repaired arguments cannot bypass successful-history
  comparison. Request identity, successful history, checkpoint serialization and
  Controller/driver duplicate checks are unchanged.

## Lifecycle and malformed-output distinction

Invalid lifecycle arguments retain the established immediate strict rejection.
They are validated as part of `validate_native_call()` before executable-schema
classification, and the existing tests intentionally treat malformed lifecycle
claims/envelopes as non-correctable. This fix targets authorized executable
arguments; it does not broaden lifecycle recovery. A correction can select a
valid bound lifecycle action, but an invalid lifecycle action in either response
cannot produce completion, failure or replan evidence.

Malformed call envelopes, unknown/unauthorized tools, non-object arguments,
non-JSON values, invalid native parser channels and unavailable/dictionary-only
schemas retain their previous handling. Incidental text alongside a structurally
valid executable call does not hide a repairable schema error; its original
response remains intact in exchange logging.

## Observability and tests

The existing invocation, token-usage and exchange hooks run for both actual
provider calls. Debug compliance logging adds the bounded reason
`correction_reason=invalid_native_tool_arguments`. Console output remains concise
and never prints validation traces. No Planner, Finalizer, ExactCollection,
capability-card, graph, mutation, provenance, schema-limit or retry-count changes
are part of this fix.

`tests/test_brain_argument_correction.py` covers the reproduced web limit and
one actual mocked HTTP execution, all supported schema error classes, nested and
bounded diagnostics, persistent invalid output, valid/no-overhead behavior,
atomic batches, duplicate interaction, lifecycle strictness, invocation/usage
counts, real Ollama serialization, and checkpoint/replay. Existing cardinality,
incidental-text, oversized-batch, schema-ownership and read-file batch tests now
distinguish repairable schema errors from malformed candidates.

Deferred work includes recovering malformed native JSON/envelopes, lifecycle
argument correction, adding a dictionary-only JSON Schema validator, and
multi-stage repair for responses with several independent disqualifiers. The
current single correction budget is intentionally retained.

## Validation results

Validation on 2026-10-16, with mocked HTTP/model transports and no live calls:

| Check | Result |
| --- | --- |
| New argument-correction regressions | 38 passed |
| Focused provider/normalization/schema/driver/batch/duplicate/web/replay suite | 564 passed |
| Broader offline suite (`--ignore=tests/live`) | 1,506 passed, 6 pre-existing failures, 3 skipped |
| Core/tools coverage | 87.09%; required 71% passed |
| `git diff --check` | Passed |

The broader failures are the same six baseline failures previously reproduced
against HEAD: async factory compatibility, default planner observability,
failed-write artifact accounting, two terminal escaping expectations, and a
terminal-memory mock missing an accepted final answer. Their exact test names and
baseline reproduction are recorded in [web-fetch validation](web-fetch.md#offline-validation).
No new suite failures were introduced.

Focused command:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q -p no:cacheprovider --no-cov --tb=short tests/test_brain_argument_correction.py tests/test_brain_outcomes.py tests/test_brain_input_authority.py tests/test_brain_ollama_boundary.py tests/test_brain_cardinality_correction.py tests/test_brain_incidental_content.py tests/test_brain_oversized_correction.py tests/test_brain_batch_policy_ownership.py tests/test_brain_read_file_batch.py tests/test_brain_metadata_batch.py tests/test_brain_mixed_batch.py tests/test_duplicate_successful_call.py tests/test_execution_driver.py tests/test_web_ops.py tests/test_web_fetch_runtime.py tests/test_planner_runtime_flow.py
```

Broader command:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q -p no:cacheprovider --cov=core --cov=tools --cov-report=term --cov-fail-under=71 --tb=short --ignore=tests/live
```

Pytest ran outside the sandbox because sandboxed temporary directories were
inaccessible, as established during the preceding web-fetch implementation.
