# Bounded native-action correction at the Brain provider boundary

## Original argument-validation gap

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
   already-bound model. One shared budget permits at most two correction
   responses after the initial response: three provider invocations maximum.
   Every correction response re-enters the same missing-call, batch, duplicate
   and argument checks. It creates no ToolRequest or execution record and
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
- Persistently invalid: three provider invocations maximum, then `INVALID_OUTPUT`
  with `invalid_native_tool_arguments` and the existing message. Controller's
  configured retry/failure policy is unchanged. Correction exceptions remain
  `PROVIDER_FAILURE` and receive no new provider retry.
- Valid first call: one provider invocation, unchanged schema and proposal.
- Multi-call: every member must pass before any proposal is executable. A valid
  member of an invalid response never executes early. The corrected response is
  checked as a whole against existing eligibility, limits and duplicate rules.
  Repairing arguments but retaining an oversized/disallowed batch can consume
  the remaining correction opportunity; triggers never receive separate budgets.
- Duplicate protection: repaired arguments cannot bypass successful-history
  comparison. Request identity, successful history, checkpoint serialization and
  Controller/driver duplicate checks are unchanged.
- A duplicate followed by a textual pseudo-call is classified again as
  `missing_native_call`. The second correction must emit a real native action;
  text is never converted into a tool or lifecycle call. Repeated duplicate
  calls or textual responses exhaust the same two-correction budget and fail closed.

## Lifecycle and malformed-output distinction

Invalid lifecycle arguments retain the established immediate strict rejection.
They are validated as part of `validate_native_call()` before executable-schema
classification, and the existing tests intentionally treat malformed lifecycle
claims/envelopes as non-correctable. This fix targets authorized executable
arguments; it does not broaden lifecycle recovery. A correction can select a
valid bound lifecycle action, but an invalid lifecycle action in any response
cannot produce completion, failure or replan evidence.

Malformed call envelopes, unknown/unauthorized tools, non-object arguments,
non-JSON values, invalid native parser channels and unavailable/dictionary-only
schemas retain their previous handling. Incidental text alongside a structurally
valid executable call does not hide a repairable schema error; its original
response remains intact in exchange logging.

## Observability and tests

The invocation, token-usage and raw exchange hooks run for every actual provider
call, including both correction responses. Raw responses are logged before
incidental-content canonicalization. Debug compliance logging reports attempt
number, correction reason, whether a correction was triggered, budget exhaustion,
and final acceptance. Console output remains concise
and never prints validation traces. No Planner, Finalizer, ExactCollection,
capability-card, graph, mutation, provenance, schema-limit or Controller retry-count changes
are part of this fix.

`tests/test_brain_argument_correction.py` covers the reproduced web limit and
one actual mocked HTTP execution, all supported schema error classes, nested and
bounded diagnostics, persistent invalid output, valid/no-overhead behavior,
atomic batches, duplicate interaction, lifecycle strictness, invocation/usage
counts, real Ollama serialization, and checkpoint/replay. Existing cardinality,
incidental-text, oversized-batch, schema-ownership and read-file batch tests now
distinguish repairable schema errors from malformed candidates.

Deferred work includes recovering malformed native JSON/envelopes, lifecycle
argument correction, and adding a dictionary-only JSON Schema validator. Switching
between the existing correction classifications uses the same bounded loop;
malformed native envelopes and lifecycle arguments retain strict rejection.

## Validation results

Validation on 2026-10-18, with mocked HTTP/model transports and no live calls:

| Check | Result |
| --- | --- |
| New shared correction-loop regressions | 14 passed |
| Focused Brain provider/native/batch/argument/duplicate suite | 453 passed |
| Broader offline suite (`--ignore=tests/live`) | 1,710 passed, 6 pre-existing failures, 3 skipped |
| Core/tools coverage | 87.57%; required 71% passed |
| `git diff --check` | Passed |

The broader failures are the same six baseline failures previously reproduced
against HEAD: async factory compatibility, default planner observability,
failed-write artifact accounting, two terminal escaping expectations, and a
terminal-memory mock missing an accepted final answer. Their exact test names and
baseline reproduction are recorded in [web-fetch validation](web-fetch.md#offline-validation).
No new suite failures were introduced.

Focused command:

```powershell
$brainTests = @(rg --files tests -g 'test_brain*.py')
& .\.venv\Scripts\python.exe -m pytest -o addopts= -q -p no:cacheprovider --no-cov @brainTests tests/test_duplicate_successful_call.py --tb=short
```

Broader command:

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q -p no:cacheprovider --cov=core --cov=tools --cov-report=term --cov-fail-under=71 --tb=short --ignore=tests/live
```

Pytest ran outside the sandbox because sandboxed temporary directories were
inaccessible, as established during the preceding web-fetch implementation.
