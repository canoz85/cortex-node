# Sequential homogeneous Brain read_file actions

The pre-implementation trace found the singular proposal in
`BrainOutcome.tool_request`. ControllerInput, ControllerDecision and
ProtocolVisibleState each have one `pending_tool_request`; that slot remains
singular. ToolInput also continues to carry one ordinary request.

Production dispatch is `create_controller_node` → PortableExecutionRuntime →
ExecutionDriver → GraphWorkerRuntimePorts → SerializedToolRuntimePort. Injected
ports can instead invoke the wrapped ToolNode and capture adapter; deferred graph
nodes use the same Controller decisions. StateGraph's configured checkpointer
serializes the typed ExecutionState rather than a separate batch transport list.

The singular result projections are WorkingState/BrainInput.last_tool_result,
bridge construction, ExecutionDriver Brain dispatch, graph_brain consumption,
record-integration repeat-failure accounting and observability. Brain's evidence
messages and Controller completion provenance use tool_execution_history, so batch
evidence is not reduced to the latest result. Neither singular result projection
nor record/provenance schemas need a batch variant.

One Brain action proposes either `tool_request` or `tool_requests` (an ordered
tuple of ordinary requests, length at least two). The payloads are exclusive.
No batch ID, lifecycle outcome, dispatch kind, result type or record type is added.
Each member uses the existing deterministic ToolRequest ID generation.

## Acceptance

The provider distinguishes valid correction candidates from executable groups.
An executable group must contain 2–24 known, well-formed, authorized calls to
`read_file` only, with empty canonical response content and no invalid native calls.
Every call must validate against the bound tool schema and ReadFileRequest.
The explicit eligibility policy also requires the production manifest entry to
be non-mutating; non-mutating metadata alone never enables batching.
ReadFileRequest's literal path, offset and limit inputs have no result references,
so each invocation can execute independently. Lifecycle and async tools are excluded.

Batch policy owns only the immutable `_BATCH_ELIGIBLE_TOOLS` name set, the existing
maximum, and batch-specific normalization and duplicate checks. Registry-owned
`get_tool_definition` and `get_tool_argument_schema` provide metadata and schemas.
The production `read_file` definition owns its schema association; the executable
decorator consumes that declaration. Bound executable schemas are used when a
registry is supplied, without falling back for a missing executable or schema.
Provider candidate validation consumes schemas without selecting eligible tool
names; shared batch policy separately decides execution permission.

Controller independently validates the complete proposal against the accepted
plan's capability ceiling and active execution/plan revision/step attempt before
authorizing any member. Request IDs must be distinct and not already recorded.
Its configured async-submission guard is retained because batch dispatch bypasses
singleton job-identity, active-job and submission-attempt preparation. This checks
execution-mode compatibility, rather than defining static batch eligibility.
Duplicate effective calls are rejected after argument coercion, defaults and
lexical path normalization (`normpath`/`normcase`, without filesystem lookup).
Oversized groups are rejected whole; no filtering, truncation or first-call selection.

Valid disallowed groups use the existing one-shot “choose one action” correction.
The size-only exception is an otherwise valid homogeneous read-only group above
`MAX_READ_FILE_BATCH`: shared policy checks every member against all non-size
constraints before the provider requests one native action containing at most
that maximum of the original useful calls, using the same tool. Other disallowed
groups retain the choose-one-call wording. The corrected proposal passes normal
validation, and the runtime never chunks, truncates or selects batch members.
Malformed, unknown, unauthorized or invalid-argument groups remain invalid output.
Correction output passes validation and can never cause a third provider call.
Incidental content beside valid singleton or multi-call candidates is discarded
only from a copied canonical response, after original raw logging and before batch
classification. Invalid candidates remain strict, and canonicalization never grants
batch eligibility. Provider-level raw logging is unchanged. The native-action
contract exposes the existing maximum directly from MAX_READ_FILE_BATCH.

## Controller state and execution

`ProtocolVisibleState.tool_request_continuation` is an immutable
`ToolRequestContinuation` containing execution ID, plan ID/revision, step ID/attempt
and `remaining: tuple[ToolRequest, ...]`. It is Controller-owned accepted state,
projected through ControllerInput and written only by applied Controller decisions.
An empty remainder scopes the final pending member until its result is consumed.

Acceptance authorizes only the first member in `pending_tool_request` and stores
the remainder. Each result returns to Controller, which first applies the existing
tool failure/async/lifecycle policy. Only ordinary continuation that would otherwise
dispatch Brain can dispatch the next queued member. Member order is deterministic
execution order, not semantic priority. Physical calls execute sequentially.
Controller iteration counts still advance only on Brain dispatches.

Every authorization uses the existing DISPATCH_TOOL_RUNTIME decision, marks a
checkpoint boundary, and replaces the single pending request while consuming the
previous result. The runtime holds no separate authority-bearing batch list.
Both portable/injected and deferred graph transports expose only that authorized
member to ToolNode; batch capture uses the same ordinary record integration.

Typed tool failures remain ordinary failure records. They do not synthesize
STEP_FAILED or retry the member. Later successes remain separate records and
cannot erase failures. Repeated-failure replanning remains authoritative.
Replan, pause, cancellation, termination, invalid scope or result identity rejection
discard the remainder and the batch pending request. Runtime exceptions and
authorization invariants propagate and stop dispatch according to existing behavior;
they do not run another member or silently retry.

## Evidence and recovery

The maximum is **24** and the Brain window displays **32** current
records. A complete fresh batch fits, leaving eight slots for earlier attempts.
With ten prior attempts, two older records leave the displayed window; all 24 batch
records remain ordered and visible. Repeated batches can evict more older evidence.
Strings remain bounded to 10,000 characters and lists to 100 items. The display
bound is shared through MAX_CURRENT_ATTEMPT_RECORDS. Full completion provenance uses all eligible records and request IDs,
including failures, independent of the display limit. exact_collection still binds
one structured collection from one eligible original record.
Scoped current attempts use the same accepted execution/plan/revision filter as
Controller provenance before display indexes are assigned. Prior facts/failures,
and historical unscoped display-only evidence, retain their existing projections.

Existing typed ExecutionState/ControllerDecision checkpoint serialization includes
the continuation automatically. Resume with a captured result consumes it and
authorizes the next member; resume with an authorized member and no captured result
dispatches that same request ID. Recorded members are never skipped or reissued.
Tests cover A captured before B authorization, B authorized, B captured, typed JSON/
LangGraph serialization, and reconstructed portable/deferred graphs sharing checkpoints.

The existing crash gap between an external invocation and durable result capture
is unchanged: a pending authorization alone cannot prove whether that invocation
already happened. Read-only execution can be resumed, but this change does not
provide exactly-once external invocation across that gap. No retry machinery is added.

Duplicate detection is lexical and argument-based; it does not query the filesystem
to discover aliases such as symlinks to the same physical file.

Planner behavior/schemas/retries, memory, mutation authorization, exact_collection,
ToolRegistry architecture, and tool inventory are unchanged. No stat_files,
read_files, metadata tools, heterogeneous/mutating batches or parallel execution.

## Changed production files

- `core/brain_batch_policy.py`: explicit eligibility, bound and effective-call checks.
- `core/brain.py`: narrow native-action contract and scoped current evidence indexes.
- `core/brain_normalization.py`, `core/brain_provider.py`: proposal construction,
  separate eligibility/correction classification and bound-schema validation.
- `core/protocol/models.py`, `core/protocol/__init__.py`: exclusive proposals and
  scoped ordinary-request continuation contract.
- `core/protocol/controller.py`: complete acceptance, policy-gated continuation,
  recovery and accepted-state queue updates.
- `core/protocol/bridge.py`, `core/runtime/controller_transition.py`: authoritative
  continuation projection and stale/untracked-state rejection.
- `core/graph_authorization.py`, `core/graph_controller.py`,
  `core/graph_worker_runtime.py`: current-member transport for injected/deferred paths.
- `core/graph_capture.py`: result identity checks and shared batch record capture.
- `core/graph_brain.py`: explanatory transport comment; batch proposals carry no
  executable multi-call AIMessage.

Documentation also updates the production contract exports. The new focused batch
test module covers acceptance, ordering, evidence, interruptions and graph recovery;
existing correction tests now use valid-but-disallowed duplicate read candidates,
and prompt assertions in three existing modules use the revised native-action wording.

## Validation results

- 51 focused batch cases passed in `tests/test_brain_read_file_batch.py`.
- 389 related Brain, Controller, driver, graph, async-policy and provenance cases passed.
- Non-live suite: 944 passed, 2 skipped, 6 failed. Coverage passed at 85.89%
  against the repository's 71% requirement. `git diff --check` passed.

The six broader-suite failures remain outside the batch cases:

- `test_async_poller.py::test_build_app_runs_submission_await_poll_capture_and_resume_end_to_end`:
  injected factory returns five nodes while build_app unpacks four.
- `test_default_observability.py::test_default_renders_accepted_plan_and_coalesced_brain_tool_call`:
  expected planner rendering is absent.
- `test_direct_tool_runtime.py::test_direct_integration_preserves_artifacts_and_repeat_failure_accounting`:
  expected inferred artifact is absent from explicit result artifacts.
- `test_live_status.py::test_permanent_output_clears_then_redraws` and
  `test_stop_clears_status_and_stops_redraw`: console clear/redraw expectations differ.
- `test_terminal_memory.py::test_updater_failure_preserves_memory_and_successful_turn`:
  mocked turn supplies no accepted final answer.

These failures were isolated and reproduced. Their fixture, artifact, console and
memory behavior was left unchanged to preserve the requested implementation scope.
