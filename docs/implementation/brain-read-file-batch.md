# Sequential independent Brain actions

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
An executable group contains independent, well-formed, authorized calls with
empty canonical response content and no invalid native calls. ToolDefinition owns
max_batch_calls: the default is 1, while list_files, read_file and find_files explicitly allow
multiple independent synchronous calls. Tools may be mixed, with at most 24 calls
overall and each tool's own limit applied to its member count. Non-mutating metadata alone never
enables batching. Mutating tools, lifecycle actions and configured async submission
tools remain excluded.

core/brain_batch_policy.py is the shared generic validator for
authorization, metadata eligibility and limits, executable argument schemas, and
duplicate effective invocations. Registry-owned get_tool_definition provides
metadata. Provider and graph composition reuse existing executable registries to
supply an argument_schema_for(name) callable; Controller receives only that narrow
lookup, not a ToolRegistry or executable tools. Normalization receives the same
lookup used by Provider.

The production read_file definition declares ReadFileRequest, which its executable
decorator consumes. find_files keeps its existing inferred executable schema.
Standalone validation can use declared schemas. An explicit schema lookup returning
None fails closed without declaration fallback. Provider retains existing bound
schema and declared-schema checks for native correction candidates.

Controller independently validates the complete proposal against the accepted
plan's capability ceiling and active execution/plan revision/step attempt before
authorizing any member. Request IDs must be distinct and not already recorded.
Its configured async-submission guard remains because batch dispatch bypasses
singleton job-identity, active-job and submission-attempt preparation. This checks
execution-mode compatibility rather than defining static batch eligibility.

Duplicate effective calls are rejected after schema coercion and defaults, using
tool name and canonical JSON of all effective arguments. No path field is assumed, and no lexical
path or filesystem alias normalization occurs. Original request arguments and
deterministic request ID generation remain unchanged. Oversized groups are rejected
whole, without filtering, truncation or first-call selection.

Valid disallowed groups use the existing one-shot choose-one-action correction.
For an otherwise valid eligible batch exceeding the overall or a tool's limit, shared policy
checks all non-size constraints before Provider requests at most 24
of the original useful calls, respecting each tool's metadata limit. Other disallowed groups
retain the singleton correction. Correction output passes normal validation and
can never cause a third provider call.

Incidental content beside valid singleton or multi-call candidates is discarded
only from a copied canonical response, after original raw logging and before batch
classification. Invalid candidates remain strict, and canonicalization never
grants batch eligibility. Provider-level raw logging is unchanged.

Provider generates batch guidance from the currently authorized, bound tools with
usable executable schemas. It lists only eligible names and their actual metadata
limits; all other tools and lifecycle actions require one call. The static Brain
contract no longer implies batching support for arbitrary read-only tools.

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

The current opted-in tools each have a maximum of **24** and the Brain window displays **32** current
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

Duplicate detection compares schema-normalized arguments; it does not canonicalize
path spellings or query the filesystem to discover aliases.

Planner behavior/schemas/retries, memory, mutation authorization, exact_collection,
ToolRegistry architecture, and tool inventory are unchanged. No stat_files,
read_files, metadata tools, mutating batches or parallel execution.

## Initial batch implementation files

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

## Initial batch implementation validation results

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

## Metadata cleanup files

ToolDefinition owns batch limits in tools/registry.py. Shared validation stays in
core/brain_batch_policy.py. core/brain.py removes the broad static allowance;
core/brain_provider.py renders bound-tool guidance and metadata-driven correction.
core/brain_normalization.py and core/protocol/controller.py receive narrow schema
lookups, wired through core/graph_controller.py and core/graph_nodes.py.
No plan, checkpoint, evidence or tool implementation schema changes are needed.
