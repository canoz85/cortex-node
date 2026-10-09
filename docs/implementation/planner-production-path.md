# Planner production path

The application supplies a user turn to `run_prompt`. The runtime starts an
execution through `CortexController.start_execution`, or passes the existing
paused execution back to the graph. Controller creates the sole authorized
`PlanningRequest`. PlannerService classifies an unclassified request, narrows its
capabilities, builds context, invokes its provider, validates strictly and returns
one `PlannerResult`. Controller accepts a plan, pauses, retries, or terminates.
Runtime dispatches only the resulting authorization.

`main.py` already met the application boundary and required no changes: it reads
input, displays output, manages session memory/history and holds the opaque pause
handle. It does not choose planning operations or interpret clarification answers.

The Router classifies execution mode: `conversation` requires no runtime tools,
`info` requires read-only runtime observation, `action` requests or requires a
state change, and `clarify` means intent is ambiguous. `info + NO_PLAN_REQUIRED`
contradicts that classification. Normalization rejects it as `INVALID_OUTPUT`,
using the existing Controller-owned retry budget; Controller also rejects a
direct `info` result that bypasses normalization. A preserved request route wins
over a worker-reported route.

Capability cards describe what a tool may establish; they do not observe current
repository, file, Git or other runtime state. A question about the declared
`read_file` contract (for example whether it returns `total_chars`) can be answered
from its card on the conversation path. A question about current file contents or
the current implementation requires runtime evidence. No request keywords choose
this validation rule; it consumes the existing classified route.

## Ownership and projections

| Representation | Authoritative owner | Necessary projection / retention |
| --- | --- | --- |
| `ExecutionState` | Controller | Runtime transports the immutable object; checkpoints serialize it. |
| `PlanningRequest` | Controller's `_build_planning_request` | Planner consumes this exact authorization. PlannerService may narrow its tool ceiling in a local validation view. |
| `user_request` | Controller-authorized request context | Initial input comes from the user turn. The provider receives the request once, in the human message. |
| `original_user_request` | Controller's protocol state | Retained from first authorization through termination, so clearing a pending request does not lose the request needed by workers or terminal memory evidence. |
| `clarification_question` | Durable pause prompt, projected by Controller | Planner needs the previous question to interpret answers such as a filename, name, or confirmation. It is not a second pause marker. |
| `clarification` | Controller's resumed request | Latest answer is bound to the paused request. Protocol retains it for downstream worker context after the planning request is consumed. |
| `planner_route` | Router classifies; Controller preserves | Router is called only when request route is absent. Pause and infrastructure retry retain the observed route. REVISE does not silently override a preserved route. |
| `capabilities` | Controller | PlannerService filters the ceiling; prompt and validation use the same filtered set. A recreated Controller cannot broaden a paused execution's saved ceiling. Accepted plan tools constrain execution. |
| `execution_id` / `run_id` | Protocol execution identity / runtime thread ID | `run_id` is the logging and checkpoint projection of the same identity. Pause handle derives it rather than storing another value. |
| `recent_history` | Application's bounded trusted conversation | One data-only input slice helps resolve conversational references. Controller snapshots it for planning. It excludes the current human request and worker/tool chatter. |
| Conversation messages | Application session | Graph messages are a transport projection. Checkpoint reentry replaces this projection rather than appending recreated history. |
| Planner memory | Application memory projected by Controller | Bounded background memory is included in the authorized planning snapshot. It is not injected by the graph after authorization, and never enters Brain/Finalizer context as Planner memory. |
| `AgentState` | Graph transport | Holds observations, messages and the authoritative execution reference. It has no independent plan, lifecycle counter, clarification marker or legacy Planner-domain metadata. |
| `PendingClarification` | Application-boundary reference | Contains only `ExecutionState`; displayed prompt and run ID are derived properties. |
| `PlanningClarification` | Controller | Contains a prompt and the saved authorized request, replacing copied operation, request text, revision, episode, route and constraint fields. |
| `ControllerDecision` | Controller | No production constructor exists elsewhere. Coordinator invokes and applies it once. |
| Protocol state writes | Controller module | Execution initialization, decision application, completion projection commits and terminal reporting writes live here. Runtime emits/assembles observations. |
| Graph checkpoint | Serialization | Does not choose authorization, reconstruct planning operations, or decide whether an answer is sufficient. |

## Clarification lifecycle

1. Planner returns `NEEDS_INPUT` with a concrete question.
2. Controller pauses and retains the authorized request with its classified route.
3. Runtime exposes the question and an opaque handle to Main.
4. Next user turn supplies an explicit, one-turn `user_input` observation. Runtime
   reuses the execution identity/state and replaces checkpoint conversation history.
5. Controller validates the saved request identity, sequence and accepted base,
   constructs the resumed request through its canonical builder, and clears the
   pause marker. Original request, capability ceiling, route and operation remain
   authoritative. REVISE also retains its failure context.
6. Planner interprets the original human request with the prior question and latest
   answer. Controller accepts the result or pauses again.

No conversation-message count decides whether a reply exists. No runtime layer
decides whether the answer resolves the missing information. Repeated NEEDS_INPUT
is permitted and does not become a new user authorization.

## Provider exchange

All four cases use system policy, optional retrieved data, a bounded
Controller-authorized context message, and one human message containing the
original authorized request.

| Case | Authorized context |
| --- | --- |
| Ordinary CREATE | Operation, necessary recent history and bounded memory. |
| Direct CREATE | Same contract; direct routes skip ambient retrieval. A semantic NO_PLAN_REQUIRED answer passes through Controller acceptance. |
| Clarification resume | Same original request; prior question and latest answer in context; preserved route bypasses Router. |
| REVISE | Accepted base plan, bounded execution-progress projection, trigger/reason, retry and failure facts, and suggested constraints. |

Capability names appear once in policy and are also the exact normalization
ceiling. The request is not repeated in context JSON. Retrieval is not repeated
there either. Base identity is available in the base plan without parallel
`base_plan_id` / `base_revision` prompt keys. Raw execution history is represented
by bounded progress, without a bookkeeping block explaining omitted raw evidence.

The proposal remains `result`, `objective`, `steps`, `message`, with semantic
outcomes PLAN_PROPOSED, NO_PLAN_REQUIRED, NEEDS_INPUT and PLANNING_FAILED.
INVALID_OUTPUT and PROVIDER_FAILURE remain infrastructure outcomes. Pydantic
validation, forbidden extras, json_schema output and ProposedStep semantics remain.
Neither provider nor Router reparses raw output after structured parsing fails.

## Removed paths

- Unused `PlannerInput`, `PlanningPauseReason`, `RoutingDecision`, Planner-result
  rationale/change-summary fields, and redundant graph plan/counter/tool-result flags.
- Unused WorkingState retrieval/metadata mirrors and the legacy converter module.
- The Planner bridge reader and its duplicate authorization check. The graph now
  reads one request through the fail-closed authorization boundary.
- Graph-side Planner memory injection and the ignored Planner-node `tools_set` argument.
- Legacy graph-to-protocol state constructors, identity/cursor overrides, proposal
  to accepted-plan inference, reverse converters, Brain result reparsing, unused
  Tool-input conversion and its fallback, and `with_cursor`.
- Router wrapper layers, optional error-propagation mode, raw JSON/enum recovery,
  and silent fallback routing; duplicate provider TypeError retry and raw JSON recovery.
- PlannerService's return-value forwarding wrapper and REVISE route rewriting.
- Duplicate request/retrieval/base-identity prompt fields and raw-evidence metadata.
- Unused Controller pending-step selector, debug prints and migration comments.
- Duplicate `test_graph_planner.py`: all 35 test functions were AST-identical to
  functions in `test_planner.py`, with no unique test names. Legacy bridge tests
  asserting reconstruction of authoritative protocol state were removed.

## Intentionally retained

Router is one classifier and supplies the execution-mode fact used for read-only
filtering and retrieval eligibility. PlannerService owns these projections and
provider interaction. The graph worker ports enforce runtime authorization at the
framework boundary; they do not construct planning requests.

Original request and latest clarification remain in protocol state because workers
and terminal memory still need them after the pending planning request is cleared.
Planner's question/answer context remains because the answer alone can be ambiguous.
Direct semantic content remains distinct from a generic Planner status message, so
only Controller-accepted answer content reaches finalization.

Completion preflight still previews a revision to bind completion requirements;
Controller performs final reconciliation/acceptance. This uses the same pure
reconciliation function, and the preflight is not an accepted-plan writer. Changing
the completion-validation workflow is outside this Planner cleanup. Runtime may
write WorkingState observations; only Controller functions write protocol state.

## Validation and limitations

Focused contract/lifecycle/graph tests and portable completion/terminal tests are
run separately from deterministic provider exchanges. The production-flow tests
exercise real node composition, provider adapters, Controller and checkpoints with
scripted structured model responses; they do not claim live model adherence.

The broader offline run excludes live model tests and `test_filesystem_completion.py`,
which already fails collection by importing the removed `_build_tool_result` helper.
The remaining unrelated failures are compared with an untouched HEAD checkout,
not attributed to this refactor without evidence. Results are reported in the
task delivery. The provided live gpt-oss clarification test was updated for the
new pause-marker representation but was not run.

Focused validation: 236 contract/lifecycle/graph tests passed, plus 28 portable
completion/terminal/async tests passed. The broader offline suite reported 826
passed and 20 failed; all 20 remaining failure identities reproduced in the
untouched HEAD comparison. `git diff --check` passed.

Test classifications:

- `test_planner_production_flow.py`: production graph integration with scripted
  structured model exchanges, including checkpoint resume and history replacement.
- `test_planner.py`, contract/routing/debug/memory/progress tests: Planner service
  and provider-boundary tests; no claim of live model judgment.
- Lifecycle/request/revision/controller tests: protocol decision and snapshot tests.
- Runner injected-event tests: presentation and transport tests, renamed to make
  their injected results explicit.
- Brain graph adapter tests: graph execution with an injected Planner result,
  renamed to distinguish this from testing Planner generation.

Older migration-format serialized checkpoints are intentionally unsupported. No
compatibility parser or schema-repair path was added. The paused-execution handle
continues to be held by the current application session; no new persistence feature
was introduced.

## Changed files

Runtime and protocol changes:

```text
core/graph_authorization.py
core/graph_brain.py
core/graph_controller.py
core/graph_node_helpers.py
core/graph_nodes.py
core/graph_planner.py
core/graph_runner.py
core/planner.py
core/planner_normalization.py
core/planner_provider.py
core/planner_routing.py
core/protocol/__init__.py
core/protocol/bridge.py
core/protocol/controller.py
core/protocol/converters.py (deleted)
core/protocol/enums.py
core/protocol/models.py
core/runtime/controller_transition.py
core/runtime/portable_orchestration.py
core/state.py
```

Tests and documentation changes (some tests only update the moved writer import
or the consolidated route type):

```text
docs/implementation/planner-production-path.md (new)
docs/protocol/CEP-005-protocol-data-contracts.md
tests/test_completion_provenance.py
tests/test_controller_iteration.py
tests/test_controller_transition.py
tests/test_direct_semantic_handoff.py
tests/test_execution_driver.py
tests/test_gpu_resources.py
tests/test_graph_brain_adapter.py
tests/test_graph_planner.py (deleted duplicate)
tests/test_graph_planning_p2.py
tests/test_graph_runner.py
tests/test_graph_state_machine_apply.py
tests/test_planner.py
tests/test_planner_contract.py
tests/test_planner_debug.py
tests/test_planner_lifecycle.py
tests/test_planner_live_memory_adherence.py
tests/test_planner_memory.py
tests/test_planner_production_flow.py (new)
tests/test_planner_revision.py
tests/test_planner_routing.py
tests/test_planning_requests.py
tests/test_protocol_bridge.py
tests/live/test_clarification_lifecycle.py (existing local live experiment updated)
```
