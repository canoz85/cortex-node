# CortexNode Architecture Migration Roadmap

- Document type: Architecture migration roadmap
- Scope: Current runtime reconciliation and staged migration
- Overall status: `NOT_STARTED`
- Baseline: Accepted current-runtime architecture reconciliation
- Migration policy: Keep the runtime working after every stage and avoid a single-pass rewrite

## Authority

This document defines the approved migration direction for CortexNode.

It is not a protocol specification.

CEP documents define protocol semantics.
CIS documents define implementation mappings.
This roadmap defines the ordered migration from the current implementation
toward the approved target architecture.

## Documentation Synchronization Policy

Architecture migration may require changes to CEP, CIS, README,
architecture documentation, and tests.

Documentation MUST NOT be updated merely to describe temporary
intermediate migration states.

During Stages 1–6:

- implementation changes may identify documentation drift;
- discovered drift must be recorded for later reconciliation;
- CEP documents must not be changed to legitimize temporary behavior;
- CIS documents must not be treated as authoritative when they conflict
  with an approved architectural decision;
- temporary migration topology must not be promoted into protocol semantics.

Stage 7 is the authoritative documentation reconciliation stage.

At Stage 7:

- CEP documents are updated to describe approved framework-neutral
  protocol semantics;
- CIS documents are updated to describe the actual implementation mapping;
- README and architecture documentation are updated to describe the
  resulting current architecture;
- obsolete documentation is removed or explicitly marked historical;
- CURRENT and FUTURE requirements are clearly distinguished.

A change to LangGraph topology alone MUST NOT require a CEP change unless
the underlying protocol semantics also changed.

## 1. Purpose

This document defines the ordered migration from the reconciled current CortexNode runtime to a portable, controller-governed execution architecture.

The roadmap is deliberately incremental. Each stage has its own acceptance criteria, dependencies, rollback boundary, and commit boundary. No stage may rely on a later stage to restore basic runtime correctness.

LangGraph is an orchestration adapter and implementation detail. It is not part of CortexNode protocol semantics. The execution lifecycle, worker contracts, Controller decisions, and persistent protocol state must remain valid if LangGraph is replaced.

Controller is the semantic transition authority. This does not require Controller to be a specific LangGraph node, the LangGraph entrypoint, or even to be hosted by a graph-based runtime. A replaceable runtime driver invokes Controller and dispatches the semantic commands that Controller authorizes.

## 2. Status Vocabulary

Every migration stage uses one of these statuses:

| Status | Meaning |
| --- | --- |
| `NOT_STARTED` | No implementation work for the stage has been accepted. |
| `IN_PROGRESS` | Work has begun, but the stage acceptance criteria are not all satisfied. |
| `COMPLETED` | All stage acceptance criteria are satisfied and the stage is an accepted migration baseline. |
| `DEFERRED` | The stage has been explicitly postponed; the reason and prerequisites must be recorded. |

All stages are initially `NOT_STARTED`.

## 3. Architectural Invariants

These invariants govern every migration stage, including temporary compatibility work.

### 3.1 Portability Boundary

1. LangGraph is an orchestration adapter and implementation detail, not part of protocol semantics.
2. Protocol and domain models must not import LangGraph types.
3. `CortexController` must not depend on `GraphState`, `Command`, `Send`, `END`, graph node names, graph edges, or LangGraph routing semantics.
4. Planner, Brain, Finalizer, and Tool contracts must be usable without LangGraph.
5. Worker results and Controller decisions must be framework-neutral typed contracts.
6. LangGraph nodes must remain thin adapters between `GraphState` and protocol/domain or service types.
7. Graph topology must not define protocol semantics.
8. Execution lifecycle semantics must remain valid if LangGraph is replaced.
9. Async scheduling must depend on protocol commands and events, not graph node identities.
10. Controller is the semantic transition authority, not necessarily a LangGraph node or graph entrypoint.
11. Protocol enums and persistent state must not contain graph node names, graph routing targets, or graph resume details.

### 3.2 Execution Ownership

1. Controller alone accepts lifecycle facts and authorizes semantic transitions.
2. Workers produce typed results, facts, or requests; they do not advance protocol-visible state.
3. A runtime driver invokes Controller and dispatches authorized work.
4. Orchestration adapters may translate semantic commands into framework-specific routes, but those translations have no protocol authority.
5. Finalization, execution termination, and rolling memory are separate responsibilities.

### 3.3 Migration Safety

1. Runtime correctness precedes structural cleanup.
2. Existing topology remains in place until the contracts beneath it are stable.
3. Final-answer ownership moves only after Finalizer is covered by tests.
4. Legacy code is removed only after replacement behavior is active and verified.
5. Async ownership changes only after Controller authority is made framework-neutral.
6. Every stage must be independently testable and reversible at its declared rollback boundary.

## 4. Required Dependency Direction

The intended dependency direction is inward toward protocol/domain contracts:

```text
LangGraph adapters ---------+
Async/runtime adapters -----+---> Application services ---> Protocol/domain
Provider/tool adapters -----+              |
                                           +---> Framework-neutral ports
```

Layer responsibilities:

| Layer | Responsibilities | May depend on | Must not depend on |
| --- | --- | --- | --- |
| Protocol/domain | Lifecycle vocabulary, execution state, worker results, Controller inputs and decisions, invariants | Standard language/runtime libraries and framework-neutral value types | LangGraph, LangChain/provider messages, graph topology, runtime adapters |
| Application services | Controller, Planner, Brain, Finalizer, summary builder, memory updater, and service ports | Protocol/domain and framework-neutral ports | `GraphState`, node names, edges, `Command`, `Send`, `END` |
| Runtime | Execution driver, command dispatch, scheduling, correlation, service registry, persistence-port coordination | Protocol/domain and application service interfaces | Hard-coded graph node identities as semantic commands |
| LangGraph adapters | `GraphState` conversion, node wrappers, route mapping, graph construction, graph-specific checkpoint/resume integration | Runtime, services, protocol/domain, and LangGraph | Defining or changing protocol semantics |
| Provider/tool adapters | Model/tool/provider-specific conversion and I/O | Service ports and protocol/domain request/result contracts | Exporting provider-specific payloads as protocol contracts |

No inward layer may import an outward adapter layer. In particular, protocol/domain and application services must remain executable under a minimal non-LangGraph runtime driver.

## 5. CURRENT and FUTURE Requirements

### 5.1 CURRENT Migration Requirements

The following requirements apply throughout this migration and are not deferred:

- Correct direct-response classification.
- Consistent terminal `ExecutionStatus` handling.
- Explicit step-failure and retry semantics.
- Framework-neutral typed Brain outcomes.
- Separation of step completion, execution termination, final-answer generation, `ExecutionSummary` generation, and rolling memory.
- Controller as the single semantic transition authority.
- Worker contracts that do not depend on LangGraph.
- Thin LangGraph adapters that only translate state and routing.
- Async scheduling through semantic wake events and Controller authorization.
- CEP/CIS documentation that distinguishes protocol semantics from the current orchestration implementation.

These are migration requirements. They must not be interpreted as claims that the pre-migration runtime already satisfies them.

### 5.2 FUTURE or Separately Authorized Requirements

The following are not claimed as current runtime guarantees unless Stage 8 separately justifies and implements them:

- Durable append-only accepted event history.
- Deterministic replay from accepted protocol events.
- Crash-consistent checkpoint commit guarantees.
- Durable recovery across process or host failure.
- Distributed execution consistency.
- Audit-grade event retention and protocol-version migration.

LangGraph checkpointing or graph traversal history must not be treated as automatically satisfying these requirements. If these capabilities are approved, they must be defined through framework-neutral protocol/runtime ports first.

## 6. Migration Stage Summary

| Stage | Name | Status | Independent commit boundary |
| --- | --- | --- | --- |
| 1 | Runtime correctness | `NOT_STARTED` | Yes |
| 2 | Typed Brain outcomes | `NOT_STARTED` | Yes, after Stage 1 |
| 3 | Finalization separation | `NOT_STARTED` | Yes, as 3A, 3B, and 3C |
| 4 | Remove legacy completion code | `NOT_STARTED` | Yes |
| 5 | Controller authority cleanup | `NOT_STARTED` | Yes, as 5A and 5B |
| 6 | Async ownership | `NOT_STARTED` | Yes, after Stage 5 |
| 7 | Documentation reconciliation | `NOT_STARTED` | Yes, documentation-only |
| 8 | Compliance hardening decision | `NOT_STARTED` | Decision only; implementation requires a separate roadmap |

## 7. Stage 1 - Runtime Correctness

- Status: `NOT_STARTED`
- Portability assessment: Preserves the portability constraint when lifecycle rules are implemented in Controller and protocol contracts rather than graph routes.

### Goal

Correct direct-response classification, terminal status handling, and explicit step-failure transitions without changing graph topology.

### LangGraph Coupling Risk

Implementing these corrections only in `graph_controller.py` or routing functions would allow graph topology to define protocol behavior.

### Required Approach

- Implement lifecycle rules in framework-neutral Controller inputs and decisions.
- Direct-response context must produce a semantic `FINAL_ANSWER_READY`, or equivalent typed worker outcome, rather than being inferred from the absence of a plan inside a graph route.
- `ControllerDecision` must explicitly carry the resulting execution status where applicable:
  - `COMPLETED`
  - `FAILED`
  - `CANCELLED`
  - `NON_TERMINAL`
- `STEP_FAILED` must define:
  - active-step validation;
  - attempt and retry handling;
  - terminal failure when retries are exhausted;
  - explicit replan behavior where permitted.
- LangGraph adapters translate decisions into the existing routes. Graph topology does not change in this stage.

### Intended Dependency Direction

`LangGraph route adapter -> CortexController -> protocol/domain transition rules`

The adapter may observe a Controller decision, but it must not invent status, retry, or termination semantics.

### Files Likely Affected

- `core/protocol/controller.py`
- `core/protocol/models.py`
- `core/protocol/enums.py`
- `core/graph_controller.py`
- `core/graph_state_machine.py`
- `core/graph_brain.py`
- Relevant Controller, Brain, and routing tests

### Protocol Contracts Affected

- Controller input and worker-result classification
- Execution status
- Failure reason
- Retry metadata
- Terminal decision semantics

### Tests Required

- Controller-only tests with no LangGraph imports.
- Direct response is not classified as step failure.
- Successful, failed, and cancelled termination set the correct status.
- Failed active step is marked failed.
- Retry-allowed and retry-exhausted behavior is explicit.
- Invalid or stale step identifiers are rejected.
- Thin adapter tests confirm that existing topology maps semantic decisions correctly.

### Migration Risk

Medium. The changes affect termination and failure behavior, but topology remains stable.

### Dependencies

None.

### Rollback Boundary

Revert Stage 1 protocol models, Controller rules, and adapter mapping together. Do not retain a partial state in which an adapter expects status fields the Controller does not produce.

### Acceptance Criteria

- Lifecycle tests run without constructing `GraphState`.
- Terminal cursor and `ExecutionStatus` cannot disagree.
- Direct responses cannot enter step-failure handling.
- Failed steps follow an explicit retry, replan, or terminal-failure path.
- Graph behavior remains topologically unchanged.
- No protocol enum contains a graph node name.

### Commit Independence

Stage 1 can and should be committed independently.

## 8. Stage 2 - Typed Brain Outcomes

- Status: `NOT_STARTED`
- Portability assessment: Preserves the portability constraint when model/provider outputs are normalized before crossing the Brain service boundary.

### Goal

Replace textual prefix parsing with a framework-neutral typed Brain contract while preserving current behavior. Final-answer generation remains in Brain temporarily during this stage.

### LangGraph Coupling Risk

Using LangChain tool-call objects, LangGraph messages, or `GraphState` as the Brain contract would leak orchestration/provider types into protocol behavior.

### Required Approach

Introduce framework-neutral types such as:

- `BrainInput`
- `BrainOutcome`
- `BrainOutcomeKind`
- `ToolRequest`
- `ReplanRequest`
- `StepCompletionEvidence`
- `FinalAnswerDraft`, or an equivalent temporary final-answer result

Potential outcome variants include:

- `TOOL_REQUESTED`
- `STEP_COMPLETED`
- `STEP_FAILED`
- `REPLAN_REQUESTED`
- `FINAL_ANSWER_READY`
- An explicit invalid-output or provider-failure outcome

LangChain/Ollama tool calls must be normalized inside the provider or Brain implementation before returning `BrainOutcome`. Provider-specific tool calls are not protocol transport types.

Text such as `STEP COMPLETED` may remain user-visible during compatibility work, but it must have no control-flow authority after this stage.

### Intended Dependency Direction

`LangGraph Brain node -> Brain service -> provider adapter`

`Brain service -> protocol/domain BrainInput and BrainOutcome`

Provider output travels back through the provider adapter and is normalized before the Brain service returns a protocol/domain result.

### Files Likely Affected

- `core/protocol/models.py`
- `core/protocol/enums.py`
- `core/protocol/controller.py`
- `core/protocol/bridge.py`
- `core/graph_brain.py`
- `core/graph_constants.py`
- A new framework-neutral Brain service/interface module if separation cannot be achieved cleanly in place
- Brain and Controller tests

### Protocol Contracts Affected

- Brain input and outcome
- Tool request normalization
- Step-completion evidence
- Replan requests
- Temporary final-answer outcome

### Tests Required

- Brain contract tests without LangGraph or `GraphState`.
- Provider-output normalization tests.
- Controller behavior for every Brain outcome.
- Free-form text beginning with or omitting `STEP COMPLETED` has no control-flow effect.
- Invalid structured output follows an explicit error path.
- LangGraph adapter tests verify only state translation and service invocation.

### Migration Risk

Medium-high. This changes the worker-to-Controller contract while preserving behavior.

### Dependencies

Stage 1.

### Rollback Boundary

Keep the old parser behind a temporary compatibility adapter until typed-outcome equivalence tests pass. Remove the fallback only when Stage 2 is accepted.

### Acceptance Criteria

- No runtime decision parses an answer prefix.
- `BrainOutcome` imports no LangGraph or provider-specific message types.
- Tool calls cross the service boundary only as domain `ToolRequest` values.
- Existing functional behavior is preserved.
- Brain contracts can be invoked by a non-LangGraph test driver.

### Commit Independence

Stage 2 can be committed independently after Stage 1.

## 9. Stage 3 - Finalization Separation

- Status: `NOT_STARTED`
- Portability assessment: Preserves the portability constraint when Finalizer is an application service rather than a graph-defined terminal behavior.

### Goal

Separate:

- step completion;
- execution termination;
- final-answer generation;
- `ExecutionSummary` generation;
- rolling memory.

Introduce a Finalizer abstraction and remove final-answer responsibility from Brain only after Finalizer is covered by tests.

### LangGraph Coupling Risk

Defining Finalizer as a required terminal graph node, or equating finalization with `END`, would make its lifecycle role dependent on LangGraph.

### Required Approach

Define Finalizer as an application service with framework-neutral contracts:

- `FinalizationRequest`
- `FinalizationResult`
- `ExecutionSummaryBuilder`
- `FinalAnswerRenderer`
- Optional `MemoryUpdater`, invoked separately

The semantic flow is:

1. Controller determines the execution outcome.
2. Controller issues a semantic `FINALIZE_EXECUTION` command.
3. The runtime invokes Finalizer.
4. Finalizer builds `ExecutionSummary` and the user-facing answer.
5. Rolling memory is updated separately and cannot change the execution outcome.

A LangGraph adapter may represent finalization as a node, but a node is not required and is not protocol-visible.

Migration slices:

- Stage 3A: Add Finalizer contracts and deterministic summary tests without routing live execution through it.
- Stage 3B: Shadow-run Finalizer and compare its results with current Brain-generated answers.
- Stage 3C: Switch final-answer ownership to Finalizer, then remove final-answer generation from Brain.

### Intended Dependency Direction

`Runtime driver -> Finalizer service -> protocol/domain finalization contracts`

`LangGraph finalizer adapter -> Runtime/Finalizer service`

`MemoryUpdater -> memory persistence port`

Finalizer and MemoryUpdater must not import graph types or infer outcomes from graph termination.

### Files Likely Affected

- `core/protocol/models.py`
- `core/protocol/controller.py`
- `core/graph_brain.py`
- `core/graph_summarize.py`
- `core/graph.py`
- New framework-neutral finalization service modules
- Thin LangGraph finalization adapters
- Finalizer, summary, memory, Controller, and integration tests

### Protocol Contracts Affected

- Semantic `FINALIZE_EXECUTION` command
- Execution terminal reason
- `ExecutionSummary`
- Finalization request and result
- Answer-delivery metadata

Rolling-memory structures remain outside the protocol outcome unless a separate protocol requirement is explicitly approved.

### Tests Required

- Finalizer tests without LangGraph.
- Deterministic summary construction.
- Success, failure, cancellation, direct-response, and partial-execution answers.
- Finalizer failure does not rewrite the execution outcome.
- Memory-update failure does not rewrite execution or finalization results.
- LangGraph adapter equivalence tests.
- Shadow-mode comparison tests before ownership switches.

### Migration Risk

High. This changes final-answer ownership and separates currently conflated responsibilities.

### Dependencies

Stage 2. Stage 3C additionally depends on comprehensive Finalizer test coverage and successful Stage 3B comparison.

### Rollback Boundary

Each slice is independently reversible:

- Revert 3A without changing live runtime behavior.
- Revert 3B by disabling shadow invocation.
- Revert 3C by restoring the tested compatibility path to Brain while keeping Finalizer contracts intact.

### Acceptance Criteria

- Stage 3A: Finalizer contracts and deterministic summary construction exist independently of LangGraph.
- Stage 3B: Shadow comparisons cover all supported terminal outcomes without changing user-visible behavior.
- Stage 3C: Brain no longer generates final answers.
- `ExecutionSummary` has exactly one authoritative producer.
- Finalizer can be invoked from a non-LangGraph runtime.
- Rolling memory is explicitly outside protocol outcome determination.
- No Finalizer contract references a graph node or graph terminal marker.

### Commit Independence

Stages 3A, 3B, and 3C should be separate commits.

## 10. Stage 4 - Remove Legacy Completion Code

- Status: `NOT_STARTED`
- Portability assessment: Fully preserves the portability constraint.

### Goal

Remove the obsolete YES/NO Step Completion Checker, unreachable completion branches, stale builders, and outdated tests after typed outcomes are live.

### LangGraph Coupling Risk

There is no inherent coupling risk in deletion. A risk appears only if the removed checker is replaced by graph-route-specific completion logic.

### Required Approach

Remove:

- the old YES/NO completion prompt;
- the textual completion parser;
- commented checker branches;
- stale completion-message builders;
- unreachable routes;
- tests that assert obsolete graph-specific behavior.

Retain completion semantics only in typed `BrainOutcome` values and Controller validation.

### Intended Dependency Direction

`Brain service -> typed BrainOutcome -> CortexController`

LangGraph adapters only transport the outcome. They do not reinterpret completion.

### Files Likely Affected

- `core/graph_brain.py`
- `core/graph_state_machine.py`
- `core/graph_constants.py`
- Obsolete completion tests and fixtures

### Protocol Contracts Affected

None beyond completing the Stage 2 typed-outcome migration.

### Tests Required

- No completion decision depends on text.
- No legacy checker symbol remains referenced.
- Controller rejects invalid completion evidence.
- Adapter routing remains covered by current typed contracts.
- Replacement tests describe framework-neutral behavior.

### Migration Risk

Low-medium. The stage is primarily deletion, but only safe after replacement behavior is established.

### Dependencies

Stage 2 and preferably Stage 3C.

### Rollback Boundary

A deletion-only commit that can be reverted without reverting the typed contracts.

### Acceptance Criteria

- The old checker and parser are absent.
- No unreachable completion branch remains.
- No control-flow decision reads a completion text prefix.
- Replacement tests cover typed completion and Controller validation.

### Commit Independence

Stage 4 is an independent deletion commit after its dependencies are satisfied.

## 11. Stage 5 - Controller Authority Cleanup

- Status: `NOT_STARTED`
- Portability assessment: Preserves the portability constraint only with Controller defined as semantic authority rather than a graph node or graph entrypoint.

### Goal

Make Controller the single semantic execution-transition authority, remove protocol-visible state mutation from workers, and place invocation/dispatch responsibility in a replaceable runtime driver.

### LangGraph Coupling Risk

The statement "Controller becomes the graph entrypoint" would incorrectly turn a protocol invariant into a LangGraph topology requirement.

### Required Approach

The governing rule is:

> Every worker operation must be authorized by a framework-neutral Controller decision. The runtime execution driver invokes Controller and dispatches authorized work.

Introduce or clarify semantic Controller commands such as:

- `REQUEST_PLAN`
- `RUN_BRAIN`
- `EXECUTE_TOOLS`
- `FINALIZE_EXECUTION`
- `WAIT`
- `TERMINATE`

These are application actions, not graph destinations.

A LangGraph adapter may map them to its current implementation:

```text
REQUEST_PLAN       -> planner node
RUN_BRAIN          -> brain node
EXECUTE_TOOLS      -> tools/capture nodes
FINALIZE_EXECUTION -> finalizer adapter
TERMINATE          -> END
```

This mapping exists only in the LangGraph adapter and must never be persisted as protocol state.

Migration slices:

- Stage 5A: Remove protocol-visible state mutation from Brain, Planner, Finalizer, and tools.
- Stage 5B: Introduce or clarify runtime execution-driver ownership. Adapt LangGraph topology so each worker dispatch is authorized by Controller. Whether the graph happens to enter through a Controller adapter remains an implementation choice.

### Intended Dependency Direction

`LangGraph adapter -> ExecutionDriver/Application port -> CortexController -> protocol/domain`

`ExecutionDriver -> worker service ports`

No worker service or protocol/domain type depends on the driver or graph adapter.

### Files Likely Affected

- `core/protocol/controller.py`
- `core/graph_controller.py`
- `core/graph_brain.py`
- `core/graph_planner.py`
- `core/graph.py`
- `core/state.py`
- `core/protocol/bridge.py`
- Runtime driver and state-machine modules
- Controller, worker, graph-adapter, and non-graph driver tests

### Protocol Contracts Affected

- Semantic Controller commands
- Worker requests and results
- Execution snapshot ownership
- Runtime `ExecutionDriver` or equivalent application port

### Tests Required

- Controller tests run without LangGraph.
- Workers cannot modify protocol-visible execution state.
- Every worker invocation has a preceding Controller authorization.
- Protocol decisions contain no node names.
- Renaming adapter nodes does not alter protocol tests.
- A minimal in-memory non-LangGraph driver executes the same lifecycle.
- LangGraph topology tests verify mapping only, not semantics.

### Migration Risk

Medium-high. This changes state ownership and orchestration boundaries.

### Dependencies

Stages 1 through 4.

### Rollback Boundary

- Stage 5A is a separate single-writer cleanup commit.
- Stage 5B is a separate execution-driver and adapter-topology commit.

Do not combine topology changes with worker state-ownership changes in one rollback unit.

### Acceptance Criteria

- Controller is the single semantic transition authority.
- Controller is not typed or documented as a LangGraph node.
- Graph entrypoint is an adapter choice, not a protocol guarantee.
- Workers do not mutate protocol-visible execution state.
- Node names, graph edges, and `END` are absent from protocol models and persistent execution state.
- A minimal in-memory driver demonstrates the lifecycle without LangGraph.

### Commit Independence

Stages 5A and 5B should be separate commits.

## 12. Stage 6 - Async Ownership

- Status: `NOT_STARTED`
- Portability assessment: Preserves the portability constraint when the scheduler targets a framework-neutral runtime port rather than a graph node.

### Goal

Refactor asynchronous polling so the scheduler emits a semantic wake event and Controller authorizes polling, rather than the scheduler fabricating Controller decisions or relying on graph identities.

### LangGraph Coupling Risk

"Wake Controller" must not mean "route to the controller node." Persisting a graph node name, resume token, or graph routing target in async state would violate the portability boundary.

### Required Approach

Use framework-neutral contracts such as:

- `AsyncPollDue`
- `ExecutionWakeRequest`
- `ControllerInput`
- `ControllerCommand`
- Correlation and request identifiers

Required flow:

```text
Scheduler
  -> emits AsyncPollDue to an ExecutionDriver/application port
  -> driver invokes Controller
  -> Controller authorizes a semantic poll ToolRequest
  -> runtime executes the tool
  -> normalized ToolResult returns to Controller
```

The scheduler must not:

- refer to a graph node;
- construct `ControllerDecision`;
- mutate protocol state;
- directly execute the provider poll before Controller authorization.

A LangGraph-backed runtime may resume graph execution internally, but graph resume tokens and node identities remain private to the adapter.

### Intended Dependency Direction

`Scheduler adapter -> ExecutionDriver/application wake port -> CortexController`

`CortexController -> semantic poll command -> Tool service port`

`LangGraph resume adapter -> ExecutionDriver/application wake port`

The scheduler and Controller do not depend on LangGraph.

### Files Likely Affected

- `core/runtime/async_poller.py`
- `core/protocol/controller.py`
- `core/protocol/models.py`
- `core/protocol/enums.py`
- `core/graph_runner.py`
- `core/graph_state_machine.py`
- Async polling, Controller, runtime-driver, and LangGraph-resume tests

### Protocol Contracts Affected

- Wake event
- Asynchronous wait state
- Poll authorization
- Correlation
- Timeout and cancellation
- Normalized tool result

### Tests Required

- Scheduler tests use a fake execution-driver port, not LangGraph.
- Scheduler emits wake events but no Controller decisions.
- Controller determines whether polling remains valid.
- Stale wakes and stale poll results are ignored safely.
- Cancellation and timeout races are covered.
- LangGraph resume behavior is tested only in adapter integration tests.
- The same async lifecycle is demonstrated through a non-LangGraph driver.

### Migration Risk

High. Async timing, cancellation, stale work, and state ownership interact.

### Dependencies

Stage 5.

### Rollback Boundary

Keep the existing poller behind the same runtime interface until the new event-driven path passes parity tests. Switch implementations in a dedicated commit.

### Acceptance Criteria

- Async protocol state contains no node names or graph resume details.
- Scheduler can operate with any execution-driver implementation.
- Only Controller authorizes polling transitions.
- Poll results are framework-neutral `ToolResult` values before reaching Controller.
- Stale wake, timeout, and cancellation behavior is deterministic.

### Commit Independence

Stage 6 can be committed independently after Stage 5.

## 13. Stage 7 - Documentation Reconciliation

- Status: `NOT_STARTED`
- Portability assessment: Preserves the portability constraint when CEP is framework-neutral and CIS clearly labels LangGraph as one implementation mapping.

### Goal

Update CEP and CIS documents to match the implemented architecture and clearly distinguish CURRENT guarantees from FUTURE compliance requirements.

### LangGraph Coupling Risk

Documentation can create accidental coupling by presenting graph topology, graph nodes, or `END` as the protocol lifecycle.

### Required Approach

- CEP documents describe semantic commands, worker outcomes, lifecycle state transitions, and ownership without graph node names.
- CIS documents describe the current LangGraph realization and clearly label it as adapter-specific.
- Architecture documents state that Controller authority is semantic rather than topological.
- CURRENT implementation guarantees and FUTURE protocol/compliance requirements are explicitly separated.
- Existing stale Summary and Step Completion descriptions are reconciled with the implemented contracts.

### Intended Dependency Direction

Documentation mirrors the code dependency direction:

`CIS LangGraph realization -> runtime/services -> CEP protocol semantics`

CIS may map protocol actions to nodes. CEP must remain valid without that mapping.

### Files Likely Affected

- `docs/protocol/CEP-001-runtime-protocol.md` through `CEP-006-protocol-compliance-and-validation.md`
- Relevant documents under `docs/implementation`
- Architecture and README material

### Protocol Contracts Affected

Documentation only. Any newly discovered semantic change must be handled as a separately approved protocol change rather than hidden in documentation cleanup.

### Tests Required

- Documentation link and identifier checks.
- Import-boundary guard preventing LangGraph types in protocol packages.
- Consistency checks between enum/model names and documentation.
- Search-based checks that protocol enums and persistent-state fields do not encode node names.

### Migration Risk

Low, provided documentation follows rather than silently changes the implementation.

### Dependencies

Stages 1 through 6.

### Rollback Boundary

A documentation-only commit.

### Acceptance Criteria

- CEP lifecycle remains valid without LangGraph.
- CIS explicitly labels graph nodes and routes as implementation details.
- Controller ownership is defined semantically, not topologically.
- CURRENT and FUTURE requirements are clearly separated.
- Step completion, execution termination, final-answer generation, `ExecutionSummary`, and rolling memory have distinct documented owners.

### Commit Independence

Stage 7 is an independent documentation-only commit.

## 14. Stage 8 - Compliance Hardening Decision

- Status: `NOT_STARTED`
- Portability assessment: Preserves the portability constraint only if durability and replay semantics are defined independently of LangGraph checkpointing.

### Goal

Evaluate separately whether CEP-003 and CEP-006 requirements such as append-only accepted event history, deterministic replay, and checkpoint commit guarantees should be:

- Option A: implemented now; or
- Option B: explicitly classified as future protocol requirements.

Do not implement Option A without clear justification and a separately approved implementation roadmap.

### LangGraph Coupling Risk

Treating LangGraph checkpoints or graph traversal history as the protocol event journal would make replay and durability depend on one orchestration engine.

### Required Approach

If Option A is justified, define framework-neutral ports before selecting storage or orchestration implementations:

- `EventJournal`
- `ExecutionSnapshotStore`
- `CheckpointCommitter`
- `ReplaySource`
- Monotonic event and version identifiers

LangGraph checkpointing may implement part of these ports, but it cannot define their semantics. Graph traversal history is not automatically equivalent to accepted protocol event history.

Absent an immediate durability, audit, distributed execution, or crash-recovery requirement, the recommended decision is Option B: classify CEP-003 and CEP-006 guarantees as FUTURE protocol requirements.

### Intended Dependency Direction

`Runtime persistence adapters -> persistence ports -> protocol/domain event and snapshot contracts`

`LangGraph checkpointer adapter -> persistence ports`, where appropriate.

Protocol replay and commit semantics must not depend on LangGraph APIs or checkpoint formats.

### Files Likely Affected

Initially, documentation and an architecture decision record only. If Option A is later approved, subsequent work may affect protocol persistence ports, runtime implementations, storage adapters, and integration tests.

### Protocol Contracts Affected

- Event identity
- Accepted-event ordering
- Commit acknowledgement
- Snapshot versioning
- Replay determinism
- Recovery guarantees

### Tests Required if Option A Is Later Approved

- The same accepted events produce the same domain state without LangGraph.
- Crash-before-commit and crash-after-commit behavior.
- Duplicate and out-of-order event handling.
- Replay across different orchestration adapters.
- Protocol and storage version compatibility.
- LangGraph checkpoint adapter conformance tests.

### Migration Risk

Low for the classification decision; very high for implementation.

### Dependencies

Stage 7 for reconciled documentation. Any Option A implementation depends on a separately approved roadmap and justification.

### Rollback Boundary

The Stage 8 decision is an architecture-decision/documentation-only commit. Any later Option A implementation must be divided into separate port, journal, checkpoint, recovery, and replay rollback units.

### Acceptance Criteria

- Current guarantees are not overstated.
- Protocol durability is not equated with LangGraph checkpoint behavior.
- The selected Option A or Option B decision and rationale are recorded.
- Option A implementation does not begin without explicit justification and approval.

### Commit Independence

The classification decision is independent. Option A implementation is not one commit and is outside this roadmap until separately authorized.

## 15. Revised Commit Sequence

The required commit order is:

1. Stage 1 - Runtime correctness
2. Stage 2 - Typed Brain outcomes
3. Stage 3A - Add Finalizer contracts and tests
4. Stage 3B - Shadow-run Finalizer
5. Stage 3C - Transfer final-answer ownership
6. Stage 4 - Remove legacy completion code
7. Stage 5A - Enforce single protocol-state writer
8. Stage 5B - Establish runtime-driver ownership and adapt topology
9. Stage 6 - Refactor async ownership
10. Stage 7 - Reconcile CEP/CIS and architecture documentation
11. Stage 8 - Record the compliance decision only

No commit may encode graph node names in protocol enums or persistent protocol state. A stage is not complete merely because its LangGraph path works; its framework-neutral contracts and non-LangGraph tests must also satisfy the stage acceptance criteria.

## 16. Target Architecture at Roadmap Completion

At completion of Stages 1 through 7:

- Controller is the single semantic transition authority.
- A replaceable runtime driver invokes Controller and dispatches authorized semantic commands.
- Planner, Brain, Tool, and Finalizer are framework-neutral worker/application-service contracts.
- Workers return typed facts and results without mutating protocol-visible state.
- Step completion is a typed Brain judgment accepted or rejected by Controller under deterministic rules.
- Execution termination is a Controller decision.
- Final-answer rendering and `ExecutionSummary` generation belong to Finalizer.
- Rolling memory is a separate post-execution concern that cannot change execution outcome.
- Async scheduling emits semantic wake events and never fabricates Controller decisions.
- LangGraph nodes remain thin adapters that translate `GraphState` and semantic commands.
- LangGraph topology is replaceable and does not define protocol semantics.
- CEP documents normative protocol behavior; CIS documents the current LangGraph/runtime realization.
- Durable history, deterministic replay, and commit guarantees remain explicitly FUTURE unless Stage 8 separately authorizes them.

This target does not require Controller to be a LangGraph node or graph entrypoint. It requires every protocol-visible transition and every worker dispatch to be authorized by Controller semantics, regardless of the orchestration technology used.
