# CIS-002 Runtime State Model

- Specification Family: CortexNode Implementation Specification (CIS)
- Document ID: CIS-002
- Version: 1.0
- Status: Review Candidate
- Layer: Layer 3 (Implementation Specification)
- Series Index: docs/implementation/README.md

## 1. Purpose
CIS-002 describes how CortexNode realizes ExecutionState, checkpoints, resume, replay, and runtime state ownership defined by CEP-003.

CEP defines protocol semantics.

CIS explains runtime realization.

CIS never overrides CEP.

## 2. Scope
Included:
- runtime state representation
- LangGraph state
- execution cursor realization
- checkpoint realization
- runtime ownership
- state mutation rules
- replay state reconstruction

Excluded:
- protocol semantics
- persistence technology
- serialization
- Redis schema
- database layout
- storage implementation

## 3. Runtime State Philosophy
Runtime state exists to realize CEP ExecutionState in the CortexNode runtime.

Runtime state is organized conceptually into:
- Protocol-visible State
- Runtime Working State

Protocol-visible State participates in protocol semantics and conformance.

Runtime Working State exists solely to support runtime execution and may evolve without changing protocol semantics.

Protocol conformance is evaluated only on Protocol-visible State and accepted protocol outcomes.

## 3.1 Runtime State Classification
Runtime Objects are classified as follows.

Protocol-visible State:
- ExecutionState
- ExecutionContext
- ExecutionCursor
- Event History
- Completed Step Ledger
- Active Plan
- Active Step
- Checkpoint State (recovery snapshot content that represents accepted protocol progression)

Runtime Working State:
- role-scoped transient execution helpers
- transient node-local orchestration artifacts
- temporary runtime coordination markers used to complete legal transitions

Only Protocol-visible State participates in replay reconstruction, checkpoint continuity, compliance evaluation, and summary inputs.

Runtime Working State supports execution flow but never becomes protocol truth.

## 4. Runtime State Components

### 4.1 ExecutionState
Purpose:
- Authoritative runtime representation of accepted protocol state.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- Controller logic only.

Runtime Readers:
- Planner Runtime Role, Brain Runtime Role, Tool Runtime Role, Summary Runtime Role, runtime runner.

CEP mapping:
- CEP-003 ExecutionState and protocol invariants.

### 4.2 ExecutionContext
Purpose:
- Protocol-visible context assembled for role-scoped execution decisions.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- Context Assembly Service under controller authority.

Runtime Readers:
- Planner Runtime Role, Brain Runtime Role, Summary Runtime Role.

CEP mapping:
- CEP-003 context and cursor-aligned state reconstruction requirements.

### 4.3 ExecutionCursor
Purpose:
- Runtime realization of the next legal continuation position after restore.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- Controller progression logic.

Runtime Readers:
- Runtime runner and controller dispatch path.

CEP mapping:
- CEP-003 Execution Cursor.

### 4.4 Event History
Purpose:
- Append-only runtime realization of accepted protocol events for legality checks, replay reconstruction, and summary input.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- Runtime history append path under controller authority.

Runtime Readers:
- Controller logic, Planner/Brain/Summary role inputs, replay validation paths.

CEP mapping:
- CEP-003 immutable execution history and replay behavior.

Editorial note:
- Event History remains protocol truth in runtime realization.
- Checkpoint State accelerates recovery and never replaces Event History as protocol truth.

### 4.5 Completed Step Ledger
Purpose:
- Runtime record of completed work used for immutable progress guarantees.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- Controller progression logic.

Runtime Readers:
- Brain Runtime Role, Summary Runtime Role, replay and compliance validation paths.

CEP mapping:
- CEP-003 completed-work immutability constraints.

### 4.6 Active Plan
Purpose:
- Runtime representation of accepted plan revision used for current execution.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- Planner proposes revisions; Controller accepts active revision.

Runtime Readers:
- Brain Runtime Role, Summary Runtime Role, runtime runner.

CEP mapping:
- CEP-003 plan revision continuity and checkpoint contents.

### 4.7 Active Step
Purpose:
- Runtime representation of the currently active step attempt.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- Controller progression logic.

Runtime Readers:
- Brain Runtime Role, runtime runner.

CEP mapping:
- CEP-003 cursor continuity and transition legality.

### 4.8 Checkpoint State
Purpose:
- Runtime recovery snapshot state used for legal restore and continuation.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- Checkpoint Manager Service under controller authority.

Runtime Readers:
- Resume and runtime startup paths.

CEP mapping:
- CEP-003 checkpoint contract and recovery behavior.

## 5. LangGraph State Mapping
LangGraph State is implementation state used to realize protocol behavior.

Protocol state remains authoritative.

Graph state contains runtime helpers in addition to protocol-visible state.

Graph state may evolve independently when observable protocol behavior remains unchanged.

Runtime objects persisted across node transitions include:
- ExecutionState
- Active Plan
- Active Step
- ExecutionCursor
- Event History
- Completed Step Ledger
- checkpoint-related recovery state required for resume continuity

Runtime Working State fields may change across runtime node transitions without changing protocol semantics.

### 5.1 Runtime State Visibility

Not all Runtime Objects are visible to every Runtime Role.

Controller has authoritative visibility over all Protocol-visible State.

Planner, Brain, Tool Runtime, and Summary consume only the role-scoped Runtime Objects required for their responsibilities.

Runtime Working State remains internal to the Runtime Services that own it.

## 6. State Ownership
Controller remains the sole authority for:
- ExecutionState
- Event History
- Execution Cursor
- checkpoint progression

Workers consume state and produce role-scoped outputs.

Workers never directly mutate accepted protocol state.

Ownership alignment:
- Protocol-visible State: controller-owned, controller-written
- Runtime Working State: controller-governed, role-consumed

Authoritative ownership rule:
- Controller Runtime Role is the only authoritative Runtime Writer of accepted runtime state.

## 7. Runtime Mutation Rules
Legal runtime mutations include:
- append-only accepted history growth
- checkpoint creation at legal transition boundaries
- retry metadata updates by controller progression
- active plan revision updates on accepted replanning
- active step updates on legal step progression

Mutation constraints:
- Completed work is immutable once accepted.
- Accepted facts are never rewritten.
- Runtime Working State may mutate for orchestration.
- Runtime Working State mutations must not alter accepted protocol facts.

## 7.1 Runtime State Lifecycle
This section defines lifecycle behavior for major Runtime Objects.

ExecutionState:
- Creation: initialized when execution starts.
- Updates: controller-authorized transition updates only.
- Checkpoint participation: Included only when necessary to reconstruct legal continuation context.
- Cleanup: released when execution lifecycle ends and cleanup policy runs.

ExecutionContext:
- Creation: assembled per role-scoped execution turn.
- Updates: refreshed from accepted facts and legal cursor position.
- Checkpoint participation: included when required for legal continuation context.
- Cleanup: replaced as execution advances.

ExecutionCursor:
- Creation: initialized with first legal continuation position.
- Updates: advanced only by legal controller transitions.
- Checkpoint participation: always required for resume continuity.
- Cleanup: finalized at terminal completion.

Event History:
- Creation: initialized at execution start.
- Updates: append-only accepted event recording.
- Checkpoint participation: referenced by checkpoint position alignment.
- Cleanup: retained or archived according to runtime policy after completion.

Completed Step Ledger:
- Creation: initialized as empty completed-work record.
- Updates: append-only completion confirmations.
- Checkpoint participation: required for no-rerun guarantees.
- Cleanup: retained with execution record lifecycle.

Active Plan:
- Creation: set on accepted initial plan.
- Updates: replaced only by accepted plan revision.
- Checkpoint participation: required for legal continuation and replay consistency.
- Cleanup: finalized at terminal execution.

Active Step:
- Creation: set when controller activates a legal next step.
- Updates: changed only by legal step progression.
- Checkpoint participation: required when a step attempt is active.
- Cleanup: cleared or finalized on step completion and terminal closure.

Checkpoint State:
- Creation: created at legal checkpoint trigger points.
- Updates: Each checkpoint is immutable once committed. A newer valid checkpoint becomes the preferred recovery point but does not modify previously committed checkpoints.
- Checkpoint participation: self-referential recovery artifact.
- Cleanup: managed by runtime retention policy.

## 8. Checkpoint Realization
Checkpoint realization in runtime follows CEP checkpoint semantics.

Checkpointing is a runtime optimization for recovery latency and continuity.

Protocol correctness depends on accepted Event History and legal transitions, not on a particular checkpoint mechanism.

Checkpoint trigger:
- controller-triggered at meaningful protocol transitions.

Checkpoint contents:
- execution identity and protocol version context
- accepted plan revision reference
- execution cursor
- completed-step ledger and retry progression
- ordered history position required for legal continuation

Restore process:
- restore latest valid checkpoint state
- restore execution cursor and accepted progression state
- prepare legal continuation context

Validation:
- verify checkpoint consistency with accepted history position
- verify legal continuation from cursor
- verify restoration does not contradict accepted Event History

Recovery:
- continue only from legal protocol position
- preserve completed work without re-execution

## 9. Resume Realization
Resume realization includes:
- restore state
- restore cursor
- validate checkpoint
- continue execution

Runtime resume principles:
- completed work is not rerun automatically
- continuation starts from legal cursor position
- restore and continuation preserve accepted facts

## 10. Replay Realization
Replay rebuilds runtime state from accepted history.

Replay behavior:
- replay consumes events
- replay never dispatches commands
- replay reconstructs Protocol-visible State, including ExecutionState, from accepted Event History.
- replay validates transition legality

Replay reconstructs Protocol-visible State exclusively from accepted Event History.

Runtime Working State may be recreated during replay for execution support, but recreated Runtime Working State never becomes protocol truth.

Replay realization goal:
- deterministic reconstruction of protocol-visible state for validation and auditability

## 11. Runtime Invariants
- Controller is sole state authority.
- Accepted Event History is append-only.
- Completed work never changes.
- Replay is deterministic.
- checkpoint restoration never rewrites accepted Event History.
- ExecutionCursor always references a legal continuation position.
- Runtime Working State never becomes authoritative protocol state.
- Runtime preserves CEP semantics.
- Runtime Services never bypass Controller authority.
- Worker Runtime Roles never own execution state.

## 12. Current Runtime Mapping
This section describes the current CortexNode implementation mapping only.

Runtime Modules may evolve while preserving observable protocol behavior and CEP conformance.

| Runtime Object | Current Runtime Realization |
| --- | --- |
| ExecutionState | core/state.py |
| ExecutionContext | core/graph_context.py plus core/graph_messages.py |
| ExecutionCursor | core/state.py plus core/graph_runner.py |
| Event History | runtime message history carried through graph state and runner event stream |
| Completed Step Ledger | controller progression state in graph execution path |
| Active Plan | core/graph_planner.py outputs accepted into runtime state |
| Active Step | controller progression path in core/graph_state_machine.py and brain dispatch flow |
| Checkpoint State | main.py session load/save responsibilities |
| Replay Realization | core/graph_runner.py plus core/graph_state_machine.py validation path |

## 13. Implementation Principles
- Runtime implementations may evolve while preserving observable protocol behavior.
- Protocol semantics remain authoritative.
- Runtime Objects exist solely to realize protocol concepts.
- Implementation optimizations never change protocol-visible behavior.
- Runtime architecture may change without affecting CEP compliance.

## 14. Compatibility
CIS-002 references and realizes:
- CEP-001 Runtime Protocol
- CEP-002 Execution Lifecycle
- CEP-003 State and Checkpoint
- CEP-004 Worker Contracts
- CEP-005 Protocol Data Contracts
- CEP-006 Protocol Compliance and Validation

CIS-002 realizes the runtime state model required by CEP-003 while remaining fully compatible with CEP-001 through CEP-006.

Implementation evolution may change runtime realization without changing protocol semantics.