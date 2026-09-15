# CEP-006 Protocol Compliance and Validation

- Protocol Family: CortexNode Execution Protocol (CEP)
- Document ID: CEP-006
- Version: 1.0
- Status: Review Candidate
- Layer: Layer 2 (Execution Protocol)

## 1. Purpose
This RFC defines protocol-level compliance and validation requirements for CortexNode Execution Protocol implementations.

Protocol compliance means implementations produce protocol-equivalent observable behavior, even when internal runtime design differs.

Compliance concerns protocol behavior only. Compliance does not prescribe internal architecture.

Protocol conformance is evaluated solely through observable protocol behavior.

Internal implementation details are intentionally outside the scope of protocol conformance.

These requirements are normative. They do not imply that the current CortexNode
production implementation is fully conformant; its current status is recorded in
Section 10.1.

## 2. Compliance Scope
Protocol compliance validates the following:
- command semantics
- event semantics
- lifecycle transitions
- worker authority boundaries
- state invariants
- checkpoint behavior
- replay determinism
- data contract invariants

Protocol compliance does not validate implementation technology. Transport design, persistence technology, language choice, and framework choice are outside this RFC.

## 2.1 Compliance Terminology

The keywords MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are used throughout this specification to describe protocol requirements.

These keywords express protocol compliance requirements rather than implementation preferences.

## 2.2 Protocol Certification Principles

Protocol certification evaluates only externally observable protocol behavior.

Independently implemented runtimes are compliant if they produce protocol-equivalent behavior for equivalent protocol inputs.

Implementation language, framework, persistence approach, transport approach, and internal architecture are not part of protocol compliance.

## 3. Compliance Levels
Implementations MAY claim compliance by level. Full conformance requires satisfying all levels.

### 3.1 Protocol Vocabulary Compliance
Scope:
- command and event vocabulary
- issuer, producer, consumer, and meaning

Reference:
- CEP-001 Runtime Protocol

### 3.2 Lifecycle Compliance
Scope:
- legal transitions
- terminal behavior
- failure, retry, and replanning paths

Reference:
- CEP-002 Execution Lifecycle

### 3.3 Worker Contract Compliance
Scope:
- worker permissions and restrictions
- controller coordination authority

Reference:
- CEP-004 Worker Contracts

### 3.4 State Compliance
Scope:
- ExecutionState semantics
- execution cursor semantics
- protocol invariants

Reference:
- CEP-003 State and Checkpoint

### 3.5 Replay Compliance
Scope:
- replay ordering
- replay legality checks
- replay reconstruction determinism

References:
- CEP-001 Runtime Protocol
- CEP-003 State and Checkpoint

### 3.6 Checkpoint Compliance
Scope:
- checkpoint trigger rules
- restore and resume legality
- completed-work preservation

References:
- CEP-002 Execution Lifecycle
- CEP-003 State and Checkpoint

### 3.7 Data Contract Compliance
Scope:
- canonical protocol contract semantics
- identity and ownership invariants

Reference:
- CEP-005 Protocol Data Contracts

## 4. Validation Rules

Validation evaluates protocol semantics rather than implementation mechanics.

Passing validation demonstrates protocol conformance, not implementation equivalence.

An implementation MUST satisfy all rules below to claim protocol conformance.

### 4.1 Command and Event Legality
- Every command MUST have a legal issuer and legal executor as defined by CEP-001 and CEP-004.
- Every event MUST have a legal producer as defined by CEP-001 and CEP-004.
- Commands and events MUST preserve CEP-defined meaning.

### 4.2 Transition and Authority Rules
- Illegal transitions MUST be rejected according to CEP-002.
- Controller MUST remain the sole authority that advances execution.
- Workers MUST NOT coordinate directly.
- Brain MUST NOT own execution.
- Planner MUST NOT execute work.
- Tool MUST NOT retry itself.
- Finalizer MUST NOT modify execution history or terminal outcome.
- A worker result MUST NOT advance lifecycle state until Controller accepts it.
- Mechanical dispatch by ExecutionDriver MUST be traceable to the corresponding
  Controller authorization.

### 4.3 History and Immutability Rules
- Accepted protocol facts MUST be immutable.
- Events MUST remain append-only.
- Completed work MUST NOT change.
- Replanning MUST NOT rewrite completed work.

## 5. Validation Categories
Validation categories are organizational only and do not introduce new protocol behavior.

### 5.1 Structural Validation
Validates protocol vocabulary, required contract presence, identity usage, and conformance to CEP-defined data language.

### 5.2 Behavioral Validation
Validates command/event semantics, legal transitions, terminal behavior, and authority-constrained execution outcomes.

### 5.3 State Validation
Validates state invariants, checkpoint consistency, resume legality, completed-work immutability, and replay reconstruction consistency.

### 5.4 Authority Validation
Validates single-controller coordination, worker boundary enforcement, and absence of direct worker coordination.

## 6. Determinism Validation
Determinism validation proves protocol equivalence across independent implementations.

Equivalent protocol inputs MUST produce protocol-equivalent observable behavior.

Protocol-equivalent observable behavior means:
- equivalent legal transitions
- equivalent event ordering constraints
- equivalent terminal execution state
- equivalent replay reconstruction
- equivalent observable protocol results

Internal implementation strategies MAY differ. Observable protocol behavior MUST remain equivalent.

## 7. Replay Validation
Replay validation MUST verify:
- event ordering correctness
- transition legality during reconstruction
- state reconstruction consistency
- terminal outcome consistency

Replay MUST NOT dispatch commands.

Replay MUST NOT depend on checkpoints, runtime memory, worker state, or implementation-specific storage.

## 8. Resume Validation
Resume validation MUST verify:
- execution cursor restoration correctness
- completed work preservation
- retry counter preservation
- plan revision preservation
- absence of duplicated completed steps
- continuation from a legal protocol position

Resume MUST continue from the latest valid checkpoint position and MUST preserve prior accepted facts.

Resume MAY use checkpoints as a recovery optimization but MUST restore a protocol-valid ExecutionState before dispatching new commands.

## 9. Worker Compliance
Each worker MUST satisfy CEP-004 permissions and restrictions.

### 9.1 Planner
- MUST produce plan outputs only within Planner authority.
- MUST NOT execute steps, coordinate runtime, or invoke tool execution directly.

### 9.2 Controller
- MUST remain sole protocol coordinator.
- MUST validate transitions and select legal next commands.
- MUST enforce worker boundary rules.

### 9.3 Brain
- MUST produce step-scoped outcomes only.
- MUST NOT own execution lifecycle decisions.

### 9.4 Tool
- MUST perform deterministic operation responses only.
- MUST NOT alter protocol policy or coordination.

### 9.5 Finalizer
The implemented terminal reporting role is Finalizer.

- MUST consume a Controller-authorized `FinalizationRequest`.
- MUST generate `ExecutionSummary` and the final user-facing answer from accepted
  terminal facts.
- MUST NOT alter execution history or execution outcome.

### 9.6 Asynchronous Tool Polling

- A semantic wake MUST be treated as correlation, not authorization.
- Controller MUST validate the wake and construct/authorize the poll request.
- ExecutionDriver MUST execute the authorized request through `ToolRuntimePort`.
- Poll results MUST cross the Controller boundary as portable `ToolResult` values.
- Protocol state MUST NOT depend on graph node or successor identities.

## 10. Protocol Conformance Checklist
An implementation is conformant when all items are satisfied.

- CEP-001 command semantics satisfied
- CEP-001 event semantics satisfied
- CEP-002 lifecycle semantics satisfied
- CEP-003 state semantics satisfied
- CEP-004 worker authority satisfied
- CEP-005 data contracts satisfied
- replay determinism verified
- resume legality verified
- illegal transitions rejected
- Controller authority preserved

This is a requirements checklist, not a declaration that every item currently passes.

## 10.1 Current CortexNode Conformance Status

The production runtime currently implements the Controller-authority, typed worker
boundary, accepted-state ownership, plan/revision acceptance, Finalizer ownership,
portable tool-result, and asynchronous poll-authorization behaviors described by
CEP-001, CEP-002, CEP-004, and CEP-005. This is not a claim of full conformance with
those documents' journal-dependent requirements.

It does **not** claim full CEP-003 or CEP-006 conformance. The following normative
requirements are not currently implemented end to end:

- an append-only journal containing every accepted protocol event;
- deterministic framework-neutral replay from that journal;
- protocol-level atomic checkpoint commits tied to an ordered event position.

LangGraph checkpoints and graph traversal state do not satisfy these requirements by
themselves. The gaps remain explicit non-conformance items whose disposition is
reserved for Stage 8; this document does not weaken or redefine them.

## 11. Non-Conformance
An implementation is non-conformant if it violates any mandatory protocol requirement in this RFC or in CEP-001 through CEP-005.

Representative non-conformance conditions include:
- illegal lifecycle transitions
- worker authority violations
- mutable protocol history
- replay producing different protocol state for equivalent protocol history
- resume rerunning completed work
- direct worker communication

## 12. Out of Scope
This RFC does not define:
- implementation classes
- programming language
- serialization
- network protocol
- database
- LangGraph nodes
- Redis schema
- MCP
- SCADA

## 13. Future Compatibility
Future CEP versions MAY extend validation requirements while preserving protocol compatibility unless an explicit protocol version upgrade defines otherwise.

Newer protocol versions MUST NOT invalidate correct implementations of earlier protocol versions unless explicitly required by a protocol version upgrade.

## 14. Compatibility
CEP-006 validates conformance with CEP-001 through CEP-005.

Future CEP documents MAY extend validation rules without changing existing protocol semantics.
