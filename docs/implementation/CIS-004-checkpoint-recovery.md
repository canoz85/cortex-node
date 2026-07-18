# CIS-004 Checkpoint & Recovery

- Specification Family: CortexNode Implementation Specification (CIS)
- Document ID: CIS-004
- Version: 1.0
- Status: Review Candidate
- Layer: Layer 3 (Implementation Specification)
- Series Index: docs/implementation/README.md

## 1. Purpose
CIS-004 describes how the current CortexNode runtime realizes checkpointing, recovery, resume, and replay behavior defined by CEP-003 and validated by CEP-006.

This document is implementation-oriented.

This document does not redefine protocol semantics.

CEP remains normative for checkpoint semantics, replay semantics, resume legality, execution cursor semantics, and completed-work preservation.

If implementation wording diverges from CEP wording, CEP is authoritative.

## 2. Scope
Included:
- checkpoint runtime realization
- recovery workflow realization
- resume realization
- replay realization
- runtime persistence responsibilities
- checkpoint ownership and writer boundaries
- recovery validation realization
- current runtime module mapping

Excluded:
- protocol semantics
- persistence technology selection
- Redis schema
- database schema
- serialization formats
- storage implementation details

## 3. Checkpoint Philosophy
Checkpoint State exists to accelerate runtime restoration and legal continuation.

Checkpoint State is a runtime implementation artifact.

Checkpoint State is not protocol truth.

Accepted Event History remains authoritative for accepted protocol outcomes.

Checkpoint State references accepted progression and recovery position so runtime can restore faster without recomputing full state on every resume path.

Recovery optimization boundary:
- Accepted Event History defines protocol truth.
- Checkpoint State provides recovery acceleration.
- Recovery optimization never changes protocol meaning.

## 4. Checkpoint Components
Checkpoint realization is composed of Protocol-visible State and selected Working State required for legal continuation.

### 4.1 Execution Cursor
Purpose:
- identifies the next legal continuation position after restoration.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- Controller progression logic.

Runtime Readers:
- Runtime Runner Service and controller dispatch path.

CEP mapping:
- CEP-003 execution cursor restoration and resume legality.

### 4.2 Accepted Event History
Purpose:
- append-only protocol-visible record used for legality checks, replay reconstruction, and recovery alignment.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- history append path under Controller Authority.

Runtime Readers:
- Controller Runtime Role, Replay/Validation services, Summary Runtime Role.

CEP mapping:
- CEP-003 immutable history and replay behavior.

### 4.3 Completed Step Ledger
Purpose:
- immutable record of accepted completed work for no-rerun guarantees across restore and resume.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- Controller progression logic.

Runtime Readers:
- Controller Runtime Role, Brain Runtime Role, replay/resume validation paths.

CEP mapping:
- CEP-003 completed-work immutability and resume constraints.

### 4.4 Active Plan
Purpose:
- runtime representation of the accepted plan revision active at checkpoint position.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- Planner proposes revisions; Controller accepts active revision.

Runtime Readers:
- Brain Runtime Role, Summary Runtime Role, Runtime Runner Service.

CEP mapping:
- CEP-003 plan revision continuity across resume and replay validation.

### 4.5 Active Step
Purpose:
- identifies the currently active step attempt at checkpoint time.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- Controller progression logic.

Runtime Readers:
- Brain Runtime Role and Runtime Runner Service.

CEP mapping:
- CEP-003 legal step progression and cursor continuity.

### 4.6 Retry Metadata
Purpose:
- preserves retry continuity required for legal continuation.

Canonical Owner:
- Controller Runtime Role.

Runtime Writer:
- controller-authorized progression and retry tracking logic.

Runtime Readers:
- Controller Runtime Role, Brain Runtime Role, recovery validation path.

CEP mapping:
- CEP-003 retry/attempt continuity and CEP-006 resume validation requirements.

### 4.7 Working State (When Required)
Purpose:
- provides runtime-only orchestration continuity for post-restore execution.

Canonical Owner:
- owning Runtime Service under Controller Authority.

Runtime Writer:
- runtime helper/service components under controller-governed mutation rules.

Runtime Readers:
- role-scoped Runtime Services and Runtime Modules that require continuation context.

CEP mapping:
- realization support only; Working State does not define protocol semantics.

Working State constraints:
- Working State supports continuation but never becomes protocol truth.
- Working State may change without changing protocol semantics.

## 5. Checkpoint Lifecycle
Checkpoint lifecycle is realized as a controller-governed runtime sequence.

Checkpoint creation:
- runtime captures checkpoint-eligible Protocol-visible State at legal transition boundaries.

Checkpoint validation:
- runtime verifies checkpoint internal consistency before reuse as a recovery source.

Checkpoint persistence:
- Checkpoint Manager Service persists checkpoint-related recovery content through the runtime persistence path.

Checkpoint replacement:
- a newer valid checkpoint supersedes older recovery snapshots as preferred restore source.
- supersession does not rewrite Accepted Event History.

Checkpoint cleanup:
- retained snapshots are managed by runtime retention policy.
- cleanup behavior must preserve legal recovery and conformance evaluation needs.

Lifecycle guarantee:
- checkpoint progression is controller-governed and history-preserving.

### 5.1 Checkpoint Trigger Policy

Runtime SHOULD create checkpoints only at protocol-safe boundaries.

Typical trigger points include:

- accepted plan creation
- accepted step completion
- accepted replan
- execution completion
- explicit runtime checkpoint request

Runtime MUST NOT create checkpoints during partially accepted protocol transitions.

## 6. Recovery Workflow
Recovery is realized as an ordered runtime workflow.

1. Locate latest valid checkpoint candidate.
2. Restore checkpoint-related Protocol-visible State.
3. Restore required Working State needed for legal continuation.
4. Validate checkpoint consistency and event-history alignment.
5. Validate Execution Cursor legality.
6. Validate completed-work and retry continuity.
7. Resume execution through Controller Authority.

Recovery ordering rule:
- validation occurs before command dispatch continuation.

Recovery boundary:
- restoration re-establishes legal runtime continuation state.
- restoration does not rewrite accepted protocol history.

### 6.1 Restore Source Selection

Recovery selects the newest checkpoint satisfying:

- checkpoint integrity
- history alignment
- cursor legality

If no checkpoint satisfies these conditions,
runtime falls back according to recovery policy.

## 7. Resume Realization
Resume is runtime continuation from a validated restore position.

Resume realization sequence:
- restore state from latest valid checkpoint source
- validate cursor and continuity constraints
- continue from the next legal continuation position

Resume continuity requirements:
- legal Execution Cursor restoration
- completed-work preservation
- retry continuity preservation
- active plan continuity preservation
- active step continuity preservation where applicable

No-rerun guarantee:
- resume realization must not re-run accepted completed work.

Authority guarantee:
- resume continuation is initiated and governed only by Controller Authority.

## 8. Replay Realization
Replay is realized as a deterministic reconstruction and validation path.

Replay source of truth:
- replay consumes Accepted Event History.

Replay behavior in runtime realization:
- reconstruct Protocol-visible State from accepted events in recorded order
- validate transition legality during reconstruction
- validate reconstruction consistency

Replay constraints:
- replay never dispatches commands
- replay never invokes live execution progression

Replay and recovery separation:
- replay is independent from checkpoint recovery optimization.
- checkpoints may accelerate runtime restore, but replay correctness is defined by Accepted Event History.

## 9. Recovery Validation
Recovery validation is realized through controller-governed consistency checks before continuation.

Validation categories:
- checkpoint consistency validation
- event-history alignment validation
- execution cursor legality validation
- completed-work preservation validation
- active plan revision continuity validation
- retry metadata continuity validation
- active step continuity validation when applicable

Validation outcome handling:
- valid checkpoint state enables continuation from legal cursor.
- invalid or mismatched checkpoint state triggers recovery fallback handling under runtime policy and CEP legality boundaries.

## 10. Runtime Ownership
Checkpoint and recovery ownership follows established CIS ownership boundaries.

Checkpoint Manager Service:
- provides checkpoint persistence and restore mechanics.
- operates under Controller Authority.

Controller Runtime Role:
- sole authority for recovery acceptance, cursor legality, and continuation decisions.
- sole authoritative writer of accepted progression state.

Runtime Services:
- provide state, history, context, and checkpoint support capabilities.
- do not own protocol authority.

Worker Runtime Roles (Planner, Brain, Tool Runtime, Summary):
- consume restored state in role scope.
- produce role-scoped outputs only.
- never perform recovery authority decisions.

Ownership guarantees:
- workers never restore execution.
- Controller remains sole recovery authority.

## 11. Runtime Invariants
The following invariants apply to current runtime realization.

- Accepted Event History remains authoritative.
- Checkpoints never rewrite accepted history.
- Controller owns recovery acceptance and continuation authority.
- Completed work remains immutable once accepted.
- Resume starts from a legal Execution Cursor.
- Replay never dispatches commands.
- Recovery preserves Protocol-visible State continuity.
- Working State never becomes protocol truth.
- Recovery validation must succeed before continuation dispatch.
- Checkpoint supersession never alters accepted protocol outcomes.

## 12. Failure Recovery
Failure recovery is implementation realization of interruption and checkpoint fault handling.

Interrupted execution:
- runtime attempts restore from latest valid checkpoint source and resumes through controller-governed continuation.

Partial checkpoint:
- runtime treats incomplete snapshot as non-authoritative recovery source.
- runtime selects latest valid checkpoint candidate or fallback path.

Invalid checkpoint:
- runtime rejects invalid recovery source during validation.
- runtime follows fallback handling while preserving accepted history.

Corrupted checkpoint:
- runtime treats corrupted snapshot as invalid and does not continue from it.

Checkpoint mismatch:
- runtime rejects mismatched checkpoint/event alignment and prevents illegal continuation.

Recovery fallback:
- runtime may fall back to safe startup state or non-resume continuation path according to runtime policy.
- fallback handling must preserve CEP legality and accepted history immutability.

Failure-handling boundary:
- this section describes runtime realization patterns only.
- protocol meaning remains defined by CEP-002, CEP-003, and CEP-006.

## 13. Current Runtime Realization
This section maps logical responsibilities to current Runtime Modules without making module names normative.

Checkpoint/session persistence surface:
- main.py

Runtime state model surface:
- core/state.py

Execution loop and recovery-adjacent run orchestration surface:
- core/graph_runner.py

Controller decision and legality helper surface:
- core/graph_state_machine.py

Additional cooperating runtime realization surfaces:
- core/graph.py
- core/graph_capture.py
- core/graph_summarize.py
- core/graph_context.py

Mapping principles:
- one logical responsibility may span multiple Runtime Modules
- one Runtime Module may realize multiple responsibilities
- module layout may evolve without changing CIS semantics

## 14. Implementation Principles
- checkpointing is a runtime realization artifact, not protocol truth
- Accepted Event History remains the authoritative protocol record
- Controller Authority governs restore acceptance and continuation
- recovery must preserve accepted Protocol-visible State
- runtime optimizations must not change observable protocol behavior
- replay and resume remain distinct runtime realizations
- implementation structure may evolve while CEP semantics remain unchanged

## 15. Compatibility
CIS-004 realizes checkpoint and recovery behavior compatible with:
- CEP-001 Runtime Protocol
- CEP-002 Execution Lifecycle
- CEP-003 State and Checkpoint
- CEP-004 Worker Contracts
- CEP-005 Protocol Data Contracts
- CEP-006 Protocol Compliance and Validation

Conformance statement:
- CIS realizes CEP and never overrides CEP.
- If runtime behavior diverges from this CIS mapping while CEP semantics remain unchanged, CIS or implementation must be updated to restore alignment.
