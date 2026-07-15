# CEP-003 State and Checkpoint

- Protocol Family: CortexNode Execution Protocol (CEP)
- Document ID: CEP-003
- Version: 1.0
- Status: Review Candidate
- Layer: Layer 2 (Execution Protocol)

## 1. Purpose
This RFC defines protocol-level runtime state semantics, checkpoint requirements, resume behavior, and replay behavior.

## 2. ExecutionState (Protocol View)
ExecutionState is the authoritative protocol state managed by Controller. It is reconstructed from accepted facts and updated only through valid transitions.

ExecutionState contains two conceptual categories:
- Logical Execution State: execution identity, active plan revision, lifecycle phase, completed/pending/failed steps, retry counters, and terminal decision.
- Recovery State: checkpoint reference, execution cursor, event position, and protocol version.

These are conceptual categories within ExecutionState. They do not introduce new runtime objects.

ExecutionState must represent:
- execution identity and protocol version
- active plan revision identity
- current phase and resume cursor
- step progress ledger (completed, failed, pending)
- retry counters and terminal decision state
- latest checkpoint marker
- append-only event history reference

ExecutionState constraints:
- single writer: Controller
- monotonic transition progression
- no in-place rewrite of historical outcomes

## 2.1 Execution Cursor
The Execution Cursor is a protocol concept that identifies the next legal protocol action after restoration.

It exists so Controller can resume execution from a precise protocol position without re-executing completed work.

Controller uses the Execution Cursor to determine the next legal transition after a valid restoration.

The Execution Cursor is protocol state. It is not a program counter, memory pointer, LangGraph node, or database offset.

## 3. Checkpoint Contract

### 3.1 Checkpoint Objective
Checkpointing guarantees that execution can continue from the last committed transition without rerunning completed work.

A checkpoint is a recovery snapshot. Execution history is the immutable protocol record. Checkpoint improves recovery performance and is never the authoritative source of protocol truth.

### 3.2 Checkpoint Trigger Points
Controller MUST checkpoint after each meaningful transition, including:
- PlanCreated or PlanRevised acceptance
- StepCompleted and StepFailed decisions
- ToolCompleted and ToolFailed integration
- ExecutionPaused and ExecutionResumed
- ExecutionCompleted or ExecutionCancelled

### 3.3 Required Checkpoint Contents
A valid checkpoint includes:
- protocol version marker
- execution identity
- accepted plan revision reference
- current resume cursor
- immutable completed-step ledger
- retry counters and current attempt metadata
- latest terminal or non-terminal decision
- ordered event position for replay continuation

### 3.4 Checkpoint Invariants
- Checkpoint commits are atomic at protocol level.
- A checkpoint references an ordered event position.
- Any resume starts from the latest valid checkpoint only.

## 3.5 Protocol Invariants

- ExecutionState is reconstructed from accepted protocol events and updated only through valid protocol transitions.
- Execution history is immutable.
- Checkpoints never modify execution history.
- Resume never creates historical events.
- Replay never dispatches commands.
- Execution identity never changes during execution.
- Checkpoint identity is immutable.
- Controller remains the single writer of ExecutionState.

## 4. Resume Protocol

ResumeExecution semantics:
- ResumeExecution is a command accepted only from paused or interrupted non-terminal states.
- Resume consists of two conceptual phases:
	1. Restore ExecutionState from the latest valid checkpoint.
	2. Resume protocol execution beginning at the Execution Cursor.
- Controller records ExecutionResumed when restoration is successful.

Resume constraints:
- completed steps are never rerun automatically
- unresolved in-flight operations are reconciled before new commands
- no direct worker-to-worker recovery

## 5. Replay Protocol

### 5.1 Replay Objective
Replay reconstructs execution history for audit, analysis, or deterministic verification. Replay is not the same as live continuation.

Replay is historical reconstruction followed by deterministic validation.

Resume is runtime continuation followed by command dispatch.

### 5.2 Replayable Messages
Replay operates on protocol events only, including:
- ExecutionStarted
- PlanCreated
- PlanRevised
- StepStarted
- ToolRequested
- ToolStarted
- ToolCompleted
- ToolFailed
- StepCompleted
- StepFailed
- ReplanRequested
- ExecutionPaused
- ExecutionResumed
- ExecutionCheckpointed
- ExecutionCompleted
- ExecutionCancelled
- SummaryGenerated

Commands are never replayed as authoritative facts.

Replay rebuilds protocol state only. Replay never resumes execution. Replay never issues commands. Replay never invokes workers.

### 5.3 Replay Behavior
Replay processor must:
- consume protocol events in recorded order
- enforce transition validity while rebuilding state
- reject illegal event sequences as invalid history
- produce same final ExecutionState for same event stream

### 5.4 Replay vs Resume
- Replay: reconstructs protocol state from execution history and performs deterministic validation without dispatching commands.
- Resume: restores checkpointed runtime state and continues dispatching commands from the Execution Cursor.

## 6. Checkpoint Recovery Behavior
When runtime interruption occurs:
- Controller restores the latest valid checkpoint.
- Controller validates that event position and state cursor align.
- Controller records ExecutionResumed for successful recovery.
- If alignment fails, Controller records terminal cancellation/failure path according to CEP-002 transition rules.

## 7. Determinism Guarantees
Deterministic behavior requires:
- append-only immutable execution history
- one coordinator (Controller)
- explicit transition table enforcement
- stable interpretation of terminal and non-terminal events
- immutable completed-step ledger across resume and replay
- identical event history produces identical ExecutionState.
