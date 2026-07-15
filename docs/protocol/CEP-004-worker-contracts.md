# CEP-004 Worker Contracts

- Protocol Family: CortexNode Execution Protocol (CEP)
- Document ID: CEP-004
- Version: 1.0
- Status: Review Candidate
- Layer: Layer 2 (Execution Protocol)

## 1. Purpose
This RFC defines protocol-facing contracts for Planner, Controller, Brain, Tool, and Summary workers. It does not redefine architecture responsibilities.

## 2. Cross-Worker Rules

- Workers never communicate directly.
- Controller is the only coordinator.
- Workers consume commands and emit events only.
- Workers do not mutate protocol history.
- Emitted events are immutable facts.

### 2.1 Worker Isolation

Workers are isolated protocol participants.

Workers do not depend on the internal implementation of other workers.

Workers communicate exclusively through Controller-mediated commands and events.

Worker correctness must not rely on implementation details, internal state, or execution strategy of other workers.

A worker implementation may be replaced without requiring protocol changes, provided it continues to satisfy the CEP contract.

## 3. Planner Contract

Planner always produces plan revisions.

Planner never produces runtime decisions.

### Owns
- plan generation
- plan revision

### Does Not Own
- execution state
- runtime coordination
- step execution

### Inputs
- CreatePlan command
- optional replanning context from Controller

### Outputs
- PlanCreated event
- PlanRevised event

### Preconditions
- valid execution identity exists
- planning request is accepted by Controller

### Postconditions
- plan revision is emitted as a new immutable fact
- planner does not dispatch step or tool work

### Failure Behavior
- planner failure must be surfaced as inability to emit PlanCreated or PlanRevised
- Controller decides retry, cancellation, or alternate terminal path

### Non-Permissions
- Planner must not execute steps
- Planner must not invoke tools
- Planner must not coordinate lifecycle

## 4. Controller Contract

Controller is the only execution coordinator.
Controller is the sole owner of ExecutionState as defined in CEP-003.

No other worker may create, modify, or transition ExecutionState.

### Owns
- ExecutionState
- protocol transitions
- checkpoint decisions
- command dispatch

### Does Not Own
- planning
- reasoning
- tool execution
- summary generation

### Inputs
- external execution intents
- all runtime events from workers
- ResumeExecution, CancelExecution, RetryStep intents

### Outputs
- commands: CreatePlan, ExecuteStep, RunTool, RetryStep, PauseExecution, ResumeExecution, CancelExecution, GenerateSummary
- events: ExecutionStarted, StepStarted, ToolStarted, ExecutionPaused, ExecutionResumed, ExecutionCheckpointed, ExecutionCompleted, ExecutionCancelled

### Preconditions
- execution identity and protocol version are established

### Postconditions
- each transition is validated and checkpointed
- deterministic next action chosen according to CEP-002 tables

### Failure Behavior
- on invalid transition, enforce protocol violation handling path
- on unrecoverable state mismatch, emit terminal cancellation/failure path

### Non-Permissions
- Controller must not delegate lifecycle ownership
- Controller must not permit direct worker-to-worker signaling

## 5. Brain Contract

Brain executes one step attempt.

Brain never advances execution.

### Owns
- step reasoning
- step validation
- tool requests

### Does Not Own
- execution lifecycle
- retry policy
- checkpointing
- plan mutation

### Inputs
- ExecuteStep command
- relevant tool outcomes routed through Controller

### Outputs
- ToolRequested event
- StepCompleted event
- StepFailed event
- ReplanRequested event

### Preconditions
- an active step attempt exists
- Brain receives step-scoped context only

### Postconditions
- exactly one deterministic step outcome path is emitted per attempt

### Failure Behavior
- if step cannot proceed safely, emit StepFailed or ReplanRequested
- Controller decides retry/replan/cancel path

### Non-Permissions
- Brain must not reorder or edit active plan
- Brain must not own execution loop
- Brain must not dispatch tools directly

## 6. Tool Contract

Tool performs deterministic operations.

Tool never coordinates execution.

### Owns
- deterministic tool execution

### Does Not Own
- planning
- protocol state
- execution coordination

### Inputs
- RunTool command

### Outputs
- ToolCompleted event
- ToolFailed event

### Preconditions
- ToolStarted already recorded for this operation

### Postconditions
- one tool outcome event emitted for requested operation

### Failure Behavior
- failures are returned as ToolFailed facts
- retries are controller decisions only

### Non-Permissions
- Tool must not plan
- Tool must not coordinate lifecycle
- Tool must not trigger replanning directly

## 7. Summary Contract

Summary interprets execution history.

Summary never changes execution history.

### Owns
- execution summary generation

### Does Not Own
- execution history
- execution state
- runtime decisions

### Inputs
- GenerateSummary command
- execution facts from protocol history

### Outputs
- SummaryGenerated event

### Preconditions
- execution has entered terminal state

### Postconditions
- summary reflects protocol facts only

### Failure Behavior
- summary generation failure is returned to Controller for terminal handling policy

### Non-Permissions
- Summary must not inspect hidden reasoning traces
- Summary must not alter historical events

## 8. Contract Compliance Checklist

A worker implementation is CEP-004 compliant only if:
- it uses command/event interaction only
- it never bypasses Controller coordination
- it emits immutable events only
- it respects permissions and non-permissions for its role
- it preserves deterministic behavior under retry, replan, resume, and cancellation
