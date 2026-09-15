# CEP-004 Worker Contracts

- Protocol Family: CortexNode Execution Protocol (CEP)
- Document ID: CEP-004
- Version: 1.0
- Status: Review Candidate
- Layer: Layer 2 (Execution Protocol)

## 1. Purpose
This RFC defines protocol-facing contracts for Planner, Controller, Brain, Tool Runtime, and Finalizer workers. It does not redefine architecture responsibilities.

## 2. Cross-Worker Rules

- Workers never communicate directly.
- Controller is the only coordinator.
- Workers consume typed requests and return typed proposals/results.
- Controller alone accepts the protocol and lifecycle meaning of worker results.
- Workers do not mutate protocol history.
- Emitted events are immutable facts.

### 2.1 Worker Isolation

Workers are isolated protocol participants.

Workers do not depend on the internal implementation of other workers.

Workers communicate exclusively through Controller-mediated commands and events.

Worker correctness must not rely on implementation details, internal state, or execution strategy of other workers.

A worker implementation may be replaced without requiring protocol changes, provided it continues to satisfy the CEP contract.

## 3. Planner Contract

Planner produces exactly one structured result for a Controller-authorized request.
Planner never produces runtime decisions.

### Owns
- plan proposal generation
- revision proposal generation

### Does Not Own
- execution state
- runtime coordination
- step execution

### Inputs
- Controller-owned `PlanningRequest` (`CREATE` or `REVISE`)

### Outputs
- request-bound `PlannerResult`: `PLAN_PROPOSED`, `NO_PLAN_REQUIRED`,
  `NEEDS_INPUT`, or `PLANNING_FAILED`

### Preconditions
- valid execution identity exists
- planning request is accepted by Controller

### Postconditions
- a proposal/result is emitted without changing accepted state
- Controller alone accepts or rejects the proposal and determines revision identity
- planner does not dispatch step or tool work

### Failure Behavior
- planner failure must use a typed planning failure category
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
- worker authorization and request construction

### Does Not Own
- planning
- reasoning
- tool execution
- summary generation
- mechanical worker execution

### Inputs
- external execution intents
- all runtime events from workers
- ResumeExecution, CancelExecution, RetryStep intents

### Outputs
- commands: CreatePlan, ExecuteStep, RunTool, RetryStep, PauseExecution, ResumeExecution, CancelExecution, GenerateFinalization
- events: ExecutionStarted, StepStarted, ToolStarted, ExecutionPaused, ExecutionResumed, ExecutionCheckpointed, ExecutionCompleted, ExecutionCancelled

### Preconditions
- execution identity and protocol version are established

### Postconditions
- each transition is validated and checkpointed
- deterministic next action chosen according to CEP-002 tables

ExecutionDriver applies the authorized transition and mechanically invokes the
selected worker. PortableExecutionRuntime owns turn preparation and reconciliation.
Neither component independently chooses lifecycle meaning.

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
- typed step-scoped judgments
- tool-request proposals

### Does Not Own
- execution lifecycle
- retry policy
- checkpointing
- plan mutation

### Inputs
- ExecuteStep command
- relevant tool outcomes routed through Controller

### Outputs
- typed `BrainResult`: tool requested, step completed, step failed, replan requested,
  direct final-answer readiness, continuation, invalid output, or provider failure

### Preconditions
- an active step attempt exists
- Brain receives step-scoped context only

### Postconditions
- exactly one typed outcome is returned per Brain invocation
- Controller validates the active step and binds accepted completion provenance

### Failure Behavior
- if step cannot proceed safely, emit StepFailed or ReplanRequested
- Controller decides retry/replan/cancel path

### Non-Permissions
- Brain must not reorder or edit active plan
- Brain must not own execution loop
- Brain must not dispatch tools directly

## 6. Tool Contract

Tool Runtime performs authorized operations and returns portable results.

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
- portable `ToolResult`, including asynchronous non-terminal or terminal state where
  the authorized tool supports asynchronous execution

### Preconditions
- ToolStarted already recorded for this operation

### Postconditions
- one tool outcome event emitted for requested operation
- result identity matches the exact authorized `ToolRequest`

### Failure Behavior
- failures are returned as ToolFailed facts
- retries are controller decisions only

### Non-Permissions
- Tool must not plan
- Tool must not coordinate lifecycle
- Tool must not trigger replanning directly

## 7. Finalizer Contract

Finalizer constructs terminal reporting from accepted execution facts.

Finalizer never changes execution history or lifecycle outcome.

### Owns
- execution summary generation
- final user-facing answer generation

### Does Not Own
- execution history
- execution state
- runtime decisions

### Inputs
- Controller-authorized `FinalizationRequest`
- accepted terminal execution facts and tool execution history

### Outputs
- `FinalizationResult` containing `ExecutionSummary`, final answer, and optional
  rendering error

### Preconditions
- execution has entered terminal state

### Postconditions
- execution summary reflects accepted protocol facts only
- answer-rendering failure does not alter terminal execution state

### Failure Behavior
- finalization failure is reported without reopening execution

### Non-Permissions
- Finalizer must not inspect hidden reasoning traces
- Finalizer must not alter historical events

## 7.1 Asynchronous Poll Boundary

A scheduler or adapter may emit a correlated poll-due wake, but it cannot construct a
Controller decision or authorize tool execution. Controller validates the wake and
constructs the status `ToolRequest`; ExecutionDriver invokes `ToolRuntimePort`; the
portable result is integrated before another Controller turn.

## 8. Contract Compliance Checklist

A worker implementation is CEP-004 compliant only if:
- it uses command/event interaction only
- it never bypasses Controller coordination
- it emits immutable events only
- it respects permissions and non-permissions for its role
- it preserves deterministic behavior under retry, replan, resume, and cancellation
