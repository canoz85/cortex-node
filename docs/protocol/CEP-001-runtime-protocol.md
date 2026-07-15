# CEP-001 Runtime Protocol

- Protocol Family: CortexNode Execution Protocol (CEP)
- Document ID: CEP-001
- Version: 1.0
- Status: Review Candidate
- Layer: Layer 2 (Execution Protocol)

## 1. Protocol Overview

### 1.1 Purpose
CEP-001 defines the runtime message protocol used to coordinate execution across existing CortexNode workers. It specifies command and event semantics so independent implementations produce identical behavior.

### 1.2 Why CortexNode Uses a Protocol
A protocol is required to make runtime behavior deterministic, replayable, and checkpoint-friendly. The protocol provides:
- explicit coordination boundaries
- immutable execution facts
- reproducible transition history
- implementation independence

### 1.3 Why Workers Never Communicate Directly
Workers never communicate directly because direct worker-to-worker signaling creates hidden control paths and nondeterministic behavior. All coordination is mediated through protocol messages handled by Controller.

### 1.4 Why Controller Owns Execution
Controller is the only coordinator for sequencing, retries, pause/resume, cancellation, replanning handoff, and completion decisions. This ensures one control authority per execution.

The Controller is the sole authority responsible for execution state transitions. Workers emit facts and requests, but only the Controller advances execution.

### 1.5 Why Protocol Messages Are Immutable
Immutable protocol messages preserve auditability and replay correctness. Once emitted, a message is historical fact and may not be edited.

## 2. Protocol Vocabulary

### 2.0 Common Event Metadata
Every event in CEP-001 shares a common protocol identity. This identity is conceptual and does not define payload structure.

Common event metadata includes:
- Event ID
- Execution ID
- Event Type
- Protocol Version
- Timestamp
- Correlation ID, or an equivalent execution correlation identifier

### 2.1 Command Semantics
Commands request work. Commands are intent messages and are not historical facts.

For each command, the issuer requests the command and the executor performs the work.

- CreatePlan: Issuer Controller; Executor Planner. Request Planner to produce an execution plan for a new or resumed execution.
- ExecuteStep: Issuer Controller; Executor Brain. Request Brain to execute exactly one selected step.
- RunTool: Issuer Controller; Executor Tool. Request Tool worker to perform one deterministic operation.
- GenerateSummary: Issuer Controller; Executor Summary. Request Summary worker to generate final execution summary from execution facts.
- PauseExecution: Issuer Controller; Executor Controller. Request Controller to pause active execution at a protocol-safe boundary.
- ResumeExecution: Issuer Controller; Executor Controller. Request Controller to continue an execution from checkpointed position.
- CancelExecution: Issuer Controller; Executor Controller. Request Controller to terminate execution.
- RetryStep: Issuer Controller; Executor Controller. Request Controller to re-attempt current step within retry policy.

### 2.2 Event Semantics
Events describe facts that already happened. Events are append-only historical facts.

CEP-001 has two logical event categories:
- Runtime Events
- Domain Events

All protocol events are replayable in CEP-001. Future protocol versions may introduce operational-only events.

#### Runtime Events

Runtime Events describe execution lifecycle facts.

#### Domain Events

Domain Events describe plan, step, tool, and summary facts.

#### ExecutionStarted
- Purpose: Marks the beginning of an execution instance.
- Category: Runtime Event.
- Producer: Controller.
- Consumer: Planner, Brain, Summary, observers.
- When it occurs: After execution request accepted and runtime initialized.
- Expected outcome: Execution has a valid runtime identity and can accept CreatePlan.

#### PlanCreated
- Purpose: Confirms initial plan creation.
- Category: Domain Event.
- Producer: Planner.
- Consumer: Controller, Brain, Summary, observers.
- When it occurs: After CreatePlan succeeds for first plan revision.
- Expected outcome: Controller can begin step scheduling.

#### PlanRevised
- Purpose: Confirms plan revision after replanning.
- Category: Domain Event.
- Producer: Planner.
- Consumer: Controller, Brain, Summary, observers.
- When it occurs: After replanning request accepted and revised plan produced.
- Expected outcome: Controller resumes with new active plan revision while preserving completed work.

#### StepStarted
- Purpose: Records start of a specific step attempt.
- Category: Domain Event.
- Producer: Controller.
- Consumer: Brain, Summary, observers.
- When it occurs: Immediately before ExecuteStep dispatch.
- Expected outcome: One step attempt is active.

#### ToolRequested
- Purpose: Records that Brain requires tool execution for current step.
- Category: Domain Event.
- Producer: Brain.
- Consumer: Controller, Tool, Summary, observers.
- When it occurs: During ExecuteStep when tool call is required.
- Expected outcome: Controller may dispatch RunTool.

#### ToolStarted
- Purpose: Records start of tool operation.
- Category: Domain Event.
- Producer: Controller.
- Consumer: Tool, Summary, observers.
- When it occurs: Immediately before RunTool dispatch.
- Expected outcome: One deterministic tool operation is active.

#### ToolCompleted
- Purpose: Records successful tool completion.
- Category: Domain Event.
- Producer: Tool.
- Consumer: Brain, Controller, Summary, observers.
- When it occurs: Tool operation returns success.
- Expected outcome: Brain can continue step validation.

#### ToolFailed
- Purpose: Records tool failure fact.
- Category: Domain Event.
- Producer: Tool.
- Consumer: Brain, Controller, Summary, observers.
- When it occurs: Tool operation returns failure.
- Expected outcome: Controller evaluates retry, step failure, or replanning path.

#### StepCompleted
- Purpose: Records successful completion of one step.
- Category: Domain Event.
- Producer: Brain.
- Consumer: Controller, Summary, observers.
- When it occurs: Brain validates step success.
- Expected outcome: Controller advances to next step or completes execution.

#### StepFailed
- Purpose: Records step failure after evaluation.
- Category: Domain Event.
- Producer: Brain.
- Consumer: Controller, Summary, observers.
- When it occurs: Brain determines current step cannot be completed in current attempt.
- Expected outcome: Controller decides retry, replan, or fail execution.

#### ExecutionPaused
- Purpose: Records execution pause.
- Category: Runtime Event.
- Producer: Controller.
- Consumer: Planner, Brain, Summary, observers.
- When it occurs: Pause requested or policy-triggered pause accepted.
- Expected outcome: No new ExecuteStep or RunTool commands until resume.

#### ExecutionResumed
- Purpose: Records resumed execution after pause.
- Category: Runtime Event.
- Producer: Controller.
- Consumer: Planner, Brain, Summary, observers.
- When it occurs: ResumeExecution accepted and checkpoint restored.
- Expected outcome: Controller may continue dispatch from resume cursor.

#### ReplanRequested
- Purpose: Records need for plan revision.
- Category: Runtime Event.
- Producer: Brain.
- Consumer: Controller, Planner, Summary, observers.
- When it occurs: Brain determines current plan cannot safely complete remaining intent.
- Expected outcome: Controller evaluates and may issue CreatePlan in replan mode.

#### ExecutionCheckpointed
- Purpose: Records checkpoint commit.
- Category: Runtime Event.
- Producer: Controller.
- Consumer: Controller, Summary, observers.
- When it occurs: After each meaningful transition commit.
- Expected outcome: Execution can be resumed without rerunning completed work.

#### ExecutionCompleted
- Purpose: Records successful terminal completion.
- Category: Runtime Event.
- Producer: Controller.
- Consumer: Summary, observers.
- When it occurs: All required steps completed successfully.
- Expected outcome: GenerateSummary may be issued.

#### ExecutionCancelled
- Purpose: Records terminal cancellation.
- Category: Runtime Event.
- Producer: Controller.
- Consumer: Summary, observers.
- When it occurs: CancelExecution accepted.
- Expected outcome: No further execution commands allowed.

#### SummaryGenerated
- Purpose: Records completion of final summary generation.
- Category: Domain Event.
- Producer: Summary.
- Consumer: Controller, observers.
- When it occurs: GenerateSummary succeeds.
- Expected outcome: Execution report is finalized.

## 3. Global Protocol Rules

### 3.1 Execution Authority

- Only Controller advances execution.
- Workers never coordinate directly.
- Workers produce requests and facts.
- Controller produces execution decisions.

### 3.2 Execution Invariants

- Completed execution facts are immutable.
- Replanning never changes completed work.
- Events represent historical facts.
- Commands represent execution intent.
- Commands never mutate runtime state by themselves.
- Events are immutable facts.
- Events are append-only.
- Commands are not replayed.
- All protocol events are replayable in CEP-001.
- Completed step outcomes are immutable once recorded.
- Replanning may change future steps but never retroactively edits completed facts.
