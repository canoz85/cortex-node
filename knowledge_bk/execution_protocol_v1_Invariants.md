# 6. Invariants

The following invariants are mandatory for every CortexNode Execution Protocol implementation. An implementation that violates any invariant is not compliant with Protocol v1.

---

## 6.1 Controller is the Only Loop Owner

The Controller exclusively owns the execution lifecycle.

**Rules**

* Only the Controller determines the next transition.
* Only the Controller decides whether execution continues, pauses, retries, replans, or terminates.
* No worker may invoke another worker directly.
* All execution requests pass through the Controller.

---

## 6.2 Brain Cannot Mutate the Plan or Reorder Steps

The Brain executes exactly one step.

**Rules**

* The Brain must never create a new execution plan.
* The Brain must never modify an existing execution plan.
* The Brain must never reorder execution steps.
* The Brain may only request replanning by emitting a `ReplanRequested` event.
* Plan changes become effective only after the Planner emits a new `PlanCreated` or `PlanReplaced` event.

---

## 6.3 One Writer Per Field

Every mutable field has exactly one authoritative owner.

**Rules**

* Each mutable field has a single writer.
* Multiple components may read a field.
* No component may modify a field owned by another component.
* Ownership is defined by the protocol, not by implementation.

**Examples**

* Planner owns execution plans.
* Controller owns execution progress.
* Brain owns reasoning results.
* Tool owns tool results.
* Summary owns the final execution summary.

---

## 6.4 Completed Steps are Immutable

Execution history is append-only.

**Rules**

* A completed step cannot be modified.
* A completed step cannot be re-executed unless explicitly invalidated by protocol policy.
* Historical tool results remain immutable.
* Historical execution events remain immutable.
* Replanning must preserve completed work.

---

## 6.5 Every Transition is Checkpointable

Every successful protocol transition produces a recoverable execution state.

**Rules**

* Every state transition must be checkpoint-safe.
* A checkpoint represents a consistent execution boundary.
* Checkpoints must not require replaying partial transitions.
* Recovery must always start from the latest completed transition.

---

## 6.6 Resume Must Not Rerun Completed Work

Resuming execution must preserve previously completed work.

**Rules**

* Resume begins from the current execution cursor.
* Completed steps remain completed.
* Completed tool executions are not repeated.
* Plan revisions preserve completed execution history.
* Recovery continues forward rather than restarting execution.

---

## 6.7 Protocol Events are Immutable

Protocol events represent facts.

**Rules**

* Events cannot be edited after publication.
* Events may only be appended.
* Event ordering must be preserved.
* Events are replayable for diagnostics and auditing.

---

## 6.8 Deterministic Tool Contract

Tools are deterministic execution units.

**Rules**

* Tools never perform planning.
* Tools never retry autonomously.
* Tools never invoke other protocol actors.
* Tools only execute the requested operation and return a structured result.

---

## 6.9 Summary is Read-Only

The Summary actor is observational.

**Rules**

* Summary consumes execution artifacts only.
* Summary cannot modify execution state.
* Summary cannot trigger replanning.
* Summary cannot influence execution flow.

---

## 6.10 Protocol Before Implementation

The protocol defines behavior.

**Rules**

* Runtime frameworks (e.g., LangGraph) implement the protocol but do not define it.
* Checkpoint implementations (e.g., Redis) persist protocol state but do not alter protocol semantics.
* Replacing orchestration, storage, or LLM providers must not change protocol behavior.
