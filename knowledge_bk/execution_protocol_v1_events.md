# CortexNode Execution Protocol v1

## Event Catalog (Core Events)

---

### 1. PlanCreated

**Purpose**

Represents acceptance of a new execution plan.

**Producer**

Planner

**Required Fields**

* execution_id
* plan_id
* plan_revision
* goal
* execution_plan
* created_at

**Allowed Next Events**

* ExecuteStep
* ExecutionFailed

---

### 2. ExecuteStep

**Purpose**

Requests execution of exactly one step from the active execution plan.

**Producer**

Controller

**Required Fields**

* execution_id
* plan_revision
* step_id
* step_index
* execution_context
* emitted_at

**Allowed Next Events**

* ToolRequested
* StepCompleted
* ReplanRequested
* ExecutionFailed

---

### 3. ToolRequested

**Purpose**

Requests deterministic execution of a single tool operation.

**Producer**

Brain

**Required Fields**

* execution_id
* step_id
* tool_name
* tool_arguments
* request_id
* emitted_at

**Allowed Next Events**

* ToolCompleted
* ExecutionFailed

---

### 4. ToolCompleted

**Purpose**

Represents completion of a deterministic tool invocation.

**Producer**

Tool

**Required Fields**

* execution_id
* step_id
* request_id
* tool_name
* tool_result
* success
* completed_at

**Allowed Next Events**

* StepCompleted
* ToolRequested
* ReplanRequested
* ExecutionFailed

---

### 5. StepCompleted

**Purpose**

Marks a single execution step as complete.

**Producer**

Brain

**Required Fields**

* execution_id
* step_id
* validation_status
* observations
* completed_at

**Allowed Next Events**

* ExecuteStep
* ExecutionCompleted
* ReplanRequested
* ExecutionFailed

---

### 6. ReplanRequested

**Purpose**

Requests revision of the active execution plan while preserving completed work.

**Producer**

Brain

**Required Fields**

* execution_id
* current_plan_revision
* current_step_id
* reason
* supporting_observations
* requested_at

**Allowed Next Events**

* PlanReplaced
* ExecutionFailed

---

### 7. PlanReplaced

**Purpose**

Publishes a new execution plan revision.

**Producer**

Planner

**Required Fields**

* execution_id
* previous_plan_revision
* new_plan_revision
* execution_plan
* preserved_completed_steps
* created_at

**Allowed Next Events**

* ExecuteStep
* ExecutionFailed

---

### 8. ExecutionCompleted

**Purpose**

Signals successful completion of the execution lifecycle.

**Producer**

Controller

**Required Fields**

* execution_id
* final_plan_revision
* completed_steps
* completion_time

**Allowed Next Events**

* None (Terminal Event)

---

### 9. ExecutionFailed

**Purpose**

Signals that execution terminated without successful completion.

**Producer**

Controller

**Required Fields**

* execution_id
* failure_stage
* failure_reason
* terminal
* occurred_at

**Allowed Next Events**

* None (Terminal Event)
