# CIS-005 Worker Runtime

- Specification Family: CortexNode Implementation Specification (CIS)
- Document ID: CIS-005
- Version: 1.0
- Status: Review Candidate
- Layer: Layer 3 (Implementation Specification)
- Series Index: docs/implementation/README.md

## 1. Purpose
CIS-005 describes how CortexNode currently realizes runtime workers defined conceptually by CEP-004.

CIS-005 is an implementation mapping document.

Protocol authority remains defined by CEP.

CIS-005 does not redefine protocol worker semantics.

Runtime implementation may evolve while preserving CEP-conformant observable behavior.

If CIS wording and CEP wording diverge, CEP is authoritative.

## 2. Scope
Included:
- runtime worker realization
- supporting runtime services relevant to worker execution
- worker lifecycle realization
- worker interaction realization
- runtime ownership boundaries
- worker execution model realization
- current runtime module mapping

Excluded:
- protocol semantics
- execution-graph topology specification
- checkpoint implementation details
- replay implementation details
- runtime state model semantics
- infrastructure technology implementation details

## 3. Runtime Worker Philosophy
Workers are runtime execution roles that realize protocol responsibilities through implementation components.

Workers consume controller-governed runtime context.

Workers produce role-scoped outputs.

Workers do not coordinate execution directly with one another.

Controller Authority remains the sole orchestration authority for continuation and transition progression.

Workers are replaceable Runtime Modules when observable protocol behavior remains unchanged.

## 4. Runtime Worker Classification
Runtime worker realization is separated into Runtime Roles and Supporting Runtime Services.

### 4.1 Runtime Roles
- Controller
- Planner
- Brain
- Tool Runtime
- Summary

### 4.2 Supporting Runtime Services
- Context Assembly Service
- Routing Service
- State Machine Service
- Tool Capture Service
- Checkpoint Manager Service
- Runtime Runner Service

Classification rule:
- Runtime Services support worker execution and worker coordination.
- Runtime Services are not workers.

## 5. Worker Realizations
This section describes runtime realization responsibilities only.

### 5.1 Controller Runtime Role
Purpose:
- realize orchestration authority for legal transition selection and continuation control.

Responsibilities:
- evaluate role outputs under legality constraints
- select legal next continuation action
- govern accepted runtime state progression
- enforce worker boundary rules

Canonical Owner:
- Controller Runtime Role.

Runtime Inputs:
- Protocol-visible State
- accepted role outputs
- transition decision context

Runtime Outputs:
- legal continuation decision
- controller-governed state progression outcomes

Current Runtime Module Mapping:
- core/graph_runner.py
- core/graph_state_machine.py
- core/graph_routing.py

Interaction Boundaries:
- dispatches or enables worker execution through controller-governed flow
- receives worker outputs through runtime state and event flow

Authority Limitations:
- must remain inside CEP-defined authority boundaries
- must not redefine protocol semantics

### 5.2 Planner Runtime Role
Purpose:
- produce planning outputs and route-oriented execution posture for controller evaluation.

Responsibilities:
- generate plan text and route metadata
- provide planning candidate outputs for controller-governed continuation

Canonical Owner:
- Planner Runtime Role.

Runtime Inputs:
- controller-governed context slice
- user request context
- context assembly outputs when available

Runtime Outputs:
- planning output
- route and planning metadata

Current Runtime Module Mapping:
- core/graph_planner.py

Interaction Boundaries:
- does not dispatch execution directly
- outputs are consumed by controller-governed continuation logic

Authority Limitations:
- does not own execution progression
- does not own transition legality
- does not execute tool work directly

### 5.3 Brain Runtime Role
Purpose:
- produce step-scoped reasoning outcomes and actionable execution outputs for controller evaluation.

Responsibilities:
- generate step-level outcomes
- produce tool-request outputs when action is required
- produce terminal/direct answer outputs when action is not required

Canonical Owner:
- Brain Runtime Role.

Runtime Inputs:
- controller-governed context
- active plan and step context
- tool capture outputs when present

Runtime Outputs:
- step-level reasoning outputs
- optional executable tool request outputs
- optional direct terminal response outputs

Current Runtime Module Mapping:
- core/graph_brain.py

Interaction Boundaries:
- receives normalized context and prior outputs via runtime state
- never invokes workers directly

Authority Limitations:
- does not own execution lifecycle
- does not own transition legality
- does not own checkpoint or replay authority

### 5.4 Tool Runtime Role
Purpose:
- perform deterministic tool execution requested through controller-governed flow.

Responsibilities:
- execute deterministic runtime operations
- return operation outcomes for capture and controller evaluation

Canonical Owner:
- Tool Runtime Role.

Runtime Inputs:
- executable tool request output from controller-governed flow
- controller-governed runtime context as required

Runtime Outputs:
- tool execution outcomes
- deterministic operation results

Current Runtime Module Mapping:
- tools/*.py
- core/graph_capture.py

Interaction Boundaries:
- executes only when routed through controller-governed continuation
- emits results into capture/state flow rather than direct worker-to-worker calls

Authority Limitations:
- does not own retries as policy authority
- does not own transition legality
- does not own lifecycle continuation

### 5.5 Summary Runtime Role
Purpose:
- produce terminal reporting output from accepted protocol-visible runtime facts.

Responsibilities:
- generate summary/terminal reporting artifacts
- reflect accepted execution progression state for terminal output

Canonical Owner:
- Summary Runtime Role.

Runtime Inputs:
- accepted protocol-visible history/context
- controller-governed terminal context

Runtime Outputs:
- terminal summary output

Current Runtime Module Mapping:
- core/graph_summarize.py

Interaction Boundaries:
- executes on controller-governed terminal path
- does not alter accepted history

Authority Limitations:
- does not own execution progression
- does not own transition legality
- does not mutate accepted runtime history

## 6. Worker Lifecycle
Worker lifecycle is realized as controller-governed activation and completion cycles.

Typical lifecycle:
- worker activation under controller-governed continuation
- context preparation through supporting runtime services
- worker execution in role scope
- role-scoped output production
- controller evaluation of output and legality
- worker completion and next continuation selection

Role lifecycle notes:
- Controller: continuously governs lifecycle progression across worker turns.
- Planner: activates for planning/replanning phases and completes after plan output.
- Brain: activates for step-level reasoning turns and completes after output emission.
- Tool Runtime: activates only when executable tool request is present and completes after deterministic result emission.
- Summary: activates on controller-governed terminal path and completes terminal reporting.

Lifecycle property:
- Workers SHOULD remain stateless execution participants.
- Execution continuity SHOULD be realized through
- Controller-governed Runtime Objects rather than worker-local mutable state.
- Worker-local transient state MAY exist during execution but MUST NOT become authoritative runtime state.

## 7. Worker Interaction Model
Worker interaction is realized through controller-governed dispatch and runtime state mediation.

Interaction model:
- Controller dispatches or enables worker progression.
- Workers do not invoke one another directly.
- Worker communication occurs through controller-governed Runtime Objects.
- Interaction flow uses accepted runtime objects and legal continuation context.

Interaction principles:
- no direct worker-to-worker coordination authority
- no worker-owned continuation decisions
- interaction legality evaluated under Controller Authority
- service-assisted interaction remains subordinate to worker authority boundaries

## 8. Runtime Ownership
Ownership follows CIS ownership boundaries and CEP authority constraints.

Controller owns:
- execution authority
- accepted state mutation acceptance
- continuation and transition decisions

Workers own:
- role-scoped reasoning/execution behavior
- role-scoped output production

Workers never own:
- execution lifecycle authority
- transition legality authority
- replay authority
- checkpoint acceptance authority

Supporting Runtime Services:
- provide orchestration support and data preparation
- do not acquire worker authority
- operate under Controller Authority where applicable

## 9. Worker Invariants
The following invariants apply to current runtime realization.

- Workers never coordinate directly with one another.
- Controller remains sole orchestration authority.
- Workers never mutate accepted runtime state directly.
- Workers consume only controller-approved context slices.
- Workers produce role-scoped outputs for controller evaluation.
- Runtime Services never acquire worker authority.
- Worker replacement must preserve observable protocol behavior.
- Worker realization changes must not alter CEP-defined authority boundaries.

## 10. Current Runtime Module Mapping
This section maps logical worker/service responsibilities to current Runtime Modules.

Controller Runtime Role:
- core/graph_runner.py
- core/graph_state_machine.py
- core/graph_routing.py

Planner Runtime Role:
- core/graph_planner.py

Brain Runtime Role:
- core/graph_brain.py

Tool Runtime Role:
- tools/*.py
- core/graph_capture.py

Summary Runtime Role:
- core/graph_summarize.py

Context Assembly Service:
- core/graph_context.py
- core/graph_messages.py

Routing Service:
- core/graph_routing.py

State Machine Service:
- core/graph_state_machine.py

Tool Capture Service:
- core/graph_capture.py

Checkpoint Manager Service:
- main.py

Runtime Runner Service:
- core/graph_runner.py

Mapping note:
- mappings describe current implementation realization only.
- one logical responsibility may span multiple Runtime Modules.
- module organization may evolve without changing CIS semantics.

## 11. Runtime Evolution
Worker implementation may evolve while preserving protocol-conformant observable behavior.

Allowed evolution patterns:
- worker implementation refactoring
- worker role split into finer-grained runtime components
- worker role merge into consolidated runtime components
- distributed worker realization
- asynchronous worker realization
- Runtime Module migration or reorganization

Evolution constraints:
- observable protocol behavior MUST remain unchanged.
- CEP authority boundaries MUST remain unchanged.
- Controller Authority MUST remain sole orchestration authority.
- supporting services MUST remain non-worker authority participants.

## 12. Compatibility
CIS-005 realizes runtime worker implementation consistent with:
- CEP-001 Runtime Protocol
- CEP-002 Execution Lifecycle
- CEP-003 State and Checkpoint
- CEP-004 Worker Contracts
- CEP-005 Protocol Data Contracts
- CEP-006 Protocol Compliance and Validation

CIS-005 realizes runtime worker implementation and never overrides CEP semantics.
