# CIS-001 Execution Mapping

- Specification Family: CortexNode Implementation Specification (CIS)
- Document ID: CIS-001
- Version: 1.0
- Status: Review Candidate
- Layer: Layer 3 (Implementation Specification)
- Series Index: docs/implementation/README.md

## 1. Purpose
CIS documents map CortexNode Execution Protocol concepts into the CortexNode runtime.

CIS-001 explains how CortexNode realizes CEP-001 through CEP-006 in the current workflow engine while preserving protocol authority.

CIS-001 is an implementation mapping document. It does not redefine protocol behavior.

### 1.1 Mapping Principles
- CIS maps protocol concepts into runtime realizations.
- CIS may describe implementation optimizations provided observable protocol behavior remains CEP-compliant.
- CEP documents remain normative for protocol semantics.
- Runtime implementation may evolve while protocol semantics remain unchanged.
- If implementation wording and protocol wording diverge, CEP is authoritative.
- A runtime component MAY realize multiple CEP concepts.
- A CEP concept MAY be realized by multiple cooperating runtime modules.
- These mappings do not change protocol ownership.

## 2. Scope
Included:
- LangGraph mapping
- runtime responsibilities
- node ownership
- state ownership
- command and event flow realization
- module responsibilities

Excluded:
- protocol semantics defined in CEP documents
- infrastructure implementation details
- Redis schema
- MCP
- SCADA
- LLM prompt design

## 3. Runtime Architecture Overview
CortexNode runtime is organized around the protocol roles defined by CEP and implemented through coordinated graph nodes and runtime helpers.

Logical Runtime Roles:
- Controller: runtime orchestration authority for transition selection, legality checks, and progression decisions
- Planner: plan generation and plan revision producer
- Brain: step-level decision producer (step completion, failure, tool request, replan request)
- Tool Runtime: deterministic tool execution surface
- Summary: terminal reporting from protocol-visible execution facts

Supporting Runtime Services:
- Execution State Service: shared runtime state representation aligned with CEP state semantics
- Event History Service: ordered protocol-visible history used for progression, replay reasoning, and summary inputs
- Checkpoint Manager Service: restore and checkpoint commit responsibility for resumable execution
- Context Assembly Service: protocol-visible context preparation for Planner and Brain

Cooperation model:
- Controller remains the sole execution coordinator.
- Planner, Brain, Tool Runtime, and Summary operate as Runtime Roles.
- Supporting Runtime Services provide state, history, checkpoint, and context capabilities.
- Runtime orchestration preserves CEP authority boundaries and transition legality.

## 4. Protocol to Runtime Mapping

| CEP Concept | Runtime Role/Service | Current Realization | Responsibility |
| --- | --- | --- | --- |
| ExecutionIdentity | Run lifecycle coordinator | core/graph_runner.py | Assign and propagate run identity for each execution session |
| ExecutionState | Execution State Service | core/state.py | Represent protocol-visible runtime state across runtime node transitions |
| ExecutionPlan | Planner output in state | core/graph_planner.py | Produce plan text and routing metadata for execution |
| ExecutionStep | Step progression model | core/graph_brain.py + core/graph_state_machine.py | Realize one-step-at-a-time execution decisions |
| ExecutionContext | Context Assembly Service | core/graph_context.py + core/graph_messages.py | Provide protocol-visible context slices for Planner and Brain |
| Controller | Controller Runtime Role | core/graph_state_machine.py + core/graph_routing.py + core/graph_runner.py | Decide legal next action and enforce execution flow |
| Planner | Planner Runtime Role | core/graph_planner.py | Produce plan and routing outputs |
| Brain | Brain Runtime Role | core/graph_brain.py | Produce step outcomes and tool requests |
| Tool | Tool Runtime Role | tools/*.py + core/graph_capture.py | Execute deterministic operations and normalize tool outcomes |
| Summary | Summary Runtime Role | core/graph_summarize.py | Generate terminal summary from accepted execution facts |
| ExecutionCursor | Resume position representation | core/state.py + core/graph_runner.py | Resume execution from legal continuation point |
| Checkpoint | Checkpoint Manager Service | main.py (load_session/save_session) | Persist and restore resumable runtime state |
| Replay | Deterministic reconstruction path | core/graph_runner.py + core/graph_state_machine.py | Reconstruct and validate execution progression from recorded history |

Mapping note:
- Each CEP concept above maps to one primary runtime realization for ownership clarity.
- Supporting modules may participate, but they do not redefine ownership.
- Implementation modules MAY change without affecting this conceptual mapping.

## 5. LangGraph Mapping
CortexNode maps CEP workers to LangGraph nodes and edges while keeping Controller authority in decision helpers and routing logic.

LangGraph runtime nodes are execution containers for Runtime Roles and Runtime Services. They are not protocol actors by themselves.

Controller protocol authority is independent of graph topology. Graph shape can evolve without changing protocol ownership.

Node mapping:
- Controller Runtime Node (logical): realized by routing and state-machine decision functions
- Planner Runtime Node: `planner`
- Brain Runtime Node: `brain`
- Tool Runtime Node: `tools`
- Summary Runtime Node: `summarize_memory`
- Tool Capture Runtime Node: `capture_tool_output` (runtime normalization helper)

Node-level mapping details:
- Controller Runtime Node
  - Inputs: current ExecutionState, latest protocol-visible outcomes
  - Outputs: next legal transition decision and command dispatch target
  - Ownership: transition authority and orchestration policy
  - Transition authority: only Controller may advance execution

- Planner Runtime Node
  - Inputs: user request context and route constraints
  - Outputs: plan and planning metadata
  - Ownership: planning and plan revision production
  - Transition authority: none

- Brain Runtime Node
  - Inputs: active step context and latest tool observations
  - Outputs: step outcome or tool request/replan request
  - Ownership: step-level reasoning outcome production
  - Transition authority: none

- Tool Runtime Node
  - Inputs: deterministic tool request
  - Outputs: tool result
  - Ownership: operation execution
  - Transition authority: none

- Summary Runtime Node
  - Inputs: terminal execution history and summary context
  - Outputs: execution summary
  - Ownership: reporting
  - Transition authority: none

## 6. Runtime Control Loop
Controller implements the control loop as a deterministic execution cycle:

Controller
while execution is non-terminal

    validate current state

    determine next legal transition

    dispatch command

    receive protocol outcome

    validate emitted event

    checkpoint runtime state

end

Runtime interpretation:
- Controller owns orchestration and transition legality.
- Runtime Roles remain protocol participants and do not coordinate directly.
- Loop termination follows terminal execution outcomes defined by CEP.

## 7. Runtime State Ownership

| Runtime Object | Owner | Writer | Readers |
| --- | --- | --- | --- |
| ExecutionState | Controller | Controller logic | Planner, Brain, Tool runtime, Summary, runner |
| ExecutionPlan | Controller (accepted plan) | Planner produces candidates, Controller accepts active plan | Brain, Summary, runner |
| ExecutionContext | Controller | Controller context assembly path | Planner, Brain, Summary |
| ExecutionCursor | Controller | Controller progression and resume logic | Runner, brain dispatch path |
| Completed Step Ledger | Controller | Controller progression logic | Brain, Summary, compliance/replay validation |
| Event History | Controller-governed runtime history | Runtime history append path | Planner, Brain, Summary, replay and validation paths |
| Checkpoint | Controller-governed recovery state | Session checkpoint manager | Resume and runtime startup paths |

Ownership rules:
- Controller is sole runtime authority for accepted state transitions and accepted Runtime Object mutation.
- Accepted Event History is append-only.
- Completed work remains immutable once accepted.
- Runtime checkpoints never rewrite accepted history.
- Runtime Modules in worker roles consume state and produce role-scoped outputs.
- Worker Runtime Modules never mutate accepted runtime state directly.

## 8. Repository Mapping

Conceptual repository mapping to protocol responsibilities:
- `core/`: protocol runtime orchestration, state logic, and graph composition
- `docs/protocol/`: normative protocol specifications (CEP series)
- `docs/implementation/`: implementation mappings (CIS series)
- `tools/`: deterministic tool operations used by Tool Runtime
- `tests/`: protocol and runtime conformance regression coverage
- `knowledge/`: retrieval content used by planner/brain context assembly
- `scripts/`: evaluation and quality validation utilities

Logical mapping aliases used in this document:
- protocol/: normative behavior definitions
- runtime/: execution orchestration and state progression (implemented primarily in `core/`)
- graph/: graph composition and node routing (implemented in `core/graph*.py`)
- workers/: planner/brain/summary/tool role realizations (implemented in `core/graph_*.py` and `tools/*.py`)
- state/: state representation and transition decisions (implemented in `core/state.py` and `core/graph_state_machine.py`)
- memory/: summary/session context and resume persistence (implemented in `core/graph_summarize.py` and `main.py`)

Maintainability note:
- The current repository layout is sufficient for protocol-preserving evolution.
- If future scale requires restructuring, boundaries should continue to separate Runtime Roles from Supporting Runtime Services.

## 9. Module Responsibilities
This section describes responsibilities only. The following mappings describe the current CortexNode runtime implementation.

Future runtime refactoring MAY relocate responsibilities while preserving CIS behavior.

Logical Runtime Roles and current Runtime Module realizations:
- Controller Runtime Role: logical controller boundary currently realized by `core/graph_state_machine.py`, `core/graph_routing.py`, and `core/graph_runner.py`.
- Planner Runtime Role: currently realized by `core/graph_planner.py`.
- Brain Runtime Role: currently realized by `core/graph_brain.py`.
- Tool Runtime Role: logical tool boundary currently realized by `tools/*.py` with normalization integration in `core/graph_capture.py`.
- Summary Runtime Role: logical summary boundary currently realized by `core/graph_summarize.py`.

Supporting Runtime Services and current Runtime Module realizations:
- Execution State Service: currently realized by `core/state.py`.
- Runtime Runner Service: currently realized by `core/graph_runner.py`.
- Context Assembly Service: currently realized by `core/graph_context.py` and `core/graph_messages.py`.
- Checkpoint Manager Service: currently realized by `main.py` session load/save responsibilities.

## 10. Runtime Invariants
- Controller remains the sole coordinator.
- Accepted history remains append-only.
- Runtime checkpoints never rewrite accepted history.
- Accepted completed work remains immutable.
- Runtime preserves CEP semantics during orchestration.
- Runtime authority always follows CEP-defined worker ownership.

## 11. Implementation Principles
- Controller owns orchestration.
- Runtime authority always follows CEP-defined worker ownership.
- Runtime Roles are deterministic protocol participants.
- Protocol remains authoritative.
- Runtime never bypasses protocol boundaries.
- Runtime may evolve without changing CEP semantics.
- Transition legality is explicit and centrally enforced.
- Replay and resume are first-class runtime responsibilities.

## 12. Compatibility
CIS-001 maps runtime realization to:
- CEP-001 Runtime Protocol
- CEP-002 Execution Lifecycle
- CEP-003 State and Checkpoint
- CEP-004 Worker Contracts
- CEP-005 Protocol Data Contracts
- CEP-006 Protocol Compliance and Validation


CIS realizes CEP and never overrides CEP.

If CIS and runtime implementation differ, runtime MUST be considered non-conformant until CIS or implementation is updated.

Runtime evolution may change CIS mappings without changing CEP semantics.

CIS-001 does not modify protocol semantics. It describes how CortexNode realizes those semantics in the current runtime architecture.
