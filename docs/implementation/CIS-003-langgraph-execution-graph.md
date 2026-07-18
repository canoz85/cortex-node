# CIS-003 LangGraph Execution Graph

- Specification Family: CortexNode Implementation Specification (CIS)
- Document ID: CIS-003
- Version: 1.0
- Status: Review Candidate
- Layer: Layer 3 (Implementation Specification)
- Series Index: docs/implementation/README.md

## 1. Purpose
CIS-003 specifies how the current CortexNode runtime realizes the execution graph used to implement CEP-001 through CEP-006.

This document is implementation-oriented.

This document does not redefine protocol semantics.

Protocol behavior, legality, and authority remain defined by CEP.

## 2. Scope
Included:
- graph topology realization
- node responsibilities
- edge responsibilities
- conditional routing realization
- controller realization in graph execution
- execution cycle realization
- termination realization
- runtime helper node realization
- graph evolution principles
- current runtime module mapping

Excluded:
- protocol semantics already defined by CEP
- framework API details
- language-level implementation details
- infrastructure and persistence internals
- storage schema design

## 3. Graph Design Principles
The following principles govern implementation behavior.

### 3.1 Protocol Authority
- CEP owns protocol semantics, legal transitions, and conformance requirements.
- CIS describes how current runtime structure realizes CEP semantics.
- If implementation wording diverges from CEP semantics, CEP is authoritative.

### 3.2 Realization Boundary
- The execution graph realizes protocol behavior; it does not define protocol behavior.
- Runtime graph structure is an implementation mechanism for protocol execution.

### 3.3 Topology Independence
- Graph topology MAY evolve over time.
- Topology changes MUST preserve CEP semantics and legal transition outcomes.

### 3.4 Controller Independence From Layout
- Controller authority is a logical runtime authority.
- Controller authority is independent of any single graph node identity.
- Controller decisions MAY be realized across multiple cooperating runtime modules.

### 3.5 Helper Role Boundary
- Runtime helper nodes are implementation services.
- Runtime helper nodes are not protocol actors.
- Runtime helper nodes MUST NOT acquire protocol authority.

### 3.6 Execution Graph Principles
The execution graph SHOULD satisfy the following implementation principles:

- deterministic execution routing
- explicit transition boundaries
- observable execution progression
- checkpoint-safe execution
- replay-safe execution
- controller-mediated transitions
- replaceable graph topology

### 3.7 Runtime Lifecycle

Graph construction
↓
Runtime initialization
↓
Execution
↓
Termination
↓
Cleanup

## 4. Graph Topology Overview
Current runtime graph topology is a directed execution flow with conditional routing and an action loop.

Conceptual topology:
Entry
↓
Planner
↓
Controller Decision
↓
Brain
↓
(optional)
Tool Runtime
↓
Tool Capture
↓
Controller Decision
↓
Repeat
or
↓
Summary
↓
End

Topology purpose:
- enable one-step planning and action progression
- enforce controller-governed transition selection
- support iterative tool-assisted execution when required
- converge on terminal summary and completion

## 5. Graph Components
Graph components are divided into Runtime Role Nodes and Runtime Helper Nodes.

### 5.1 Runtime Role Nodes
Planner:
- Produces plan output and route classification metadata for the turn.
- Proposes action posture; does not own transition authority.

Brain:
- Produces step-level execution outcomes.
- Emits either executable tool requests or direct terminal answer content.
- Does not own transition authority.

Tool Runtime:
- Executes deterministic tools when requested by Brain output.
- Produces tool execution outcomes.
- Does not own transition authority.

Summary:
- Produces terminal summary/memory output from accepted runtime facts and recent execution context.
- Finalizes terminal reporting for the turn.
- Does not own transition authority.

### 5.2 Runtime Helper Nodes And Services
Tool Capture:
- Normalizes tool outputs into runtime state fields used by subsequent decision turns.
- Records tool success/failure and signature-tracking helpers.
- Acts as runtime normalization support, not protocol authority.

Context Assembly:
- Assembles role-scoped context and retrieval context used by Planner and Brain.
- Supports legal and scoped runtime inputs.

Routing:
- Realizes conditional transition selection after Brain outcomes.
- Delegates transition legality to controller decision helpers.

State Machine:
- Realizes transition decision helpers and execution guard decisions.
- Implements legal next-node selection logic used by routing.

Checkpoint Helpers:
- Realize resume/session persistence behavior outside protocol semantics.
- Preserve runtime continuity without changing protocol authority boundaries.

Runtime Role Nodes and Runtime Service Nodes operate on both Protocol-visible State and Working State as defined in CIS-002. Only Protocol-visible State contributes to protocol semantics and conformance.

## 6. Node Classification
Node and helper responsibilities are classified as follows.

### 6.1 Protocol Role Nodes
- Planner
- Brain
- Tool Runtime
- Summary

Responsibility model:
- Produce role-scoped outputs.
- Consume controller-governed state/context.
- Never self-authorize protocol progression.

### 6.2 Runtime Service Nodes
- Tool Capture
- Context Assembly
- Routing helpers
- State machine decision helpers

Responsibility model:
- Support orchestration and legality realization.
- Prepare/normalize runtime data for role execution.
- Never own protocol authority.

### 6.3 Infrastructure Nodes And Surfaces
- Graph runner stream/execution driver
- Checkpoint/session load-save helpers
- Terminal output framing and observability helpers

Responsibility model:
- Execute and observe runtime flow.
- Provide continuity and traceability.
- Must preserve controller authority boundaries.

## 7. Edge Classification
Edges are classified by protocol purpose.

Execution edges:
- Represent deterministic progression between role executions.
- Example purpose: Planner output to Brain execution.

Conditional edges:
- Select legal next target based on controller decision logic after Brain.
- Realize branch selection without changing protocol semantics.

Loop edges:
- Realize repeated action cycle when additional tool-assisted work is required.
- Example purpose: Tool Runtime -> Tool Capture -> Brain.

Terminal edges:
- Realize legal convergence to completion.
- Example purpose: Summary -> End.

Helper edges:
- Connect runtime helper functions needed for normalization, routing, and control continuity.
- Must remain subordinate to protocol authority boundaries.

## 8. Controller Realization
Controller is realized as a logical runtime authority, not as a mandatory one-node implementation. Controller authority is realized through cooperating runtime components rather than through ownership of a single execution node.

Controller realization in current runtime:
- Transition decision helpers determine legal next-node progression.
- Routing layer applies controller decision outcomes to graph transitions.
- Execution runner drives cycle progression using those decisions.

Controller authority guarantees:
- Controller logic MUST be the only authority that advances execution.
- Worker role outputs MUST be interpreted through controller decision logic before transition.
- Graph routing MUST realize controller decisions, not replace them.

## 9. Graph Execution Cycle
Current runtime realizes the following cycle for each turn. Each execution cycle MUST begin and end with a controller validation point.

1. Planner produces plan and route metadata.
2. Controller decision context is established from planner route and state.
3. Brain executes with role-scoped context.
4. Tool Runtime executes only when Brain emits executable tool calls.
5. Tool Capture normalizes tool outcomes for subsequent reasoning.
6. Controller re-evaluates legal transition after each Brain pass.
7. Cycle repeats until controller-selected terminal path is reached.

Conceptual cycle:
Planner
    ↓
Controller Decision
    ↓
Brain
    ↓
Tool Runtime
    ↓
Tool Capture
    ↓
Controller Decision
    ↓
Summary
    ↓
End

Cycle properties:
- execution remains controller-governed
- worker roles do not directly coordinate transition control
- loop continuation is conditional and legality-driven

## 10. Routing Strategy
Routing strategy is controller-governed and condition-based. Routing is a realization of controller decisions rather than an independent decision mechanism.

Conditional routing:
- Branching occurs from Brain outcomes.
- Executable tool-call presence routes to Tool Runtime.
- Non-tool terminal answer path routes to Summary.

Legal transitions:
- Only controller decision helpers determine legal next node.
- Illegal transition bypass is not a valid runtime behavior.

Termination routing:
- Termination is realized by routing to Summary and then End.
- Max-step guard may force termination path when configured limit is reached.

Replanning routing:
- Planner route metadata influences Brain execution posture and control guidance.
- Replanning behavior is realized through repeated control-loop passes.

Tool routing:
- Tool path is selected only when executable tool calls are present.
- Tool outputs re-enter control loop through Tool Capture.

Retry routing:
- Runtime decision helpers support empty-response and repeated-signature recovery behavior.
- Retry behavior remains subordinate to controller authority and legal progression.

## 11. Termination Realization
Termination is realized through controller-selected terminal routing and end-state convergence. Termination MUST preserve all accepted protocol-visible state.

Primary termination path:
- Brain output no longer requires tool execution.
- Routing selects Summary.
- Summary emits terminal reporting state.
- Graph reaches End.

Guard termination path:
- Configured execution step limit is reached.
- Runtime forces non-unbounded termination behavior.

Termination guarantees:
- Runtime MUST terminate through legal terminal routing.
- Runtime MUST NOT bypass summary/terminal realization when closing execution.

## 12. Graph Invariants
The following invariants apply to current runtime realization.

- Only controller logic advances execution progression.
- Worker roles never coordinate transition control directly.
- Graph routing never bypasses protocol legality boundaries.
- Runtime helper nodes never own protocol authority.
- Accepted execution history remains authoritative for accepted outcomes.
- Tool output normalization supports progression but does not become protocol authority.
- Topology choices remain subordinate to CEP semantics.
- Every graph execution begins from a legal protocol entry point.
- Every graph execution terminates at a legal protocol terminal state.
- Graph topology never changes Controller authority.
- Runtime helper nodes never mutate accepted protocol-visible state.

## 13. Evolution Principles
Execution graph realization may evolve while preserving protocol semantics.

Allowed evolution patterns:
- topology changes that preserve legal behavior and observable protocol outcomes
- helper node split into finer-grained helper nodes
- helper node merge into consolidated helper nodes
- routing strategy refinement for maintainability or robustness
- module relocation of responsibilities

Representative Examples:
- Planner may be split into multiple planning nodes.
- Brain may become multiple reasoning stages.
- Tool Runtime may become parallel.
- Context Assembly may become asynchronous.
- Routing helpers may be reorganized.
provided observable protocol behavior remains unchanged.

Non-negotiable constraint:
- CEP semantics and authority boundaries MUST remain unchanged.

## 14. Current Runtime Realization 
This mapping describes current realization and does not prescribe permanent file structure.

Graph composition and topology:
- core/graph.py

Execution loop driver and stream realization:
- core/graph_runner.py

Controller decision helpers and state-machine transition logic:
- core/graph_state_machine.py

Routing realization:
- core/graph_routing.py

Brain role realization:
- core/graph_brain.py

Planner role realization:
- core/graph_planner.py

Tool capture helper realization:
- core/graph_capture.py

Summary role realization:
- core/graph_summarize.py

Context assembly helper realization:
- core/graph_context.py

Checkpoint/session continuity helpers:
- main.py

Mapping note:
- One conceptual responsibility may span multiple runtime modules.
- One module may realize multiple conceptual responsibilities.
- This mapping reflects current runtime organization only.

## 15. Compatibility
CIS-003 realizes execution graph behavior for:
- CEP-001 Runtime Protocol
- CEP-002 Execution Lifecycle
- CEP-003 State and Checkpoint
- CEP-004 Worker Contracts
- CEP-005 Protocol Data Contracts
- CEP-006 Protocol Compliance and Validation

Conformance statement:
- CIS realizes CEP and never overrides CEP.
- If runtime behavior diverges from this CIS mapping while CEP semantics are unchanged, CIS or implementation must be updated to restore alignment.
- Future runtime implementations MAY replace LangGraph entirely while preserving CEP conformance.
