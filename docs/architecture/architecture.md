## CortexNode Architecture

### Planner lifecycle

The canonical planning flow is:

`Controller -> PlanningRequest -> PlannerService -> PlannerProvider -> PlannerResult -> Controller`

The graph enters through Controller. Planner runs only while the durable cursor and
pending `PlanningRequest` authorize it, and Controller consumes each bound result once.
Planner proposes; it never mutates protocol state, executes tools, renders a user
answer, retries itself, or selects the next graph transition.

`PlanningRequest.operation` is either `CREATE` or `REVISE`. Provider output is the
strict structured P3 contract with exactly one result:

- `PLAN_PROPOSED`: a structured candidate plan. Controller accepts CREATE candidates
  and sends REVISE candidates through P4 reconciliation.
- `NO_PLAN_REQUIRED`: Controller skips Brain/tools and routes to Finalizer; Planner
  supplies context, not the user-facing answer.
- `NEEDS_INPUT`: Controller consumes the request/result and checkpoints an explicit
  clarification pause. A later user clarification starts a new Controller-authorized
  planning episode with a new request identity.
- `PLANNING_FAILED`: typed `INVALID_OUTPUT`, `PROVIDER_FAILURE`, or `UNPLANNABLE`.
  Controller owns the bounded retry/terminal policy.

Every `PlannerResult` is bound to the exact pending request ID. Numbered or free-form
prose plans are unsupported and normalize to `INVALID_OUTPUT`.

### Revision handling

REVISE requests are bound to the accepted execution, plan ID, and base revision. P4
deterministically preserves completed work and its provenance, permits replacement of
failed/interrupted work, rejects stale/mismatched or structurally ineffective
proposals, and commits an accepted revision only after reconciliation succeeds.

P4.1A supplies Planner with a bounded deterministic projection of observable tool
actions. Raw tool history remains durable audit/provenance data and is not converted
into semantic facts. Semantic revision novelty and cross-tool strategy equivalence are
not implemented.

### Conversation and checkpoint semantics

`context.user_request` is the current objective. `recent_history` contains prior user
messages and accepted Finalizer answers only; Brain, tool, and Controller transport are
excluded. A clarification is a typed current input, not tool evidence.

Pending requests, Planner results awaiting Controller consumption, planning attempt
budgets, and clarification pauses are durable. Resume re-enters the same Controller-
owned state transition and cannot authorize Planner or consume a result twice.

### Execution ownership

- Controller owns plans once accepted, lifecycle transitions, retries, checkpointing,
  reconciliation, and finalization dispatch.
- Planner owns only proposal generation.
- Brain executes one active step and emits typed step/tool/replan outcomes.
- Tool Runtime executes authorized `ToolRequest` values and returns `ToolResult`.
- Finalizer owns the final user-facing answer.

The runtime graph remains Controller-first:

`START -> Controller -> Planner (when authorized) -> Controller -> Brain / Planner / Finalizer / pause / terminal`
