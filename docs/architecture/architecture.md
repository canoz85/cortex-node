## CortexNode Architecture

### Production ownership

CortexController is the sole lifecycle and authorization authority. It accepts
protocol-visible facts, owns accepted execution state, selects legal transitions,
and authorizes every worker dispatch. Mechanical invocation does not transfer that
authority:

- `PortableExecutionRuntime` prepares each portable turn, evaluates completion
  coverage, reconciles revision proposals, and coordinates application-level turn
  sequencing.
- `ExecutionDriver` applies the Controller transition and dispatches exactly the
  worker authorized by that decision.
- Planner produces request-bound plan or revision proposals. Controller alone accepts
  a plan and its revision.
- Brain executes one active step and returns one typed outcome per invocation.
  Controller accepts its lifecycle meaning and binds accepted completion provenance.
- Tool Runtime executes an authorized `ToolRequest` and returns a portable
  `ToolResult`.
- Finalizer exclusively produces both `ExecutionSummary` and the final user-facing
  answer from a Controller-authorized `FinalizationRequest`.

### Planner lifecycle and revision handling

The canonical planning flow is:

`Controller -> ExecutionDriver -> PlanningRequest -> Planner -> PlannerResult -> Controller`

Planner proposes; it never mutates protocol state, executes tools, renders a user
answer, retries itself, or chooses a lifecycle transition. `PlanningRequest.operation`
is `CREATE` or `REVISE`, and every result is bound to the pending request identity.

For a revision, portable reconciliation validates the accepted execution, plan ID,
and base revision; preserves completed work and provenance; and rejects stale,
mismatched, or ineffective proposals. Controller alone commits the reconciled plan.
Deferred semantic novelty and Finalizer evidence-handoff debts are unchanged.

### Brain completion boundary

Brain returns typed tool-request, completion, failure, replan, direct-response,
continuation, invalid-output, or provider-failure outcomes. It does not run a separate
lifecycle-owning completion checker. For completion, Brain supplies a semantic
step-scoped judgment; Controller validates completion coverage and binds execution,
plan revision, step, and eligible tool-result provenance before accepting it.

The retained completion service is deterministic completion-requirement and coverage
validation. It is not the removed legacy Brain checker.

### Production topology

The Stage 5B production LangGraph topology is effectively:

```text
START -> controller -> controller -> ... -> END
```

The Controller node is a thin presentation adapter around portable runtime turns.
Planner and Brain graph functions are callable worker transports invoked by
ExecutionDriver within those turns; they are not production graph successors.
Normal executable tools use direct `ToolRuntimePort` execution through
`SerializedToolRuntimePort`, not LangGraph `ToolNode`.

An injected multi-node topology containing Planner, Brain, ToolNode, Capture, and
Summary nodes remains for test/integration compatibility. It is compatibility-only
and is not the production architecture.

### Async wake and continuation

Async wake is correlation, not authorization:

```text
poll-due wake
  -> PortableExecutionRuntime
  -> CortexController validates wait and constructs/authorizes poll ToolRequest
  -> ExecutionDriver invokes ToolRuntimePort
  -> portable ToolResult integration
  -> portable Controller turns until next wait or terminal decision
```

The continuation has no LangGraph node or successor semantics. The LangGraph async
adapter loads a snapshot, presents accumulated turn updates, and persists the result.
Provider-local telemetry and resource handoff remain adapter concerns and do not gain
protocol authority.

### Conversation and persistence boundary

`context.user_request` is the current objective. `recent_history` contains prior user
messages and accepted Finalizer answers; Brain, tool, and Controller transport are
excluded. Clarification is typed current input, not tool evidence.

LangGraph supplies current production snapshot persistence and presentation. It is
not lifecycle or worker-routing authority, and its checkpoints or traversal history
are not a CEP accepted-event journal. Production does not currently claim the
append-only event-journal, framework-neutral replay, or protocol-level atomic
checkpoint/event-position guarantees required by CEP-003 and CEP-006. Those gaps are
reserved for the separate Stage 8 decision.
