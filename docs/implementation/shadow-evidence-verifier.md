# Shadow evidence verifier

The optional observer on `ExecutionDriver.transition` runs after
`ControllerCoordinator.transition` has decided and applied its decision. Production
graph construction installs `ShadowEvidenceVerifier`; standalone drivers default
to no observer. No Controller, Planner or Brain policy is changed.

The observer schedules only `STEP_COMPLETED` inputs that produced new accepted
completion provenance for the active step. Rejected proposals, final-answer
requests, other worker outcomes and already-present bindings do not schedule a
verification. It uses the accepted binding's summary, not parsed terminal output
or a replacement Brain response.

Evidence selection reuses `completion_provenance_records` on the original
Controller input: matching execution, accepted plan/revision and active step,
preserving history order. The full evidence fingerprint and ordered tool request
IDs are checked against the accepted binding before any filtering; a mismatch is
unavailable. A verifier-only projection then keeps exact-collection source records,
records explicitly cited by request ID, error code or result message, and successful
primary-tool evidence. Successful records referenced by tool name or resource
(path, URL, artifact path or prompt ID) are also retained. Failed attempts otherwise
stay out. For older steps without a primary tool, substantive successful records
are retained conservatively, excluding uncited discovery/preparation operations
such as list_files, find_files, install_package and index refresh. Discovery is
retained when it is itself the primary step deliverable or explicitly referenced.

This deterministic projection does not infer claim-level support or rank records
by agreement with the answer: conflicting primary-tool results remain visible.
The existing binding contains no per-claim relevance links, so selection is a
conservative heuristic. It neither changes that binding nor adds an evidence
store. The prompt includes the original selected records, not generated summaries.
Dependency-step records outside the bound provenance are not silently added.

`VERIFIER_PROMPT` in `core/shadow_verifier.py` is the exact system prompt. It
asks only about material unsupported, contradicted, overstated or cross-record
misattributed claims. Input is serialized data; output is plain text, without
tools or structured-response schemas. Only the first non-empty, trimmed line is
parsed: `PASS` becomes `pass`, `SUSPICIOUS` becomes `suspicious`; anything else
becomes `unavailable`. Following lines remain raw audit text, not machine status.
Non-text content, tool calls, provider/parsing errors and timeout are unavailable.

Production uses a separate ChatOllama client with the configured Brain model and
endpoint, temperature zero, 512 generated-token limit and 45-second transport
timeout. This neither binds tools nor changes the original model. Custom
non-Ollama graph model factories are unavailable for this production verifier;
tests can inject a model factory directly into the isolated observer.

The execution thread only schedules work. A daemon audit consumer selects and
serializes evidence, then invokes the model on a separate daemon thread with a
45-second deadline. At most one provider call is in flight per verifier. If a
provider ignores its timeout, later queued audits log unavailable while that call
remains busy. Late responses cannot change the timeout verdict or execution.
Provider context does not inherit execution LiveStatus. No result is returned to
execution, awaited, fed into state, or used for retry/replan decisions. Scheduling,
model, parsing and logging failures are isolated. No shutdown join is introduced.

Audit records use the existing sanitized `log_llm_exchange` JSONL path
(`CORTEX_RAW_LLM_FILE`, otherwise `logs/raw_llm.jsonl`), with worker
`shadow_verifier`, operation `verify_accepted_completion`, execution ID, step/plan
binding, status, reason, model/provider, raw response and available usage. They do
not overwrite Brain records and are not printed into session output.

This is best-effort observation, not a durability guarantee: pending audits can
be lost when the process exits, and failed logging cannot be guaranteed to record
its own failure. A slow audit sink can accumulate pending references. Local GPU
and model-server contention may still affect performance even though execution
never waits for a verdict. No live-model accuracy claim is made by the mocked
provider tests.
