# Brain LLM ownership and native binding audit

Audited 11 October 2026. Recommendation: **A — benchmark models with current binding unchanged**, using separate role experiments and fixed inputs. The current ownership boundary is sound. Binding breadth has a measurable repeated schema cost, but the source does not establish a safe, trivial binding cleanup. No production behavior was changed.

**1. Current Brain LLM ownership path**

| Stage | Current source and behavior |
| --- | --- |
| Configuration | `main.py:32`, `main.py:80`, `main.py:107`, `main.py:425`: defaults, environment/CLI settings, then `build_app(model=..., model_planner=...)`. |
| Construction | `core/graph.py:103`: default factory constructs `ChatOllama(model=model, temperature=temperature)`. `build_app`, lines 315–316, separately calls the injected factory for Planner and Brain, both at temperature 0. Brain context size, seed, and generation limits are not explicitly set here. |
| Injection | `core/graph.py:324` -> `core/graph_nodes.py:18` -> `core/graph_brain.py:13`. The composition root passes the base Brain model and enabled executable tools. |
| Storage | The Brain node closure retains `BrainService`; `BrainService.provider` stores its provider (`core/brain.py:32`); `LangChainBrainProvider.brain_llm` stores the injected base model (`core/brain_provider.py:346`). Brain does not construct the model. |
| Invocation semantics | `BrainService.run`, `core/brain.py:39`, requires an active step, builds execution messages, and invokes the provider. Finalization and missing-step paths return without a model call. |
| Runtime binding | `LangChainBrainProvider.generate`, `core/brain_provider.py:364–378`, resolves this input's accepted ceiling and creates local `llm = self.brain_llm.bind_tools(...)`. |
| Invocation | `core/brain_provider.py:378`: `llm.invoke(provider_messages)`. One native-call correction may reuse this local binding and invoke it again (`core/brain_provider.py:86–88`). |

The production portable path is Controller -> `ExecutionDriver.dispatch` (`core/runtime/execution_driver.py:156`) -> `GraphWorkerRuntimePorts.run_authorized` (`core/graph_worker_runtime.py:53`) -> Brain node -> `require_brain_authorization` -> `build_brain_input` -> service/provider. `build_brain_input` copies the accepted plan and active step from authoritative execution state (`core/protocol/bridge.py:81`).

Installed `ChatOllama.bind_tools` converts tools with `convert_to_openai_tool` and returns `super().bind(tools=...)`, a `RunnableBinding` referencing the base model. It does not overwrite the base model. There is no assignment of a bound model back to `self.brain_llm`, no shared binding cache, and no bound model in checkpoint state. The diagnostic retained the same base/provider across five calls with changing ceilings; each call created a distinct binding, the base configuration remained unchanged, and an unbound base request had no tools. Reusing a binding for a correction within one `generate` call is appropriate.

Keep model/provider/configuration construction in the composition root and authorization-sensitive binding in Brain's provider. No concrete ownership problem justifies moving construction into Brain.

**2. Current tool-binding path**

```text
production executable factories
  -> full ToolRegistry
  -> default_enabled_tools / injected tool_list_factory selection
  -> build_app's enabled registry names
  -> Controller PlanningCapabilities snapshot
  -> PlanningRequest
  -> Router execution-mode classification
  -> Planner route projection / proposal schema
  -> normalize_planner_proposal -> ExecutionPlan.available_tools
  -> Controller plan validation / revision reconciliation / acceptance
  -> authoritative active_plan -> BrainInput
  -> accepted names intersect Brain registry names
  -> registry.resolve in registration order
  -> bind executable tools + all three lifecycle schemas
  -> invoke -> normalize -> Controller authorization -> executable runtime
```

`_resolve_authorized_tools` (`core/brain_provider.py:300`) reads `active_plan.available_tools`, intersects with `registry.names`, then resolves real executable objects. It does not read `primary_tool`. Final normalization also uses the registered-and-authorized intersection (`core/brain_provider.py:412`). Schema visibility does not itself dispatch a tool.

**3. Exact source of `available_tools`**

| Boundary | Source | Exact effect |
| --- | --- | --- |
| Inventory | `tools/registry.py:85`, `:269` | 28 production definitions and actual executable factory results; inventory membership does not independently grant authority. |
| Deployment selection | `tools/registry.py:305`; `core/graph.py:59`, `:300–310` | Default selection uses `enabled_by_default`; all current 28 definitions are enabled. An injected tool factory can select fewer. `build_app` derives names from its selected executable list, not the full manifest. |
| Controller ceiling | `core/graph_nodes.py:31–36`; `core/protocol/controller.py:96` | Validate supplied names against the enabled registry, then store `PlanningCapabilities(available_tools=tuple(sorted(tools_set)))`. Default `unavailable_tools` is empty. |
| Planning authorization | `core/protocol/controller.py:1287–1335` | Put the ceiling in an identity/sequence-bound `PlanningRequest`. A continued planning episode preserves the previous request's capability snapshot; a fresh request uses Controller's configured snapshot. |
| Route filtering | `core/planner_routing.py:10`; `core/planner.py:134`, `:184–212` | Router classifies execution mode. `filter_planner_tools` removes `MUTATING_TOOLS` only for `info`. `action` retains all authorized names; it does not add capabilities beyond the ceiling. No domain or active-step filtering occurs. |
| Mutating metadata | `core/graph_constants.py:40`; `tools/registry.py:85` | Eight names are marked mutating: `write_file`, `make_directory`, `run_python`, `install_package`, `rag_refresh_index`, `execute_abap_report`, `run_comfy_workflow`, `download_comfy_output_image`. All remain in default action plans, including during an inspection step. |
| Planner contract | `core/planner_contract.py:16–45`, `:70`; `core/planner_provider.py:108` | `primary_tool` is constrained to authorized names. The model-facing proposal has no `available_tools` field and no supporting-tool list. The model does not choose the plan ceiling. |
| Unavailable filtering | `core/planner_validation.py:74` | `authorized_plan_tools` applies the effective `info` effect restriction and subtracts `request.capabilities.unavailable_tools`. `PlanningCapabilities` requires available/unavailable sets to be disjoint (`core/protocol/models.py:719`); subtraction is defensive for normal valid snapshots. |
| Normalization | `core/planner_normalization.py:51–70` | Set `ExecutionPlan.available_tools = tuple(sorted(authorized_plan_tools(...)))`: the entire route-authorized set, rather than the union of step primaries. |
| Controller validation | `core/protocol/controller.py:329–397`; `core/planner_validation.py:115–176` | Check subset of request ceiling, unavailable names, route restrictions, primary membership, step/dependency/lifecycle invariants; reconcile revisions, then validate again before acceptance. A valid injected proposal may have a smaller explicit ceiling; Controller does not expand it. |
| Durable acceptance | `core/protocol/controller.py:1656`; `core/protocol/models.py:170`, `:982` | Store the accepted plan in protocol-visible execution state. `available_tools` is durable authorization for that plan revision. |
| Brain projection/binding | `core/protocol/bridge.py:81`; `core/brain_provider.py:300`, `:376` | Project the accepted plan; intersect with executable registration; append lifecycle schemas; bind per invocation. |

There is **no runtime health or endpoint availability filter** in this path. Comfy server availability, the `llava` model, a Git repository, or live SAP connectivity do not remove schemas. SAP/SCADA placeholders are still registered. `read_knowledge_file` is omitted by registry construction only when no knowledge root is supplied; the default `build_app` supplies a resolved root. A direct no-knowledge registry diagnostic had 27 executables. `DOMAIN_TOOL_MAP` exists in `core/graph_constants.py:33` but has no production consumer in this authorization path; its presence must not be mistaken for domain filtering.

**4. Representative bound counts**

The offline diagnostic passed scripted proposals through the real Planner normalization and Controller acceptance paths, then used the production Brain service/provider and real ChatOllama binding/serialization. Only routing/proposal responses and the HTTP transport were scripted; executable schemas were not approximated.

| Case | Executable schemas | Lifecycle schemas | Total |
| --- | ---: | ---: | ---: |
| Default `info`, configuration discovery/read step | 20 | 3 | 23 |
| Default `action`, initial configuration inspection step | 28 | 3 | 31 |
| Injected filesystem ceiling `{find_files,list_files,read_file,write_file}`, `info` | 3 | 3 | 6 |
| Injected ceiling `{read_file}`, `action` | 1 | 3 | 4 |
| Default `action` with one native-call correction | 28 per request | 3 per request | 31 per request; two model requests |

The narrower injected cases demonstrate existing dependency-injection/authorization behavior; they are not proposed binding policies. Default `info` names are `agent_info`, `current_time`, `describe_image`, `find_files`, `get_comfy_history`, `get_report_data`, `git_changed_files`, `git_diff`, `git_log`, `git_show`, `git_status`, `list_files`, `lookup_material`, `query_abap_table`, `rag_search`, `read_file`, `read_knowledge_file`, `scada_status`, `search_text`, `token_usage`. Default `action` additionally includes the eight mutating names above.

**5. Schema character/token measurements**

Measurements use compact JSON, `json.dumps(value, ensure_ascii=False, separators=(",", ":"))`, on the tools array. All measured schema characters were ASCII, so character and UTF-8 byte counts coincide. “Bound” means the actual `RunnableBinding.kwargs["tools"]`; “request” means the same schemas after the installed Ollama `ChatRequest` model serializes them. The latter matches `ollama.Client.chat` before HTTP serialization. The SDK discards some validation keywords and defaults, accounting for the smaller request values. Argument validation remains local.

| Schema set | Bound characters | Request characters | Rough request tokens, ceil(chars/4) |
| --- | ---: | ---: | ---: |
| `info` executables, 20 | 7,846 | 7,306 | 1,827 |
| `action` executables, 28 | 12,307 | 11,196 | 2,799 |
| Lifecycle actions, 3 | 2,318 | 2,163 | 541 |
| Default `info` total, 23 | 10,163 | 9,468 | 2,367 |
| Default `action` total, 31 | 14,624 | 13,358 | 3,340 |
| Injected filesystem `info` total, 6 | 3,701 | 3,388 | 847 |
| Injected singleton `action` total, 4 | 2,869 | 2,639 | 660 |

Each row is a separately serialized array: combining two arrays removes one bracket/comma overhead character, so component character totals exceed the combined array by one. Rounded token estimates need not add exactly.

No model tokenizer is installed (`tiktoken`, `transformers`, and `tokenizers` are absent). These token values are **size proxies, not measured model prompt tokens**. A chars/5 to chars/3 sensitivity range is 1,894–3,156 for default info and 2,672–4,453 for default action. Ollama's model-specific chat template can render tool definitions differently from JSON; framing and other messages are excluded. No live inference was run to estimate an exact marginal token cost or cache benefit.

Lifecycle schemas separately, as individual objects excluding array overhead:

| Schema | Bound characters | Request characters |
| --- | ---: | ---: |
| `brain_step_completed` | 1,504 | 1,435 |
| `brain_step_failed` | 335 | 292 |
| `brain_replan_requested` | 475 | 432 |

Lifecycle schemas account for approximately 23% of info schema characters and 16% of action schema characters. They remain material even with a smaller executable ceiling.

At unchanged breadth, 50 model invocations carry 473,400 info or 667,900 action schema characters in repeated tools arrays: roughly 118k or 167k tokens under the same chars/4 proxy. The executable-only proxy is about 91k or 140k; lifecycle arrays contribute about 27k. These are workload projections, not measured tokens or compute time.

The existing raw log contains 22 Brain model calls across two executions of 11 calls each, all labeled `qwen3.8:27b`: 166,654 input tokens and 12,530 output tokens; median input per call is 4,409. Logged message content has a median of 6,624 characters. Those records omit bound schemas, accepted ceilings, and schema hashes, so they cannot establish what fraction of historical tokens came from today's tool inventory. Repeated schemas are a plausible material contributor, alongside the repeatedly projected request, evidence/history, and correction context (`core/brain.py:67`, `:297`). There is no evidence here that schemas explain all or most cumulative usage.

**6. Whether schemas are resent per invocation**

Yes. Every `generate` creates a binding from the base and invokes it with its full tools list. Installed `ChatOllama._chat_params` includes `tools` in each request. Installed `ollama.Client.chat` serializes a complete `ChatRequest` with tools on each `/api/chat` call (`ollama/_client.py:387–402`). There is no application-level tool registration handle, delta transmission, or persistent bound-model session.

Native-call corrections reuse the local binding and resend the identical tools list along with the augmented messages. The diagnostic verified this through real transport serialization. Server-side prompt/KV caching is not established by this source audit; resending does not prove repeated uncached computation or a specific latency/billing cost.

**7. Active-step relevance and architectural smells**

The current contract provides a primary hint, step title/description, completion requirement, evidence, and a plan ceiling. It does not declare a mechanically complete supporting-tool set. The Brain prompt explicitly says `primary_tool is a hint, not an allowlist` (`core/graph_constants.py:15`). Brain's active brief intentionally omits plan-wide state while tool visibility remains plan-wide (`core/brain.py:319`).

These are manual audit examples under stated evidence, not a proposed selector:

| Active step | Primary | Clearly plausible support | Unrelated under the example's evidence |
| --- | --- | --- | --- |
| `info`: find configuration files and read them to identify settings | `find_files` (1) | `list_files`, `read_file`, `search_text` (3) | Remaining 16 info executables, 5,296 request characters as a separate array |
| `action`: initial discovery/read step before an update | `find_files` (1) | Same three (3) | Remaining 24 action executables, including mutators reserved for later work, 9,186 characters |
| `action`: locate configuration, update settings, create missing directories, verify saved text | `write_file` (1) | `find_files`, `list_files`, `read_file`, `search_text`, `make_directory` (5) | Remaining 22 action executables, 8,693 characters |

“Unrelated” is contextual: a later observed blocker, explicit user requirement, different file type, or changed step may make an additional tool useful. For the info example, unrelated names are exactly `agent_info`, `current_time`, `describe_image`, `get_comfy_history`, `get_report_data`, `git_changed_files`, `git_diff`, `git_log`, `git_show`, `git_status`, `lookup_material`, `query_abap_table`, `rag_search`, `read_knowledge_file`, `scada_status`, `token_usage`. Lifecycle actions are a separate required control surface, not unrelated executables.

Findings:

- No Brain construction/binding ownership defect or accidental long-lived bound-model state was found.
- Broad default deployment plus whole-route plan normalization exposes cross-domain capabilities and stub integrations during narrow steps. This is a measurable cost and potential model-selection distraction; it is consistent with current authorization semantics, not evidence of an authority bypass.
- Runtime availability is not reflected in schema visibility. Failed choices can therefore consume recovery calls. This audit did not test actual endpoint health.
- The unused `DOMAIN_TOOL_MAP` can mislead readers about actual filtering. Removing it would not reduce bindings and is outside this audit.
- Role configuration is coupled: `core/graph_nodes.py:40–42` injects Brain's base model into Finalizer; `:45–48` injects Planner's model into Router. Finalizer stores/invokes the unbound base (`core/finalizer_provider.py:194`, `:241`), so sharing does not leak tools, but global model changes do not isolate those roles.
- Router usage is accumulated with `worker="planner"` (`core/planner_routing.py:58`); raw logs distinguish `operation="route"` and `"plan"`. Aggregate Planner totals alone cannot benchmark those roles separately.

**8. Invariants and risks of future narrowing**

| Invariant | What currently enforces it | Dependence on plan-wide visibility / risk |
| --- | --- | --- |
| Controller authority | Controller decides worker dispatch; graph/driver validate authorization and request identity (`core/graph_authorization.py`; `core/runtime/execution_driver.py:187`). | Does not require every authorized schema to be visible. Narrowing must never replace Controller authorization. |
| Plan capability ceiling | Planner/Controller plan validation; provider intersection; Controller single-call membership (`controller.py:685`), batch membership (`:826`), runtime recheck (`execution_driver.py:198`). | A visible subset can remain within the same ceiling. Do not overwrite durable `available_tools` with a temporary presentation subset or widen authorization through registration. |
| Supporting-tool freedom | Prompt permits any authorized tool that advances the active step; Controller does not restrict calls to `primary_tool`. | Requires supporting tools to remain available for model selection. Primary-only binding would break the existing contract even though calls still pass authorization checks. |
| Heterogeneous multi-call | `core/brain_batch_policy.py`: explicit metadata permits independent synchronous read-only calls for `list_files`, `read_file`, `find_files`, up to 24 total; Controller validates the group and dispatches members serially with continuations. | Does not require every plan tool, but requires all useful batch-member tools to remain visible together. Narrowing can prevent currently valid mixed proposals. Mutators and lifecycle actions remain singleton calls. |
| Lifecycle actions | Always bind completed/failed/replan schemas; normalize into typed outcomes; Controller owns transitions and completion validation. | Independent of executable breadth. Preserve all three, including on retries; do not filter them based on the primary tool. |
| Mutating access on action routes | Manifest effect metadata, route normalization, accepted ceiling, Controller membership. | Action plans intentionally retain eight default mutators. A presentation subset could prevent authorized writes, installs, async submissions/downloads, or directory creation. Permission safety alone does not preserve capability. |
| Replan behavior | Brain replan lifecycle -> Controller REVISE request (`controller.py:748`, `:1344`); reconciliation preserves completed definitions and accepts a new ceiling (`core/planner_revision.py:84`). | Full visibility is not necessary to request replan; available alternatives influence whether replanning is needed. Rebuild visibility from the newly accepted revision. A fresh replan may use Controller's broader ceiling; do not treat an old presentation subset as the next planning ceiling. |
| Recovery after failed choices | One provider correction using bound tools; Controller step retry; tool failure redispatch or repeated-signature replan (`controller.py:530`, `:1179`). | The retry mechanics do not need full breadth, but selecting a different supporting tool does. A static subset can trap recovery or add unnecessary replans. |
| Checkpoint/resume | Durable accepted plan/step/cursor/request/continuation; batch resume checks scope, current ceiling, and request IDs (`controller.py:881`, `:899`; `core/graph_async_resume.py:19`). | Checkpoint correctness does not depend on full schema visibility. Pending authorized calls must continue independently of a new presentation subset. Any future subset calculation must be reproducible from restored accepted state and refreshed for revisions. No bound model is checkpointed. |
| Provenance | Runtime captures typed records; Controller binds completion to execution/plan/revision/step and original records (`controller.py:615`; `core/protocol/completion_identity.py`). | Does not depend on full visibility or primary equality. Supporting and mixed-tool calls remain valid evidence. Narrowing must not discard existing evidence or require it to come only from the primary tool. |

Controller authorization protects scope/capability and lifecycle facts; it does not independently decide whether each plan-authorized tool is semantically relevant to the active step. That choice currently belongs to Brain under its prompt. Narrowing would therefore change execution competence and policy, even if per-call authorization remains intact.

**9. Benchmark readiness: recommendation A**

Benchmark models now with current binding unchanged. The schema overhead is deterministic for a given accepted ceiling and registry order, making it a controlled part of the production workload. Differences in distraction tolerance, native-call compliance, supporting-tool choice, and mixed-call quality are relevant Brain model differences. Source evidence does not identify a redundant binding entry or safe deterministic step subset whose removal would preserve today's behavior. B is not justified; exact marginal-token measurements can accompany benchmarking rather than block it under C.

Use matched role experiments:

- For Brain, replay identical accepted `BrainInput` fixtures, system prompt, evidence, ceiling, schema order/hash, and tool-result fixtures across candidates. Include separate info/action, supporting-tool, mixed-call, failed-choice recovery, replan, and resumed-execution cases.
- Hold Router/Planner outputs fixed while benchmarking Brain; hold Brain outcomes/evidence fixed while benchmarking Finalizer. Benchmark Router and Planner independently through their existing providers/contracts. Do not add another runtime selector or Planner.
- Record success, tool/argument validity, supporting-call quality, corrections, model-invocation count, input/output tokens, latency, and schema size. Keep raw `route` and `plan` operations separate. Report per-invocation and cumulative usage; a model that needs extra calls resends schemas more often.
- Control context/generation limits, provider/model versions and comparable sampling settings in the benchmark harness; temperature 0 alone does not ensure identical settings or deterministic behavior across models. Distinguish cold/warm measurements where cache state affects latency. Compare quality against tokens/latency, rather than treating equal tokenizer counts as fairness.
- Do not use the current global `--model`/`--model-planner` end-to-end run as proof of role-specific superiority: those settings change paired roles. `scripts/benchmark.py:83` launches `main.py --prompt` and checks whole-app output, so it is an end-to-end baseline, not four isolated role benchmarks.

**10. Files inspected**

Composition and runtime: `main.py`; `core/graph.py`; `core/graph_nodes.py`; `core/graph_brain.py`; `core/graph_planner.py`; `core/graph_controller.py`; `core/graph_worker_runtime.py`; `core/graph_authorization.py`; `core/graph_async_resume.py`; `core/graph_constants.py`; `core/graph_node_helpers.py`; `core/runtime/execution_driver.py`; `core/runtime/controller_transition.py`; `core/runtime/portable_orchestration.py`; `core/runtime/state_propagation.py`.

Contracts/reasoning: `core/brain.py`; `core/brain_provider.py`; `core/brain_batch_policy.py`; `core/planner.py`; `core/planner_routing.py`; `core/planner_provider.py`; `core/planner_contract.py`; `core/planner_normalization.py`; `core/planner_validation.py`; `core/planner_revision.py`; `core/protocol/models.py`; `core/protocol/controller.py`; `core/protocol/bridge.py`; `core/protocol/completion_identity.py`; `core/finalizer_provider.py`; `core/debug.py`; `core/logging/live_status.py`.

Tools and supporting configuration: `tools/registry.py`; `tools/file_ops.py`; `tools/rag_ops.py`; `tools/vision_ops.py`; `tools/comfy_ops.py`; `tools/sap_ops.py`; `tools/discovery_ops.py`; `tools/sandbox_paths.py`; source searches in `tools/exec_ops.py` and `tools/info_ops.py`; real registered schemas from all ten executable factories, including `tools/git_ops.py` and `tools/scada_ops.py`; `requirements.txt`; `pytest.ini`; `.gitignore`; `scripts/benchmark.py`; `benchmarks/README.md`.

Tests: `tests/conftest.py`; `tests/test_tool_registry.py`; `tests/test_brain_outcomes.py`; `tests/test_brain_ollama_boundary.py`; inspected tests/searches for ownership, batches, planning, Controller, driver and graph adapters listed below. Existing `logs/raw_llm.jsonl` was read only to aggregate usage; its prompts/results are not copied into this report.

Installed dependency source: `langchain_ollama.ChatOllama.bind_tools` and `_chat_params`; `.venv/Lib/site-packages/ollama/_client.py`; `.venv/Lib/site-packages/ollama/_types.py`. Versions: langchain-ollama 1.1.0, langchain-core 1.3.3, ollama 0.6.2, pydantic 2.13.4.

**11. Diagnostics and tests run**

Ignored temporary artifacts remain under `.tmp/brain-binding-audit-20261011/`:

- `audit.py`: rerunnable offline helper, using real executable objects, real Planner normalization/Controller acceptance, Brain service/provider, ChatOllama binding, and Ollama serialization. No model server or executable tool is called; production logging is patched only inside the helper.
- `measurements.json`: counts, character/byte/token-proxy costs, per-schema costs, manual relevance classifications, dependency versions, base-model isolation assertions, retry schema-equality assertions, and aggregate historical usage.
- `production-bound-schemas.json`, `production-ollama-schemas.json`: actual schema snapshots for reproducibility. Pretty-print whitespace in these files is excluded from compact measurements.

Commands:

```powershell
.\.venv\Scripts\python.exe .tmp/brain-binding-audit-20261011/audit.py

.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --no-cov --basetemp=.tmp/brain-binding-audit-20261011/pytest-verified tests/test_tool_registry.py tests/test_brain_input_authority.py tests/test_brain_ollama_boundary.py tests/test_brain_mixed_batch.py tests/test_brain_cardinality_correction.py tests/test_brain_batch_policy_ownership.py tests/test_planner_routing.py tests/test_planner_boundary_hardening.py tests/test_planner_revision.py tests/test_controller_execution.py tests/test_execution_driver.py tests/test_graph.py tests/test_graph_brain_adapter.py
```

All helper assertions passed; five Brain `generate` calls captured six simulated model requests. **195 focused tests passed**; one pre-existing LangGraph pending-deprecation warning about `allowed_objects`. The first sandboxed pytest attempt failed because Windows denied access to pytest's temporary directory; the authorized rerun outside the sandbox completed successfully. No live benchmark or live model-token ablation was run. The helper's first development attempt needed a class-level rather than instance-level method patch because ChatOllama is a Pydantic model; the corrected helper completed successfully.

**12. Diff and change scope**

`git diff --check` completed with exit code 0 and no whitespace errors. Git emitted LF-to-CRLF notices for the three pre-existing modified files. The new report was separately whitespace-checked.

Only this report and ignored audit artifacts were added. Pre-existing modifications to `core/finalizer_provider.py`, `tests/test_finalizer.py`, and `tests/test_finalizer_rendering.py` were left untouched. No Planner, Router, Brain, Finalizer, Controller, executable-tool, lifecycle, checkpoint, or heterogeneous multi-call behavior was changed.
