# Frozen Planner test suite

Production source and contracts were unchanged during consolidation. Tests use
scripted providers or injected results unless explicitly named live.

The scoped inventory went from 22 files / 148 functions / 210 parametrized cases
to 17 files / 109 functions / 160 cases. The complete non-live inventory is
939 → 889 cases: the initial collection found 846 cases, excluding the separate
opt-in live-memory case, with 92 blocked by a deleted capture helper import.
That import now targets the existing tool normalization boundary.
The scoped counts include seven mandatory live cases and one separate opt-in
memory adherence smoke; neither adds non-live provider calls.

Every scoped test has exactly one primary category below. Parameter instances
inherit the function's category. A scripted provider tests transport and semantic
handoff, not live model judgment.

| Invariant | Authoritative file | Representative test |
| --- | --- | --- |
| Four semantic proposal outcomes | test_planner_contract.py | test_all_semantic_variants_validate_without_failure_category |
| Invalid variants / extras / executable step schema | test_planner_contract.py | test_invalid_proposal_variant_shapes_are_rejected |
| json_schema parsing failures / missing parsed output / provider exceptions | test_planner_provider.py | test_infrastructure_failures_are_normalized_at_provider_boundary |
| Strict route transport, no raw recovery | test_planner_routing.py | test_router_does_not_reparse_raw_enum_or_json |
| Preserved route bypasses Router | test_planner_service.py | test_preserved_clarification_route_skips_router_and_plans_original_request |
| Filtered capabilities agree across prompt and normalization | test_planner_service.py | test_info_planner_prompt_and_validation_use_one_authorized_tool_set |
| CREATE / immutable authorization snapshots | test_controller_planning.py | test_initial_controller_authorizes_create_without_revision_facts |
| REVISE / completed progress / failure context | test_controller_planning.py | test_controller_authorizes_revision_with_failure_and_progress_context |
| Pause / same execution resume / repeated pause / route and capability ceiling | test_controller_planning.py | test_clarification_pause_resume_controller_lifecycle |
| Sole production request/decision constructors | test_controller_planning.py | test_planning_requests_and_controller_decisions_have_one_production_constructor_owner |
| Retry / terminal decision / accepted plan | test_controller_planning.py | test_retryable_failures_have_two_attempts_and_unplannable_does_not_retry |
| Completed work survives revision acceptance | test_planner_revision.py | test_rejection_decision_is_atomic_and_provenance_survives_acceptance_round_trip |
| Executable graph flow with real PlannerService | test_planner_runtime_flow.py | test_executable_runtime_flow_with_scripted_provider |
| Direct graph flow / single human request | test_planner_runtime_flow.py | test_create_direct_response_uses_one_human_request_and_controller_terminal_state_with_scripted_provider |
| Clarification / bounded checkpoint history replacement | test_planner_runtime_flow.py | test_checkpointed_clarification_resumes_same_authorization_without_duplicate_history_with_scripted_provider |
| Runtime revision worker observation consumption | test_controller_planning.py | test_revision_runtime_flow_with_scripted_provider |
| Opaque pause handle / canonical session projection | test_graph_runner.py | test_run_prompt_transports_injected_pause_and_execution_reference_on_reply |
| Authorized memory and bounded execution progress | test_planner_memory.py / test_planner_progress.py | test_planner_consumes_authorized_memory_snapshot_without_graph_injection |
| Accepted step semantics reach Brain | test_direct_semantic_handoff.py | test_accepted_step_semantics_reach_brain_with_scripted_provider |
| Adapter fails closed without protocol state | test_protocol_bridge.py | test_bridge_rejects_legacy_protocol_reconstruction |
| Live adherence table / real clarification | live/test_planner_stability.py / live/test_clarification_lifecycle.py | test_planner_stability_live / test_clarification_resume_live |

Raw logging is isolated by the autouse fixture in conftest.py. The production
logger also writes when CORTEX_RAW_LLM_FILE is absent, so tests point it at
os.devnull. Diagnostic tests override it with tmp_path. Live logging requires
the explicit --live-raw-llm-file option; a developer's existing raw-log setting
is never inherited. A focused-run sentinel check confirmed no appended records.

Coverage holes found: executable graph generation previously relied on an
injected PlannerResult. One scripted structured-provider flow now covers
PlannerService → accepted plan → actual tool → completion → final answer.
Provider exception tests now exercise the real provider adapter rather than
merely injecting a service exception. No additional outcome cross-products
were added.

Consolidation removed repeated schema/tool rejection tests, canned semantic
passthrough cases that implied generation, duplicate direct Controller outcomes,
CREATE/REVISE × malformed result combinations, route × memory combinations,
Brain direct × native/JSON combinations, helper-call spies, exact graph topology
assertions, and deleted AgentState field tests. The checkpoint interruption test
was retained because it exercises a durable boundary. Separate memory/progress
and revision reconciliation files remain because their invariants are distinct.

Merges/renames:

- test_planner.py → test_planner_service.py.
- test_planning_requests.py + test_planner_lifecycle.py → test_controller_planning.py.
- test_planner_debug.py → test_planner_provider.py, including strict provider failures.
- test_planner_production_flow.py + checkpoint coverage from test_graph_planning_p2.py → test_planner_runtime_flow.py.
- test_planning_context_isolation.py → test_graph_runner.py and test_controller_planning.py.
- test_protocol_controller_stage1.py → test_controller_execution.py, removing duplicate Planner lifecycle cases.
- test_graph_brain.py and test_graph_brain_stage1.py removed: stale fields/helper call counts and direct-Brain behavior already covered at BrainService / typed adapter boundaries.

Misleading names now identify scripted provider transport, injected event
projection, reconciliation, or live adherence. In particular, the canned Router
test no longer claims semantic classification, and the accepted step handoff
test no longer claims that a fake provider resolved memory.

Validation:

- Focused regression run: 244 passed, including all 92 previously uncollectable filesystem completion cases.
- Non-live collection: clean, 889 cases (including one opt-in live-memory case that skips by default).
- Full non-live run: 873 passed, 15 failed, 1 skipped. The 15 failures predate this consolidation: one stale async graph factory; one obsolete Brain brief field assertion; eight Ollama nested-schema serialization expectations; one old accepted-plan rendering expectation; one artifact fixture; two terminal escape rendering checks; one application fake missing an accepted final answer. Production was left frozen.
- Live stability: six cases passed, ten runs each (60 Planner generations).
- Live clarification: failed twice; gpt-oss:20b repeated NEEDS_INPUT after “Can”. Captured resumed input has the original request, prior clarification question, answer “Can”, preserved conversation route, and unchanged capabilities. The assertion remains unchanged; no retries were added to the test and no production semantics were changed.
- Production Python source hashes match the initial snapshot; scoped AST duplicate audit is clean.

Run deterministic tests with `python -m pytest -o addopts= -q -p no:cacheprovider --no-cov --ignore=tests/live`.
Run the general-purpose live production Planner harness (PowerShell):

```powershell
$env:CORTEX_LIVE_PLANNER = '1'
$env:CORTEX_LIVE_PLANNER_PROMPT = '20 tane Michael Jackson şarkısı yaz'
$env:CORTEX_LIVE_PLANNER_RUNS = '1'
python -m pytest -o addopts= -q -s -p no:cacheprovider --no-cov tests/live/test_planner_stability.py
```

Change only `CORTEX_LIVE_PLANNER_PROMPT` to inspect another arbitrary request.
The convenience default is `List files`. `CORTEX_LIVE_PLANNER_RUNS` defaults to
one; set it to ten for ten independent observations of the same prompt. Without
the existing `CORTEX_LIVE_PLANNER=1` opt-in the harness skips during offline tests.
Application defaults and environment settings select the production model,
workspace, knowledge directory, embedding model and RAG depth. The current
default Planner is `gpt-oss:20b`; the provider/model and generation
settings are printed. Application CLI/config-file overrides are not supplied.

Call path: application settings -> production chat/tool/RAG factories -> fresh
Controller execution -> `build_controller_input` -> `CortexController.decide`
-> Controller-built `PlanningRequest` -> `apply_controller_decision_to_state`
-> `create_planner_node` / `require_planner_authorization` -> `PlannerService.run`
-> production Router, filtering, capability projection and message assembly
-> `LangChainPlannerProvider.generate` -> authorized schema bound through
`with_structured_output(method="json_schema", include_raw=True)` -> configured
ChatOllama transport -> production normalization / `PlannerResult`.

Only Planner dispatch decisions are executed. Controller decides whether an
invalid/provider-failed attempt receives its normal bounded retry, preserving
its own feedback and route. Brain, runtime tools, Finalizer, memory saving and
the full graph/Controller runtime are never executed. Tools are constructed only
to obtain the deployment's enabled registry; normal ambient Planner retrieval
is retained. No prompt, capability definition, schema or retry policy is copied.

Each attempt and final result are printed in full, including step titles,
descriptions, tools, dependencies and messages. The normalized PlannerResult is
also saved to `planner-observations.json` under pytest's temporary test directory.
Existing `log_llm_exchange` JSONL
diagnostics supply per-attempt usage, model and `done_reason`, including length
limits. `--live-raw-llm-file` still selects an explicit retained raw log; otherwise
raw exchanges remain in the temporary directory. Missing exchange metrics on
provider failure are unknown (`null`), not zero. Token totals include all Planner
attempts and exclude Router/embedding tokens. Elapsed time covers the Planner
node, including routing/context retrieval, and totals all attempts in a run.

This is an observation/debug harness. It does not classify plan quality, require
an expected step shape, or fail because Planner returned a particular production
outcome. Semantic results and infrastructure failures are displayed directly:
PLAN_PROPOSED, NO_PLAN_REQUIRED, NEEDS_INPUT, PLANNING_FAILED, INVALID_OUTPUT or
PROVIDER_FAILURE. Every attempt is visible, including length-limited attempts and
Controller-authorized retries. Public production component types and completed
exchange diagnostics establish fidelity without private HTTP transport checks.
Offline display/lifecycle checks live in `test_planner_stability_measurement.py`.

The historical six-case live results above describe the previous stability
test, which called production PlannerService/provider with a unit-test request
fixture, no Controller retry handling, and no opt-in guard.
The separate clarification lifecycle smoke remains available with
`python -m pytest -o addopts= -q -p no:cacheprovider --no-cov tests/live/test_clarification_lifecycle.py`.

## Primary classification

### Contract/schema

- `tests/test_controller_execution.py::test_controller_decision_rejects_terminal_status_disagreement`
- `tests/test_controller_planning.py::test_request_operation_and_capability_contracts_reject_inconsistent_values`
- `tests/test_direct_semantic_handoff.py::test_direct_result_is_not_a_tool_or_step_authority`
- `tests/test_graph_brain_adapter.py::test_all_brain_prompts_align_with_the_outcome_contract`
- `tests/test_graph_brain_adapter.py::test_prompt_examples_are_complete_json_envelopes`
- `tests/test_planner_contract.py::test_all_semantic_variants_validate_without_failure_category`
- `tests/test_planner_contract.py::test_executable_step_rejects_missing_tool_and_extra_arguments`
- `tests/test_planner_contract.py::test_executable_step_requires_a_primary_tool`
- `tests/test_planner_contract.py::test_invalid_proposal_variant_shapes_are_rejected`
- `tests/test_planner_contract.py::test_message_bearing_variants_reject_empty_messages`
- `tests/test_planner_contract.py::test_proposal_extra_fields_are_forbidden`
- `tests/test_planner_contract.py::test_system_policy_exposes_only_semantic_proposal_outcomes`
- `tests/test_planner_routing.py::test_router_schema_rejects_removed_fields`
- `tests/test_planner_service.py::test_planner_results_are_bound_and_outcome_payloads_are_strict`
- `tests/test_planner_service.py::test_planning_request_requires_explicit_episode_identity`
- `tests/test_protocol_bridge.py::test_controller_input_async_evidence_queries_respect_terminal_latest_result`
- `tests/test_protocol_bridge.py::test_controller_input_async_evidence_queries_use_latest_valid_observation`
- `tests/test_protocol_bridge.py::test_controller_input_evidence_predicates_and_consecutive_failures`

### Planner unit

- `tests/test_controller_planning.py::test_revise_context_projection_with_scripted_provider`
- `tests/test_planner_memory.py::test_authorized_memory_is_projected_in_one_scripted_provider_exchange`
- `tests/test_planner_memory.py::test_counts_and_record_limits_keep_whole_records`
- `tests/test_planner_memory.py::test_empty_projection_and_authority_classes_are_explicit`
- `tests/test_planner_memory.py::test_expired_continuity_is_omitted_without_dropping_durable_facts`
- `tests/test_planner_memory.py::test_total_budget_prefers_strong_sources_then_recent_within_rank`
- `tests/test_planner_progress.py::test_exact_signatures_group_but_different_signatures_do_not`
- `tests/test_planner_progress.py::test_foreign_executions_excluded_and_legacy_unscoped_retained`
- `tests/test_planner_progress.py::test_missing_signatures_are_request_specific_and_round_trip_is_stable`
- `tests/test_planner_progress.py::test_projection_bounds_newest_groups_and_each_payload_deterministically`
- `tests/test_planner_progress.py::test_source_ids_are_bounded_while_occurrence_count_and_revisions_remain_complete`
- `tests/test_planner_progress.py::test_successful_not_found_output_is_observation_not_fact`
- `tests/test_planner_provider.py::test_infrastructure_failures_are_normalized_at_provider_boundary`
- `tests/test_planner_provider.py::test_planner_provider_writes_one_plan_exchange`
- `tests/test_planner_provider.py::test_router_writes_one_exchange_record`
- `tests/test_planner_routing.py::test_router_does_not_reparse_raw_enum_or_json`
- `tests/test_planner_routing.py::test_router_failures_propagate_without_fallback_route`
- `tests/test_planner_routing.py::test_supported_route_transport_with_scripted_provider`
- `tests/test_planner_service.py::test_comfy_guidance_uses_action_route_and_authorized_capability_only`
- `tests/test_planner_service.py::test_direct_routes_remain_ambient_rag_ineligible`
- `tests/test_planner_service.py::test_empty_authorized_set_is_consistent_and_can_return_unplannable`
- `tests/test_planner_service.py::test_explicit_result_variants`
- `tests/test_planner_service.py::test_info_planner_cannot_validate_tool_outside_authorized_set`
- `tests/test_planner_service.py::test_info_planner_prompt_and_validation_use_one_authorized_tool_set`
- `tests/test_planner_service.py::test_invalid_plan_constraints_rejected`
- `tests/test_planner_service.py::test_knowledge_request_and_uncertain_request_remain_ambient_rag_eligible`
- `tests/test_planner_service.py::test_malformed_structured_output_is_invalid`
- `tests/test_planner_service.py::test_new_request_and_existing_replan_still_route_normally`
- `tests/test_planner_service.py::test_planner_context_includes_retrieved_background_once`
- `tests/test_planner_service.py::test_preserved_clarification_route_skips_router_and_plans_original_request`
- `tests/test_planner_service.py::test_revise_preserves_direct_route_and_skips_ambient_retrieval`
- `tests/test_planner_service.py::test_revise_preserves_request_and_versions_candidate`
- `tests/test_planner_service.py::test_runtime_discovery_bypasses_ambient_retrieval_with_scripted_provider`
- `tests/test_planner_service.py::test_runtime_intent_without_matching_authorized_capability_keeps_retrieval`
- `tests/test_planner_service.py::test_service_runs_with_framework_imports_blocked`
- `tests/test_planner_service.py::test_valid_dependent_plan`
- `tests/test_planner_service.py::test_valid_independent_steps`

### Controller lifecycle

- `tests/test_controller_execution.py::test_accepting_replacement_plan_clears_old_retry_history`
- `tests/test_controller_execution.py::test_cancelled_and_failed_termination_carry_matching_status_and_cursor`
- `tests/test_controller_execution.py::test_final_answer_after_completed_plan_does_not_require_active_step`
- `tests/test_controller_execution.py::test_replan_request_rejects_stale_failed_step_identifier`
- `tests/test_controller_execution.py::test_step_completion_clears_retry_history_but_preserves_budget`
- `tests/test_controller_execution.py::test_step_failed_marks_step_failed_and_terminates_when_retries_exhausted`
- `tests/test_controller_execution.py::test_step_failed_retries_same_step_when_budget_remains`
- `tests/test_controller_execution.py::test_step_results_reject_missing_or_stale_active_step_identifiers`
- `tests/test_controller_planning.py::test_clarification_pause_resume_controller_lifecycle`
- `tests/test_controller_planning.py::test_controller_authorizes_revision_with_failure_and_progress_context`
- `tests/test_controller_planning.py::test_controller_rejects_stale_request_sequence_on_resume`
- `tests/test_controller_planning.py::test_create_plan_and_no_plan_have_distinct_terminal_semantics`
- `tests/test_controller_planning.py::test_initial_controller_authorizes_create_without_revision_facts`
- `tests/test_controller_planning.py::test_planner_result_binding_rejects_stale_and_unbound_results`
- `tests/test_controller_planning.py::test_planning_requests_and_controller_decisions_have_one_production_constructor_owner`
- `tests/test_controller_planning.py::test_request_serializes_and_resume_reuses_authorization`
- `tests/test_controller_planning.py::test_retryable_failures_have_two_attempts_and_unplannable_does_not_retry`
- `tests/test_controller_planning.py::test_revise_clarification_preserves_operation_base_and_failure_context`
- `tests/test_controller_planning.py::test_revise_keeps_typed_execution_facts_outside_recent_history`
- `tests/test_controller_planning.py::test_revise_no_plan_is_rejected_and_keeps_accepted_plan`
- `tests/test_planner_revision.py::test_create_path_is_not_revision_reconciled`
- `tests/test_planner_revision.py::test_failed_step_can_be_replaced_but_completed_definition_cannot_change`
- `tests/test_planner_revision.py::test_only_structurally_carried_completed_evidence_crosses_revision_boundary`
- `tests/test_planner_revision.py::test_pending_copy_of_completed_step_is_reconciled_not_rerun`
- `tests/test_planner_revision.py::test_rejection_decision_is_atomic_and_provenance_survives_acceptance_round_trip`
- `tests/test_planner_revision.py::test_revision_reconciliation_versions_candidate_and_preserves_completed_work`
- `tests/test_planner_revision.py::test_structurally_identical_remaining_plan_is_ineffective`
- `tests/test_planner_revision.py::test_wrong_plan_id_and_stale_base_are_rejected`

### Runtime/graph integration

- `tests/test_controller_planning.py::test_adapter_rejects_missing_and_stale_authorizations`
- `tests/test_controller_planning.py::test_revision_runtime_flow_with_scripted_provider`
- `tests/test_direct_semantic_handoff.py::test_accepted_step_semantics_reach_brain_with_scripted_provider`
- `tests/test_direct_semantic_handoff.py::test_model_backed_finalizer_presents_accepted_direct_content_without_model_call`
- `tests/test_direct_semantic_handoff.py::test_stale_or_unaccepted_direct_semantics_cannot_reach_finalizer`
- `tests/test_graph_brain_adapter.py::test_adapter_only_translates_input_output_and_consumed_tool_evidence`
- `tests/test_graph_brain_adapter.py::test_bridge_preserves_typed_payloads_without_reparsing_messages`
- `tests/test_graph_brain_adapter.py::test_graph_brain_execution_with_injected_planner_result`
- `tests/test_planner_memory.py::test_planner_consumes_authorized_memory_snapshot_without_graph_injection`
- `tests/test_planner_memory.py::test_projection_does_not_enter_brain_or_finalizer_context`
- `tests/test_planner_runtime_flow.py::test_checkpointed_clarification_resumes_same_authorization_without_duplicate_history_with_scripted_provider`
- `tests/test_planner_runtime_flow.py::test_create_direct_response_uses_one_human_request_and_controller_terminal_state_with_scripted_provider`
- `tests/test_planner_runtime_flow.py::test_executable_runtime_flow_with_scripted_provider`
- `tests/test_planner_runtime_flow.py::test_runtime_checkpoint_preserves_authorization_with_scripted_provider`
- `tests/test_protocol_bridge.py::test_bridge_rejects_legacy_protocol_reconstruction`
- `tests/test_protocol_bridge.py::test_build_brain_input_transfers_tool_execution_history_from_working_state`
- `tests/test_protocol_bridge.py::test_build_brain_input_uses_initial_user_request_without_durable_override`
- `tests/test_protocol_bridge.py::test_build_controller_input_leaves_optional_worker_outputs_none_when_missing`
- `tests/test_protocol_bridge.py::test_build_controller_input_preserves_checkpointed_async_policy`

### Application/session transport

- `tests/test_graph_runner.py::test_new_create_request_has_conversation_but_no_prior_execution_artifacts`
- `tests/test_graph_runner.py::test_run_prompt_renders_portable_controller_tool_result_concisely`
- `tests/test_graph_runner.py::test_run_prompt_transports_injected_pause_and_execution_reference_on_reply`
- `tests/test_graph_runner.py::test_runner_projects_session_history_from_injected_finalization_events`

### Live provider smoke

- `tests/live/test_clarification_lifecycle.py::test_clarification_resume_live`
- `tests/live/test_planner_stability.py::test_planner_stability_live`
- `tests/test_planner_live_memory_adherence.py::test_real_planner_resolves_memory_into_accepted_step_before_tools`

