from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

from .enums import (
    AsyncJobStatus,
    BrainOutcome,
    CancellationSource,
    ControllerDecisionType,
    ExecutionPhase,
    ExecutionStatus,
    PlannerOutcome,
    StepStatus,
    WorkerRole,
)
from .models import (
    BrainResult,
    ControllerDecision,
    ControllerInput,
    ExecutionCursor,
    ExecutionPlan,
    ExecutionStep,
    PlannerResult,
    RetryMetadata,
    ToolRequest,
    ToolResult,
)


class CortexController:
    """Protocol decision engine.

    The Controller owns execution decisions but never executes workers.
    It evaluates protocol inputs and returns the next legal continuation.
    """

    def __init__(
        self,
        max_reasoning_steps: int,
        *,
        now_utc: Callable[[], datetime] | None = None,
        async_submission_tool_names: Iterable[str] = ("run_comfy_workflow",),
        ambiguous_submission_error_codes: Iterable[str] = (
            "COMFY_API_ERROR",
            "COMFY_CONNECTION_FAILED",
            "COMFY_PROMPT_FAILED",
        ),
    ):
        self._max_reasoning_steps = max_reasoning_steps
        self._now_utc = now_utc or (lambda: datetime.now(timezone.utc))
        self._async_submission_tool_names = frozenset(async_submission_tool_names)
        self._ambiguous_submission_error_codes = frozenset(
            ambiguous_submission_error_codes
        )
        
    def decide(
        self,
        controller_input: ControllerInput,
    ) -> ControllerDecision:
        self._validate(controller_input)

        if controller_input.cancel_requested:
            return self._cancel(
                controller_input.cursor,
                "local_async_cancellation_requested",
                source=CancellationSource.LOCAL,
            )

        active_async_job_ids = controller_input.get_active_async_job_ids()
        if len(active_async_job_ids) > 1:
            return self._pause(
                controller_input.cursor,
                reason=(
                    "multiple_active_async_jobs_for_step:"
                    f"{','.join(active_async_job_ids)}"
                ),
                reconciliation_required=True,
                clear_pending_tool_request=True,
            )

        # print("=== CONTROLLER INPUT ===")
        # print("planner_result:", controller_input.planner_result)
        # print("brain_result:", controller_input.brain_result)
        # print("tool_result:", controller_input.tool_result)
        # print("active_step:", controller_input.active_step.step_id if controller_input.active_step else None)
        # print("cursor.step_id:", controller_input.cursor.step_id)
        # print("========================")

        if (
            controller_input.cursor.controller_iteration is not None
            and controller_input.cursor.controller_iteration >= self._max_reasoning_steps
        ):
            return self._terminate(controller_input.cursor, "max_steps")

        if controller_input.planner_result is not None:
            return self._decide_from_planner(controller_input)
        
        if controller_input.brain_result is not None:
            return self._decide_from_brain(controller_input)

        if controller_input.tool_result is not None:
            return self._decide_from_tool(controller_input)

        return self._decide_initial(controller_input)

    def _validate(self, controller_input: ControllerInput) -> None:
        """Validate protocol invariants."""
        results = (
            controller_input.planner_result is not None,
            controller_input.brain_result is not None,
            controller_input.tool_result is not None,
        )

        if sum(results) > 1:
            raise ValueError(
                "ControllerInput may contain only one worker result."
            )

    def _decide_initial(
        self,
        controller_input: ControllerInput,
    ) -> ControllerDecision:
        if controller_input.active_plan is None:
            return self._dispatch_planner(
                controller_input.cursor,
                "No active plan.",
            )

        next_step = self._find_next_executable_step(controller_input.active_plan)
        if next_step is None:
            return self._dispatch_summary(controller_input.cursor, "Plan completed.")

        return self._dispatch_brain(
            cursor=controller_input.cursor,
            reason="Executing next step.",
            next_step_id=next_step.step_id,
        )

    def _decide_from_planner(
        self,
        controller_input: ControllerInput,
    ) -> ControllerDecision:
        planner_result = controller_input.planner_result

        match planner_result.outcome:

            case PlannerOutcome.DIRECT_RESPONSE:
                return self._dispatch_brain(
                    cursor=controller_input.cursor,
                    reason="Direct response.",
                    direct_response=True,
                )

            case PlannerOutcome.EXECUTION_PLAN:
                plan = planner_result.proposed_plan

                if plan is None:
                    return self._terminate(
                        controller_input.cursor,
                        "Planner returned PLAN_CREATED without a plan.",
                    )

                # next_step = self._find_next_pending_step(controller_input)
                next_step = next(
                    (step for step in plan.steps if step.status == StepStatus.PENDING),
                        None,
                    )

                if next_step is None:
                    return self._dispatch_summary(
                        controller_input.cursor,
                        "Plan contains no executable steps.",
                    )

                cursor = controller_input.cursor.model_copy(
                    update={
                        "phase": ExecutionPhase.EXECUTING,
                        "step_id": next_step.step_id,
                        "step_attempt": next_step.attempt,
                        "current_worker": WorkerRole.BRAIN,
                        "plan_revision": plan.revision,
                    }
                )

                return ControllerDecision(
                    decision_type=ControllerDecisionType.DISPATCH_BRAIN,
                    next_worker=WorkerRole.BRAIN,
                    reason="Plan accepted.",
                    accepted_plan=plan,
                    next_step_id=next_step.step_id,
                    cursor=cursor,
                    retry=RetryMetadata(max_retries=controller_input.retry.max_retries),
                )
            
            case PlannerOutcome.CLARIFICATION_REQUIRED:
                return self._dispatch_brain(
                    cursor=controller_input.cursor,
                    reason="Clarification required.",
                )

            case PlannerOutcome.FAILED:
                failure_reason = (
                    planner_result.message.strip() or "planner_failed"
                )
                return self._terminate(
                    controller_input.cursor,
                    "planner_failed",
                    failure_reason=failure_reason,
                )

        raise ValueError(f"Unsupported planner outcome: {planner_result.outcome}")

    def _update_active_step_status(
        self,
        controller_input: ControllerInput,
        status: StepStatus,
    ) -> ExecutionPlan | None:

        plan = controller_input.active_plan
        active_step = controller_input.active_step

        if plan is None or active_step is None:
            return None

        return plan.model_copy(
            update={
                "steps": tuple(
                    step.model_copy(update={"status": status})
                    if step.step_id == active_step.step_id
                    else step
                    for step in plan.steps
                )
            }
        )

    def _validate_active_step(
        self,
        controller_input: ControllerInput,
        *,
        transition: str,
    ) -> tuple[ExecutionPlan, ExecutionStep]:
        plan = controller_input.active_plan
        active_step = controller_input.active_step

        if plan is None:
            raise ValueError(f"{transition} requires an active plan")
        if active_step is None:
            raise ValueError(f"{transition} requires an active step")
        if controller_input.cursor.step_id != active_step.step_id:
            raise ValueError(
                f"{transition} cursor step does not match active step"
            )

        matching_steps = tuple(
            step for step in plan.steps if step.step_id == active_step.step_id
        )
        if len(matching_steps) != 1:
            raise ValueError(
                f"{transition} active step is missing or duplicated in active plan"
            )

        retry_step_id = controller_input.retry.step_id
        if retry_step_id is not None and retry_step_id != active_step.step_id:
            raise ValueError(
                f"{transition} retry metadata does not match active step"
            )

        return plan, matching_steps[0]

    @staticmethod
    def _replace_plan_step(
        plan: ExecutionPlan,
        replacement: ExecutionStep,
    ) -> ExecutionPlan:
        return plan.model_copy(
            update={
                "steps": tuple(
                    replacement if step.step_id == replacement.step_id else step
                    for step in plan.steps
                )
            }
        )

    def _decide_step_failure(
        self,
        controller_input: ControllerInput,
    ) -> ControllerDecision:
        plan, active_step = self._validate_active_step(
            controller_input,
            transition="step failure",
        )
        brain_result = controller_input.brain_result
        failure_reason = (
            brain_result.message.strip()
            if brain_result is not None and brain_result.message.strip()
            else "step_failed"
        )
        retry = controller_input.retry.model_copy(
            update={
                "step_id": active_step.step_id,
                "last_error_message": failure_reason,
            }
        )

        if retry.retry_count < retry.max_retries:
            next_attempt = max(
                active_step.attempt,
                controller_input.cursor.step_attempt or 0,
            ) + 1
            retry = retry.model_copy(
                update={"retry_count": retry.retry_count + 1}
            )
            retry_step = active_step.model_copy(
                update={
                    "status": StepStatus.ACTIVE,
                    "attempt": next_attempt,
                }
            )
            retry_plan = self._replace_plan_step(plan, retry_step)
            cursor = controller_input.cursor.model_copy(
                update={
                    "phase": ExecutionPhase.EXECUTING,
                    "step_id": active_step.step_id,
                    "step_attempt": next_attempt,
                    "current_worker": WorkerRole.BRAIN,
                }
            )
            return ControllerDecision(
                accepted_plan=retry_plan,
                decision_type=ControllerDecisionType.DISPATCH_BRAIN,
                reason="retry_step",
                next_worker=WorkerRole.BRAIN,
                cursor=cursor,
                failed_step_id=active_step.step_id,
                failure_reason=failure_reason,
                next_step_id=active_step.step_id,
                retry=retry,
                requires_checkpoint=True,
            )

        failed_step = active_step.model_copy(update={"status": StepStatus.FAILED})
        failed_plan = self._replace_plan_step(plan, failed_step)
        return self._terminate(
            controller_input.cursor,
            "step_failed_retries_exhausted",
            accepted_plan=failed_plan,
            failed_step_id=active_step.step_id,
            failure_reason=failure_reason,
            retry=retry,
        )

    def _decide_from_brain(
        self,
        controller_input: ControllerInput,
    ) -> ControllerDecision:
        
        brain_result = controller_input.brain_result
        if brain_result is None:
            raise RuntimeError(
                "Controller dispatched to Brain without BrainResult."
        )

        match brain_result.outcome:
            case BrainOutcome.TOOL_REQUEST:
                tool_request = brain_result.tool_request
                if tool_request is None:
                    return self._terminate(
                        controller_input.cursor,
                        "brain_tool_request_missing_payload",
                    )

                if tool_request.tool_name in self._async_submission_tool_names:
                    tool_request = self._prepare_async_submission_request(
                        controller_input,
                        tool_request,
                    )
                    active_job_ids = controller_input.get_active_async_job_ids()
                    if active_job_ids:
                        return self._pause(
                            controller_input.cursor,
                            reason=(
                                "active_async_job_prevents_submission:"
                                f"{active_job_ids[-1]}"
                            ),
                            reconciliation_required=True,
                        )

                    submission_attempts = (
                        controller_input.get_async_submission_attempt_count(
                            submission_tool_names=self._async_submission_tool_names,
                            ambiguous_error_codes=(
                                self._ambiguous_submission_error_codes
                            ),
                        )
                    )
                    if (
                        submission_attempts
                        >= controller_input.async_policy.max_submission_attempts
                    ):
                        return self._pause(
                            controller_input.cursor,
                            reason=(
                                "async_submission_attempt_limit_reached:"
                                f"{submission_attempts}"
                            ),
                            reconciliation_required=True,
                        )

                cursor = controller_input.cursor.model_copy(
                    update={
                        "phase": ExecutionPhase.EXECUTING,
                        "step_id": controller_input.active_step.step_id,
                        "step_attempt": controller_input.active_step.attempt,
                        "current_worker": WorkerRole.TOOL_RUNTIME,
                    }
                )
                return ControllerDecision(
                        accepted_plan=self._update_active_step_status(
                            controller_input,
                            StepStatus.ACTIVE,
                        ),
                        decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
                        reason="tool_request",
                        next_worker=WorkerRole.TOOL_RUNTIME,
                        cursor=cursor,
                        pending_tool_request=tool_request,
                        next_step_id=controller_input.active_step.step_id,
                    )

            case BrainOutcome.REPLAN_REQUEST:
                plan, active_step = self._validate_active_step(
                    controller_input,
                    transition="replan request",
                )
                replan_request = brain_result.replan_request
                if (
                    replan_request is not None
                    and replan_request.failed_step_id is not None
                    and replan_request.failed_step_id != active_step.step_id
                ):
                    raise ValueError(
                        "replan request failed step does not match active step"
                    )

                failure_reason = (
                    replan_request.reason
                    if replan_request is not None
                    else brain_result.message.strip() or "replan_requested"
                )
                failed_plan = self._replace_plan_step(
                    plan,
                    active_step.model_copy(update={"status": StepStatus.FAILED}),
                )
                retry = RetryMetadata(max_retries=controller_input.retry.max_retries)
                cursor = controller_input.cursor.model_copy(
                    update={
                        "phase": ExecutionPhase.REPLANNING,
                        "current_worker": WorkerRole.PLANNER,
                        "step_id": None,
                        "step_attempt": None,
                    }
                )

                return ControllerDecision(
                    accepted_plan=failed_plan,
                    decision_type=ControllerDecisionType.DISPATCH_PLANNER,
                    reason="replan_request",
                    next_worker=WorkerRole.PLANNER,
                    requires_checkpoint=True,
                    requires_replan=True,
                    cursor=cursor,
                    failed_step_id=active_step.step_id,
                    failure_reason=failure_reason,
                    retry=retry,
                    clear_active_step=True,
                )

            case BrainOutcome.FINAL_ANSWER:
                accepted_plan = None
                completed_step_id = None
                if (
                    controller_input.active_step is not None
                    or controller_input.cursor.step_id is not None
                ):
                    _, active_step = self._validate_active_step(
                        controller_input,
                        transition="final answer",
                    )
                    accepted_plan = self._update_active_step_status(
                        controller_input,
                        StepStatus.COMPLETED,
                    )
                    completed_step_id = active_step.step_id

                return self._dispatch_summary(
                    controller_input.cursor,
                    "final_answer",
                    accepted_plan=accepted_plan,
                    completed_step_id=completed_step_id,
                )

            case BrainOutcome.STEP_COMPLETED:
                return self._advance_to_next_step(controller_input)

            case BrainOutcome.STEP_FAILED:
                return self._decide_step_failure(controller_input)

            case BrainOutcome.CONTINUE:
                return self._dispatch_brain(cursor=controller_input.cursor, reason="continue")

        raise ValueError(f"Unsupported brain outcome: {brain_result.outcome}")

    def _decide_from_tool(
        self,
        controller_input: ControllerInput,
    ) -> ControllerDecision:
        tool_result: ToolResult = controller_input.tool_result
        pending_tool_request = controller_input.pending_tool_request

        if (
            pending_tool_request is not None
            and tool_result.request_id != pending_tool_request.request_id
        ):
            return self._terminate(
                controller_input.cursor,
                "tool_result_request_id_mismatch",
            )

        if (
            pending_tool_request is not None
            and pending_tool_request.tool_name in self._async_submission_tool_names
            and not tool_result.success
            and not tool_result.is_async_job
            and (
                tool_result.error_code is None
                or tool_result.error_code in self._ambiguous_submission_error_codes
            )
        ):
            return self._pause(
                controller_input.cursor,
                reason=(
                    "ambiguous_async_submission_requires_reconciliation:"
                    f"{pending_tool_request.tool_name}"
                ),
                reconciliation_required=True,
                clear_pending_tool_request=True,
            )

        if tool_result.is_async_job:
            async_result = self._select_async_result(
                controller_input,
                current_result=tool_result,
            )

            if async_result.async_job_status in {
                AsyncJobStatus.SUBMITTED,
                AsyncJobStatus.RUNNING,
                AsyncJobStatus.UNKNOWN,
            }:
                async_job_id = async_result.async_job_id
                if async_job_id is None:
                    raise ValueError("Async observation requires async_job_id.")

                started_at = (
                    controller_input.get_async_job_started_at_utc(async_job_id)
                    or async_result.async_observed_at_utc
                    or self._now_utc()
                )
                started_at = self._as_utc(started_at)
                observed_at = self._as_utc(
                    async_result.async_observed_at_utc or self._now_utc()
                )
                now_utc = self._as_utc(self._now_utc())
                deadline_utc = started_at + timedelta(
                    seconds=controller_input.async_policy.execution_timeout_seconds,
                )

                poll_failures = (
                    controller_input.get_consecutive_async_poll_failures(
                        async_job_id,
                        excluded_tool_names=self._async_submission_tool_names,
                    )
                )
                is_submission_observation = bool(
                    pending_tool_request is not None
                    and pending_tool_request.tool_name
                    in self._async_submission_tool_names
                )
                if (
                    not async_result.success
                    and poll_failures == 0
                    and not is_submission_observation
                ):
                    poll_failures = 1
                if poll_failures >= controller_input.async_policy.max_poll_failures:
                    return self._pause(
                        controller_input.cursor,
                        reason=(
                            "async_poll_failure_budget_exhausted:"
                            f"{poll_failures}"
                        ),
                        async_job_id=async_job_id,
                        execution_deadline_utc=deadline_utc,
                        reconciliation_required=True,
                        clear_pending_tool_request=(pending_tool_request is not None),
                    )

                if now_utc >= deadline_utc:
                    if observed_at < deadline_utc or not async_result.success:
                        return self._await_async_job(
                            controller_input,
                            controller_input.cursor,
                            async_result=async_result,
                            clear_pending_tool_request=(pending_tool_request is not None),
                            reason=(
                                f"Final reconciliation for timed-out async job "
                                f"{async_job_id}."
                            ),
                            execution_deadline_utc=deadline_utc,
                            force_immediate=True,
                        )
                    return self._pause(
                        controller_input.cursor,
                        reason="async_execution_timeout_after_reconciliation",
                        async_job_id=async_job_id,
                        execution_deadline_utc=deadline_utc,
                        reconciliation_required=False,
                        clear_pending_tool_request=(pending_tool_request is not None),
                    )

                reason = f"Awaiting async job {async_job_id}."
                if (
                    async_result.async_job_status == AsyncJobStatus.UNKNOWN
                    and async_result.success
                    and (
                        now_utc - started_at
                    ).total_seconds()
                    >= controller_input.async_policy.visibility_grace_seconds
                ):
                    if controller_input.is_async_job_confirmed_absent(async_job_id):
                        submission_attempts = (
                            controller_input.get_async_submission_attempt_count(
                                submission_tool_names=self._async_submission_tool_names,
                                ambiguous_error_codes=(
                                    self._ambiguous_submission_error_codes
                                ),
                            )
                        )
                        if (
                            submission_attempts
                            >= controller_input.async_policy.max_submission_attempts
                        ):
                            return self._pause(
                                controller_input.cursor,
                                reason=(
                                    "async_submission_absent_after_reconciliation:"
                                    f"attempts={submission_attempts}"
                                ),
                                async_job_id=async_job_id,
                                execution_deadline_utc=deadline_utc,
                                reconciliation_required=False,
                                clear_pending_tool_request=(
                                    pending_tool_request is not None
                                ),
                            )
                        return self._dispatch_brain(
                            cursor=controller_input.cursor,
                            reason=(
                                "Async submission was not found after provider "
                                "reconciliation; retry policy permits a new attempt."
                            ),
                            clear_pending_tool_request=(
                                pending_tool_request is not None
                            ),
                        )
                    reason = (
                        f"Async job {async_job_id} remains unobserved after "
                        "the visibility grace window."
                    )
                return self._await_async_job(
                    controller_input,
                    controller_input.cursor,
                    async_result=async_result,
                    clear_pending_tool_request=(pending_tool_request is not None),
                    reason=reason,
                    execution_deadline_utc=deadline_utc,
                )

            if async_result.async_job_status == AsyncJobStatus.COMPLETED:
                return self._dispatch_brain(
                    cursor=controller_input.cursor,
                    reason="Async job completed.",
                    clear_pending_tool_request=(pending_tool_request is not None),
                )

            if async_result.async_job_status == AsyncJobStatus.CANCELLED:
                return self._cancel(
                    controller_input.cursor,
                    "remote_async_job_cancelled",
                    source=CancellationSource.PROVIDER,
                    clear_pending_tool_request=(pending_tool_request is not None),
                )

            if async_result.async_job_status == AsyncJobStatus.FAILED:
                return self._decide_tool_failure(
                    controller_input,
                    tool_result=async_result,
                    clear_pending_tool_request=(pending_tool_request is not None),
                )

        if not tool_result.success:
            return self._decide_tool_failure(
                controller_input,
                tool_result=tool_result,
                clear_pending_tool_request=(pending_tool_request is not None),
            )

        return self._dispatch_brain(
            cursor=controller_input.cursor,
            reason="Tool completed.",
            clear_pending_tool_request=(pending_tool_request is not None),
        )

    def _select_async_result(
        self,
        controller_input: ControllerInput,
        *,
        current_result: ToolResult,
    ) -> ToolResult:
        """Use history's monotonic view once the current observation is recorded."""
        current_is_recorded = any(
            record.result.request_id == current_result.request_id
            for record in controller_input.tool_execution_history
        )
        if not current_is_recorded:
            return current_result

        return controller_input.get_latest_async_result() or current_result

    def _decide_tool_failure(
        self,
        controller_input: ControllerInput,
        *,
        tool_result: ToolResult,
        clear_pending_tool_request: bool,
    ) -> ControllerDecision:
        """Apply the existing Controller-owned retry/replan policy to a failure."""
        # Evidence-driven check: consecutive failures for the same signature
        if not tool_result.success and tool_result.signature:
            consecutive_fails = controller_input.get_consecutive_failures(tool_result.signature)
            max_allowed_fails = (
                controller_input.retry.max_retries
                if controller_input.retry.max_retries > 0
                else 3
            )
            if consecutive_fails >= max_allowed_fails:
                cursor = controller_input.cursor.model_copy(
                    update={
                        "phase": ExecutionPhase.REPLANNING,
                        "current_worker": WorkerRole.PLANNER,
                    }
                )
                return ControllerDecision(
                    decision_type=ControllerDecisionType.DISPATCH_PLANNER,
                    reason=f"Repeated tool failure threshold ({consecutive_fails}) reached for {tool_result.signature}",
                    next_worker=WorkerRole.PLANNER,
                    requires_checkpoint=True,
                    cursor=cursor,
                    clear_pending_tool_request=clear_pending_tool_request,
                )

        return self._dispatch_brain(
            cursor=controller_input.cursor,
            reason="Tool failed.",
            clear_pending_tool_request=clear_pending_tool_request,
        )

    def _await_async_job(
        self,
        controller_input: ControllerInput,
        cursor: ExecutionCursor,
        *,
        async_result: ToolResult,
        clear_pending_tool_request: bool,
        reason: str,
        execution_deadline_utc: datetime,
        force_immediate: bool = False,
    ) -> ControllerDecision:
        async_job_id = async_result.async_job_id
        if async_job_id is None:
            raise ValueError("Async wait requires an async_job_id.")

        poll_observations = controller_input.get_async_observation_count(
            async_job_id,
            excluded_tool_names=self._async_submission_tool_names,
        )
        delay_seconds = controller_input.async_policy.poll_interval_seconds
        for _ in range(poll_observations):
            delay_seconds = min(
                delay_seconds * 2,
                controller_input.async_policy.max_poll_interval_seconds,
            )
            if delay_seconds == controller_input.async_policy.max_poll_interval_seconds:
                break
        observed_at = self._as_utc(
            async_result.async_observed_at_utc or self._now_utc()
        )
        if force_immediate:
            resume_after_utc = self._as_utc(self._now_utc())
        else:
            resume_after_utc = min(
                observed_at + timedelta(seconds=delay_seconds),
                execution_deadline_utc,
            )

        waiting_cursor = cursor.model_copy(
            update={
                "phase": ExecutionPhase.WAITING,
                "current_worker": WorkerRole.CONTROLLER,
            }
        )
        return ControllerDecision(
            decision_type=ControllerDecisionType.AWAIT_ASYNC_JOB,
            reason=reason,
            next_worker=WorkerRole.CONTROLLER,
            cursor=waiting_cursor,
            async_job_id=async_job_id,
            resume_after_utc=resume_after_utc,
            execution_deadline_utc=execution_deadline_utc,
            requires_checkpoint=True,
            clear_pending_tool_request=clear_pending_tool_request,
        )

    def _dispatch_planner(
        self,
        cursor: ExecutionCursor,
        reason: str,
    ) -> ControllerDecision:
        planning_cursor = cursor.model_copy(
            update={
                "phase": ExecutionPhase.PLANNING,
                "current_worker": WorkerRole.PLANNER,
                "step_id": None,
                "step_attempt": None,
            }
        )
        return ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_PLANNER,
            next_worker=WorkerRole.PLANNER,
            reason=reason,
            cursor=planning_cursor,
            requires_checkpoint=True,
        )

    def _prepare_async_submission_request(
        self,
        controller_input: ControllerInput,
        tool_request: ToolRequest,
    ) -> ToolRequest:
        """Attach a stable provider ID before POST so ambiguous results are queryable."""
        arguments = dict(tool_request.arguments)
        correlation_key = (
            f"cortex:{controller_input.identity.execution_id}:"
            f"{controller_input.cursor.step_id or 'no-step'}:{tool_request.request_id}"
        )
        arguments.setdefault(
            "prompt_id",
            str(uuid5(NAMESPACE_URL, correlation_key)),
        )
        arguments.setdefault("client_id", controller_input.identity.execution_id)
        return tool_request.model_copy(update={"arguments": arguments})

    def _dispatch_brain(
        self,
        cursor: ExecutionCursor,
        reason: str,
        completed_step_id: str | None = None,
        next_step_id: str | None = None,
        clear_pending_tool_request: bool = False,
        clear_active_step: bool = False,
        direct_response: bool = False,
    ) -> ControllerDecision:

        resolved_step_id = next_step_id
        if resolved_step_id is None and not clear_active_step:
            resolved_step_id = cursor.step_id

        next_cursor = cursor.model_copy(
            update={
                "current_worker": WorkerRole.BRAIN,
                "step_id": resolved_step_id,
                "phase": ExecutionPhase.EXECUTING,
            }
        )

        return ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_BRAIN,
            next_worker=WorkerRole.BRAIN,
            clear_pending_tool_request=clear_pending_tool_request,
            cursor=next_cursor,
            completed_step_id=completed_step_id,
            next_step_id=next_step_id,
            reason=reason,
            clear_active_step=clear_active_step,
            direct_response=direct_response,
        )


    def _dispatch_summary(
        self,
        cursor: ExecutionCursor,
        reason: str,
        *,
        accepted_plan: ExecutionPlan | None = None,
        completed_step_id: str | None = None,
    ) -> ControllerDecision:
        completed_cursor = cursor.model_copy(
            update={
                "phase": ExecutionPhase.COMPLETED,
                "current_worker": WorkerRole.SUMMARY,
                "step_id": None,
                "step_attempt": None,
            }
        )
        return ControllerDecision(
            accepted_plan=accepted_plan,
            decision_type=ControllerDecisionType.DISPATCH_SUMMARY,
            next_worker=WorkerRole.SUMMARY,
            reason=reason,
            execution_status=ExecutionStatus.COMPLETED,
            cursor=completed_cursor,
            completed_step_id=completed_step_id,
            clear_active_step=True,
            clear_pending_tool_request=True,
            terminal=True,
        )

    def _terminate(
        self,
        cursor: ExecutionCursor,
        reason: str,
        *,
        accepted_plan: ExecutionPlan | None = None,
        failed_step_id: str | None = None,
        failure_reason: str | None = None,
        retry: RetryMetadata | None = None,
    ) -> ControllerDecision:
        failed_cursor = cursor.model_copy(
            update={
                "phase": ExecutionPhase.FAILED,
                "current_worker": WorkerRole.CONTROLLER,
                "step_id": None,
                "step_attempt": None,
            }
        )
        return ControllerDecision(
            accepted_plan=accepted_plan,
            decision_type=ControllerDecisionType.TERMINATE,
            reason=reason,
            execution_status=ExecutionStatus.FAILED,
            cursor=failed_cursor,
            failed_step_id=failed_step_id,
            failure_reason=failure_reason or reason,
            retry=retry,
            clear_active_step=True,
            clear_pending_tool_request=True,
            terminal=True,
        )

    def _pause(
        self,
        cursor: ExecutionCursor,
        *,
        reason: str,
        reconciliation_required: bool,
        async_job_id: str | None = None,
        execution_deadline_utc: datetime | None = None,
        clear_pending_tool_request: bool = False,
    ) -> ControllerDecision:
        waiting_cursor = cursor.model_copy(
            update={
                "phase": ExecutionPhase.WAITING,
                "current_worker": WorkerRole.CONTROLLER,
            }
        )
        return ControllerDecision(
            decision_type=ControllerDecisionType.PAUSE,
            reason=reason,
            next_worker=WorkerRole.CONTROLLER,
            cursor=waiting_cursor,
            async_job_id=async_job_id,
            execution_deadline_utc=execution_deadline_utc,
            reconciliation_required=reconciliation_required,
            requires_checkpoint=True,
            clear_pending_tool_request=clear_pending_tool_request,
        )

    def _cancel(
        self,
        cursor: ExecutionCursor,
        reason: str,
        *,
        source: CancellationSource,
        clear_pending_tool_request: bool = True,
    ) -> ControllerDecision:
        cancelled_cursor = cursor.model_copy(
            update={
                "phase": ExecutionPhase.CANCELLED,
                "current_worker": WorkerRole.CONTROLLER,
                "step_id": None,
                "step_attempt": None,
            }
        )
        return ControllerDecision(
            decision_type=ControllerDecisionType.CANCEL,
            reason=reason,
            next_worker=WorkerRole.CONTROLLER,
            execution_status=ExecutionStatus.CANCELLED,
            cursor=cancelled_cursor,
            requires_checkpoint=True,
            clear_active_step=True,
            clear_pending_tool_request=clear_pending_tool_request,
            cancellation_source=source,
            terminal=True,
        )

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _find_next_executable_step(
        self,
        plan: ExecutionPlan,
    ) -> ExecutionStep | None:
        """Return the first pending step whose dependencies are completed."""

        completed_step_ids = {
            step.step_id
            for step in plan.steps
            if step.status == StepStatus.COMPLETED
        }

        for step in plan.steps:
            if step.status != StepStatus.PENDING:
                continue

            if all(
                dependency_id in completed_step_ids
                for dependency_id in step.depends_on_step_ids
            ):
                return step

        return None
    
    def _find_next_pending_step(
        self,
        controller_input: ControllerInput,
    ) -> ExecutionStep | None:
        """Return the next executable step in the active plan."""

        plan = controller_input.active_plan
        if plan is None:
            return None

        completed = set(
            controller_input.context.completed_step_ids
            if hasattr(controller_input.context, "completed_step_ids")
            else ()
        )

        for step in plan.steps:
            if step.step_id in completed:
                continue

            if step.status == StepStatus.COMPLETED:
                continue

            return step

        return None

    def _advance_to_next_step(
        self,
        controller_input: ControllerInput,
    ) -> ControllerDecision:

        _, current = self._validate_active_step(
            controller_input,
            transition="step completion",
        )

        updated_plan = self._update_active_step_status(
            controller_input,
            StepStatus.COMPLETED,
        )

        if updated_plan is None:
            return self._terminate(
                controller_input.cursor,
                "Cannot complete active step.",
            )

        next_step = self._find_next_executable_step(updated_plan)
        # Retry history belongs to the completed step; only the policy carries on.
        retry = RetryMetadata(max_retries=controller_input.retry.max_retries)

        if next_step is None:
            cursor = controller_input.cursor.model_copy(
                update={
                    "phase": ExecutionPhase.EXECUTING,
                    "step_id": None,
                    "step_attempt": None,
                    "current_worker": WorkerRole.BRAIN,
                }
            )
            return ControllerDecision(
                accepted_plan=updated_plan,
                decision_type=ControllerDecisionType.DISPATCH_BRAIN,
                next_worker=WorkerRole.BRAIN,
                reason="Generate final answer.",
                cursor=cursor,
                completed_step_id=current.step_id if current else None,
                next_step_id=None,
                clear_active_step=True,
                retry=retry,
            )

        cursor = controller_input.cursor.model_copy(
            update={
                "phase": ExecutionPhase.EXECUTING,
                "step_id": next_step.step_id,
                "step_attempt": next_step.attempt,
                "current_worker": WorkerRole.BRAIN,
            }
        )
        return ControllerDecision(
            accepted_plan=updated_plan,
            decision_type=ControllerDecisionType.DISPATCH_BRAIN,
            next_worker=WorkerRole.BRAIN,
            reason="Advance to next executable step.",
            cursor=cursor,
            completed_step_id=current.step_id if current else None,
            next_step_id=next_step.step_id,
            retry=retry,
        )
