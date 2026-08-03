from __future__ import annotations

from .enums import (
    BrainOutcome,
    ControllerDecisionType,
    ExecutionPhase,
    PlannerOutcome,
    WorkerRole,
    StepStatus,
)
from .models import (
    BrainResult,
    ControllerDecision,
    ControllerInput,
    ExecutionCursor,
    ExecutionPlan,
    ExecutionStep,
    PlannerResult,
    ToolResult,
)


class CortexController:
    """Protocol decision engine.

    The Controller owns execution decisions but never executes workers.
    It evaluates protocol inputs and returns the next legal continuation.
    """

    def __init__(self, max_reasoning_steps: int):
        self._max_reasoning_steps = max_reasoning_steps
        
    def decide(
        self,
        controller_input: ControllerInput,
    ) -> ControllerDecision:
        self._validate(controller_input)

        if (
            controller_input.cursor.controller_iteration is not None
            and controller_input.cursor.controller_iteration >= self._max_reasoning_steps
        ):
            return self._terminate("max_steps")

        worker = controller_input.cursor.current_worker

        if worker == WorkerRole.PLANNER:
            return self._decide_from_planner(controller_input)
        
        if worker == WorkerRole.BRAIN:
            return self._decide_from_brain(controller_input)

        if worker == WorkerRole.TOOL_RUNTIME:
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
            return self._dispatch_planner("No active plan.")

        next_step = self._find_next_pending_step(controller_input)

        if next_step is None:
            return self._dispatch_summary("Plan completed.")

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
                )

            case PlannerOutcome.EXECUTION_PLAN:
                plan = planner_result.proposed_plan

                if plan is None:
                    return self._terminate("Planner returned PLAN_CREATED without a plan.")

                # next_step = self._find_next_pending_step(controller_input)
                next_step = next(
                    (step for step in plan.steps if step.status == StepStatus.PENDING),
                        None,
                    )

                if next_step is None:
                    return self._dispatch_summary("Plan contains no executable steps.")

                return ControllerDecision(
                    decision_type=ControllerDecisionType.DISPATCH_BRAIN,
                    next_worker=WorkerRole.BRAIN,
                    reason="Plan accepted.",
                    accepted_plan=plan,
                    next_step_id=next_step.step_id,
                )
            
            case PlannerOutcome.CLARIFICATION_REQUIRED:
                return self._dispatch_brain(
                    cursor=controller_input.cursor,
                    reason="Clarification required.",
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
                return ControllerDecision(
                        accepted_plan=self._update_active_step_status(
                            controller_input,
                            StepStatus.ACTIVE,
                        ),
                        decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
                        reason="tool_request",
                        next_worker=WorkerRole.TOOL_RUNTIME,
                        next_step_id=controller_input.active_step.step_id,
                    )

                # return self._dispatch_tool("tool_request")

            case BrainOutcome.REPLAN_REQUEST:
                cursor = controller_input.cursor.model_copy(
                    update={
                        "phase": ExecutionPhase.REPLANNING,
                        "current_worker": WorkerRole.PLANNER,
                    }
                )

                return ControllerDecision(
                    decision_type=ControllerDecisionType.DISPATCH_PLANNER,
                    reason="replan_request",
                    next_worker=WorkerRole.PLANNER,
                    requires_checkpoint=True,
                    cursor=cursor,
                )

            case BrainOutcome.FINAL_ANSWER:

                cursor = controller_input.cursor.model_copy(
                    update={
                        "phase": ExecutionPhase.TERMINATING,
                        "current_worker": WorkerRole.SUMMARY,
                    }
                )

                return ControllerDecision(

                    accepted_plan=self._update_active_step_status(
                        controller_input,
                        StepStatus.COMPLETED,
                    ),
                    decision_type=ControllerDecisionType.DISPATCH_SUMMARY,
                    reason="final_answer",
                    next_worker=WorkerRole.SUMMARY,
                    cursor=cursor,
                    terminal=True,
                    clear_last_tool_result=True,
                )
                #return self._dispatch_summary("final_answer")

            case BrainOutcome.STEP_COMPLETED:
                return self._advance_to_next_step(controller_input)

            case BrainOutcome.STEP_FAILED:
                return self._terminate("step_failed")

            case BrainOutcome.CONTINUE:
                return self._dispatch_brain(cursor=controller_input.cursor, reason="continue")

        raise ValueError(f"Unsupported brain outcome: {brain_result.outcome}")

    def _decide_from_tool(
        self,
        controller_input: ControllerInput,
    ) -> ControllerDecision:
        tool_result: ToolResult = controller_input.tool_result

        if tool_result.success:
            return self._dispatch_brain(cursor=controller_input.cursor, reason="Tool completed.")

        return self._dispatch_brain(cursor=controller_input.cursor, reason="Tool failed.")

    def _dispatch_planner(self, reason: str) -> ControllerDecision:
        return ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_PLANNER,
            next_worker=WorkerRole.PLANNER,
            reason=reason,
            requires_checkpoint=True,
        )

    def _dispatch_brain(
        self,
        cursor: ExecutionCursor,
        reason: str,
        next_step_id: str | None = None,
        clear_last_tool_result: bool = False,
        clear_active_step: bool = False,
    ) -> ControllerDecision:

        next_cursor = cursor.model_copy(
            update={
                "current_worker": WorkerRole.BRAIN,
                "step_id": next_step_id,
                "phase": ExecutionPhase.EXECUTING,
            }
        )

        return ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_BRAIN,
            next_worker=WorkerRole.BRAIN,
            clear_last_tool_result=clear_last_tool_result,
            cursor=next_cursor,
            next_step_id=next_step_id,
            reason=reason,
            clear_active_step=clear_active_step,
        )

    def _dispatch_tool(self, reason: str) -> ControllerDecision:
        return ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_TOOL_RUNTIME,
            next_worker=WorkerRole.TOOL_RUNTIME,
            reason=reason,
        )

    def _dispatch_summary(self, reason: str) -> ControllerDecision:
        return ControllerDecision(
            decision_type=ControllerDecisionType.DISPATCH_SUMMARY,
            next_worker=WorkerRole.SUMMARY,
            reason=reason,
            terminal=True,
        )

    def _terminate(self, reason: str) -> ControllerDecision:
        return ControllerDecision(
            decision_type=ControllerDecisionType.TERMINATE,
            reason=reason,
            terminal=True,
        )

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
        """Advance execution to the next pending step."""

        plan = controller_input.active_plan

        if plan is None:
            return self._dispatch_summary("No active plan.")

        current = controller_input.active_step

        if current is None:
            if not plan.steps:
                return self._dispatch_summary("Plan completed.")

            next_step = plan.steps[0]

        else:
            try:
                index = next(
                    i
                    for i, step in enumerate(plan.steps)
                    if step.step_id == current.step_id
                )
            except StopIteration:
                return self._terminate("Active step not found in plan.")

            if index + 1 >= len(plan.steps):
                return self._dispatch_brain(
                    cursor=controller_input.cursor,
                    reason="Generate final answer.",
                    next_step_id=None,
                    clear_active_step=True,
                    clear_last_tool_result=True,
                )
            
            next_step = plan.steps[index + 1]

        return self._dispatch_brain(
            cursor=controller_input.cursor,
            reason="Advance to next step.",
            next_step_id=next_step.step_id,
            clear_last_tool_result=True,
        )