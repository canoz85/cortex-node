"""Planner semantics cross only the Controller-accepted downstream boundary."""

import pytest

from core.brain import _build_brain_execution_brief
from core.finalizer import Finalizer
from core.finalizer_provider import LangChainFinalAnswerRenderer
from core.planner import PlannerRouting, PlannerService, PLANNER_SYSTEM_PROMPT
from core.planner_contract import PlannerProposal, PlannerProposalResultType, ProposedStep
from core.protocol.controller import CortexController
from core.protocol.enums import ControllerDecisionType, ExecutionStatus, PlannerOutcome
from core.protocol.models import (
    AcceptedDirectResponse, ControllerInput, ExecutionContext, ExecutionCursor, ExecutionIdentity,
    ExecutionState, FinalizationRequest, PlannerMemoryContext, PlannerMemoryFact, PlannerResult,
    PlanningCapabilities, ProtocolVisibleState,
)
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver


IDENTITY = ExecutionIdentity(execution_id="handoff-run", protocol_version="1")
MEMORY = PlannerMemoryContext(user_facts=(PlannerMemoryFact(
    category="user_profile", text="The user's preferred signature is Amber.",
    authority="explicit_user", source_turn_index=1,
),))


class Planner:
    def __init__(self, semantic, *, planned=False):
        self.semantic = semantic
        self.planned = planned
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        worker_request = request.model_copy(update={"context": request.context.model_copy(update={
            "planner_memory_context": MEMORY,
        })})

        class Provider:
            def route(self, user_request):
                return PlannerRouting("action" if self_planned else "conversation")

            def generate(self, messages):
                assert "Amber" in messages[-2].content
                if self_planned:
                    return PlannerProposal(
                        result=PlannerProposalResultType.PLAN_PROPOSED,
                        steps=(ProposedStep(
                            step_id="write", title="Write the requested note",
                            description=f"Write the note using the signature {self_semantic}.",
                            primary_tool="write_file",
                        ),),
                    )
                return PlannerProposal(
                    result=PlannerProposalResultType.NO_PLAN_REQUIRED,
                    message=self_semantic,
                )

        self_planned, self_semantic = self.planned, self.semantic
        return PlannerService(
            provider=Provider(), tools_set={"write_file"}, domain_tool_map={},
            mutating_tools=set(), system_capabilities_text="",
        ).run(worker_request)


class CaptureBrain:
    def __init__(self):
        self.inputs = []

    def run(self, value):
        self.inputs.append(value)
        raise RuntimeError("Stop after inspecting authorized Brain input")


class CaptureFinalizer:
    def __init__(self):
        self.requests = []

    def finalize(self, request):
        self.requests.append(request)
        return Finalizer().finalize(request)


def _input(state, request_text, planner_result=None):
    protocol = state.protocol_visible
    return ControllerInput(
        identity=protocol.identity, cursor=protocol.cursor,
        context=ExecutionContext(user_request=request_text),
        active_plan=protocol.active_plan, active_step=protocol.active_step,
        planning_request=protocol.planning_request,
        planning_sequence=protocol.planning_sequence,
        planner_result=planner_result,
    )


def _start(request_text, planner):
    finalizer = CaptureFinalizer()
    brain = CaptureBrain()
    driver = ExecutionDriver(
        coordinator=ControllerCoordinator(CortexController(
            20, planning_capabilities=PlanningCapabilities(available_tools=("write_file",)),
        )), planner=planner, brain=brain, tool_runtime=object(), finalizer=finalizer,
    )
    initial = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=IDENTITY, cursor=ExecutionCursor(),
    ))
    first = driver.turn(initial, _input(initial, request_text))
    assert first.decision.decision_type == ControllerDecisionType.DISPATCH_PLANNER
    assert first.worker_result.request_id == first.decision.planning_request.request_id
    return driver, first, brain, finalizer


def test_memory_backed_direct_semantics_are_accepted_bound_and_presented():
    planner = Planner("Your preferred signature is Amber.")
    driver, first, brain, finalizer = _start("What is my preferred signature?", planner)
    second = driver.turn(first.execution_state, _input(
        first.execution_state, "What is my preferred signature?", first.worker_result,
    ))
    accepted = second.execution_state.protocol_visible.accepted_direct_response
    assert second.decision.decision_type == ControllerDecisionType.DISPATCH_SUMMARY
    assert accepted.content == "Your preferred signature is Amber."
    assert accepted.request_id == first.decision.planning_request.request_id
    assert accepted.execution_id == IDENTITY.execution_id
    restored = ExecutionState.model_validate_json(second.execution_state.model_dump_json())
    assert restored.protocol_visible.accepted_direct_response == accepted
    assert finalizer.requests[0].accepted_direct_response == accepted
    assert finalizer.requests[0].context.planner_memory_context is None
    assert second.worker_result.final_answer == accepted.content
    assert not brain.inputs
    assert second.execution_state.protocol_visible.active_plan is None
    assert second.execution_state.protocol_visible.completed_step_ids == ()
    assert second.execution_state.protocol_visible.pending_tool_request is None


def test_empty_no_plan_and_generic_message_create_no_semantic_answer():
    planner = Planner("")
    driver, first, _, finalizer = _start("Just talk", planner)
    assert first.worker_result.message == "No execution plan required."
    assert first.worker_result.direct_response_content is None
    second = driver.turn(first.execution_state, _input(first.execution_state, "Just talk", first.worker_result))
    assert second.execution_state.protocol_visible.accepted_direct_response is None
    assert finalizer.requests[0].accepted_direct_response is None

    generic = PlannerResult(
        outcome=PlannerOutcome.DIRECT_RESPONSE,
        request_id=first.decision.planning_request.request_id,
        message="No tools required",
    )
    third = driver.turn(first.execution_state, _input(first.execution_state, "Just talk", generic))
    assert third.execution_state.protocol_visible.accepted_direct_response is None


def test_stale_or_unaccepted_direct_semantics_cannot_reach_finalizer():
    driver, first, _, finalizer = _start("Question", Planner("Private answer"))
    stale = first.worker_result.model_copy(update={"request_id": "another-request"})
    with pytest.raises(ValueError, match="request identity"):
        driver.turn(first.execution_state, _input(first.execution_state, "Question", stale))
    assert finalizer.requests == []
    with pytest.raises(ValueError, match="completed plan-free execution"):
        ProtocolVisibleState(
            identity=IDENTITY, cursor=ExecutionCursor(),
            status=ExecutionStatus.COMPLETED,
            accepted_direct_response=AcceptedDirectResponse(
                execution_id="another-execution", request_id="another-request",
                content="Private answer",
            ),
        )


def test_memory_resolved_step_reaches_brain_without_planner_projection():
    planner = Planner("Amber", planned=True)
    driver, first, brain, finalizer = _start("Write a note using my signature", planner)
    assert "Amber" in first.worker_result.proposed_plan.steps[0].description
    with pytest.raises(RuntimeError, match="Stop after"):
        driver.turn(first.execution_state, _input(
            first.execution_state, "Write a note using my signature", first.worker_result,
        ))
    assert len(brain.inputs) == 1
    brain_input = brain.inputs[0]
    assert "Amber" in brain_input.active_step.description
    assert "Amber" in _build_brain_execution_brief(brain_input)
    assert brain_input.context.planner_memory_context is None
    assert brain_input.active_plan.steps[0] == brain_input.active_step
    assert finalizer.requests == []
    assert "put the concrete value in the responsible step's title or description" in PLANNER_SYSTEM_PROMPT


def test_direct_result_is_not_a_tool_or_step_authority():
    with pytest.raises(ValueError, match="direct response content"):
        PlannerResult(
            outcome=PlannerOutcome.EXECUTION_PLAN, request_id="request",
            direct_response_content="A fact",
        )
    assert not hasattr(MEMORY, "tool_request")
    assert not hasattr(MEMORY, "completed_step_ids")


def test_model_backed_finalizer_presents_accepted_direct_content_without_model_call():
    class Model:
        def invoke(self, messages):
            raise AssertionError("accepted direct response must not require another model decision")

    planner = Planner("Your preferred signature is Amber.")
    driver, first, _, _ = _start("What is my preferred signature?", planner)
    second = driver.turn(first.execution_state, _input(
        first.execution_state, "What is my preferred signature?", first.worker_result,
    ))
    request = FinalizationRequest(
        identity=IDENTITY, status=second.execution_state.protocol_visible.status,
        context=ExecutionContext(user_request="What is my preferred signature?"),
        accepted_direct_response=second.execution_state.protocol_visible.accepted_direct_response,
    )
    result = Finalizer(answer_renderer=LangChainFinalAnswerRenderer(llm=Model())).finalize(request)
    assert result.final_answer == "Your preferred signature is Amber."
