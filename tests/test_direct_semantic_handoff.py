"""Planner semantics cross only the Controller-accepted downstream boundary."""

import pytest

from core.brain import _build_brain_execution_brief
from core.finalizer import Finalizer
from core.finalizer_provider import LangChainFinalAnswerRenderer
from core.planner import PlannerService, PLANNER_SYSTEM_PROMPT
from core.planner_contract import PlannerProposal, PlannerProposalResultType, ProposedStep
from core.planner import PlannerRoute
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

class FakePlannerRouter:
    def __init__(self, route: str = "action"):
        self.route_value = route

    def route(self, user_request: str):
        return PlannerRoute(route=self.route_value)

class Planner:
    def __init__(self, semantic, *, planned=False):
        self.semantic = semantic
        self.planned = planned
        self.requests = []

    def run(self, request):
        self.requests.append(request)

        class Provider:
        
            def generate(self, messages):
                memory_message = next(message.content for message in messages
                                      if message.content.startswith("PLANNER MEMORY CONTEXT"))
                assert "Amber" in memory_message
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
            provider=Provider(), router=FakePlannerRouter(), mutating_tools=set(),
        ).run(request)


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
        planner_memory_context=MEMORY,
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


def test_accepted_step_semantics_reach_brain_with_scripted_provider():
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
    assert "Any value required for execution must appear in the responsible step semantics" in PLANNER_SYSTEM_PROMPT


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
