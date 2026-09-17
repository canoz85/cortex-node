"""Opt-in real-model check of Planner memory resolution before any tool runs."""

import os
from uuid import uuid4

import pytest
from langchain_ollama import ChatOllama

from core.planner import PlannerRouting, PlannerService
from core.planner_provider import LangChainPlannerProvider
from core.protocol.controller import CortexController
from core.protocol.enums import PlannerOutcome, WorkerRole
from core.protocol.models import (
    BrainInput, ControllerInput, ExecutionContext, ExecutionCursor,
    ExecutionIdentity, ExecutionState, PlannerMemoryContext, PlannerMemoryFact,
    PlanningCapabilities, ProtocolVisibleState,
)
from core.runtime.controller_transition import apply_controller_decision_to_state
from main import DEFAULT_SETTINGS


@pytest.mark.skipif(
    os.getenv("CORTEX_LIVE_PLANNER") != "1",
    reason="Set CORTEX_LIVE_PLANNER=1 to call the configured local Planner model",
)
def test_real_planner_resolves_memory_into_accepted_step_before_tools(tmp_path, monkeypatch):
    # No workspace evidence or tool runtime exists in this test.
    monkeypatch.chdir(tmp_path)
    marker = f"Marker-{uuid4().hex[:12]}"
    user_request = "Write a short note in note.txt using my preferred marker."
    projection = PlannerMemoryContext(user_facts=(PlannerMemoryFact(
        category="user_profile",
        text=f"The user's preferred marker is {marker}.",
        authority="explicit_user", source_turn_index=1,
    ),))
    identity = ExecutionIdentity(execution_id=f"live-{uuid4().hex}", protocol_version="1")
    context = ExecutionContext(user_request=user_request)
    controller = CortexController(
        20, planning_capabilities=PlanningCapabilities(available_tools=("write_file",)),
    )
    dispatch = controller.decide(ControllerInput(
        identity=identity, cursor=ExecutionCursor(), context=context,
    ))
    request = dispatch.planning_request
    worker_request = request.model_copy(update={"context": request.context.model_copy(update={
        "planner_memory_context": projection,
    })})

    model = os.getenv("CORTEX_MODEL_PLANNER") or DEFAULT_SETTINGS["model_planner"]
    transport = LangChainPlannerProvider(planner_llm=ChatOllama(
        model=model, temperature=0, client_kwargs={"timeout": 120},
    ))

    class LiveProvider:
        raw_proposal = None

        def route(self, user_request):
            return PlannerRouting("action")

        def generate(self, messages):
            self.raw_proposal = transport.generate(messages)
            return self.raw_proposal

    provider = LiveProvider()
    result = PlannerService(
        provider=provider, tools_set={"write_file"}, domain_tool_map={},
        mutating_tools=set(), system_capabilities_text="",
    ).run(worker_request, retrieve=lambda _: ())

    assert provider.raw_proposal is not None
    print("raw Planner proposal:", provider.raw_proposal.model_dump_json())
    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    print("normalized ExecutionPlan:", result.proposed_plan.model_dump_json())
    assert any(marker in f"{step.title} {step.description}" for step in provider.raw_proposal.steps)
    assert any(marker in f"{step.title} {step.description}" for step in result.proposed_plan.steps)

    accepted = controller.decide(ControllerInput(
        identity=identity, cursor=dispatch.cursor, context=context,
        planning_request=request, planning_sequence=request.sequence,
        planner_result=result,
    ))
    state = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=identity, cursor=dispatch.cursor,
        planning_request=request, planning_sequence=request.sequence,
    ))
    protocol = apply_controller_decision_to_state(state, accepted).protocol_visible
    print("Controller-accepted active step:", protocol.active_step.model_dump_json())
    assert marker in f"{protocol.active_step.title} {protocol.active_step.description}"

    brain_input = BrainInput(
        identity=identity, cursor=protocol.cursor,
        context=ExecutionContext(user_request=user_request, role=WorkerRole.BRAIN),
        active_plan=protocol.active_plan, active_step=protocol.active_step,
    )
    print("BrainInput active step:", brain_input.active_step.model_dump_json())
    assert marker in f"{brain_input.active_step.title} {brain_input.active_step.description}"
    assert brain_input.context.planner_memory_context is None
