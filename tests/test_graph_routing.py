from langchain_core.messages import AIMessage
from langgraph.graph import END

from core.graph_routing import route_after_brain, route_after_planner
from core.graph_state_machine import apply_controller_decision_to_state, decide_controller_decision


from core.protocol.enums import BrainOutcome, ControllerDecisionType, ExecutionPhase, WorkerRole
from core.protocol.models import (
    BrainResult,
    ControllerDecision,
    ControllerInput,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionStep,
    PlannerResult,
    ToolRequest,
)

def test_decide_controller_decision_dispatches_tool_runtime_for_tool_request():
    controller_input = ControllerInput(
        identity=ExecutionIdentity(execution_id="run-1", protocol_version="1.0"),
        cursor=ExecutionCursor(phase="executing", controller_iteration=1),
        context=ExecutionContext(user_request="create file", role=WorkerRole.CONTROLLER),
        brain_result=BrainResult(
            outcome=BrainOutcome.TOOL_REQUEST,
            tool_request=ToolRequest(request_id="req-1", tool_name="write_file", arguments={"path": "x.py"}),
        ),
    )

    decision = decide_controller_decision({"messages": [AIMessage(content="tool needed")]}, controller_input=controller_input)

    assert decision.decision_type == ControllerDecisionType.DISPATCH_TOOL_RUNTIME
    assert decision.next_worker == WorkerRole.TOOL_RUNTIME
    assert decision.requires_checkpoint is False

def test_route_after_brain_returns_end_when_no_history():
    assert route_after_brain({"messages": [], "steps": 0}) == END


def test_route_after_brain_returns_tools_when_tool_calls_present():
    message = AIMessage(content="", tool_calls=[{"name": "list_files", "args": {}, "id": "1", "type": "tool_call"}])
    assert route_after_brain({"messages": [message], "steps": 1}) == "tools"


def test_route_after_brain_prefers_protocol_brain_tool_request_when_present():
    message = AIMessage(content="No tool calls in this message.")
    state = {
        "messages": [message],
        "steps": 1,
        "brain_outcome": "tool_request",
        "tool_request": {
            "request_id": "req-1",
            "tool_name": "list_files",
            "arguments": {"path": "."},
            "requested_by": "brain",
        },
    }

    assert route_after_brain(state) == "tools"


def test_route_after_brain_returns_end_when_step_limit_reached():
    message = AIMessage(content="", tool_calls=[{"name": "list_files", "args": {}, "id": "1", "type": "tool_call"}])
    assert route_after_brain({"messages": [message], "steps": 999}) == END


def test_decide_controller_decision_dispatches_brain_for_planner_result():
    controller_input = ControllerInput(
        identity=ExecutionIdentity(execution_id="run-2", protocol_version="1.0"),
        cursor=ExecutionCursor(phase="planning", controller_iteration=1),
        context=ExecutionContext(user_request="inspect workspace", role=WorkerRole.CONTROLLER),
        planner_result=PlannerResult(
            proposed_plan=ExecutionPlan(
                plan_id="plan-1",
                revision=1,
                objective="Inspect workspace",
                steps=(ExecutionStep(step_id="step-1", title="Inspect", description="Inspect"),),
            ),
            message="Plan generated",
        ),
    )

    decision = decide_controller_decision({"messages": [AIMessage(content="planner output")]}, controller_input=controller_input)

    assert decision.decision_type == ControllerDecisionType.DISPATCH_BRAIN
    assert decision.next_worker == WorkerRole.BRAIN


def test_route_after_planner_returns_brain_when_planner_result_is_present():
    planner_result = PlannerResult(
        proposed_plan=ExecutionPlan(
            plan_id="plan-1",
            revision=1,
            objective="Inspect workspace",
            steps=(ExecutionStep(step_id="step-1", title="Inspect", description="Inspect"),),
        ),
        message="Plan generated",
    )
    state = {"messages": [AIMessage(content="planner output")], "planner_result": planner_result}

    assert route_after_planner(state) == "brain"


def test_decide_controller_decision_requests_replan_for_replan_outcome():
    controller_input = ControllerInput(
        identity=ExecutionIdentity(execution_id="run-3", protocol_version="1.0"),
        cursor=ExecutionCursor(phase="executing", controller_iteration=2),
        context=ExecutionContext(user_request="recover from failure", role=WorkerRole.CONTROLLER),
        brain_result=BrainResult(outcome=BrainOutcome.REPLAN_REQUEST),
    )

    decision = decide_controller_decision({"messages": [AIMessage(content="replan")]} , controller_input=controller_input)

    assert decision.decision_type == ControllerDecisionType.REQUEST_REPLAN
    assert decision.next_worker == WorkerRole.PLANNER
    assert decision.requires_checkpoint is True
    assert decision.cursor is not None
    assert decision.cursor.phase == "replanning"


def test_decide_controller_decision_marks_summary_cursor_for_final_answer():
    controller_input = ControllerInput(
        identity=ExecutionIdentity(execution_id="run-4", protocol_version="1.0"),
        cursor=ExecutionCursor(phase="executing", controller_iteration=3),
        context=ExecutionContext(user_request="finish task", role=WorkerRole.CONTROLLER),
        brain_result=BrainResult(outcome=BrainOutcome.FINAL_ANSWER),
    )

    decision = decide_controller_decision({"messages": [AIMessage(content="final")]}, controller_input=controller_input)

    assert decision.decision_type == ControllerDecisionType.DISPATCH_SUMMARY
    assert decision.cursor is not None
    assert decision.cursor.phase == "terminating"
    assert decision.cursor.current_worker == WorkerRole.SUMMARY


def test_route_after_brain_routes_summary_for_final_answer():
    controller_input = ControllerInput(
        identity=ExecutionIdentity(execution_id="run-4", protocol_version="1.0"),
        cursor=ExecutionCursor(phase="executing", controller_iteration=3),
        context=ExecutionContext(user_request="finish task", role=WorkerRole.CONTROLLER),
        brain_result=BrainResult(outcome=BrainOutcome.FINAL_ANSWER),
    )

    state = {"messages": [AIMessage(content="final")], "brain_result": BrainResult(outcome=BrainOutcome.FINAL_ANSWER)}
    state["execution_state"] = {"protocol_visible": {"cursor": controller_input.cursor}, "working": {}}

    assert route_after_brain(state) == "summarize_memory"


def test_route_after_brain_applies_controller_state_updates_during_routing():
    state = {
        "messages": [AIMessage(content="replan")],
        "run_id": "run-6",
        "steps": 1,
        "brain_result": BrainResult(outcome=BrainOutcome.REPLAN_REQUEST),
    }

    routed = route_after_brain(state)

    assert routed == "planner"
    assert state["phase"] == ExecutionPhase.REPLANNING.value
    assert state["current_worker"] == WorkerRole.PLANNER.value
    assert state["steps"] == 1
    assert state["checkpoint_id"].startswith("run-6")


def test_apply_controller_decision_updates_execution_state_for_resume():
    state = {"messages": [AIMessage(content="resume me")], "steps": 1, "run_id": "run-5"}
    decision = ControllerDecision(
        decision_type=ControllerDecisionType.REQUEST_REPLAN,
        reason="resume_checkpoint",
        next_worker=WorkerRole.PLANNER,
        requires_checkpoint=True,
        cursor=ExecutionCursor(phase="replanning", controller_iteration=2, current_worker=WorkerRole.PLANNER),
    )

    updated_state = apply_controller_decision_to_state(state, decision)

    assert updated_state["current_worker"] == WorkerRole.PLANNER.value
    assert updated_state["phase"] == ExecutionPhase.REPLANNING.value
    assert updated_state["steps"] == 2
    assert updated_state["checkpoint_id"].startswith("run-5")
    execution_state = updated_state["execution_state"]
    assert execution_state.protocol_visible.cursor.phase == ExecutionPhase.REPLANNING
    assert execution_state.protocol_visible.cursor.current_worker == WorkerRole.PLANNER
