
from langchain_core.messages import ToolMessage

from core.graph_messages import tool_message_content
from core.protocol.models import ToolExecutionRecord, ToolRequest, ToolResult
from core.state import AgentState
from core.tool_output import build_artifact_records, compute_repeat_fail_count, normalize_tool_output

def _build_tool_execution_record(
    *,
    execution_state,
    request: ToolRequest,
    result: ToolResult,
) -> ToolExecutionRecord:
    protocol = execution_state.protocol_visible
    active_step = protocol.active_step
    active_plan = protocol.active_plan
    step_id = active_step.step_id 

    return ToolExecutionRecord(
        execution_id=protocol.identity.execution_id,
        plan_id=active_plan.plan_id if active_plan is not None else None,
        plan_revision=active_plan.revision if active_plan is not None else None,
        step_id=step_id,
        tool_name=request.tool_name,
        arguments=request.arguments,
        result=result,
        artifacts=build_artifact_records(
            result.artifacts,
            step_id=step_id,
        ),
    )

def create_capture_tool_output_node():
    def capture_tool_output_node(state: AgentState):

        execution_state = state["execution_state"]
        decision = state.get("controller_decision")
        history = state.get("messages", [])

        active_step = execution_state.protocol_visible.active_step
        if active_step is None:
            raise RuntimeError(
                "Capture executed without an active step."
            )

        if not history:
            return {}

        last_message = history[-1]
        if not isinstance(last_message, ToolMessage):
            return {}

        if decision is None or decision.pending_tool_request is None:
            raise RuntimeError(
                "Capture executed without pending_tool_request."
            )

        raw_content = tool_message_content(last_message)

        tool_result = normalize_tool_output(
            raw_content=raw_content,
            request=decision.pending_tool_request,
        )

        working = execution_state.working
        active_step = execution_state.protocol_visible.active_step

        repeat_fail_count = compute_repeat_fail_count(
            previous=working.last_tool_result,
            previous_repeat_count=working.repeat_fail_count,
            current=tool_result,
        )

        tool_execution_record = _build_tool_execution_record(
            execution_state=execution_state,
            request=decision.pending_tool_request,
            result=tool_result,
        )

        updated_history = (
            *working.tool_execution_history,
            tool_execution_record,
        )

        working = working.model_copy(
            update={
                "last_tool_result": tool_result,
                "tool_execution_history": updated_history,
                "repeat_fail_count": repeat_fail_count,
            }
        )

        updated_execution_state = execution_state.model_copy(
            update={
                "working": working,
            }
        )


        return {
            "execution_state": updated_execution_state,
        }

    return capture_tool_output_node
