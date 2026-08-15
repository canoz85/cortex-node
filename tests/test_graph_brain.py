from langchain_core.messages import AIMessage, HumanMessage

import core.graph_brain as graph_brain
from core.protocol.enums import ExecutionPhase
from core.protocol.models import (
    BrainInput,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionState,
    ExecutionStep,
    ProtocolVisibleState,
    ToolExecutionRecord,
    ToolResult,
)


class FakeLLM:
    def __init__(self, reply_text: str):
        self.reply_text = reply_text
        self.invocations: list[list] = []

    def invoke(self, messages):
        self.invocations.append(list(messages))
        return AIMessage(content=self.reply_text)


def test_brain_node_constructs_brain_input_once(monkeypatch):
    bridge_calls: list[dict] = []

    def fake_build_brain_input(state):
        bridge_calls.append(state)
        return BrainInput(
            identity=ExecutionIdentity(execution_id="run-1", protocol_version="1.0"),
            cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING),
            context=ExecutionContext(user_request="hello"),
        )

    monkeypatch.setattr(graph_brain, "build_brain_input", fake_build_brain_input)

    brain_llm = FakeLLM("discussion reply")
    tool_brain_llm = FakeLLM("action reply")

    brain_node = graph_brain.create_brain_node(
        brain_llm=brain_llm,
        tool_brain_llm=tool_brain_llm,
        agent_system_prompt="agent prompt",
        final_answer_system_prompt="final prompt",
        tool_completed_system_prompt="tool completed prompt",
        casual_system_prompt="casual prompt",
        tools_set=set(),
        show_raw_llm=False,
    )

    state = {
        "messages": [HumanMessage(content="hello")],
        "plan": "",
        "steps": 0,
        "retrieval_messages": [],
        "rolling_summary": "",
        "last_tool_output": "",
        "last_tool_signature": "",
    }

    result = brain_node(state)

    assert len(bridge_calls) == 1
    assert bridge_calls[0] is state
    assert result["steps"] == 1
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "discussion reply"
    assert len(brain_llm.invocations) == 1
    assert len(tool_brain_llm.invocations) == 0


def _execution_state_with_step(step_id: str) -> ExecutionState:
    return ExecutionState(
        protocol_visible=ProtocolVisibleState(
            identity=ExecutionIdentity(
                execution_id="run-1",
                protocol_version="1.0",
            ),
            cursor=ExecutionCursor(
                phase=ExecutionPhase.EXECUTING,
                step_id=step_id,
            ),
            active_plan=ExecutionPlan(
                plan_id="p-1",
                revision=1,
                objective="demo objective",
                steps=(
                    ExecutionStep(step_id="s1", title="step 1"),
                    ExecutionStep(step_id="s2", title="step 2"),
                ),
            ),
            active_step=ExecutionStep(
                step_id=step_id,
                title=step_id,
                description="demo step",
            ),
        ),
    )


def test_tool_completed_messages_include_structured_tool_progress(monkeypatch):
    tool_history = (
        ToolExecutionRecord(
            step_id="s1",
            tool_name="list_files",
            arguments={"path": "."},
            result=ToolResult(
                request_id="req-1",
                signature='list_files:{"path": "."}',
                success=True,
                message="Listing for .",
                rendered_output="Files under .:\n- a.py",
                data={"entries": ["a.py"]},
            ),
        ),
    )

    brain_input = BrainInput(
        identity=ExecutionIdentity(execution_id="run-1", protocol_version="1.0"),
        cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING, step_id="s2"),
        context=ExecutionContext(user_request="read all python files"),
        active_plan=ExecutionPlan(
            plan_id="p-1",
            revision=1,
            objective="demo objective",
            steps=(ExecutionStep(step_id="s2", title="read all"),),
        ),
        active_step=ExecutionStep(
            step_id="s2",
            title="Read all Python files",
            description="Use read_file for each discovered .py file",
        ),
        last_tool_result=ToolResult(
            request_id="req-2",
            signature='read_file:{"path": "a.py"}',
            success=True,
            message="Read file: a.py",
            rendered_output="print('a')",
            data={"path": "a.py", "content": "print('a')"},
        ),
        tool_execution_history=tool_history,
    )

    monkeypatch.setattr(graph_brain, "build_brain_input", lambda _state: brain_input)

    brain_llm = FakeLLM("step-check")
    tool_brain_llm = FakeLLM("unused")

    brain_node = graph_brain.create_brain_node(
        brain_llm=brain_llm,
        tool_brain_llm=tool_brain_llm,
        agent_system_prompt="agent prompt",
        final_answer_system_prompt="final prompt",
        tool_completed_system_prompt="tool completed prompt",
        casual_system_prompt="casual prompt",
        tools_set=set(),
        show_raw_llm=False,
    )

    state = {
        "execution_state": _execution_state_with_step("s2"),
        "messages": [HumanMessage(content="read all python files")],
        "retrieval_messages": [],
        "rolling_summary": "",
        "steps": 0,
    }

    brain_node(state)

    assert len(brain_llm.invocations) == 1
    rendered_messages = [str(getattr(m, "content", "")) for m in brain_llm.invocations[0]]
    assert any("Tool execution progress (structured):" in content for content in rendered_messages)
    assert len(tool_brain_llm.invocations) == 0


def test_normal_execution_messages_include_structured_tool_progress(monkeypatch):
    tool_history = (
        ToolExecutionRecord(
            step_id="s1",
            tool_name="list_files",
            arguments={"path": "."},
            result=ToolResult(
                request_id="req-1",
                signature='list_files:{"path": "."}',
                success=True,
                message="Listing for .",
                rendered_output="Files under .:\n- a.py\n- b.py",
                data={"entries": ["a.py", "b.py"]},
            ),
        ),
    )

    brain_input = BrainInput(
        identity=ExecutionIdentity(execution_id="run-1", protocol_version="1.0"),
        cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING, step_id="s2"),
        context=ExecutionContext(user_request="read all python files"),
        active_plan=ExecutionPlan(
            plan_id="p-1",
            revision=1,
            objective="demo objective",
            steps=(ExecutionStep(step_id="s2", title="read all"),),
        ),
        active_step=ExecutionStep(
            step_id="s2",
            title="Read all Python files",
            description="Use read_file for each discovered .py file",
        ),
        last_tool_result=None,
        tool_execution_history=tool_history,
    )

    monkeypatch.setattr(graph_brain, "build_brain_input", lambda _state: brain_input)

    brain_llm = FakeLLM("unused")
    tool_brain_llm = FakeLLM("next tool request")

    brain_node = graph_brain.create_brain_node(
        brain_llm=brain_llm,
        tool_brain_llm=tool_brain_llm,
        agent_system_prompt="agent prompt",
        final_answer_system_prompt="final prompt",
        tool_completed_system_prompt="tool completed prompt",
        casual_system_prompt="casual prompt",
        tools_set=set(),
        show_raw_llm=False,
    )

    state = {
        "execution_state": _execution_state_with_step("s2"),
        "messages": [HumanMessage(content="read all python files")],
        "retrieval_messages": [],
        "rolling_summary": "",
        "steps": 0,
    }

    brain_node(state)

    assert len(tool_brain_llm.invocations) == 1
    rendered_messages = [str(getattr(m, "content", "")) for m in tool_brain_llm.invocations[0]]
    assert any("Tool execution progress (structured):" in content for content in rendered_messages)
