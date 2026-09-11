from langchain_core.messages import AIMessage, HumanMessage

import core.graph_brain as graph_brain
from core.protocol.enums import BrainOutcome, ExecutionPhase
from core.protocol.models import (
    BrainInput,
    ExecutionContext,
    ExecutionCursor,
    ExecutionIdentity,
)


class FakeLLM:
    def __init__(self, reply_text: str):
        self.reply_text = reply_text

    def invoke(self, _messages):
        return AIMessage(content=self.reply_text)


def _brain_node(monkeypatch, *, direct_response: bool):
    brain_input = BrainInput(
        identity=ExecutionIdentity(
            execution_id="stage-1",
            protocol_version="1.0",
        ),
        cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING),
        context=ExecutionContext(user_request="hello"),
        direct_response=direct_response,
    )
    monkeypatch.setattr(graph_brain, "build_brain_input", lambda _state: brain_input)
    return graph_brain.create_brain_node(
        brain_llm=FakeLLM("ordinary direct reply"),
        tool_brain_llm=FakeLLM("unused"),
        agent_system_prompt="agent",
        casual_system_prompt="casual",
        tools_set=set(),
        show_raw_llm=False,
    )


def test_explicit_direct_response_context_requests_finalization_without_answer(
    monkeypatch,
):
    result = _brain_node(monkeypatch, direct_response=True)(
        {"messages": [HumanMessage(content="hello")], "steps": 0}
    )

    assert result["brain_result"].outcome == BrainOutcome.FINAL_ANSWER
    assert not hasattr(result["brain_result"], "final_answer")
    assert result["messages"][0].content == "Finalization requested."


def test_absence_of_plan_alone_does_not_claim_final_answer(monkeypatch):
    result = _brain_node(monkeypatch, direct_response=False)(
        {"messages": [HumanMessage(content="hello")], "steps": 0}
    )

    assert result["brain_result"].outcome == BrainOutcome.STEP_FAILED
