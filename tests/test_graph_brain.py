from langchain_core.messages import AIMessage, HumanMessage

import core.graph_brain as graph_brain


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
        return object()

    monkeypatch.setattr(graph_brain, "build_brain_input", fake_build_brain_input)

    brain_llm = FakeLLM("discussion reply")
    tool_brain_llm = FakeLLM("action reply")

    brain_node = graph_brain.create_brain_node(
        brain_llm=brain_llm,
        tool_brain_llm=tool_brain_llm,
        agent_system_prompt="agent prompt",
        casual_system_prompt="casual prompt",
        tool_name_set=set(),
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
