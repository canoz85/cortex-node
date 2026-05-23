from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from core.graph_planner import create_planner_node


class DummyPlannerLLM:
    def __init__(self, response_text: str = "1. plan\n2. tools\n3. done"):
        self.response_text = response_text
        self.invocations: list[list] = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return SimpleNamespace(content=self.response_text)


class DummyRAG:
    def __init__(self, context: str = ""):
        self.context = context

    def format_context(self, query: str, top_k: int):
        if self.context:
            return self.context
        return ""


def test_planner_node_info_route_uses_synthetic_plan():
    llm = DummyPlannerLLM("unused")
    planner_node = create_planner_node(planner_llm=llm, rag_service=DummyRAG(), rag_top_k=4)

    state = {"messages": [HumanMessage(content="show token usage")], "rolling_summary": ""}
    result = planner_node(state)

    assert result["planner_route"] == "info"
    assert result["planner_plan_source"] == "synthetic"
    assert "call token_usage tool" in result["plan"]


def test_planner_node_action_route_uses_llm_plan():
    llm = DummyPlannerLLM("1. create file\n2. run python\n3. summarize")
    planner_node = create_planner_node(planner_llm=llm, rag_service=DummyRAG(), rag_top_k=4)

    state = {"messages": [HumanMessage(content="create hello.py and run it")], "rolling_summary": ""}
    result = planner_node(state)

    assert result["planner_route"].startswith("action")
    assert result["planner_plan_source"] == "llm"
    assert "create file" in result["plan"]
    assert len(llm.invocations) >= 1


def test_planner_node_preserves_domain_metadata():
    llm = DummyPlannerLLM("unused")
    planner_node = create_planner_node(planner_llm=llm, rag_service=DummyRAG(), rag_top_k=4)

    state = {"messages": [HumanMessage(content="[domain:sap] create report")], "rolling_summary": ""}
    result = planner_node(state)

    assert result["planner_domain"] == "sap"
    assert result["planner_domain_enforced"] is True
