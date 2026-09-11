"""Current Planner adapter characterization, including legacy limitations."""
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from core.graph_planner import create_planner_node
from core.protocol.enums import PlannerOutcome
from core.protocol.models import (
    ExecutionCursor, ExecutionIdentity, ExecutionState, PlannerResult, ProtocolVisibleState,
)


class DummyPlannerLLM:
    def __init__(self, text, *, route="action", domain="workspace", confidence=0.95):
        self.text = text
        self.routing = SimpleNamespace(route=route, domain=domain, confidence=confidence,
                                       enforced=False, reason="fixture")
        self.invocations = []
        self.routes = []

    def with_structured_output(self, schema, method):
        def invoke(messages):
            self.routes.append(messages)
            return self.routing
        return SimpleNamespace(invoke=invoke)

    def invoke(self, messages):
        self.invocations.append(messages)
        if isinstance(self.text, Exception):
            raise self.text
        return SimpleNamespace(content=self.text)


class DummyRAG:
    def __init__(self):
        self.calls = []

    def format_context(self, query, top_k):
        self.calls.append((query, top_k))
        return "retrieved context"


def make_node(llm, rag):
    return create_planner_node(planner_llm=llm, router_llm=llm, rag_service=rag,
                               rag_top_k=4, tools_set={"list_files", "write_file", "query_abap_table"})


def test_current_numbered_plan_and_retrieval():
    text = "1. Inspect – Use `list_files` to inspect.\n2. Write - Use `write_file` to create."
    llm, rag = DummyPlannerLLM(text), DummyRAG()
    state = {"messages": [HumanMessage(content="create a file")]}
    update = make_node(llm, rag)(state)
    result = update["planner_result"]
    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    assert result.proposed_plan.objective == text
    assert [s.primary_tool for s in result.proposed_plan.steps] == ["list_files", "write_file"]
    assert result.proposed_plan.steps[1].depends_on_step_ids == ("step-1",)
    assert rag.calls == [("create a file", 4)]
    assert [m.content for m in update["retrieval_messages"]] == ["retrieved context"]
    assert llm.invocations[0][1].content == "retrieved context"
    assert set(update) == {"planner_result", "retrieval_messages"}
    assert set(state) == {"messages"}


def test_legacy_parser_skips_unmatched_lines_without_repair():
    llm = DummyPlannerLLM("intro\n1. Inspect – Use `list_files`.\n2. malformed\n3. Verify - Use `read_file`.")
    result = make_node(llm, DummyRAG())({"messages": [HumanMessage(content="inspect")]})["planner_result"]
    assert [s.step_id for s in result.proposed_plan.steps] == ["step-1", "step-3"]
    assert result.proposed_plan.steps[1].depends_on_step_ids == ("step-1",)


@pytest.mark.parametrize("route", ["conversation", "clarify_domain"])
def test_direct_routes_do_not_generate_or_retrieve(route):
    llm, rag = DummyPlannerLLM("unused", route=route), DummyRAG()
    update = make_node(llm, rag)({"messages": [HumanMessage(content="hello")]})
    assert update["planner_result"].outcome == PlannerOutcome.DIRECT_RESPONSE
    assert update["retrieval_messages"] == []
    assert llm.invocations == []
    assert rag.calls == []


@pytest.mark.parametrize("route,has_write", [("info", False), ("action", True)])
def test_current_tool_filtering(route, has_write):
    llm = DummyPlannerLLM("1. Inspect – Use `list_files`.", route=route)
    make_node(llm, DummyRAG())({"messages": [HumanMessage(content="inspect")]})
    prompt = llm.invocations[0][0].content
    tools = prompt.split("AVAILABLE TOOLS FOR THIS REQUEST", 1)[1].split("PLANNING RULES:", 1)[0]
    assert "- list_files" in tools
    assert ("- write_file" in tools) == has_write
    assert "- query_abap_table" not in tools
    assert all(f"- {tool}" in tools for tool in ("agent_info", "token_usage", "current_time"))


def test_low_confidence_router_preserves_conversation_fallback():
    llm, rag = DummyPlannerLLM("unused", confidence=0.5), DummyRAG()
    result = make_node(llm, rag)({"messages": [HumanMessage(content="inspect")]})["planner_result"]
    assert result.outcome == PlannerOutcome.DIRECT_RESPONSE
    assert llm.invocations == []


def execution_state():
    return ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=ExecutionIdentity(execution_id="p1-adapter", protocol_version="1"),
        cursor=ExecutionCursor(),
    ))


def test_adapter_delegates_and_preserves_execution_state():
    expected = PlannerResult(outcome=PlannerOutcome.DIRECT_RESPONSE, message="fake result")
    calls = []
    class FakeService:
        def run(self, planner_input, *, retrieve):
            calls.append(planner_input)
            assert retrieve(planner_input.context.user_request) == ("retrieved context",)
            return expected
    state = {"execution_state": execution_state(), "messages": [HumanMessage(content="hello")]}
    before = state["execution_state"].model_dump(mode="json")
    node = create_planner_node(planner_service=FakeService(), rag_service=DummyRAG(),
                               rag_top_k=4, tools_set=set())
    update = node(state)
    assert update["planner_result"] is expected
    assert len(calls) == 1
    assert calls[0].identity == state["execution_state"].protocol_visible.identity
    assert set(update) == {"planner_result", "retrieval_messages"}
    assert state["execution_state"].model_dump(mode="json") == before


@pytest.mark.parametrize("content", ["", "completely malformed", RuntimeError("offline")])
def test_failure_reaches_existing_controller_path(content):
    from core.graph_controller import create_controller_node
    from core.protocol.enums import ExecutionStatus
    state = {"execution_state": execution_state(), "messages": [HumanMessage(content="inspect")]}
    before = state["execution_state"].model_dump(mode="json")
    llm = DummyPlannerLLM(content)
    update = make_node(llm, DummyRAG())(state)
    assert update["planner_result"].outcome == PlannerOutcome.FAILED
    assert state["execution_state"].model_dump(mode="json") == before
    applied = create_controller_node()({**state, **update})
    assert applied["controller_decision"].terminal is True
    assert applied["execution_state"].protocol_visible.status == ExecutionStatus.FAILED
    assert applied["planner_result"] is None
    assert len(llm.invocations) == 1


def test_router_provider_exception_is_failed_not_direct_response():
    class BrokenRouter(DummyPlannerLLM):
        def with_structured_output(self, schema, method):
            raise RuntimeError("router offline")
    llm, rag = BrokenRouter("unused"), DummyRAG()
    update = make_node(llm, rag)({"messages": [HumanMessage(content="inspect")]})
    assert update["planner_result"].outcome == PlannerOutcome.FAILED
    assert update["planner_result"].message == "Planner provider failed (RuntimeError)."
    assert llm.invocations == []
    assert rag.calls == []


def test_missing_router_retains_direct_fallback():
    llm, rag = DummyPlannerLLM("unused"), DummyRAG()
    node = create_planner_node(planner_llm=llm, rag_service=rag, rag_top_k=4, tools_set=set())
    assert node({"messages": [HumanMessage(content="inspect")]})["planner_result"].outcome == PlannerOutcome.DIRECT_RESPONSE
    assert llm.invocations == []


def test_retrieval_is_per_invocation_not_shared_service_state():
    llm, rag = DummyPlannerLLM("1. Inspect - Use list_files."), DummyRAG()
    node = make_node(llm, rag)
    first = node({"messages": [HumanMessage(content="first")]})
    llm.routing.route = "conversation"
    second = node({"messages": [HumanMessage(content="second")]})
    assert len(first["retrieval_messages"]) == 1
    assert second["retrieval_messages"] == []
    assert rag.calls == [("first", 4)]
