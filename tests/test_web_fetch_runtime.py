"""web_fetch through authorized cards, native binding and runtime evidence."""

from copy import deepcopy
from functools import partial
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama
from ollama._types import ChatRequest

from core.brain_normalization import normalize_brain_output
from core.brain_provider import LangChainBrainProvider
from core.graph_runner import run_prompt
from core.models import WebFetchRequest
from core.protocol.bridge import build_controller_input
from core.protocol.controller import CortexController
from core.protocol.enums import BrainOutcomeKind as Kind, ControllerDecisionType as Decision, ExecutionStatus
from core.protocol.models import BrainOutcome, ExecutionState, ProtocolVisibleState, ToolRequest
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver
from core.runtime.tool_result_integration import SerializedToolRuntimePort, integrate_tool_result
from test_brain_outcomes import brain_input, native_action
from test_planner_runtime_flow import StructuredModel, production_app
from test_web_ops import dns_records, html_response
from tools import web_ops as web
from tools.registry import (
    ToolRegistry, build_tool_registry, default_enabled_tools, get_tool_argument_schema,
    get_tool_definition, planning_capability_projection,
)


URL = "https://example.com/"
TEXT = "Current public page evidence Ω."


@pytest.fixture(autouse=True)
def network(monkeypatch):
    monkeypatch.setattr(web.socket, "getaddrinfo", Mock(return_value=dns_records("93.184.216.34")))
    monkeypatch.setattr(web, "create_connection", Mock(side_effect=AssertionError("Unexpected network")))
    send = Mock(side_effect=lambda *_args, **_kwargs: html_response(
        f"<title>Current page</title><body><article><p>{TEXT}</p></article></body>".encode(),
    ))
    monkeypatch.setattr(web._PublicHTTPAdapter, "send", send)
    return send


def web_context():
    context = brain_input()
    step = context.active_step.model_copy(update={
        "title": "Read and summarize the page", "description": f"Read {URL} and summarize it.",
        "primary_tool": "web_fetch",
    })
    return context.model_copy(update={
        "active_step": step,
        "active_plan": context.active_plan.model_copy(update={"steps": (step,), "available_tools": ("web_fetch",)}),
    })


def test_schema_registry_defaults_and_authorized_capability_card(tmp_path):
    registry = build_tool_registry(str(tmp_path), None, SimpleNamespace(), "test")
    assert "web_fetch" in registry.names and "search_web" not in registry.names and "web_search" not in registry.names
    assert "web_fetch" in {tool.name for tool in default_enabled_tools(registry)}
    definition = get_tool_definition("web_fetch")
    assert not definition.mutating and not definition.exact_collection_outputs
    assert definition.args_schema is get_tool_argument_schema("web_fetch", registry=registry) is WebFetchRequest
    schema = convert_to_openai_tool(registry.by_name["web_fetch"])["function"]
    assert schema["name"] == "web_fetch"
    parameters = schema["parameters"]
    assert parameters["required"] == ["url"]
    # LangChain/Ollama may drop validation keywords from transported schemas;
    # executable Pydantic validation remains the authority for extra fields.
    assert WebFetchRequest.model_json_schema()["additionalProperties"] is False
    assert set(parameters["properties"]) == {"url", "max_chars"}
    assert parameters["properties"]["max_chars"] == {
        "type": "integer", "default": 50_000, "minimum": 1, "maximum": 200_000,
        "description": "Maximum extracted text characters; excess text is explicitly truncated.",
    }
    card, = planning_capability_projection(("web_fetch",))
    assert card["name"] == "web_fetch" and card["mutating"] is False
    assert "pagination" not in card and "exact_collection" not in card
    assert "not a search engine" in " ".join(card["limits"])
    assert planning_capability_projection(("current_time",))[0]["name"] != "web_fetch"


def test_real_ollama_native_schema_binding_accepts_web_fetch_without_network(network):
    transported = []

    def chat(**kwargs):
        transported.append(ChatRequest(**deepcopy(kwargs)).model_dump(exclude_none=True))
        return iter([{"message": {"role": "assistant", "content": "", "tool_calls": [{
            "function": {"name": "web_fetch", "arguments": {"url": URL}},
        }]}, "done": True}])

    model = ChatOllama(model="offline-test", temperature=0)
    model._client = SimpleNamespace(chat=chat)
    provider = LangChainBrainProvider(brain_llm=model, executable_tools=web.get_web_tools())
    result = provider.generate(web_context(), ())
    assert result.kind == Kind.TOOL_REQUESTED
    assert result.tool_request.tool_name == "web_fetch" and result.tool_request.arguments == {"url": URL}
    schema = next(tool["function"] for tool in transported[0]["tools"] if tool["function"]["name"] == "web_fetch")
    assert schema["parameters"]["required"] == ["url"]
    assert {key: value["type"] for key, value in schema["parameters"]["properties"].items()} == {
        "url": "string", "max_chars": "integer",
    }
    network.assert_not_called()


@pytest.mark.parametrize("max_chars", [50_000, 10])
def test_info_request_fetches_once_exposes_evidence_and_completes_through_controller(max_chars, network):
    arguments = {"url": URL, "max_chars": max_chars}
    replies = iter([
        native_action("web_fetch", arguments),
        native_action("web_fetch", arguments),  # Successful duplicates must be corrected before dispatch.
        native_action("brain_step_completed", {"message": "Summarized the available public page evidence."}),
    ])

    class Model(StructuredModel):
        def __init__(self):
            super().__init__([{"result": "PLAN_PROPOSED", "objective": f"Read {URL}", "steps": [{
                "step_id": "read", "title": "Read and summarize page", "description": f"Read {URL} and summarize it.",
                "deliverable_kind": "semantic", "primary_tool": "web_fetch", "dependencies": [],
            }]}], route="info")
            self.brain_calls = []

        def invoke(self, messages):
            if any(message.content.startswith("Active step:") for message in messages):
                self.brain_calls.append(messages)
                return next(replies)
            return AIMessage(content="Summary of the fetched public page.")

    model = Model()
    app = production_app(model, web.get_web_tools())
    run_id = f"web-fetch-info-{max_chars}"
    history, _ = run_prompt(app, f"Read {URL} and summarize it.", run_id=run_id)
    state = app.get_state({"configurable": {"thread_id": run_id}}).values["execution_state"]
    assert state.protocol_visible.status == ExecutionStatus.COMPLETED
    assert state.protocol_visible.active_plan.steps[0].primary_tool == "web_fetch"
    assert state.protocol_visible.completed_step_ids == ("read",)
    assert len(state.working.tool_execution_history) == 1 and network.call_count == 1
    record, = state.working.tool_execution_history
    assert record.tool_name == "web_fetch" and record.result.success
    assert record.result.data["text"] == TEXT[:max_chars]
    assert record.result.data["total_chars"] == len(TEXT)
    assert record.result.integrity.is_truncated == (max_chars < len(TEXT))
    assert record.result.integrity.original_bytes == len(TEXT.encode())
    assert record.result.integrity.captured_bytes == len(TEXT[:max_chars].encode())
    assert record.result.pagination is None
    evidence_message = next(message.content for message in model.brain_calls[1]
                            if message.content.startswith("Execution evidence v1:"))
    evidence = json.loads(evidence_message.split("\n", 1)[1])["current_attempts"][0]["evidence"]
    assert evidence["text"] == TEXT[:max_chars] and evidence["truncated"] == (max_chars < len(TEXT))
    assert "duplicate_successful_call" in model.brain_calls[2][-1].content
    planner_system = model.exchanges[1][1][0].content  # Router exchange then Planner exchange.
    assert '"name":"web_fetch"' in planner_system and '"mutating":false' in planner_system
    assert '"name":"web_search"' not in planner_system
    assert history[-1].content == "Summary of the fetched public page."


def test_controller_and_driver_reject_duplicate_even_if_native_guard_is_bypassed(network):
    tools = web.get_web_tools()
    registry = ToolRegistry.from_tools(tools)
    lookup = partial(get_tool_argument_schema, registry=registry)
    context = web_context()
    state = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=context.identity, cursor=context.cursor, active_plan=context.active_plan,
        active_step=context.active_step, retry=context.retry, original_user_request=f"Read {URL}",
    ))
    controller = CortexController(24, argument_schema_for=lookup)
    driver = ExecutionDriver(coordinator=ControllerCoordinator(controller), planner=Mock(), brain=Mock(),
                             tool_runtime=SerializedToolRuntimePort(tools), finalizer=Mock())
    proposal = BrainOutcome(outcome=Kind.TOOL_REQUESTED, step_id=context.active_step.step_id, tool_request=ToolRequest(
        request_id="first", tool_name="web_fetch", arguments={"url": URL},
    ))
    value = build_controller_input({"execution_state": state, "brain_result": proposal})
    turn = driver.turn(state, value)
    assert turn.decision.decision_type == Decision.DISPATCH_TOOL_RUNTIME and turn.worker_result.success
    captured = integrate_tool_result(turn.execution_state, turn.decision, turn.worker_result)
    assert captured.working.tool_execution_history[0].result.data["text"] == TEXT
    duplicate = proposal.model_copy(update={"tool_request": proposal.tool_request.model_copy(update={
        "request_id": "different-id", "arguments": {"url": URL, "max_chars": 50_000},
    })})
    value = build_controller_input({"execution_state": captured, "brain_result": duplicate}).model_copy(update={"tool_result": None})
    rejected = driver.turn(captured, value, dispatch_worker=False)
    assert rejected.decision.decision_type == Decision.DISPATCH_BRAIN
    assert rejected.decision.retry.last_error_code == "duplicate_successful_call"
    assert rejected.decision.pending_tool_request is None and network.call_count == 1
    # The native guard compares the same executable-schema defaults too.
    context = context.model_copy(update={"tool_execution_history": captured.working.tool_execution_history})
    outcome = normalize_brain_output(native_action("web_fetch", {"url": URL, "max_chars": 50_000}),
                                     context, {"web_fetch"}, argument_schema_for=lookup)
    assert outcome.error_code == "duplicate_successful_call"
