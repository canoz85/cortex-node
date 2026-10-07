"""Exercise real ChatOllama binding/serialization; replace only the HTTP transport."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_ollama import ChatOllama
from ollama._types import ChatRequest

from core.brain import BrainMessage, BrainService, BRAIN_OUTPUT_PROTOCOL
from core.brain_provider import LangChainBrainProvider, LIFECYCLE_ACTION_SCHEMAS
from core.protocol.enums import BrainOutcomeKind as Kind
from tools.git_ops import get_git_tools
from test_brain_outcomes import brain_input


def boundary(replies):
    requests = []
    replies = iter(replies)
    def chat(**kwargs):
        # Serialize with Ollama's real request model too, not only LangChain kwargs.
        requests.append(ChatRequest(**deepcopy(kwargs)).model_dump(exclude_none=True))
        return iter([{"message": {"role": "assistant", **next(replies)}, "done": True}])
    llm = ChatOllama(model="gemma4:26b", temperature=0)
    llm._client = SimpleNamespace(chat=chat)
    return LangChainBrainProvider(
        brain_llm=llm, executable_tools=get_git_tools("."),
    ), requests


def git_brain_input():
    context = brain_input()
    return context.model_copy(update={
        "active_plan": context.active_plan.model_copy(update={
            "available_tools": ("git_status", "git_diff", "git_log", "git_show"),
        }),
    })


def native(name, args):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


@pytest.mark.parametrize("name,args,kind", [
    ("git_status", {}, Kind.TOOL_REQUESTED),
    ("brain_step_completed", {"message": "Done"}, Kind.STEP_COMPLETED),
    ("brain_step_failed", {"message": "Cannot proceed"}, Kind.STEP_FAILED),
    ("brain_replan_requested", {"reason": "Plan cannot proceed", "constraints": []}, Kind.REPLAN_REQUESTED),
])
@pytest.mark.parametrize("retry", [False, True])
def test_native_actions_survive_real_ollama_boundary(name, args, kind, retry):
    replies = ([{"content": 'brain_step_completed{"message":"Done"}'}]
               if retry else []) + [native(name, args)]
    provider, requests = boundary(replies)
    protocol = BRAIN_OUTPUT_PROTOCOL
    outcome = provider.generate(git_brain_input(), (BrainMessage("system", protocol),))
    assert outcome.kind == kind
    assert len(requests) == (2 if retry else 1)
    for request in requests:
        schemas = {tool["function"]["name"]: tool["function"] for tool in request["tools"]}
        assert schemas["git_status"]["parameters"]["properties"] == {}
        for schema in LIFECYCLE_ACTION_SCHEMAS:
            actual = schemas[schema["function"]["name"]]
            # Ollama retains top-level arguments but strips nested object schemas
            # and validation keywords. Normalization must enforce those locally.
            expected = schema["function"]["parameters"]
            parameters = actual["parameters"]
            assert parameters["type"] == "object"
            assert parameters["required"] == expected["required"]
            assert parameters["properties"].keys() == expected["properties"].keys()
            for field, definition in expected["properties"].items():
                transported = parameters["properties"][field]
                assert transported["type"] == definition["type"]
                assert transported["description"] == definition["description"]
                if definition["type"] == "array":
                    assert transported["items"] == definition["items"]
        assert "tool_choice" not in request
    if retry:
        assert requests[0]["tools"] == requests[1]["tools"]
        assert requests[0]["messages"] == requests[1]["messages"][:-2]
        assert requests[1]["messages"][-2]["role"] == "assistant"
        assert requests[1]["messages"][-2]["content"].startswith('brain_step_completed{')
        correction = requests[1]["messages"][-1]["content"]
        assert "previous response contained no native call and was not accepted" in correction
        assert "exactly one native tool call" in correction
        assert "same active-step decision" not in correction


def test_exhausted_textual_pseudo_calls_are_typed_failure():
    provider, requests = boundary([{"content": 'brain_step_completed{"message":"Done"}'}] * 2)
    outcome = provider.generate(git_brain_input(), ())
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.error_code == "unexpected_response_content"
    assert len(requests) == 2
    assert requests[0]["tools"] == requests[1]["tools"]


def test_native_protocol_does_not_ask_for_textual_outcome_object():
    assert "Return one native action" in BRAIN_OUTPUT_PROTOCOL
    assert '"kind"' not in BRAIN_OUTPUT_PROTOCOL


def test_active_task_is_last_user_turn_at_ollama_boundary_without_filtering_tools():
    provider, requests = boundary([native("git_status", {})])
    context = git_brain_input()
    context = context.model_copy(update={
        "active_step": context.active_step.model_copy(update={
            "title": "Inspect status", "description": "Inspect repository status", "primary_tool": "git_status",
        }),
        "context": context.context.model_copy(update={
            "user_request": "Inspect status, then inspect diffs and summarize changes",
        }),
    })
    result = BrainService(provider=provider, agent_system_prompt="Execute only the active step").run(context)
    assert result.kind == Kind.TOOL_REQUESTED
    request = requests[0]
    assert request["messages"][-1]["role"] == "user"
    assert '"primary_tool": "git_status"' in request["messages"][-1]["content"]
    assert context.context.user_request not in request["messages"][-1]["content"]
    assert request["messages"][-2]["content"].lstrip().startswith("BRAIN NATIVE CALL CONTRACT:")
    assert any(context.context.user_request in m["content"] and m["role"] == "system"
               for m in request["messages"])
    assert {t["function"]["name"] for t in request["tools"]} == {
        "git_status", "git_diff", "git_log", "git_show",
        "brain_step_completed", "brain_step_failed", "brain_replan_requested",
    }

