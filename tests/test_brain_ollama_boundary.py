"""Exercise real ChatOllama binding/serialization; replace only the HTTP transport."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_ollama import ChatOllama
from ollama._types import ChatRequest

from core.brain import BrainMessage, BrainService, build_brain_output_protocol
from core.brain_provider import LangChainBrainProvider, native_brain_tools, LIFECYCLE_ACTION_SCHEMAS
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
    bound = llm.bind_tools(native_brain_tools(get_git_tools(".")))
    return LangChainBrainProvider(brain_llm=llm, tool_brain_llm=bound,
                                 tools_set={"git_status", "git_diff", "git_log", "git_show"}), requests


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
    protocol = build_brain_output_protocol(supports_native_tool_calls=True, tools_enabled=True)
    outcome = provider.generate(brain_input(), (BrainMessage("system", protocol),), tools_enabled=True)
    assert outcome.kind == kind
    assert len(requests) == (2 if retry else 1)
    for request in requests:
        schemas = {tool["function"]["name"]: tool["function"] for tool in request["tools"]}
        assert schemas["git_status"]["parameters"]["properties"] == {}
        for schema in LIFECYCLE_ACTION_SCHEMAS:
            actual = schemas[schema["function"]["name"]]
            # The installed Ollama SDK drops additionalProperties, but retains
            # every argument, array item type, and required field.
            expected = {k: v for k, v in schema["function"]["parameters"].items()
                        if k != "additionalProperties"}
            assert actual["parameters"] == expected
        assert "tool_choice" not in request
    if retry:
        assert requests[0]["tools"] == requests[1]["tools"]
        assert requests[0]["messages"] == requests[1]["messages"][:-2]
        assert requests[1]["messages"][-2]["role"] == "assistant"
        assert requests[1]["messages"][-2]["content"].startswith('brain_step_completed{')
        correction = requests[1]["messages"][-1]["content"]
        assert "Lifecycle actions returned in content are text" in correction
        assert "Keep the same active-step decision" in correction


def test_exhausted_textual_pseudo_calls_are_typed_failure():
    provider, requests = boundary([{"content": 'brain_step_completed{"message":"Done"}'}] * 2)
    outcome = provider.generate(brain_input(), (), tools_enabled=True)
    assert outcome.kind == Kind.INVALID_OUTPUT
    assert outcome.error_code == "expected_structured_outcome"
    assert len(requests) == 2
    assert requests[0]["tools"] == requests[1]["tools"]


def test_native_protocol_does_not_ask_for_textual_outcome_object():
    native_protocol = build_brain_output_protocol(supports_native_tool_calls=True, tools_enabled=True)
    compatibility = build_brain_output_protocol(supports_native_tool_calls=False, tools_enabled=True)
    assert "Return exactly one outcome object." not in native_protocol
    assert "Leave content empty" in native_protocol
    assert "Return exactly one outcome object." in compatibility
    assert '\"kind\":\"STEP_COMPLETED\"' in compatibility


def test_active_task_is_last_user_turn_at_ollama_boundary_without_filtering_tools():
    provider, requests = boundary([native("git_status", {})])
    context = brain_input()
    context = context.model_copy(update={
        "active_step": context.active_step.model_copy(update={
            "title": "Inspect status", "description": "Inspect repository status", "primary_tool": "git_status",
        }),
        "context": context.context.model_copy(update={
            "user_request": "Inspect status, then inspect diffs and summarize changes",
        }),
    })
    result = BrainService(provider=provider, agent_system_prompt="Execute only the active step",
                          casual_system_prompt="Converse").run(context)
    assert result.kind == Kind.TOOL_REQUESTED
    request = requests[0]
    assert request["messages"][-1]["role"] == "user"
    assert '"primary_tool": "git_status"' in request["messages"][-1]["content"]
    assert context.context.user_request not in request["messages"][-1]["content"]
    assert request["messages"][-2]["content"].startswith("BRAIN OUTCOME CONTRACT:")
    assert any(context.context.user_request in m["content"] and m["role"] == "system"
               for m in request["messages"])
    assert {t["function"]["name"] for t in request["tools"]} == {
        "git_status", "git_diff", "git_log", "git_show",
        "brain_step_completed", "brain_step_failed", "brain_replan_requested",
    }

