from collections import Counter
from pathlib import Path

import pytest
from langchain_ollama import ChatOllama

from core.graph import (
    _build_tools,
    _default_rag_factory,
    _default_tool_list_factory,
    _tool_names,
)
from core.graph_constants import MUTATING_TOOLS
from core.planner import PlannerService
from core.planner_provider import LangChainPlannerProvider
from core.planner_routing import LangChainPlannerRouter
from core.protocol.models import PlanningCapabilities

from tests.test_planner_service import planner_input


RUNS = 10


CASES = [
    {
        "name": "list_files",
        "request": "List files",
        "expected_outcome": "PlannerOutcome.EXECUTION_PLAN",
        "expected_steps": [
            {
                "primary_tool": "list_files",
                "contains": ["list"],
            }
        ],
    },
    {
        "name": "git_diff",
        "request": "Get git diff",
        "expected_outcome": "PlannerOutcome.EXECUTION_PLAN",
        "expected_steps": [
            {
                "primary_tool": "git_diff",
                "contains": ["diff"],
            }
        ],
    },
    {
        "name": "git_diff_explain",
        "request": "Get git diff and explain the changes",
        "expected_outcome": "PlannerOutcome.EXECUTION_PLAN",
        "expected_steps": [
            {
                "primary_tool": "git_diff",
                "contains": ["diff", "explain"],
            }
        ],
    },
    {
        "name": "direct_response",
        "request": "What is the capital of France?",
        "expected_outcome": "PlannerOutcome.DIRECT_RESPONSE",
        "expected_steps": [],
    },
    {
        "name": "needs_input",
        "request": "What is my name?",
        "expected_outcome": "PlannerOutcome.CLARIFICATION_REQUIRED",
        "expected_steps": [],
    },
    {
        "name": "planning_failed",
        "request": "Send an email to test@example.com saying hello",
        "expected_outcome": "PlannerOutcome.FAILED",
        "expected_steps": [],
    },
]


def build_production_tool_names():
    workspace_root = Path("workspace").resolve()
    knowledge_root = Path("knowledge").resolve()

    rag_service = _default_rag_factory(
        knowledge_root,
        "nomic-embed-text",
        4,
    )

    tools = _build_tools(
        tool_list_factory=_default_tool_list_factory,
        workspace_root=str(workspace_root),
        knowledge_root=str(knowledge_root),
        rag_service=rag_service,
        model="gpt-oss:20b",
        resource_coordinator=None,
    )

    return _tool_names(tools)


def build_planner_service():
    llm = ChatOllama(
        model="gpt-oss:20b",
        temperature=0,
    )

    return PlannerService(
        provider=LangChainPlannerProvider(
            planner_llm=llm,
            show_raw_llm=False,
        ),
        router=LangChainPlannerRouter(
            router_llm=llm,
            show_raw_llm=False,
        ),
        mutating_tools=MUTATING_TOOLS,
        show_raw_llm=False,
    )


def assert_steps(result, expected_steps):
    
    proposed_plan = result.proposed_plan

    if not expected_steps:
        assert proposed_plan is None
        return

    assert proposed_plan is not None

    actual_steps = proposed_plan.steps

    assert len(actual_steps) == len(expected_steps)

    for actual, expected in zip(actual_steps, expected_steps):
        assert actual.primary_tool == expected["primary_tool"]

        text = f"{actual.title} {actual.description}".lower()

        for required_text in expected.get("contains", []):
            assert required_text.lower() in text


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[case["name"] for case in CASES],
)
def test_planner_stability_live(case):
    service = build_planner_service()
    tool_names = build_production_tool_names()

    request = planner_input()

    request = request.model_copy(
        update={
            "context": request.context.model_copy(
                update={
                    "user_request": case["request"],
                }
            ),
            "capabilities": PlanningCapabilities(
                available_tools=tuple(sorted(tool_names)),
            ),
        }
    )

    results = []

    print("\n" + "=" * 80)
    print(f"PLANNER STABILITY: {case['name']}")
    print(f"REQUEST: {case['request']}")
    print(f"EXPECTED: {case['expected_outcome']}")
    print("=" * 80)

    for i in range(RUNS):
        result = service.run(request)

        value = str(result.outcome)
        results.append(value)

        print(f"{i + 1:02d}: {value}")
        print(f"    message={result.message!r}")

        if result.proposed_plan is not None:
            for step in result.proposed_plan.steps:
                print(
                    f"    step={step.step_id} "
                    f"tool={step.primary_tool!r} "
                    f"title={step.title!r}"
                )
                print(f"        description={step.description!r}")

        assert value == case["expected_outcome"]
        assert_steps(result, case["expected_steps"])

    print("\nSUMMARY")
    print(Counter(results))

    assert len(results) == RUNS