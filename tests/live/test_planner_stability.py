from collections import Counter

from langchain_ollama import ChatOllama

from core.graph_constants import MUTATING_TOOLS
from core.planner import PlannerService
from core.planner_provider import LangChainPlannerProvider
from core.planner_routing import LangChainPlannerRouter

from tests.test_planner import planner_input


RUNS = 10


def test_planner_stability():
    llm = ChatOllama(
        model="gpt-oss:20b",
        temperature=0,
    )

    tools_set = set(tool_registry.keys())


    service = PlannerService(
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

    request = planner_input()

    request = request.model_copy(
        update={
            "context": request.context.model_copy(
                update={
                    "user_request":     "Get git diff and explain the changes",
                }
            )
        }
    )

    results = []

    print("\n" + "=" * 80)
    print("PLANNER STABILITY")
    print("=" * 80)

    for i in range(RUNS):
        result = service.run(request)

        value = str(result.outcome)

        results.append(value)

        print(f"{i + 1:02d}: {value}")
        print(f"    message={result.message!r}")

    print("\nSUMMARY")
    print(Counter(results))

    assert len(results) == RUNS