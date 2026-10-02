from collections import Counter

from langchain_ollama import ChatOllama

from core.planner_routing import LangChainPlannerRouter


def test_router_stability():
    llm = ChatOllama(
        model="gpt-oss:20b",
        temperature=0,
    )

    router = LangChainPlannerRouter(
        router_llm=llm,
        show_raw_llm=False,
    )

    prompt = "create a file named test.txt"  # "What is the capital of France?" in Turkish
    results = []

    for i in range(20):
        try:
            route = router.route(prompt).route
            results.append(route)
            print(f"{i + 1:02d}: {route}")
        except Exception as exc:
            result = f"ERROR:{type(exc).__name__}"
            results.append(result)
            print(f"{i + 1:02d}: {result} -> {exc}")

    print("\nSUMMARY")
    print(Counter(results))

    assert len(results) == 20