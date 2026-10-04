from core.graph import build_app
from core.graph_messages import ACCEPTED_FINALIZER_PROVENANCE, CONVERSATION_PROVENANCE_KEY
from core.graph_runner import run_prompt
from core.memory.models import ConversationMemory
from core.planner_memory import project_planner_memory
from core.protocol.models import PlannerMemoryContext


def test_clarification_resume_live():
    app = build_app(
        workspace_dir="workspace",
        knowledge_dir="knowledge",
        model="qwen3.8:27b",
        model_planner="gpt-oss:20b",
        embedding_model="nomic-embed-text",
        rag_top_k=4,
        show_raw_llm=False,
    )

    planner_memory = project_planner_memory(
        ConversationMemory(),
        current_turn_index=1,
    )

    clarification_updates = []

    # Turn 1
    history, summary = run_prompt(
        app,
        "What is my name?",
        history=[],
        rolling_summary="",
        planner_memory_context=planner_memory,
        clarification_sink=clarification_updates,
    )

    assert len(clarification_updates) == 1

    pending = clarification_updates[0]

    print("\nFIRST TURN")
    print(f"run_id={pending.run_id}")
    print(f"prompt={pending.prompt!r}")
    print(
        "original_request=",
        pending.execution_state.protocol_visible
        .planning_clarification.request.context.user_request,
    )

    assert (
        pending.execution_state.protocol_visible
        .planning_clarification.request.context.user_request
        == "What is my name?"
    )

    # Turn 2
    clarification_updates = []

    history, summary = run_prompt(
        app,
        "Can",
        history=history,
        rolling_summary=summary,
        planner_memory_context=planner_memory,
        pending_clarification=pending,
        clarification_sink=clarification_updates,
    )

    print("\nSECOND TURN")
    print(f"remaining clarification={clarification_updates}")

    # Clarification must be resolved.
    assert clarification_updates == []

    # Look for accepted final answer.
    accepted_answers = [
        message
        for message in history
        if (
            getattr(message, "type", "") == "ai"
            and message.additional_kwargs.get(CONVERSATION_PROVENANCE_KEY)
            == ACCEPTED_FINALIZER_PROVENANCE
        )
    ]

    assert accepted_answers

    final_answer = str(accepted_answers[-1].content)

    print(f"FINAL ANSWER: {final_answer}")

    assert "Can" in final_answer
