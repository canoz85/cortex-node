import io

from langchain_core.messages import AIMessage

from core.conversation_memory_updater import MemoryProposal, MemoryUpdateRequest
from core.logging.live_status import LiveStatus
from core.memory import TurnStatus
from core.memory_provider import LangChainMemoryProposalProvider


def test_memory_provider_usage_contributes_once_to_turn_total():
    raw = AIMessage(
        content='{"facts": []}',
        usage_metadata={"input_tokens": 800, "output_tokens": 40, "total_tokens": 840},
    )

    class Structured:
        def invoke(self, _messages):
            return {"raw": raw, "parsed": MemoryProposal(facts=[]), "parsing_error": None}

    class Model:
        def with_structured_output(self, *_args, **_kwargs):
            return Structured()

    request = MemoryUpdateRequest.model_construct(
        turn_id="turn", turn_index=1, execution_id="execution",
        user_request="remember nothing", terminal_status=TurnStatus.COMPLETED,
        accepted_answer=None, accepted_completions=(), existing_facts=(),
    )
    status = LiveStatus(stream=io.StringIO(), refresh_interval=60, enabled=True)
    status.start("memory", "saving session")
    try:
        result = LangChainMemoryProposalProvider(Model()).generate(request)
        assert result.facts == ()
        assert status.total_tokens == 840
        assert status.usage_by_worker["memory"]["calls"] == 1
    finally:
        status.stop()
