
from langchain_core.messages import SystemMessage
from core.rag import WorkspaceRAG


def retrieval_message(rag_service: WorkspaceRAG, query: str, top_k: int) -> list[SystemMessage]:
    context = rag_service.format_context(query=query, top_k=top_k)
    if not context:
        return []
    return [SystemMessage(content=context)]


