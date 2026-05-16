from langchain_core.tools import tool

from core.models import ToolResult
from core.rag import WorkspaceRAG


def get_rag_tools(rag_service: WorkspaceRAG):
    @tool
    def rag_search(query: str, top_k: int = 4) -> str:
        """Search the knowledge folder for relevant context."""
        try:
            payload = rag_service.to_payload(query=query, top_k=top_k)
            results = payload.get("results", [])
            if not results:
                return ToolResult(
                    success=False,
                    message="No relevant knowledge found.",
                    data=payload,
                ).to_tool_output()

            return ToolResult(
                success=True,
                message=f"Found {len(results)} relevant chunk(s).",
                data=payload,
            ).to_tool_output()
        except Exception as exc:
            return ToolResult(success=False, message=f"Error searching knowledge: {exc}").to_tool_output()

    @tool
    def rag_refresh_index() -> str:
        """Rebuild the in-memory knowledge index."""
        try:
            chunk_count = rag_service.refresh()
            return ToolResult(
                success=True,
                message="Knowledge index refreshed.",
                data={"chunks_indexed": chunk_count},
            ).to_tool_output()
        except Exception as exc:
            return ToolResult(success=False, message=f"Error refreshing knowledge index: {exc}").to_tool_output()

    return [rag_search, rag_refresh_index]