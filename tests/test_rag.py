from core.rag import WorkspaceRAG
from tools.rag_ops import get_rag_tools

from conftest import get_tool, parse_result


class FakeEmbeddings:
    instances: list["FakeEmbeddings"] = []

    def __init__(self, model: str, validate_model_on_init: bool):
        self.model = model
        self.validate_model_on_init = validate_model_on_init
        self.embed_documents_calls = 0
        self.embed_query_calls = 0
        FakeEmbeddings.instances.append(self)

    def embed_documents(self, texts):
        self.embed_documents_calls += 1
        return [[float(len(text))] for text in texts]

    def embed_query(self, text):
        self.embed_query_calls += 1
        return [float(len(text))]


class FailingQueryEmbeddings(FakeEmbeddings):
    def embed_query(self, text):
        self.embed_query_calls += 1
        raise RuntimeError("query embedding unavailable")


def test_workspace_rag_caches_search_results_and_clears_cache_on_refresh(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "example.md").write_text("alpha beta gamma", encoding="utf-8")

    monkeypatch.setattr("core.rag.OllamaEmbeddings", FakeEmbeddings)
    FakeEmbeddings.instances.clear()

    rag = WorkspaceRAG(knowledge_dir, embed_model="fake-model", top_k=1)
    embeddings = FakeEmbeddings.instances[-1]

    first_results = rag.search("alpha", top_k=1)
    second_results = rag.search("alpha", top_k=1)
    alternate_results = rag.search("alpha", top_k=2)

    assert embeddings.embed_documents_calls == 1
    assert embeddings.embed_query_calls == 2
    assert first_results == second_results
    assert alternate_results == first_results

    (knowledge_dir / "example.md").write_text("alpha beta gamma delta", encoding="utf-8")
    rag.refresh()

    refreshed_results = rag.search("alpha", top_k=1)

    assert embeddings.embed_documents_calls == 2
    assert embeddings.embed_query_calls == 3
    assert refreshed_results


def test_workspace_rag_reads_invalid_json_file_without_crashing(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "broken.json").write_text('{"a": 1,', encoding="utf-8")

    monkeypatch.setattr("core.rag.OllamaEmbeddings", FakeEmbeddings)
    FakeEmbeddings.instances.clear()

    rag = WorkspaceRAG(knowledge_dir, embed_model="fake-model", top_k=1)

    assert len(rag._chunks) >= 1
    assert rag._chunks[0].text == '{"a": 1,'


def test_workspace_rag_search_falls_back_to_lexical_when_query_embedding_fails(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "doc.md").write_text("alpha beta gamma", encoding="utf-8")

    monkeypatch.setattr("core.rag.OllamaEmbeddings", FailingQueryEmbeddings)
    FailingQueryEmbeddings.instances.clear()

    rag = WorkspaceRAG(knowledge_dir, embed_model="fake-model", top_k=1)
    embeddings = FailingQueryEmbeddings.instances[-1]

    matches = rag.search("alpha", top_k=1)

    assert embeddings.embed_documents_calls == 1
    assert embeddings.embed_query_calls == 1
    assert len(matches) == 1
    assert matches[0].chunk.path == "doc.md"
    assert matches[0].score > 0


class _NoResultsRagService:
    def to_payload(self, query: str, top_k: int = 4):
        return {"query": query, "top_k": top_k, "results": []}

    def refresh(self):
        return 0


class _FailingRagService:
    def to_payload(self, query: str, top_k: int = 4):
        raise RuntimeError("index unavailable")

    def refresh(self):
        return 0


class _FailingRefreshRagService:
    def to_payload(self, query: str, top_k: int = 4):
        return {"query": query, "top_k": top_k, "results": []}

    def refresh(self):
        raise RuntimeError("refresh failed")


class _RefreshingRagService:
    def __init__(self, chunks_indexed: int):
        self._chunks_indexed = chunks_indexed

    def to_payload(self, query: str, top_k: int = 4):
        return {"query": query, "top_k": top_k, "results": []}

    def refresh(self):
        return self._chunks_indexed


def test_rag_search_returns_structured_failure_when_no_results_found():
    tools = get_rag_tools(_NoResultsRagService())
    rag_search = get_tool(tools, "rag_search")

    result = parse_result(rag_search.invoke({"query": "nothing", "top_k": 3}))

    assert result["success"] is False
    assert result["message"] == "No relevant knowledge found."
    assert isinstance(result.get("data"), dict)
    assert result["data"]["results"] == []


def test_rag_search_returns_structured_failure_when_backend_raises():
    tools = get_rag_tools(_FailingRagService())
    rag_search = get_tool(tools, "rag_search")

    result = parse_result(rag_search.invoke({"query": "anything"}))

    assert result["success"] is False
    assert "Error searching knowledge:" in result["message"]
    assert "index unavailable" in result["message"]


def test_rag_refresh_index_returns_structured_failure_when_backend_raises():
    tools = get_rag_tools(_FailingRefreshRagService())
    rag_refresh_index = get_tool(tools, "rag_refresh_index")

    result = parse_result(rag_refresh_index.invoke({}))

    assert result["success"] is False
    assert "Error refreshing knowledge index:" in result["message"]
    assert "refresh failed" in result["message"]


def test_rag_refresh_index_returns_chunk_count_on_success():
    tools = get_rag_tools(_RefreshingRagService(chunks_indexed=42))
    rag_refresh_index = get_tool(tools, "rag_refresh_index")

    result = parse_result(rag_refresh_index.invoke({}))

    assert result["success"] is True
    assert result["message"] == "Knowledge index refreshed."
    assert isinstance(result.get("data"), dict)
    assert result["data"]["chunks_indexed"] == 42