from core.rag import WorkspaceRAG


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