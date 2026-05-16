from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_ollama import OllamaEmbeddings


DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 160
DEFAULT_TOP_K = 4

TEXT_EXTENSIONS = {".md", ".json"}
IGNORED_DIRS = {".git", ".venv", ".vscode", "__pycache__", "node_modules"}


@dataclass(slots=True)
class RagChunk:
    source: str
    path: str
    chunk_index: int
    text: str
    embedding: list[float] | None = None


@dataclass(slots=True)
class RagMatch:
    chunk: RagChunk
    score: float


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _normalize_json_text(raw_text: str) -> str:
    try:
        parsed = json.loads(raw_text)
    except Exception:
        return raw_text
    return json.dumps(parsed, indent=2, ensure_ascii=True, sort_keys=True)


def _read_text_file(path: Path) -> str:
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".json":
        return _normalize_json_text(raw_text)
    return raw_text


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []

    paragraphs = [piece.strip() for piece in cleaned.split("\n\n") if piece.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            flush()
            words = paragraph.split()
            window: list[str] = []
            window_length = 0
            for word in words:
                addition = len(word) + (1 if window else 0)
                if window and window_length + addition > chunk_size:
                    chunks.append(" ".join(window).strip())
                    if overlap > 0:
                        window = window[-max(1, min(len(window), overlap // 8)) :]
                        window_length = sum(len(item) for item in window) + max(len(window) - 1, 0)
                    else:
                        window = []
                        window_length = 0
                window.append(word)
                window_length += addition
            if window:
                chunks.append(" ".join(window).strip())
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            flush()
            current = paragraph

    flush()
    return chunks


def _cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if not left_values or not right_values:
        return 0.0

    dot_product = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


class WorkspaceRAG:
    def __init__(
        self,
        knowledge_dir: str | Path,
        *,
        embed_model: str = "nomic-embed-text",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir).resolve()
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self._embeddings: OllamaEmbeddings | None = None
        self._chunks: list[RagChunk] = []
        self.refresh()

    def _get_embeddings(self) -> OllamaEmbeddings:
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(model=self.embed_model, validate_model_on_init=False)
        return self._embeddings

    def _iter_source_files(self) -> Iterable[Path]:
        if not self.knowledge_dir.exists():
            return []

        for path in self.knowledge_dir.rglob("*"):
            if not path.is_file() or _is_ignored(path) or path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            yield path

    def _build_chunks(self) -> list[RagChunk]:
        chunks: list[RagChunk] = []
        if not self.knowledge_dir.exists():
            return chunks

        for path in self._iter_source_files():
            try:
                text = _read_text_file(path)
            except Exception:
                continue

            if not text.strip():
                continue

            relative_path = str(path.relative_to(self.knowledge_dir)).replace("\\", "/")
            pieces = _chunk_text(text, self.chunk_size, self.chunk_overlap)
            for index, piece in enumerate(pieces):
                chunks.append(
                    RagChunk(
                        source=relative_path,
                        path=relative_path,
                        chunk_index=index,
                        text=piece,
                    )
                )

        return chunks

    def refresh(self) -> int:
        self._chunks = self._build_chunks()
        if not self._chunks:
            return 0

        try:
            vectors = self._get_embeddings().embed_documents([chunk.text for chunk in self._chunks])
            for chunk, vector in zip(self._chunks, vectors, strict=False):
                chunk.embedding = list(vector)
        except Exception:
            for chunk in self._chunks:
                chunk.embedding = None

        return len(self._chunks)

    def _lexical_score(self, query: str, chunk: RagChunk) -> float:
        query_tokens = _tokenize(query)
        chunk_tokens = _tokenize(chunk.text)
        if not query_tokens or not chunk_tokens:
            return 0.0

        query_counts = Counter(query_tokens)
        chunk_counts = Counter(chunk_tokens)
        overlap = sum(min(query_counts[token], chunk_counts.get(token, 0)) for token in query_counts)
        if overlap == 0:
            return 0.0
        return overlap / math.sqrt(sum(value * value for value in query_counts.values()))

    def search(self, query: str, top_k: int | None = None) -> list[RagMatch]:
        if not self._chunks:
            return []

        effective_top_k = max(1, top_k or self.top_k)
        query_text = query.strip()
        if not query_text:
            return []

        matches: list[RagMatch] = []
        try:
            query_embedding = list(self._get_embeddings().embed_query(query_text))
            for chunk in self._chunks:
                if not chunk.embedding:
                    continue
                score = _cosine_similarity(query_embedding, chunk.embedding)
                if score > 0:
                    matches.append(RagMatch(chunk=chunk, score=score))
        except Exception:
            matches = []

        if not matches:
            for chunk in self._chunks:
                score = self._lexical_score(query_text, chunk)
                if score > 0:
                    matches.append(RagMatch(chunk=chunk, score=score))

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:effective_top_k]

    def format_context(self, query: str, top_k: int | None = None) -> str:
        matches = self.search(query, top_k=top_k)
        if not matches:
            return ""

        lines = ["Relevant knowledge context:"]
        for index, match in enumerate(matches, start=1):
            snippet = " ".join(match.chunk.text.split())
            if len(snippet) > 900:
                snippet = f"{snippet[:900].rstrip()}..."
            lines.append(
                f"[{index}] {match.chunk.path}#chunk-{match.chunk.chunk_index} (score={match.score:.3f})\n{snippet}"
            )
        return "\n\n".join(lines)

    def to_payload(self, query: str, top_k: int | None = None) -> dict[str, object]:
        matches = self.search(query, top_k=top_k)
        return {
            "query": query,
            "top_k": max(1, top_k or self.top_k),
            "knowledge_dir": str(self.knowledge_dir),
            "total_chunks": len(self._chunks),
            "results": [
                {
                    "source": match.chunk.path,
                    "chunk_index": match.chunk.chunk_index,
                    "score": round(match.score, 6),
                    "content": match.chunk.text,
                }
                for match in matches
            ],
        }