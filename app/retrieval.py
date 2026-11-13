from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from app.config import Settings
from app.guidance import load_guidance


@dataclass(frozen=True)
class RetrievedChunk:
    document: Document
    relevance_score: float | None

    @property
    def chunk_id(self) -> str:
        return str(self.document.metadata["chunk_id"])


class GuidanceRetriever:
    def __init__(self, vector_store: Any, k: int = 4) -> None:
        self.vector_store = vector_store
        self.k = k

    def search(self, text: str) -> list[RetrievedChunk]:
        if not text.strip():
            return []
        matches = self.vector_store.similarity_search_with_relevance_scores(
            text, k=self.k
        )
        return [
            RetrievedChunk(document=document, relevance_score=float(score))
            for document, score in matches
        ]


def create_embeddings(settings: Settings) -> Any:
    import os

    from langchain_openai import OpenAIEmbeddings

    # OpenAI-compatible proxies (e.g. LiteLLM) expect raw text, not token IDs.
    custom_endpoint = bool(
        os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        check_embedding_ctx_length=not custom_endpoint,
    )


def _index_path(settings: Settings) -> Path:
    return settings.vector_store_dir / "guidance.json"


def open_vector_store(settings: Settings) -> InMemoryVectorStore:
    embeddings = create_embeddings(settings)
    path = _index_path(settings)
    if path.exists():
        return InMemoryVectorStore.load(str(path), embeddings)
    return InMemoryVectorStore(embeddings)


def rebuild_index(settings: Settings) -> int:
    documents = load_guidance(
        settings.guidance_dir, settings.chunk_size, settings.chunk_overlap
    )
    store = InMemoryVectorStore(create_embeddings(settings))
    store.add_documents(documents)
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    store.dump(str(_index_path(settings)))
    return len(documents)
