from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

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


def open_vector_store(settings: Settings) -> Any:
    from langchain_chroma import Chroma

    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name="guidance",
        embedding_function=create_embeddings(settings),
        persist_directory=str(settings.vector_store_dir),
    )


def rebuild_index(settings: Settings) -> int:
    documents = load_guidance(
        settings.guidance_dir, settings.chunk_size, settings.chunk_overlap
    )
    store = open_vector_store(settings)
    store.reset_collection()
    store.add_documents(documents)
    return len(documents)
