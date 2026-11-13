from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.config import Settings


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


class ChromaGuidanceStore:
    """Read-only view of the index written by `python -m app.ingest`."""

    def __init__(self, settings: Settings) -> None:
        from app.embeddings import COLLECTION, create_embeddings, open_client

        self.collection = open_client(settings).get_or_create_collection(COLLECTION)
        self.embeddings = create_embeddings(settings)

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def similarity_search_with_relevance_scores(
        self, text: str, k: int
    ) -> list[tuple[Document, float]]:
        if self.is_empty():
            return []
        result = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(text)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            # Chroma's default L2 distance on unit vectors maps to 0..1.
            (Document(page_content=content, metadata=metadata), 1.0 - distance / 2**0.5)
            for content, metadata, distance in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]
