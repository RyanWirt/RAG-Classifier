from __future__ import annotations

import os
from typing import Any

from app.config import Settings


def create_embeddings(settings: Settings) -> Any:
    from langchain_openai import OpenAIEmbeddings

    # OpenAI-compatible proxies (e.g. LiteLLM) expect raw text, not token IDs.
    custom_endpoint = bool(
        os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        check_embedding_ctx_length=not custom_endpoint,
    )


COLLECTION = "guidance"


def open_client(settings: Settings) -> Any:
    import chromadb

    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.vector_store_dir))
