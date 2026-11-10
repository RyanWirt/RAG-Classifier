from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    guidance_dir: Path = Path("guidance")
    vector_store_dir: Path = Path(".vector_store")
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_k: int = 4
    llm_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    tool_timeout_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            guidance_dir=Path(os.getenv("GUIDANCE_DIR", "guidance")),
            vector_store_dir=Path(os.getenv("VECTOR_STORE_DIR", ".vector_store")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
            llm_provider=os.getenv("LLM_PROVIDER", "openai").casefold(),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMPERATURE", "0")),
            tool_timeout_seconds=float(os.getenv("TOOL_TIMEOUT_SECONDS", "5")),
        )
        if settings.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero")
        if not 0 <= settings.chunk_overlap < settings.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be in [0, CHUNK_SIZE)")
        if settings.retrieval_k <= 0:
            raise ValueError("RETRIEVAL_K must be greater than zero")
        if settings.llm_provider not in {"openai", "anthropic"}:
            raise ValueError("LLM_PROVIDER must be openai or anthropic")
        if settings.tool_timeout_seconds <= 0:
            raise ValueError("TOOL_TIMEOUT_SECONDS must be greater than zero")
        return settings
