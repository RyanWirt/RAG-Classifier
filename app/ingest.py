"""Vectorization tool: embeds guidance text into a local Chroma database.

Run with `python -m app.ingest`. The classifier only reads the result.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Sequence

from app.config import Settings
from app.embeddings import COLLECTION, create_embeddings, open_client
from app.guidance import load_guidance


def rebuild_index(settings: Settings) -> int:
    documents = load_guidance(
        settings.guidance_dir, settings.chunk_size, settings.chunk_overlap
    )
    texts = [document.page_content for document in documents]
    vectors = create_embeddings(settings).embed_documents(texts)

    client = open_client(settings)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)
    collection.add(
        ids=[document.metadata["chunk_id"] for document in documents],
        documents=texts,
        embeddings=vectors,
        metadatas=[document.metadata for document in documents],
    )
    return len(documents)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argparse.ArgumentParser(
        description="Rebuild the Chroma guidance index from GUIDANCE_DIR"
    ).parse_args(argv)
    try:
        count = rebuild_index(Settings.from_env())
    except Exception:
        logging.exception("Indexing failed")
        return 1
    print(json.dumps({"indexed_chunks": count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
