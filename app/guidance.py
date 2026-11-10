from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

HEADING = re.compile(
    r"^\s{0,3}(?:#{1,6}\s+.+|\d+(?:\.\d+)*[.)]?\s+[^\n]+|"
    r"[A-Z][A-Z0-9 /&-]{2,}:?)\s*$"
)


def _sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        if HEADING.match(line.strip()):
            if "\n".join(current_lines).strip():
                sections.append((current_heading, current_lines))
            current_heading = re.sub(r"^#{1,6}\s*", "", line.strip()).rstrip(":")
            current_lines = []
        else:
            current_lines.append(line)
    if "\n".join(current_lines).strip():
        sections.append((current_heading, current_lines))
    return [(heading, "\n".join(lines).strip()) for heading, lines in sections]


def load_guidance(
    directory: Path, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Load text guidance as overlapping LangChain documents with traceable metadata."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Guidance directory does not exist: {directory}")
    files = sorted(directory.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt guidance files found in {directory}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    documents: list[Document] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        chunk_number = 0
        for section_id, section_text in _sections(text):
            for chunk in splitter.split_text(section_text):
                chunk_id = f"{path.name}#chunk-{chunk_number:04d}"
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source_file": path.name,
                            "section_id": section_id or "",
                            "chunk_id": chunk_id,
                            "original_text": chunk,
                        },
                    )
                )
                chunk_number += 1
    if not documents:
        raise ValueError(f"No usable guidance text found in {directory}")
    return documents
