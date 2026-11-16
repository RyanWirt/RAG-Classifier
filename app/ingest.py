"""Ingestion: guidance PDFs -> cleaned text -> chunks -> embeddings -> Chroma.

Run with `python -m app.ingest`. The classifier only reads the result.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.embeddings import COLLECTION, create_embeddings, open_client

MIN_TEXT_CHARS = 25  # fewer extracted characters than this means a scanned page
OCR_DPI = 300

HEADING = re.compile(
    r"^\s{0,3}(?:#{1,6}\s+.+|\d+(?:\.\d+)*[.)]?\s+[^\n]+|"
    r"[A-Z][A-Z0-9 /&-]{2,}:?)\s*$"
)
# Two-column PDFs extract as "paragraph text    ## Label" on the paragraph's first line.
RIGHT_LABEL = re.compile(r"^(?P<left>.*\S)(?:\s{2,}|\t)(?P<label>#{1,6}\s+\S.*)$")


# --- PDF text extraction -------------------------------------------------


def normalize_line(line: str) -> str:
    """Digits become '#', so 'Page 3 of 9' and 'Page 4 of 9' compare equal."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.casefold())).strip()


def _edge_indexes(lines: list[str], edge_lines: int) -> set[int]:
    filled = [i for i, line in enumerate(lines) if line.strip()]
    return set(filled[:edge_lines]) | set(filled[-edge_lines:])


def strip_repeated_lines(
    pages: list[list[str]],
    edge_lines: int = 3,
    min_fraction: float = 0.4,
    min_pages: int = 3,
) -> tuple[list[list[str]], dict[str, int]]:
    """Remove lines at the top/bottom of pages that repeat across many pages."""
    if len(pages) < min_pages:
        return pages, {}
    edges = [_edge_indexes(lines, edge_lines) for lines in pages]
    seen: Counter[str] = Counter()
    for lines, indexes in zip(pages, edges):
        seen.update({normalize_line(lines[i]) for i in indexes})
    threshold = max(min_pages, min_fraction * len(pages))
    repeated = {key: n for key, n in seen.items() if key and n >= threshold}
    cleaned = [
        [
            line
            for i, line in enumerate(lines)
            if not (i in indexes and normalize_line(line) in repeated)
        ]
        for lines, indexes in zip(pages, edges)
    ]
    return cleaned, repeated


def clean_text(pages: list[list[str]]) -> str:
    """Join pages, rejoin hyphenated line breaks, and collapse blank-line runs."""
    text = "\n\n".join("\n".join(line.rstrip() for line in lines) for lines in pages)
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


@functools.cache
def _load_ocr() -> Callable[[Any], list[str]]:
    try:
        import numpy
        from rapidocr import RapidOCR
    except ImportError:
        raise SystemExit("OCR needs: pip install -r requirements-ingest.txt") from None
    engine = RapidOCR(params={"Global.log_level": "warning"})

    def run(image: Any) -> list[str]:
        result = engine(numpy.asarray(image.convert("RGB")))
        return list(result.txts or [])

    return run


def _page_lines(page: Any, ocr_mode: str) -> tuple[list[str], bool]:
    text = page.get_textpage().get_text_range().replace("\ufffe", "")  # pdfium's soft-hyphen marker
    needs_ocr = ocr_mode == "always" or len(text.strip()) < MIN_TEXT_CHARS
    if ocr_mode != "never" and needs_ocr:
        return _load_ocr()(page.render(scale=OCR_DPI / 72).to_pil()), True
    return text.splitlines(), False


def pdf_to_text(
    path: Path, ocr_mode: str = "auto", edge_lines: int = 3, min_fraction: float = 0.4
) -> str:
    """Extract text from a PDF, OCRing scanned pages and dropping running headers/footers."""
    try:
        import pypdfium2
    except ImportError:
        raise SystemExit("PDF reading needs: pip install -r requirements-ingest.txt") from None

    document = pypdfium2.PdfDocument(path)
    try:
        pages: list[list[str]] = []
        ocr_pages = 0
        for index in range(len(document)):
            page = document[index]
            try:
                lines, used_ocr = _page_lines(page, ocr_mode)
            finally:
                page.close()
            pages.append(lines)
            ocr_pages += used_ocr
    finally:
        document.close()
    pages, removed = strip_repeated_lines(pages, edge_lines, min_fraction)
    logging.info(
        "%s: %d pages, %d OCR, %d header/footer patterns removed",
        path.name, len(pages), ocr_pages, len(removed),
    )
    for pattern, count in sorted(removed.items()):
        logging.info("%s: removed %r on %d pages", path.name, pattern, count)
    return clean_text(pages)


# --- Sectioning and chunking ---------------------------------------------


def _sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        left_text = ""
        side_label = RIGHT_LABEL.match(line)
        if side_label:
            line, left_text = side_label["label"], side_label["left"]
        if HEADING.match(line.strip()):
            if "\n".join(current_lines).strip():
                sections.append((current_heading, current_lines))
            current_heading = re.sub(r"^#{1,6}\s*", "", line.strip()).rstrip(":")
            current_lines = [left_text] if left_text else []
        else:
            current_lines.append(line)
    if "\n".join(current_lines).strip():
        sections.append((current_heading, current_lines))
    return [(heading, "\n".join(lines).strip()) for heading, lines in sections]


def load_guidance(
    directory: Path,
    chunk_size: int,
    chunk_overlap: int,
    ocr_mode: str = "auto",
    edge_lines: int = 3,
    min_fraction: float = 0.4,
) -> list[Document]:
    """Read every PDF (and .txt) in `directory` into chunked documents with traceable metadata."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Guidance directory does not exist: {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".pdf", ".txt"})
    if not files:
        raise FileNotFoundError(f"No .pdf or .txt guidance files found in {directory}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    documents: list[Document] = []
    for path in files:
        if path.suffix.lower() == ".pdf":
            text = pdf_to_text(path, ocr_mode, edge_lines, min_fraction)
        else:
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


# --- Embedding and storage -----------------------------------------------


def rebuild_index(documents: list[Document], settings: Settings) -> int:
    """Embed the chunks and replace the whole Chroma collection with them."""
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
    parser = argparse.ArgumentParser(
        description="Rebuild the Chroma guidance index from the PDFs in GUIDANCE_DIR"
    )
    parser.add_argument(
        "--ocr",
        choices=["auto", "always", "never"],
        default="auto",
        help="auto OCRs only pages with no text layer (default)",
    )
    parser.add_argument("--edge-lines", type=int, default=3, help="lines checked at each page edge")
    parser.add_argument("--min-fraction", type=float, default=0.4, help="share of pages a line must repeat on")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the chunks without embedding or writing"
    )
    args = parser.parse_args(argv)
    if args.edge_lines < 1 or not 0 < args.min_fraction <= 1:
        parser.error("--edge-lines must be >= 1 and --min-fraction in (0, 1]")

    try:
        settings = Settings.from_env()
        documents = load_guidance(
            settings.guidance_dir,
            settings.chunk_size,
            settings.chunk_overlap,
            args.ocr,
            args.edge_lines,
            args.min_fraction,
        )
        if args.dry_run:
            for document in documents:
                metadata = document.metadata
                print(f"[{metadata['chunk_id']}] ({metadata['section_id']})\n{document.page_content}\n")
            return 0
        count = rebuild_index(documents, settings)
    except Exception:
        logging.exception("Indexing failed")
        return 1
    print(json.dumps({"indexed_chunks": count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
