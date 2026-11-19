"""Vision OCR ingestion: two-column guidance PDFs -> (label, guidance) pairs -> Chroma.

Each page is rendered to an image and sent to a vision model served by the
LiteLLM proxy (OPENAI_BASE_URL). The model returns the left-column description
and right-column label for every row. Entries are upserted into the same Chroma
collection the classifier and MCP server read.

Run with `python -m app.vision_ingest path/to/pdfs`.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.embeddings import COLLECTION, create_embeddings, open_client

RENDER_DPI = 200

SYSTEM_PROMPT = """You transcribe pages of a classification guidance document.
Each page is laid out in two columns: the LEFT column is a guidance description,
the RIGHT column is the classification label that the description belongs to.

Return ONLY a JSON object of the form:
{"entries": [{"label": "<right column text>", "guidance": "<left column text>"}]}

Rules:
- One entry per row, in reading order, top to bottom.
- Transcribe text verbatim; do not summarise or invent text.
- Join lines that wrap within one cell into a single paragraph.
- If the first description on the page has no label beside it (it continues
  from the previous page), use null for its label.
- Ignore page headers, footers, page numbers, and column titles.
- If the page has no guidance rows, return {"entries": []}."""


@dataclass(frozen=True)
class GuidanceEntry:
    label: str
    guidance: str
    source_file: str
    page: int


# --- Vision model --------------------------------------------------------


def create_vision_model(model: str) -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=0)


def _image_message(png: bytes) -> HumanMessage:
    data = base64.b64encode(png).decode("ascii")
    return HumanMessage(
        content=[
            {"type": "text", "text": "Extract the label/guidance rows from this page."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
        ]
    )


def parse_entries(reply: str) -> list[dict[str, str | None]]:
    """Parse the model's JSON reply, tolerating code fences and surrounding prose."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        raise ValueError(f"Vision model reply contained no JSON: {reply[:200]!r}")
    entries = json.loads(match.group(0)).get("entries", [])
    parsed = []
    for entry in entries:
        guidance = str(entry.get("guidance") or "").strip()
        label = entry.get("label")
        label = str(label).strip() if label else None
        if guidance or label:
            parsed.append({"label": label, "guidance": guidance})
    return parsed


def extract_page(model: Any, png: bytes) -> list[dict[str, str | None]]:
    reply = model.invoke([SystemMessage(content=SYSTEM_PROMPT), _image_message(png)])
    content = reply.content if isinstance(reply.content, str) else "".join(
        part.get("text", "") for part in reply.content if isinstance(part, dict)
    )
    return parse_entries(content)


# --- PDF handling --------------------------------------------------------


def render_pages(path: Path, dpi: int = RENDER_DPI) -> list[bytes]:
    try:
        import pypdfium2
    except ImportError:
        raise SystemExit("PDF reading needs: pip install -r requirements-ingest.txt") from None

    document = pypdfium2.PdfDocument(path)
    images = []
    try:
        for index in range(len(document)):
            page = document[index]
            try:
                buffer = io.BytesIO()
                page.render(scale=dpi / 72).to_pil().save(buffer, format="PNG")
                images.append(buffer.getvalue())
            finally:
                page.close()
    finally:
        document.close()
    return images


def merge_entries(
    pages: list[list[dict[str, str | None]]], source_file: str
) -> list[GuidanceEntry]:
    """Attach unlabeled continuations to the previous entry and merge repeated labels."""
    merged: list[GuidanceEntry] = []
    for page_number, entries in enumerate(pages, start=1):
        for entry in entries:
            label, guidance = entry["label"], entry["guidance"] or ""
            if not label:
                if merged and guidance:
                    last = merged[-1]
                    merged[-1] = GuidanceEntry(
                        last.label, f"{last.guidance} {guidance}".strip(), source_file, last.page
                    )
                elif guidance:
                    logging.warning("%s p%d: unlabeled text dropped", source_file, page_number)
                continue
            if merged and merged[-1].label == label and not guidance:
                continue
            merged.append(GuidanceEntry(label, guidance, source_file, page_number))
    return [entry for entry in merged if entry.guidance]


def extract_pdf(
    path: Path, model: Any, render: Callable[[Path], list[bytes]] = render_pages
) -> list[GuidanceEntry]:
    pages = []
    for page_number, png in enumerate(render(path), start=1):
        entries = extract_page(model, png)
        logging.info("%s p%d: %d rows", path.name, page_number, len(entries))
        pages.append(entries)
    return merge_entries(pages, path.name)


# --- Formatting and storage ----------------------------------------------


def to_guidance_text(entries: list[GuidanceEntry]) -> str:
    """Render entries in the same `## Label` layout as the .txt guidance files."""
    return "\n\n".join(f"## {entry.label}\n\n{entry.guidance}" for entry in entries) + "\n"


def to_documents(
    entries: list[GuidanceEntry], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents: list[Document] = []
    counters: dict[str, int] = {}
    for entry in entries:
        for chunk in splitter.split_text(entry.guidance):
            number = counters.get(entry.source_file, 0)
            counters[entry.source_file] = number + 1
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source_file": entry.source_file,
                        "section_id": entry.label,
                        "chunk_id": f"{entry.source_file}#chunk-{number:04d}",
                        "original_text": chunk,
                        "page": entry.page,
                        "extraction": "vision",
                    },
                )
            )
    return documents


def upsert_documents(documents: list[Document], settings: Settings) -> int:
    """Replace each source file's chunks in Chroma, leaving other files untouched."""
    if not documents:
        return 0
    texts = [document.page_content for document in documents]
    vectors = create_embeddings(settings).embed_documents(texts)
    collection = open_client(settings).get_or_create_collection(COLLECTION)
    for source in {document.metadata["source_file"] for document in documents}:
        collection.delete(where={"source_file": source})
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
        description="OCR two-column guidance PDFs with a vision model and add them to Chroma"
    )
    parser.add_argument("folder", type=Path, help="folder of guidance PDFs")
    parser.add_argument(
        "--model",
        default=os.getenv("VISION_MODEL"),
        help="vision model name on the LiteLLM proxy (default: $VISION_MODEL)",
    )
    parser.add_argument(
        "--save-txt",
        type=Path,
        metavar="DIR",
        help="also write <pdf name>.txt in `## Label` format to DIR",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the extracted guidance without writing"
    )
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("set --model or VISION_MODEL")
    if not args.folder.is_dir():
        parser.error(f"not a directory: {args.folder}")

    try:
        settings = Settings.from_env()
        model = create_vision_model(args.model)
        pdfs = sorted(p for p in args.folder.iterdir() if p.suffix.lower() == ".pdf")
        if not pdfs:
            raise FileNotFoundError(f"No .pdf files found in {args.folder}")
        summary: dict[str, int] = {}
        documents: list[Document] = []
        for path in pdfs:
            entries = extract_pdf(path, model)
            summary[path.name] = len(entries)
            text = to_guidance_text(entries)
            if args.dry_run:
                print(f"===== {path.name} =====\n{text}")
            if args.save_txt:
                args.save_txt.mkdir(parents=True, exist_ok=True)
                (args.save_txt / f"{path.stem}.txt").write_text(text, encoding="utf-8")
            documents.extend(to_documents(entries, settings.chunk_size, settings.chunk_overlap))
        if args.dry_run:
            return 0
        count = upsert_documents(documents, settings)
    except Exception:
        logging.exception("Vision ingestion failed")
        return 1
    print(json.dumps({"entries": summary, "indexed_chunks": count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
