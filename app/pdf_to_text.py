"""Convert PDFs to plain .txt guidance: OCR scanned pages and drop running headers/footers.

Usage: python -m app.pdf_to_text FILE_OR_DIR... [-o guidance]
"""

from __future__ import annotations

import argparse
import functools
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

MIN_TEXT_CHARS = 25  # fewer extracted characters than this means a scanned page
OCR_DPI = 300


class Conversion(NamedTuple):
    text: str
    pages: int
    ocr_pages: int
    removed: dict[str, int]


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


def convert_pdf(
    path: Path, ocr_mode: str = "auto", edge_lines: int = 3, min_fraction: float = 0.4
) -> Conversion:
    import pypdfium2

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
    return Conversion(clean_text(pages), len(pages), ocr_pages, removed)


def _collect(paths: Sequence[str]) -> list[Path]:
    pdfs: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            pdfs += sorted(p for p in path.rglob("*") if p.suffix.lower() == ".pdf")
        elif path.is_file():
            pdfs.append(path)
        else:
            raise SystemExit(f"Not found: {path}")
    return pdfs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="PDF files or folders of PDFs")
    parser.add_argument("-o", "--output", type=Path, default=Path("guidance"))
    parser.add_argument(
        "--ocr",
        choices=["auto", "always", "never"],
        default="auto",
        help="auto OCRs only pages with no text layer (default)",
    )
    parser.add_argument("--edge-lines", type=int, default=3, help="lines checked at each page edge")
    parser.add_argument("--min-fraction", type=float, default=0.4, help="share of pages a line must repeat on")
    parser.add_argument("--force", action="store_true", help="overwrite existing .txt files")
    args = parser.parse_args(argv)
    if args.edge_lines < 1 or not 0 < args.min_fraction <= 1:
        parser.error("--edge-lines must be >= 1 and --min-fraction in (0, 1]")

    pdfs = _collect(args.paths)
    if not pdfs:
        raise SystemExit("No PDF files found.")
    failures = 0
    for pdf in pdfs:
        target = args.output / f"{pdf.stem}.txt"
        if target.exists() and not args.force:
            print(f"SKIP {pdf}: {target} exists (use --force)")
            failures += 1
            continue
        try:
            result = convert_pdf(pdf, args.ocr, args.edge_lines, args.min_fraction)
        except Exception as error:
            print(f"FAIL {pdf}: {type(error).__name__}: {error}")
            failures += 1
            continue
        if not result.text.strip():
            print(f"FAIL {pdf}: no text found (try --ocr always)")
            failures += 1
            continue
        args.output.mkdir(parents=True, exist_ok=True)
        target.write_text(result.text, encoding="utf-8")
        print(
            f"OK   {pdf} -> {target} ({result.pages} pages, {result.ocr_pages} OCR, "
            f"{len(result.removed)} header/footer patterns removed)"
        )
        for pattern, count in sorted(result.removed.items()):
            print(f"       removed {pattern!r} on {count} pages")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
