from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from app.agent import ClassificationAgent, resolve_labels
from app.config import Settings
from app.mcp import terminology_tool
from app.providers import create_chat_model
from app.retrieval import ChromaGuidanceStore, GuidanceRetriever


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guidance-grounded text classifier")
    commands = parser.add_subparsers(dest="command", required=True)
    classify = commands.add_parser("classify", help="Attach a label to each sentence")
    source = classify.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="Sentence to label (repeatable)")
    source.add_argument("--file", type=Path, help="File with one sentence per line")
    classify.add_argument(
        "--labels",
        type=Path,
        help="JSON array of allowed labels (default: $LABELS_FILE, else labels in the index)",
    )
    classify.add_argument("--no-mcp", action="store_true", help="Do not give the agent the MCP tool")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        store = ChromaGuidanceStore(settings)
        if store.is_empty():
            logging.warning("Vector store is empty; run ingestion first")
        labels = resolve_labels(args.labels, store.collection)
        logging.info("Allowed labels: %s", ", ".join(labels))
        agent = ClassificationAgent(
            create_chat_model(settings),
            GuidanceRetriever(store, settings.retrieval_k),
            labels,
            terminology=None if args.no_mcp else terminology_tool(),
            tool_timeout_seconds=settings.tool_timeout_seconds,
        )
        sentences = args.text or [
            line.strip()
            for line in args.file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for sentence in sentences:
            print(agent.classify(sentence).model_dump_json(), flush=True)
        return 0
    except Exception:
        logging.exception("Classifier command failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
