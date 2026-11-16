from __future__ import annotations

import argparse
import logging
from typing import Sequence

from app.checker import ThirdPartyChecker
from app.classifier import Classifier
from app.config import Settings
from app.manager import ClassifierManager
from app.mcp import terminology_tool
from app.providers import create_chat_model
from app.retrieval import ChromaGuidanceStore, GuidanceRetriever


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guidance-grounded text classifier")
    commands = parser.add_subparsers(dest="command", required=True)
    classify = commands.add_parser("classify", help="Classify input text")
    classify.add_argument("--text", required=True, help="Text to classify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    settings = Settings.from_env()
    try:
        vector_store = ChromaGuidanceStore(settings)
        if vector_store.is_empty():
            logging.warning("Vector store is empty; run `python -m app.ingest` first")
        retriever = GuidanceRetriever(vector_store, settings.retrieval_k)
        model = create_chat_model(settings)
        manager = ClassifierManager(
            classifier=Classifier(retriever, model=model),
            checker=ThirdPartyChecker(model=model),
            mcp_tools=[terminology_tool()],
            tool_timeout_seconds=settings.tool_timeout_seconds,
        )
        result = manager.classify(args.text)
        print(result.model_dump_json(indent=2))
        return 0
    except Exception:
        logging.exception("Classifier command failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
