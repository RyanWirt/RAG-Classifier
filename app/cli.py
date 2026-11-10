from __future__ import annotations

import argparse
import json
import logging
from typing import Sequence

from app.checker import ThirdPartyChecker
from app.classifier import Classifier
from app.config import Settings
from app.manager import ClassifierManager
from app.mcp import MockTerminologyTool
from app.providers import create_chat_model
from app.retrieval import GuidanceRetriever, open_vector_store, rebuild_index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guidance-grounded text classifier")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("reindex", help="Rebuild the local guidance index")
    classify = commands.add_parser("classify", help="Classify input text")
    classify.add_argument("--text", required=True, help="Text to classify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    settings = Settings.from_env()
    try:
        if args.command == "reindex":
            count = rebuild_index(settings)
            print(json.dumps({"indexed_chunks": count}, indent=2))
            return 0

        vector_store = open_vector_store(settings)
        retriever = GuidanceRetriever(vector_store, settings.retrieval_k)
        model = create_chat_model(settings)
        manager = ClassifierManager(
            classifier=Classifier(retriever, model=model),
            checker=ThirdPartyChecker(model=model),
            mcp_tools=[MockTerminologyTool()],
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
