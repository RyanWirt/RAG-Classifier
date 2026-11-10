"""Interactive trace of the classification workflow. Run: python app_test.py"""

from __future__ import annotations

import logging
import textwrap

from app.checker import ThirdPartyChecker
from app.classifier import Classifier
from app.config import Settings
from app.manager import ClassifierManager
from app.mcp import terminology_tool
from app.models import ClassificationResult
from app.providers import create_chat_model
from app.retrieval import GuidanceRetriever, RetrievedChunk, open_vector_store

STATUS_MEANING = {
    "approved": "The checker agreed with the classifier's label.",
    "needs_review": "A person should review: label undetermined, checker unsure, or disagreement.",
    "rejected": "The checker rejected the classifier's label.",
}


class TracingRetriever(GuidanceRetriever):
    """Remembers the chunks returned for the latest query so they can be shown."""

    last: list[RetrievedChunk] = []

    def search(self, text: str) -> list[RetrievedChunk]:
        self.last = super().search(text)
        return self.last


def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=90, initial_indent=indent, subsequent_indent=indent)


def show(result: ClassificationResult, retrieved: list[RetrievedChunk]) -> None:
    print("\n[1] Tool calls")
    if not result.mcp_evidence:
        print("  (no tools registered)")
    for entry in result.mcp_evidence:
        detail = f" [{entry['error']}]" if "error" in entry else ""
        print(f"  - {entry['tool']}: {entry['status']}{detail} ({entry['latency_ms']} ms)")
        for key, value in entry.get("result", {}).items():
            print(_wrap(f"{key}: {value}", "      "))

    print("\n[2] Retrieval (guidance chunks sent to the classifier)")
    if not retrieved:
        print("  (nothing retrieved; has `python -m app reindex` been run?)")
    for chunk in retrieved:
        section = chunk.document.metadata.get("section_id") or "-"
        score = f"{chunk.relevance_score:.2f}" if chunk.relevance_score is not None else "n/a"
        print(f"  - {chunk.chunk_id}  section={section}  score={score}")

    cited = ", ".join(c.chunk_id for c in result.citations) or "none"
    print("\n[3] Classifier agent")
    print(f"  proposed: {result.classification} (confidence {result.confidence:.2f})")
    print(_wrap(f"why: {result.rationale}", "  "))
    print(f"  cited: {cited}")

    checker = result.checker_result
    print("\n[4] Checker agent")
    print(f"  verdict: {checker.status}")
    if checker.proposed_classification:
        print(f"  checker's label: {checker.proposed_classification}")
    print(_wrap(f"why: {checker.rationale}", "  "))
    print(f"  cited: {', '.join(c.chunk_id for c in checker.citations) or 'none'}")

    print("\n=== Decision ===")
    print(f"  Classification: {result.classification}")
    print(f"  Final status:   {result.final_status}")
    print(_wrap(STATUS_MEANING[result.final_status], "  "))


def build() -> tuple[ClassifierManager, TracingRetriever]:
    settings = Settings.from_env()
    vector_store = open_vector_store(settings)
    if not vector_store.get(limit=1)["ids"]:
        print("Warning: the vector store is empty. Run `python -m app reindex` first.")
    retriever = TracingRetriever(vector_store, settings.retrieval_k)
    model = create_chat_model(settings)
    manager = ClassifierManager(
        classifier=Classifier(retriever, model=model),
        checker=ThirdPartyChecker(model=model),
        mcp_tools=[terminology_tool()],
        tool_timeout_seconds=settings.tool_timeout_seconds,
    )
    return manager, retriever


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    manager, retriever = build()
    print("RAG-Classifier trace. Enter text to classify; blank line or 'quit' exits.")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text or text.casefold() in {"quit", "exit"}:
            return 0
        print("Running tools, retrieval, classifier and checker...")
        try:
            result = manager.classify(text)
        except Exception as error:
            print(f"Error: {type(error).__name__}: {error}")
            continue
        show(result, retriever.last)


if __name__ == "__main__":
    raise SystemExit(main())
