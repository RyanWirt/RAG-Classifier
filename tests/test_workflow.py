import os
import time
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app.checker import ThirdPartyChecker
from app.classifier import Classifier
from app.config import Settings
from app.manager import ClassifierManager
from app.models import (
    CheckerResponse,
    CheckerResult,
    ClassifierResponse,
    GuidanceCitation,
)
from app.retrieval import GuidanceRetriever


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents

    def similarity_search_with_relevance_scores(self, text, k):
        return [(document, 0.9) for document in self.documents[:k]]


class FakeChain:
    def __init__(self, response):
        self.response = response
        self.inputs = None

    def invoke(self, inputs):
        self.inputs = inputs
        return self.response


class BrokenTool:
    name = "broken"

    def call(self, text):
        raise RuntimeError("unavailable")


class SlowTool:
    name = "slow"

    def call(self, text):
        time.sleep(0.05)
        return {}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.document = Document(
            page_content=(
                "Billing: choose Billing for an incorrect payment. "
                "Service Issue applies to an unavailable feature."
            ),
            metadata={
                "source_file": "policy.txt",
                "section_id": "Billing",
                "chunk_id": "policy.txt#chunk-0000",
            },
        )
        self.retriever = GuidanceRetriever(FakeVectorStore([self.document]), k=2)

    def test_classifier_parses_structured_output_and_resolves_exact_citation(self):
        chain = FakeChain(
            {
                "classification": "Billing",
                "confidence": 0.91,
                "rationale": "The input reports an incorrect payment.",
                "citation_chunk_ids": ["policy.txt#chunk-0000"],
            }
        )
        classifier = Classifier(self.retriever, chain=chain)

        response, citations = classifier.classify("I was charged incorrectly.")

        self.assertIsInstance(response, ClassifierResponse)
        self.assertEqual(response.classification, "Billing")
        self.assertEqual(citations[0].excerpt, self.document.page_content)
        self.assertEqual(citations[0].source_file, "policy.txt")

    def test_retriever_returns_relevance_scores(self):
        matches = self.retriever.search("incorrect payment")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].chunk_id, "policy.txt#chunk-0000")
        self.assertEqual(matches[0].relevance_score, 0.9)

    def test_llm_provider_is_configurable(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}):
            settings = Settings.from_env()

        self.assertEqual(settings.llm_provider, "anthropic")

    def test_classifier_rejects_malformed_structured_output(self):
        classifier = Classifier(
            self.retriever, chain=FakeChain({"classification": "Billing"})
        )

        response, citations = classifier.classify("Incorrect charge")

        self.assertEqual(response.classification, "UNDETERMINED")
        self.assertEqual(citations, [])

    def test_classifier_does_not_accept_ungrounded_labels(self):
        classifier = Classifier(
            self.retriever,
            chain=FakeChain(
                {
                    "classification": "Fraud",
                    "confidence": 0.8,
                    "rationale": "Unsupported label",
                    "citation_chunk_ids": ["policy.txt#chunk-0000"],
                }
            ),
        )

        response, _ = classifier.classify("Incorrect charge")

        self.assertEqual(response.classification, "UNDETERMINED")

    def test_checker_disagreement_requires_review(self):
        checker = ThirdPartyChecker(
            chain=FakeChain(
                CheckerResponse(
                    status="approved",
                    proposed_classification="Service Issue",
                    rationale="The report describes a service outage.",
                    citation_chunk_ids=["policy.txt#chunk-0000"],
                )
            )
        )
        citation = GuidanceCitation(
            source_file="policy.txt",
            section_id="Billing",
            chunk_id="policy.txt#chunk-0000",
            excerpt=self.document.page_content,
            relevance_score=0.9,
        )

        result = checker.check("Input", "Billing", "Reason", [citation], [])

        self.assertEqual(result.status, "approved")
        self.assertEqual(
            ClassifierManager._reconcile("Billing", result), "needs_review"
        )

    def test_checker_rejects_citations_not_in_supplied_guidance(self):
        checker = ThirdPartyChecker(
            chain=FakeChain(
                {
                    "status": "approved",
                    "rationale": "Looks right.",
                    "citation_chunk_ids": ["missing-chunk"],
                }
            )
        )

        result = checker.check("Input", "Billing", "Reason", [], [])

        self.assertEqual(result.status, "needs_review")

    def test_manager_records_mcp_tool_failures(self):
        class FakeClassifier:
            def classify(self, text, evidence):
                self.evidence = evidence
                return (
                    ClassifierResponse(
                        classification="UNDETERMINED",
                        confidence=0,
                        rationale="Not enough evidence.",
                    ),
                    [],
                )

        class FakeChecker:
            def check(self, *args):
                return CheckerResult(status="needs_review", rationale="Review")

        classifier = FakeClassifier()
        manager = ClassifierManager(
            classifier, FakeChecker(), [BrokenTool()], tool_timeout_seconds=0.5
        )

        result = manager.classify("Input")

        self.assertEqual(result.mcp_evidence[0]["status"], "error")
        self.assertEqual(result.mcp_evidence[0]["error"], "RuntimeError")
        self.assertEqual(result.final_status, "needs_review")

    def test_manager_records_mcp_timeouts(self):
        manager = ClassifierManager(
            classifier=object(),
            checker=object(),
            mcp_tools=[SlowTool()],
            tool_timeout_seconds=0.001,
        )

        evidence = manager._gather_mcp_evidence("Input")

        self.assertEqual(evidence[0]["status"], "error")
        self.assertEqual(evidence[0]["error"], "timeout")

    def test_approved_checker_approves_agreeing_result(self):
        self.assertEqual(
            ClassifierManager._reconcile(
                "Billing", CheckerResult(status="approved", rationale="Agrees")
            ),
            "approved",
        )


if __name__ == "__main__":
    unittest.main()
