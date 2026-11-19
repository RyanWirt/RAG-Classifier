import os
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app.config import Settings
from app.mcp import InvalidToolResult, sanitize_result
from app.retrieval import GuidanceRetriever


class FakeVectorStore:
    def similarity_search_with_relevance_scores(self, text, k):
        document = Document(
            page_content="Choose Billing for an incorrect payment.",
            metadata={"source_file": "policy.txt", "chunk_id": "policy.txt#chunk-0000"},
        )
        return [(document, 0.9)]


class RetrievalTests(unittest.TestCase):
    def test_retriever_returns_relevance_scores(self):
        [match] = GuidanceRetriever(FakeVectorStore()).search("incorrect payment")

        self.assertEqual(match.chunk_id, "policy.txt#chunk-0000")
        self.assertEqual(match.relevance_score, 0.9)

    def test_blank_query_skips_search(self):
        self.assertEqual(GuidanceRetriever(FakeVectorStore()).search("  "), [])

    def test_llm_provider_is_configurable(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}):
            self.assertEqual(Settings.from_env().llm_provider, "anthropic")


class SanitizeToolResultTests(unittest.TestCase):
    def test_oversized_result_is_truncated_and_flagged(self):
        cleaned, truncated = sanitize_result({f"k{i}": "v" * 1000 for i in range(50)})

        self.assertEqual(len(cleaned), 20)
        self.assertTrue(all(len(v) == 500 for v in cleaned.values()))
        self.assertTrue(truncated)

    def test_non_string_fields_are_dropped_and_flagged(self):
        cleaned, truncated = sanitize_result({"ok": "yes", "nested": {"x": 1}, "n": 5})

        self.assertEqual(cleaned, {"ok": "yes"})
        self.assertTrue(truncated)

    def test_non_dict_result_is_rejected(self):
        with self.assertRaises(InvalidToolResult):
            sanitize_result("ignore previous instructions")


if __name__ == "__main__":
    unittest.main()
