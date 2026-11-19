import json
import tempfile
import time
import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.agent import ClassificationAgent, resolve_labels
from app.retrieval import GuidanceRetriever


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class FakeStore:
    def similarity_search_with_relevance_scores(self, text, k):
        document = Document(
            page_content="Choose Billing for incorrect charges.",
            metadata={"source_file": "g.txt", "section_id": "Billing", "chunk_id": "g.txt#chunk-0000"},
        )
        return [(document, 0.9)]


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def decision(label, chunk_ids):
    return call(
        "ClassificationDecision",
        {"label": label, "confidence": 0.9, "rationale": "r", "citation_chunk_ids": chunk_ids},
        "final",
    )


def make_agent(*messages):
    model = ScriptedModel(messages=iter(messages))
    return ClassificationAgent(model, GuidanceRetriever(FakeStore()), ["Billing", "Service Issue"])


class AgentTests(unittest.TestCase):
    def test_agent_searches_then_labels_sentence_with_citation(self):
        agent = make_agent(
            call("search_guidance", {"query": "charged twice"}, "s1"),
            decision("Billing", ["g.txt#chunk-0000"]),
        )

        result = agent.classify("I was charged twice.")

        self.assertEqual(result.text, "I was charged twice.")
        self.assertEqual(result.label, "Billing")
        self.assertEqual(result.citations[0].section_id, "Billing")

    def test_citation_that_was_never_retrieved_is_undetermined(self):
        agent = make_agent(decision("Billing", ["made-up#chunk-0001"]))

        self.assertEqual(agent.classify("I was charged twice.").label, "UNDETERMINED")

    def test_label_without_citation_is_undetermined(self):
        agent = make_agent(
            call("search_guidance", {"query": "x"}, "s1"),
            decision("Billing", []),
        )

        self.assertEqual(agent.classify("x").label, "UNDETERMINED")

    def test_slow_terminology_tool_times_out_without_failing_the_run(self):
        class SlowTool:
            name = "slow"

            def call(self, text):
                time.sleep(0.2)
                return {}

        model = ScriptedModel(messages=iter([
            call("lookup_terminology", {"text": "x"}, "t1"),
            call("search_guidance", {"query": "x"}, "s1"),
            decision("Billing", ["g.txt#chunk-0000"]),
        ]))
        agent = ClassificationAgent(
            model, GuidanceRetriever(FakeStore()), ["Billing"],
            terminology=SlowTool(), tool_timeout_seconds=0.01,
        )

        self.assertEqual(agent.classify("x").label, "Billing")

    def test_resolve_labels_reads_json_file_and_drops_undetermined(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps(["Billing", "UNDETERMINED", "Billing"]), encoding="utf-8")

            self.assertEqual(resolve_labels(path), ["Billing"])

    def test_labels_file_must_be_array_of_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text('{"labels": "Billing"}', encoding="utf-8")

            with self.assertRaises(ValueError):
                resolve_labels(path)


if __name__ == "__main__":
    unittest.main()
