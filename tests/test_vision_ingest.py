import unittest
from pathlib import Path
from types import SimpleNamespace

from app.vision_ingest import extract_pdf, parse_entries, to_documents, to_guidance_text


class FakeVisionModel:
    def __init__(self, replies):
        self.replies = list(replies)

    def invoke(self, messages):
        return SimpleNamespace(content=self.replies.pop(0))


class VisionIngestTests(unittest.TestCase):
    def test_parse_entries_tolerates_code_fences(self):
        reply = '```json\n{"entries": [{"label": "Billing", "guidance": " Charges. "}]}\n```'

        self.assertEqual(parse_entries(reply), [{"label": "Billing", "guidance": "Charges."}])

    def test_unlabeled_row_continues_previous_page_entry(self):
        model = FakeVisionModel([
            '{"entries": [{"label": "Billing", "guidance": "Refunds for wrong"}]}',
            '{"entries": [{"label": null, "guidance": "charges."},'
            ' {"label": "Service Issue", "guidance": "Feature is down."}]}',
        ])

        entries = extract_pdf(Path("rules.pdf"), model, render=lambda _: [b"p1", b"p2"])

        self.assertEqual([(e.label, e.guidance, e.page) for e in entries], [
            ("Billing", "Refunds for wrong charges.", 1),
            ("Service Issue", "Feature is down.", 2),
        ])
        self.assertEqual(
            to_guidance_text(entries),
            "## Billing\n\nRefunds for wrong charges.\n\n## Service Issue\n\nFeature is down.\n",
        )

    def test_documents_carry_label_as_section_id(self):
        model = FakeVisionModel(['{"entries": [{"label": "Billing", "guidance": "Charges."}]}'])
        entries = extract_pdf(Path("rules.pdf"), model, render=lambda _: [b"p1"])

        [document] = to_documents(entries, chunk_size=800, chunk_overlap=0)

        self.assertEqual(document.metadata["section_id"], "Billing")
        self.assertEqual(document.metadata["chunk_id"], "rules.pdf#chunk-0000")
        self.assertEqual(document.metadata["page"], 1)


if __name__ == "__main__":
    unittest.main()
