import tempfile
import unittest
from pathlib import Path

from app.ingest import clean_text, load_guidance, normalize_line, strip_repeated_lines


def make_pages(count=6):
    topics = ["billing", "refunds", "outages", "logins", "privacy", "security"]
    return [
        ["ACME Corp Confidential", f"Guidance about {topics[i - 1]}.", f"Page {i} of {count}"]
        for i in range(1, count + 1)
    ]


class PdfCleanupTests(unittest.TestCase):
    def test_normalize_line_ignores_digits_and_spacing(self):
        self.assertEqual(normalize_line("Page  3  of 12"), normalize_line("page 4 of 99"))

    def test_repeated_headers_and_page_numbers_are_removed(self):
        pages, removed = strip_repeated_lines(make_pages())

        self.assertEqual(pages[0], ["Guidance about billing."])
        self.assertEqual(set(removed), {"acme corp confidential", "page # of #"})

    def test_repeated_line_in_page_body_is_kept(self):
        pages = [
            ["Header", "a", "b", "c", "d", "Confirm the change", "e", "f", "g", "Footer"]
            for _ in range(5)
        ]

        cleaned, removed = strip_repeated_lines(pages, edge_lines=1)

        self.assertIn("Confirm the change", cleaned[0])
        self.assertEqual(set(removed), {"header", "footer"})

    def test_short_documents_are_left_untouched(self):
        pages = make_pages(2)

        cleaned, removed = strip_repeated_lines(pages)

        self.assertEqual(cleaned, pages)
        self.assertEqual(removed, {})

    def test_lines_that_do_not_repeat_are_kept(self):
        pages = [[f"Unique heading {chr(97 + i)}", f"body {chr(97 + i)}"] for i in range(6)]

        cleaned, removed = strip_repeated_lines(pages)

        self.assertEqual(cleaned, pages)
        self.assertEqual(removed, {})

    def test_clean_text_rejoins_hyphenation_and_collapses_blank_lines(self):
        text = clean_text([["informa-", "tion here   ", "", "", "", "next"], ["Page two"]])

        self.assertEqual(text, "information here\n\nnext\n\nPage two\n")

    def test_clean_text_keeps_real_hyphens(self):
        self.assertEqual(clean_text([["well-", "Known"]]), "well-\nKnown\n")


class ChunkingTests(unittest.TestCase):
    def test_chunks_preserve_source_section_and_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            guidance_dir = Path(temporary_directory)
            (guidance_dir / "policy.txt").write_text(
                "# Billing\n" + ("Refunds apply to incorrect charges. " * 12),
                encoding="utf-8",
            )

            chunks = load_guidance(guidance_dir, chunk_size=90, chunk_overlap=15)

            self.assertGreater(len(chunks), 1)
            self.assertEqual(chunks[0].metadata["source_file"], "policy.txt")
            self.assertEqual(chunks[0].metadata["section_id"], "Billing")
            self.assertEqual(chunks[0].metadata["chunk_id"], "policy.txt#chunk-0000")
            self.assertEqual(
                chunks[0].metadata["original_text"], chunks[0].page_content
            )
            self.assertIn("Refunds", chunks[0].page_content)

    def test_label_in_right_column_becomes_section_heading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            guidance_dir = Path(temporary_directory)
            (guidance_dir / "policy.txt").write_text(
                "Choose Feature Request when the text suggests a new           ## Feature Request\n"
                "capability and does not report that something is broken.\n",
                encoding="utf-8",
            )

            chunks = load_guidance(guidance_dir, chunk_size=800, chunk_overlap=100)

            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].metadata["section_id"], "Feature Request")
            self.assertEqual(
                chunks[0].page_content,
                "Choose Feature Request when the text suggests a new\n"
                "capability and does not report that something is broken.",
            )

    def test_missing_guidance_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(FileNotFoundError):
                load_guidance(Path(temporary_directory), 100, 10)


if __name__ == "__main__":
    unittest.main()
