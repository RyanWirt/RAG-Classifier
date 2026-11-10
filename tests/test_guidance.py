import tempfile
import unittest
from pathlib import Path

from app.guidance import load_guidance


class GuidanceIngestionTests(unittest.TestCase):
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

    def test_missing_guidance_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(FileNotFoundError):
                load_guidance(Path(temporary_directory), 100, 10)


if __name__ == "__main__":
    unittest.main()
