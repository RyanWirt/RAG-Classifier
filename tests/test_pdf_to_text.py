import unittest

from app.pdf_to_text import clean_text, normalize_line, strip_repeated_lines


def make_pages(count=6):
    topics = ["billing", "refunds", "outages", "logins", "privacy", "security"]
    return [
        ["ACME Corp Confidential", f"Guidance about {topics[i - 1]}.", f"Page {i} of {count}"]
        for i in range(1, count + 1)
    ]


class PdfToTextTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
