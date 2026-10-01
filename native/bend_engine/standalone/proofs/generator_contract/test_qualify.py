"""Host result classification tests; not source-proof evidence."""
from __future__ import annotations

import unittest

from .qualify import safe, semantic_rejection


class ClassificationTests(unittest.TestCase):
    def result(self, code: int | None = 0, text: str = "All terms check.\n", timed_out: bool = False) -> dict:
        return {"exit_code": code, "raw_text": text, "timed_out": timed_out}

    def test_only_exact_safe_success(self):
        self.assertTrue(safe(self.result()))
        for result in (
            self.result(1), self.result(-9), self.result(None, timed_out=True),
            self.result(text=""), self.result(text="All terms check.\nWARNING: unsafe"),
        ):
            self.assertFalse(safe(result))

    def test_timeout_cannot_be_a_rejection(self):
        text = "expected x observed y Location: Closed.public_builder_matches_recipe"
        self.assertTrue(semantic_rejection(self.result(1, text)))
        for result in (
            self.result(None, text, True), self.result(-9, text),
            self.result(0, text), self.result(1, text, True),
        ):
            self.assertFalse(semantic_rejection(result))

    def test_unrelated_or_resource_diagnostic_is_not_semantic(self):
        text = "expected x observed y Location: Closed.public_builder_matches_recipe"
        for suffix in ("RangeError", "out of memory", "more than once", "unsafe", "TODO"):
            self.assertFalse(semantic_rejection(self.result(1, text + " " + suffix)))
        self.assertFalse(semantic_rejection(self.result(1, "expected x observed y Location: other")))
        self.assertFalse(semantic_rejection(self.result(1, "parse failed")))


if __name__ == "__main__":
    unittest.main()
