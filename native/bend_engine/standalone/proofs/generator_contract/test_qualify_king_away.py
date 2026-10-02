"""Fail-closed output and source policy controls; no compiler process required."""
from __future__ import annotations

import unittest

from .qualify import safe
from .qualify_castles import semantic_rejection
from .qualify_king_away import manifest
from ..table_preservation._validation import require


class GateTests(unittest.TestCase):
    def test_exact_safe_output(self):
        result = {"exit_code": 0, "timed_out": False, "raw_text": "All terms check.\n"}
        require(safe(result), "exact clean output rejected")
        for update in ({"exit_code": 1}, {"timed_out": True},
                       {"raw_text": "All terms check.\nWARNING: unsafe"}, {"raw_text": ""}):
            require(not safe({**result, **update}), "unsafe output credited")

    def test_semantic_control_cannot_credit_backend_failure(self):
        raw_text = "- expected : {{False{{}} == False{{}} : Bool}}\n- observed : {{True{{}} == False{{}} : Bool}}\nLocation: put\n"
        result = {"exit_code": 1, "timed_out": False, "raw_text": raw_text}
        require(semantic_rejection(result, "put"), "intended semantic rejection lost")
        for update in ({"exit_code": 0}, {"timed_out": True},
                       {"raw_text": "RangeError\n"},
                       {"raw_text": raw_text + "out of memory"}):
            require(not semantic_rejection({**result, **update}, "put"), "backend failure credited")
        require(not semantic_rejection(result, "side"), "wrong location credited")

    def test_complete_manifest(self):
        manifest()


if __name__ == "__main__":
    unittest.main()
