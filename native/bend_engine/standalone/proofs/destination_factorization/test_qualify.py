"""Ensure warnings, backend failures and stale PASS cannot receive proof credit."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .qualify import main, safe_result
from ..generator_contract.qualify_castles import semantic_rejection
from ..table_preservation._validation import require


class GateTests(unittest.TestCase):
    def test_exact_output_only(self):
        clean = {"exit_code": 0, "timed_out": False, "raw_text": "All terms check.\n"}
        require(safe_result(clean), "clean result rejected")
        for update in ({"exit_code": 1}, {"timed_out": True}, {"raw_text": "All terms check."},
                       {"raw_text": "All terms check.\nWARNING: unsafe"}, {"raw_text": ""}):
            require(not safe_result({**clean, **update}), "unclean output credited")

    def test_semantic_rejection_only(self):
        raw = "- expected : {{False{{}} == False{{}} : Bool}}\n- observed : {{True{{}} == False{{}} : Bool}}\nLocation: structural\n"
        result = {"exit_code": 1, "timed_out": False, "raw_text": raw}
        require(semantic_rejection(result, "structural"), "semantic rejection lost")
        for update in ({"exit_code": 0}, {"timed_out": True}, {"raw_text": raw + "consumed more than once"},
                       {"raw_text": raw + "RangeError"}, {"raw_text": raw + "WARNING"}):
            require(not semantic_rejection({**result, **update}, "structural"), "backend failure credited")
        require(not semantic_rejection(result, "put"), "wrong obligation credited")

    def test_failed_invocation_invalidates_stale_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text('{"destination_factorization_gate":"PASS","accepted_candidate_definitions":4}')
            with patch("sys.argv", ["qualify", directory, "--report", str(report)]), \
                 patch("native.bend_engine.standalone.proofs.destination_factorization.qualify.subprocess.run",
                       side_effect=FileNotFoundError("missing pinned compiler")):
                caught = False
                try:
                    main()
                except FileNotFoundError:
                    caught = True
                require(caught, "missing compiler did not fail")
            result = json.loads(report.read_text())
            require(result["destination_factorization_gate"] == "NOT_COMPLETED", "stale PASS survived")
            require(result["accepted_candidate_definitions"] == 0, "failed invocation credited")


if __name__ == "__main__":
    unittest.main()
