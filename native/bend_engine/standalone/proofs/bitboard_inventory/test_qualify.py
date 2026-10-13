"""Fail-closed result classification and stale-receipt protection."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .qualify import main, safe_result
from ..generator_contract.qualify_castles import semantic_rejection
from ..table_preservation._validation import require


class InventoryGateTests(unittest.TestCase):
    def test_exact_success_classification(self):
        clean = {"exit_code": 0, "timed_out": False, "raw_text": "All terms check.\n"}
        require(safe_result(clean), "clean result rejected")
        for update in ({"exit_code": 1}, {"timed_out": True},
                       {"raw_text": "All terms check."},
                       {"raw_text": "All terms check.\nWARNING: unsafe"},
                       {"raw_text": ""}):
            require(not safe_result({**clean, **update}), "unclean result credited")

    def test_semantic_rejection_classification(self):
        raw = (
            "Error:\\n- expected : Chess.Ply{src, dst, 0, 2} <> tail\\n"
            "- observed : Chess.Ply{src, dst, 0, tag} <> tail\\n"
            "Location: singleton_indices\\n"
        ).replace("\\n", "\n")
        result = {"exit_code": 1, "timed_out": False, "raw_text": raw}
        require(semantic_rejection(result, "singleton_indices"), "intended semantic error rejected")
        for update in ({"exit_code": 0}, {"timed_out": True},
                       {"raw_text": raw + "RangeError"},
                       {"raw_text": raw + "WARNING: unsafe"},
                       {"raw_text": raw + "consumed more than once"}):
            require(not semantic_rejection({**result, **update}, "singleton_indices"),
                    "backend failure credited as semantic rejection")
        require(not semantic_rejection(result, "scan_step_onehot"), "wrong obligation credited")

    def test_failed_start_invalidates_stale_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text('{"bitboard_inventory_gate":"PASS","accepted_candidate_definitions":9}')
            with patch("sys.argv", ["qualify", directory, "--report", str(report)]), \
                 patch.dict("os.environ", {"BUN": "bun"}), \
                 patch("native.bend_engine.standalone.proofs.bitboard_inventory.qualify.subprocess.run",
                       side_effect=FileNotFoundError("missing pinned compiler")):
                try:
                    main()
                except FileNotFoundError:
                    pass
                else:
                    require(False, "missing compiler did not fail")
            result = json.loads(report.read_text())
            require(result["bitboard_inventory_gate"] == "NOT_COMPLETED", "stale PASS survived")
            require(result["accepted_candidate_definitions"] == 0, "failed invocation received credit")


if __name__ == "__main__":
    unittest.main()
