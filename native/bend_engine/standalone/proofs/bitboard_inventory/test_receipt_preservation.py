"""CPU-only nested-process controls; no Bend checker or theorem qualification."""
# Use unittest controls deliberately: their checks survive Python -O and -OO
# without pytest assertion rewriting, matching nearby proof host tests.
# ruff: noqa: PT009, PT027
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import qualify


class ReceiptPreservationTests(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.historical = self.repo / "docs/experiments/evidence/bend-bitboard-inventory/inherited-destination-factorization.json"
        self.historical.parent.mkdir(parents=True)
        self.historical.write_bytes(b'{"historical": "PASS"}\n')
        self.original = self.historical.read_bytes()
        self.parent = self.root / "outputs/parent.json"
        self.report: dict = {"bitboard_inventory_gate": "NOT_COMPLETED"}
        self.pass_receipt: dict = {
            "destination_factorization_gate": "PASS",
            "accepted_candidate_definitions": 4,
            "controls": [{"rejected": True} for _ in range(5)],
        }
        self.real_run = subprocess.run

    def execute(self, receipt=None, code=0, mode="normal", report_path=None):
        if receipt is None:
            receipt = self.pass_receipt
        raw = receipt if isinstance(receipt, str) else json.dumps(receipt)

        def child(command, **kwargs):
            target = Path(command[-1])
            self.assertTrue(target.is_absolute())
            self.assertEqual(command[-2], "--report")
            self.assertNotEqual(target, self.historical)
            self.assertEqual(kwargs["cwd"], self.repo)
            self.assertEqual(kwargs["timeout"], 900)
            self.assertFalse(kwargs["check"])
            persisted = json.loads(self.parent.read_text())["inherited_destination_gate"]
            self.assertEqual(persisted["receipt"], str(target))
            self.assertIsNone(persisted["exit_code"])
            self.assertTrue(target.parent.is_dir())
            self.assertFalse(target.exists())
            if mode == "launch_error":
                raise FileNotFoundError("synthetic launch failure")
            # Actual bounded Python process uses the real atomic receipt writer's
            # replacement behavior, rather than a mocked return value or file read.
            source = (
                "from pathlib import Path; import os,sys,time; "
                "p=Path(sys.argv[1]); q=p.with_suffix('.tmp'); "
                "q.write_text(sys.argv[2]); os.replace(q,p); "
                + ("time.sleep(5); " if mode == "timeout" else "")
                + "sys.exit(int(sys.argv[3]))"
            )
            if mode == "missing":
                source = "import sys; sys.exit(0)"
            return self.real_run(
                [sys.executable, "-c", source, str(target), raw, str(code)],
                cwd=kwargs["cwd"], capture_output=True, text=True,
                timeout=0.5 if mode == "timeout" else 5, check=False,
            )

        try:
            with patch.object(qualify, "REPO", self.repo), patch.object(qualify.subprocess, "run", child):
                qualify.check_inherited_destination(self.root / "compiler", report_path or self.parent, self.report)
        finally:
            self.assertEqual(self.historical.read_bytes(), self.original)
            evidence = json.loads(self.parent.read_text())["inherited_destination_gate"]
            observed = Path(evidence["receipt"])
            self.assertTrue(observed.parent.exists())
            if observed.is_file():
                self.assertEqual(evidence["receipt_sha256"], hashlib.sha256(observed.read_bytes()).hexdigest())
            else:
                self.assertIsNone(evidence["receipt_sha256"])
            self.assertEqual(evidence, self.report["inherited_destination_gate"])

    def test_success_and_repeated_runs_keep_distinct_persistent_receipts(self):
        self.execute()
        first = Path(self.report["inherited_destination_gate"]["receipt"])
        original = first.read_bytes()
        self.execute()
        self.assertNotEqual(str(first), self.report["inherited_destination_gate"]["receipt"])
        self.assertEqual(first.read_bytes(), original)

    def test_relative_parent_report_outside_repo(self):
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            self.execute(report_path=Path("outputs/parent.json"))
        finally:
            os.chdir(previous)

    def test_nonzero_pass_receipt_refused_and_hashed(self):
        with self.assertRaisesRegex(AssertionError, "gate failed"):
            self.execute(code=1)

    def test_nonpass_receipt_refused_and_hashed(self):
        self.pass_receipt["destination_factorization_gate"] = "NOT_COMPLETED"
        with self.assertRaisesRegex(AssertionError, "four contracts"):
            self.execute()

    def test_wrong_contract_count_refused(self):
        self.pass_receipt["accepted_candidate_definitions"] = 3
        with self.assertRaisesRegex(AssertionError, "four contracts"):
            self.execute()

    def test_wrong_control_count_refused(self):
        self.pass_receipt["controls"].pop()
        with self.assertRaisesRegex(AssertionError, "five controls"):
            self.execute()

    def test_unrejected_control_refused(self):
        self.pass_receipt["controls"][0]["rejected"] = False
        with self.assertRaisesRegex(AssertionError, "five controls"):
            self.execute()

    def test_timeout_preserves_and_hashes_partial_receipt(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.execute(receipt={"destination_factorization_gate": "NOT_COMPLETED"}, mode="timeout")
        self.assertTrue(self.report["inherited_destination_gate"]["timed_out"])
        self.assertIsNone(self.report["inherited_destination_gate"]["exit_code"])

    def test_timeout_cannot_accept_even_a_pass_receipt(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.execute(mode="timeout")
        self.assertIsNone(self.report["inherited_destination_gate"]["exit_code"])

    def test_main_dispatches_new_gate_and_persists_failure(self):
        engine = self.repo / "native/bend_engine"
        suite = engine / "standalone/proofs/bitboard_inventory"
        suite.mkdir(parents=True)
        names = {f"fixture/{i}.bend" for i in range(249)}
        extra = {
            "standalone/toolchain.json", "standalone/verify_compiler.js",
            "standalone/proofs/generator_contract/qualify.py",
            "standalone/proofs/generator_contract/qualify_castles.py",
            "standalone/proofs/table_preservation/focused.py",
            "standalone/proofs/table_preservation/_validation.py",
            "standalone/proofs/destination_factorization/qualify.py",
        }
        for name in names | extra:
            target = engine / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture")
        prior = self.repo / "docs/experiments/evidence/bend-king-away/source.json"
        prior.parent.mkdir(parents=True)
        prior.write_text(json.dumps({"source_sha256s": {
            name: hashlib.sha256(b"fixture").hexdigest() for name in names
        }}))
        pin = subprocess.CompletedProcess([], 0,
            "aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae", "")
        with patch.object(qualify, "REPO", self.repo), \
                patch.object(qualify, "ENGINE", engine), \
                patch.object(qualify, "SUITE", suite), \
                patch.object(qualify, "load_closure", return_value=lambda *args: names), \
                patch.object(qualify, "manifest"), \
                patch.object(qualify.subprocess, "run", return_value=pin), \
                patch.dict(os.environ, {"BUN": "inert-bun"}), \
                patch.object(sys, "argv", ["qualify", str(self.root / "compiler"),
                                            "--report", str(self.parent)]), \
                patch.object(qualify, "check_inherited_destination",
                             side_effect=RuntimeError("stop before proof checks")) as gate, \
                self.assertRaisesRegex(RuntimeError, "stop before proof checks"):
            qualify.main()
        gate.assert_called_once()
        self.assertEqual(gate.call_args.args[1], self.parent)
        stored = json.loads(self.parent.read_text())
        self.assertEqual(stored["bitboard_inventory_gate"], "NOT_COMPLETED")
        self.assertIn("stop before proof checks", stored["failure"])
        self.assertEqual(self.historical.read_bytes(), self.original)

    def test_launch_error_keeps_owned_path_without_inventing_receipt(self):
        with self.assertRaises(FileNotFoundError):
            self.execute(mode="launch_error")
        self.assertIsNone(self.report["inherited_destination_gate"]["exit_code"])

    def test_missing_receipt_is_not_historical_fallback(self):
        with self.assertRaisesRegex(AssertionError, "receipt missing"):
            self.execute(mode="missing")

    def test_malformed_receipt_still_has_exact_hash(self):
        with self.assertRaises(json.JSONDecodeError):
            self.execute(receipt="{invalid")


if __name__ == "__main__":
    unittest.main()
