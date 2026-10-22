"""Exercise the same report/seal consumer used for current local reports."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts import backlog_handoff as h
from scripts import current_backlog_report as report
from test_handoff import NOW, OWNER, cycle, ev, owner, row


def crosswalk(items):
    ledger = {"schema": "backlog-assignment-crosswalk/v18", "records": [{"id": key, "original": "retained"} for key in items]}
    return h.append_cycle(ledger, cycle(items))


class ReportConsumerTests(unittest.TestCase):
    def publish(self, ledger, directory):
        path = directory / "input.json"
        raw = json.dumps(ledger).encode()
        path.write_bytes(raw)
        output = directory / "output"
        return report.publish_report(path, output, NOW, h.digest(raw)), output

    def test_unhandled_terminal_worker_prevents_actual_consumer_seal(self):
        item = row("terminal")
        item["worker"] = {"task_id": OWNER, "state": "completed", "evidence": ev("terminal-result")}
        with tempfile.TemporaryDirectory() as temp:
            result, output = self.publish(crosswalk({"commit:f1": item}), Path(temp))
            self.assertFalse(result["audit_ok"])
            self.assertFalse((output / "seal.json").exists())
            self.assertFalse((output / "report.json").exists())
            pending = json.loads((output / "pending-report.json").read_bytes())
            self.assertEqual(pending["finding_ids"], ["commit:f1"])
            self.assertEqual(pending["unresolved_ids"], ["commit:f1"])
            self.assertIn("DROPPED_HANDOFF", {i["code"] for i in pending["handoff_audit"]["issues"]})

    def test_positive_continuation_seals_same_consumer_and_pins_sources(self):
        item = row("terminal")
        item["worker"] = {"task_id": OWNER, "state": "completed", "evidence": ev("terminal-result")}
        item["continuation"] = {"kind": "review", "owner": owner(), "description": "review actual saved result",
                                "worker_result": ev("terminal-result")["sha256"], "evidence": ev("review-started")}
        with tempfile.TemporaryDirectory() as temp:
            result, output = self.publish(crosswalk({"commit:f1": item}), Path(temp))
            self.assertTrue(result["audit_ok"])
            seal = json.loads((output / "seal.json").read_bytes())
            self.assertEqual(seal["report_sha256"], h.digest((output / "report.json").read_bytes()))
            self.assertEqual(seal["ledger_sha256"], h.digest((output / "ledger.json").read_bytes()))
            self.assertEqual(seal["producer_sha256"], h.digest(Path(report.__file__).read_bytes()))
            # Clean handoff is not closure; the unresolved finding remains visible.
            self.assertEqual(result["unresolved_ids"], ["commit:f1"])

    def test_incomplete_overlay_cannot_be_clean_even_with_clean_handoff(self):
        ledger = crosswalk({"commit:f1": row()})
        ledger["current_receipt_overlay"] = {"schema": "current-receipt-overlay/v1", "canonical_tracker": h.TRACKER,
            "report_status": "PENDING_UNAUDITED_MISSING_JOINS", "rows": [{"finding_id": "commit:f1", "source_owner_task_id": None}],
            "missing_joins": {"owner_task_ids": ["commit:f1"], "current_receipts": ["commit:f1"]}}
        with tempfile.TemporaryDirectory() as temp:
            result, output = self.publish(ledger, Path(temp))
            self.assertFalse(result["audit_ok"])
            self.assertFalse((output / "seal.json").exists())
            self.assertEqual(json.loads((output / "ledger.json").read_bytes()), ledger)

    def test_missing_overlay_row_or_input_identity_refuses_output(self):
        ledger = crosswalk({"commit:f1": row()})
        ledger["current_receipt_overlay"] = {"schema": "current-receipt-overlay/v1", "canonical_tracker": h.TRACKER,
                                           "rows": [], "missing_joins": {"owner_task_ids": [], "current_receipts": []}}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "input.json"
            raw = json.dumps(ledger).encode()
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                report.publish_report(path, directory / "output", NOW, h.digest(raw))
            self.assertFalse((directory / "output").exists())
            with self.assertRaises(ValueError):
                report.publish_report(path, directory / "output", NOW, "0" * 64)
            self.assertFalse((directory / "output").exists())

    def test_extra_unresolved_join_prevents_clean_seal(self):
        ledger = crosswalk({"commit:f1": row()})
        ledger["current_receipt_overlay"] = {"schema": "current-receipt-overlay/v1", "canonical_tracker": h.TRACKER,
            "report_status": "RECONCILED", "rows": [{"finding_id": "commit:f1", "source_owner_task_id": OWNER,
                                                     "source_owner_observed_status": "active"}],
            "missing_joins": {"owner_task_ids": [], "current_receipts": [], "exact_CI_JUnit_membership": "unresolved"}}
        with tempfile.TemporaryDirectory() as temp:
            result, output = self.publish(ledger, Path(temp))
            self.assertFalse(result["audit_ok"])
            self.assertFalse((output / "seal.json").exists())
            audit = json.loads((output / "audit.json").read_bytes())
            self.assertIn("ADDITIONAL_JOIN_MISSING", {i["code"] for i in audit["issues"]})

    def test_existing_output_is_preserved_and_not_resealed(self):
        ledger = crosswalk({"commit:f1": row()})
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _, output = self.publish(ledger, directory)
            before = (output / "seal.json").read_bytes()
            with self.assertRaises(FileExistsError):
                report.publish_report(directory / "input.json", output, NOW, h.digest((directory / "input.json").read_bytes()))
            self.assertEqual((output / "seal.json").read_bytes(), before)
