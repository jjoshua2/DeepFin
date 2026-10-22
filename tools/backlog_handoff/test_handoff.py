"""Synthetic ledger fixtures; stdlib only, no project/model imports or launches."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import backlog_handoff as h

NOW = "2026-10-22T09:00:00Z"
OWNER = "01a13697-1896-7746-9f70-83f5e2d6a0db"
REVIEWER = "01a14615-00f8-7796-bcd8-a531f927e0fd"


def ev(name="fixture", kind="observation", outcome=None):
    return {"ref": "synthetic://" + name, "sha256": h.digest(name.encode()),
            "kind": kind, "outcome": outcome}


def owner(task=OWNER):
    return {"task_id": task, "status": "active", "evidence": ev("assignment-" + task)}


def row(state="ready"):
    return {"state": state, "updated_at": NOW, "owner": owner(),
            "next_action": {"description": "review bounded synthetic result", "owner": owner(REVIEWER)},
            "permission": {"state": "granted", "reason": "fixture only", "evidence": ev("scope")},
            "stop": False, "custody": "settled", "evidence": [ev()],
            "blockers": [], "depends_on": [], "credits": []}


def source():
    return {"schema": 1, "findings": [{"finding_id": "f1", "owner": None},
                                       {"finding_id": "f2", "owner": "science"}],
            "cohorts": [{"name": "retained", "finding_ids": ["f1", "f2"]}],
            "accepted_selected_cpu_cases": {"total": 127}, "original_failures": ["retained"]}


def cycle(items=None, name="cycle-1", hold=None):
    items = {"f1": row(), "f2": row()} if items is None else items
    return {"event_id": name, "at": NOW, "evidence": ev(name), "items": items,
            "selected": [k for k, r in items.items() if r["state"] == "ready"], "hold": hold}


def fixture():
    return h.append_cycle(source(), cycle())


def codes(ledger, **kwargs):
    return {i["code"] for i in h.audit(ledger, NOW, **kwargs)["issues"]}


class HandoffTests(unittest.TestCase):
    def test_existing_v22_crosswalk_schema_keeps_commit_prefix_and_source_records(self):
        ledger = {"schema": "backlog-assignment-crosswalk/v18", "records": [
            {"id": "commit:f1", "retained": {"owner": None}, "cloud_status": "new"},
            {"id": "commit:f2", "retained": {"owner": "science"}, "cloud_status": "new"}],
            "scope": "synthetic 201-shaped fixture, no actual historical totals asserted"}
        event = cycle()
        event["items"] = {"commit:" + k: v for k, v in event["items"].items()}
        event["selected"] = ["commit:" + k for k in event["selected"]]
        successor = h.append_cycle(ledger, event)
        self.assertEqual({k: v for k, v in successor.items() if k != "handoff"}, ledger)
        self.assertEqual(h.audit(successor, NOW)["candidate_ids"], ["commit:f1", "commit:f2"])
        self.assertEqual(codes(successor), set())
        successor["findings"] = [{"finding_id": "competing-id"}]
        self.assertIn("INVALID_LEDGER", codes(successor))

    def test_success_preserves_existing_ledger_and_counts(self):
        original = source()
        candidate = h.append_cycle(original, cycle())
        self.assertNotIn("handoff", original)
        self.assertEqual({k: v for k, v in candidate.items() if k != "handoff"}, original)
        self.assertEqual(codes(candidate), set())
        self.assertEqual(h.audit(candidate, NOW)["authority"], "NONE; external action gates remain mandatory")

    def test_terminal_success_failure_timeout_and_disconnect_need_continuation(self):
        for status in ("completed", "failed", "timeout", "disconnected"):
            with self.subTest(status=status):
                a = row("terminal")
                a["worker"] = {"task_id": OWNER, "state": status, "evidence": ev(status)}
                a["custody"] = "unknown" if status in {"timeout", "disconnected"} else "settled"
                event = cycle({"f1": a, "f2": row()})
                candidate = h.append_cycle(source(), event)
                self.assertIn("DROPPED_HANDOFF", codes(candidate))
                a["continuation"] = {"kind": "reconcile", "owner": owner(REVIEWER),
                                     "description": "reconcile saved result and custody",
                                     "worker_result": ev(status)["sha256"], "evidence": ev("continuation")}
                successor = h.append_cycle(candidate, cycle({"f1": a, "f2": row()}, "cycle-2"))
                self.assertNotIn("DROPPED_HANDOFF", codes(successor))

    def test_continuation_must_bind_exact_result(self):
        a = row("terminal")
        a["worker"] = {"task_id": OWNER, "state": "failed", "evidence": ev("new-failure")}
        a["continuation"] = {"kind": "review", "owner": owner(), "description": "review",
                             "worker_result": ev("old-result")["sha256"], "evidence": ev()}
        self.assertIn("DROPPED_HANDOFF", codes(h.append_cycle(source(), cycle({"f1": a}))))

    def test_restart_cannot_hide_unreconciled_terminal_result(self):
        a = row("terminal")
        a["worker"] = {"task_id": OWNER, "state": "timeout", "evidence": ev("timeout-old")}
        candidate = h.append_cycle(source(), cycle({"f1": a, "f2": row()}))
        candidate = h.append_cycle(candidate, cycle(name="restart"))
        self.assertIn("DROPPED_HANDOFF", codes(candidate))
        event = cycle(name="recover")
        event["items"]["f1"]["continuation"] = {"kind": "reconcile", "owner": owner(),
            "description": "read back saved timeout and reconcile custody", "worker_result": ev("timeout-old")["sha256"],
            "evidence": ev("actual-recovery")}
        self.assertNotIn("DROPPED_HANDOFF", codes(h.append_cycle(candidate, event)))

    def test_terminal_without_worker_result_is_not_clean(self):
        self.assertIn("UNKNOWN_WORKER", codes(h.append_cycle(source(), cycle({"f1": row("terminal")}))))

    def test_role_label_and_null_do_not_count_as_assignment(self):
        for value in (None, "science", {"task_id": "fast", "status": "active", "evidence": ev()},
                      {"task_id": OWNER, "status": "completed", "evidence": ev()}):
            with self.subTest(value=value):
                a = row()
                a["owner"] = value
                result = h.append_cycle(source(), cycle({"f1": a}))
                self.assertIn("UNOWNED", codes(result))

    def test_evidence_backed_terminal_disposition_needs_no_active_worker(self):
        a = row("done")
        a.pop("owner")
        a.pop("next_action")
        a["resolution"] = {**ev("not-applicable", "disposition"), "reason": "reviewed false positive"}
        result = h.append_cycle(source(), cycle({"f1": a, "f2": row()}))
        self.assertEqual(codes(result), set())
        a.pop("resolution")
        self.assertIn("UNSUPPORTED_TERMINAL", codes(h.append_cycle(source(), cycle({"f1": a}))))

    def test_unresolved_missing_ownership_next_action_and_evidence(self):
        a = row()
        a.pop("owner")
        a.pop("next_action")
        result = codes(h.append_cycle(source(), cycle({"f1": a})))
        self.assertTrue({"UNOWNED", "NO_NEXT_ACTION", "MISSING_HANDOFF"}.issubset(result))
        a["evidence"] = ["source review"]
        with self.assertRaises(ValueError):
            h.append_cycle(source(), cycle({"f1": a}))

    def test_unrelated_blocker_cannot_hide_independent_runnable_work(self):
        a = row("blocked")
        a["blockers"] = [{"reason": "denied external operation", "owner": owner(),
                          "resolution": "await exact scope", "evidence": ev("denial")}]
        event = cycle({"f1": a, "f2": row()}, hold={"reason": "f1 held", "blocked_by": ["f1"], "evidence": ev()})
        event["selected"] = []
        result = codes(h.append_cycle(source(), event))
        self.assertTrue({"HIDDEN_RUNNABLE", "UNSCHEDULED"}.issubset(result))
        event["selected"] = ["f2"]
        self.assertEqual(codes(h.append_cycle(source(), event)), set())

    def test_stale_ready_item_and_owned_blocker(self):
        event = cycle()
        event["items"]["f1"]["updated_at"] = "2026-10-21T09:00:00Z"
        event["items"]["f2"]["blockers"] = [{"reason": "unknown"}]
        result = codes(h.append_cycle(source(), event))
        self.assertTrue({"STALE", "UNOWNED_BLOCKER", "SELECTION_NOT_READY"}.issubset(result))

    def test_blocked_label_without_blocker_cannot_hide_ready_item(self):
        event = cycle({"f1": row("blocked"), "f2": row()})
        self.assertTrue({"UNJUSTIFIED_BLOCK", "UNSCHEDULED"}.issubset(codes(h.append_cycle(source(), event))))

    def test_no_progress_is_not_cleared_by_changed_prose_or_replayed_cycle(self):
        candidate = fixture()
        for number in (2, 3):
            event = cycle(name=f"cycle-{number}")
            event["items"]["f1"]["next_action"]["description"] += str(number)
            candidate = h.append_cycle(candidate, event)
        self.assertIn("NO_PROGRESS", codes(candidate))
        self.assertEqual(h.append_cycle(candidate, event), candidate)
        event["items"]["f1"]["evidence"].append(ev("new-result", "test", "passed"))
        event["event_id"] = "cycle-4"
        candidate = h.append_cycle(candidate, event)
        self.assertFalse(any(i["finding_id"] == "f1" and i["code"] == "NO_PROGRESS"
                             for i in h.audit(candidate, NOW)["issues"]))

    def test_duplicate_completion_credit_and_source_review_not_tests(self):
        event = cycle()
        credit = {"claim_id": "claim-1", "kind": "test", "case_id": "test_contract",
                  "source_sha256": ev("source")["sha256"], "evidence": ev("result", "test", "passed"),
                  "review": ev("independent-review")}
        event["items"]["f1"]["credits"] = [credit]
        candidate = h.append_cycle(source(), event)
        event["event_id"] = "cycle-2"
        event["items"]["f2"]["credits"] = [{**credit, "claim_id": "claim-2"}]
        self.assertIn("DUPLICATE_CREDIT", codes(h.append_cycle(candidate, event)))
        event["items"]["f2"]["credits"] = []
        event["items"]["f1"]["credits"][0]["evidence"] = ev("review-only", "source_review", "passed")
        self.assertIn("UNSUPPORTED_CREDIT", codes(h.append_cycle(source(), event)))

    def test_duplicate_credit_inside_snapshot_is_rejected(self):
        event = cycle()
        credit = {"claim_id": "finding-1", "kind": "finding", "evidence": ev("disposition"), "review": ev("review")}
        event["items"]["f1"]["credits"] = [credit, credit]
        self.assertIn("DUPLICATE_CREDIT", codes(h.append_cycle(source(), event)))

    def test_status_churn_and_fresh_zero_tallies_are_not_progress(self):
        candidate = fixture()
        for number, state in ((2, "open"), (3, "ready")):
            event = cycle(name=f"cycle-{number}")
            event["items"]["f1"]["state"] = state
            event["items"]["f1"]["evidence"].append(ev(f"zero-tally-{number}", "observation"))
            candidate = h.append_cycle(candidate, event)
        self.assertIn("NO_PROGRESS", codes(candidate))

    def test_reordered_material_evidence_is_not_progress(self):
        event = cycle()
        event["items"]["f1"]["evidence"] = [ev("result-a", "result"), ev("result-b", "source_review")]
        candidate = h.append_cycle(source(), event)
        for number in (2, 3):
            event["event_id"] = f"cycle-{number}"
            event["items"]["f1"]["evidence"].reverse()
            event["items"]["f1"]["evidence"][0]["ref"] += "/refreshed"
            candidate = h.append_cycle(candidate, event)
        self.assertIn("NO_PROGRESS", codes(candidate))

    def test_repeated_snapshot_is_not_duplicate_credit_but_rewritten_claim_is(self):
        event = cycle()
        credit = {"claim_id": "finding-1", "kind": "finding", "evidence": ev("disposition"), "review": ev("review")}
        event["items"]["f1"]["credits"] = [credit]
        candidate = h.append_cycle(source(), event)
        event["event_id"] = "cycle-2"
        self.assertNotIn("DUPLICATE_CREDIT", codes(h.append_cycle(candidate, event)))
        event["items"]["f1"]["credits"][0]["evidence"] = ev("different")
        self.assertIn("CREDIT_REWRITTEN", codes(h.append_cycle(candidate, event)))

    def test_denied_pending_uncertain_stop_and_timeout_never_become_dispatch(self):
        for state in ("denied", "pending", "uncertain", "granted"):
            a = row("terminal")
            a["permission"]["state"] = state
            a["stop"] = state == "granted"
            a["worker"] = {"task_id": OWNER, "state": "timeout", "evidence": ev("timeout")}
            a["custody"] = "unknown"
            a["continuation"] = {"kind": "dispatch", "owner": owner(), "description": "new work",
                                 "worker_result": ev("timeout")["sha256"], "evidence": ev("bad-dispatch")}
            result = h.audit(h.append_cycle(source(), cycle({"f1": a, "f2": row()})), NOW)
            self.assertIn("UNSAFE_CONTINUATION", {i["code"] for i in result["issues"]})
            self.assertNotIn("f1", result["candidate_ids"])

    def test_invalid_missing_state_and_unknown_schema_fail_closed(self):
        self.assertIn("INVALID_LEDGER", codes(source()))
        candidate = fixture()
        candidate["handoff"]["schema"] = 2
        self.assertIn("INVALID_LEDGER", codes(candidate))
        event = cycle()
        event["items"]["f1"]["state"] = "magic-done"
        with self.assertRaises(ValueError):
            h.append_cycle(source(), event)

    def test_missing_dependency_and_selection_refuse(self):
        event = cycle()
        event["items"]["f1"]["depends_on"] = ["f2"]
        self.assertIn("SELECTION_NOT_READY", codes(h.append_cycle(source(), event)))
        event["items"]["f1"]["depends_on"] = ["invented"]
        with self.assertRaises(ValueError):
            h.append_cycle(source(), event)

    def test_persist_reopen_recovery_and_optimistic_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inventory.json"
            raw = (json.dumps(source()) + "\n").encode()
            path.write_bytes(raw)
            first = h.persist_cycle(path, cycle(), h.digest(raw))
            reopened = json.loads(path.read_bytes())  # no retained in-memory state
            self.assertEqual(codes(reopened), set())
            self.assertEqual(h.persist_cycle(path, cycle(), first), first)
            with self.assertRaises(ValueError):
                h.persist_cycle(path, cycle(name="cycle-2"), h.digest(raw))
            self.assertEqual(h.digest(path.read_bytes()), first)
            second = h.persist_cycle(path, cycle(name="cycle-2"), first)
            self.assertNotEqual(first, second)
            self.assertEqual(len(json.loads(path.read_bytes())["handoff"]["cycles"]), 2)

    def test_interrupted_write_retains_old_ledger_and_lock_not_stolen(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inventory.json"
            raw = json.dumps(source()).encode()
            path.write_bytes(raw)
            with patch.object(h.os, "replace", side_effect=OSError("synthetic interrupted replace")), self.assertRaises(OSError):
                h.persist_cycle(path, cycle(), h.digest(raw))
            self.assertEqual(path.read_bytes(), raw)
            lock = path.with_name(path.name + ".handoff.lock")
            lock.write_text("unreconciled owner")
            with self.assertRaises(FileExistsError):
                h.persist_cycle(path, cycle(), h.digest(raw))
            self.assertEqual(lock.read_text(), "unreconciled owner")

    def test_report_hook_stops_emission_and_cli_reports_failure(self):
        emitted = []
        def existing_report_consumer(ledger):
            audit = h.report_hook(ledger, NOW)
            emitted.append({"findings": ledger["findings"], "audit": audit})
        with self.assertRaises(ValueError):
            existing_report_consumer(source())
        self.assertEqual(emitted, [])
        existing_report_consumer(fixture())
        self.assertTrue(emitted[0]["audit"]["ok"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inventory.json"
            path.write_text(json.dumps(source()))
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                status = h.main([str(path), "--now", NOW])
            self.assertEqual(status, 1)
            self.assertFalse(json.loads(capture.getvalue())["ok"])

    def test_malformed_event_cli_is_structured_failure_without_write(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inventory.json"
            raw = json.dumps(source()).encode()
            path.write_bytes(raw)
            event = Path(temp) / "event.json"
            event.write_text(json.dumps({"event_id": "missing-fields"}))
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                status = h.main([str(path), "--event", str(event), "--expected-sha256", h.digest(raw), "--now", NOW])
            self.assertEqual(status, 2)
            self.assertFalse(json.loads(capture.getvalue())["ok"])
            self.assertEqual(path.read_bytes(), raw)

    def test_reused_cycle_id_different_content_and_future_time_refuse(self):
        event = cycle()
        candidate = h.append_cycle(source(), event)
        event["evidence"] = ev("rewritten")
        with self.assertRaises(ValueError):
            h.append_cycle(candidate, event)
        candidate["handoff"]["cycles"][0]["at"] = "2026-10-23T09:00:00Z"
        self.assertIn("INVALID_LEDGER", codes(candidate))


if __name__ == "__main__":
    unittest.main()
