#!/usr/bin/env python3
"""Generate a local report from the existing crosswalk, preserving pending work.

Integrated successor of coordinator's current_backlog_report.py proposal. This
finite consumer gates its own clean seal only. Manual parent chat and hosted
Review Atlas reporting remain outside its mechanical enforcement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

if __package__:
    from . import backlog_handoff as handoff
else:
    # Direct script execution puts scripts/ on sys.path; package imports use the branch above.
    import backlog_handoff as handoff  # pyright: ignore[reportImplicitRelativeImport]


def require_handoff_audit(ledger: dict, now: str) -> dict:
    """Invoke the reviewed hook even for rejected reports; retain its findings."""
    try:
        result = handoff.report_hook(ledger, now)
    except ValueError as exc:
        result = json.loads(str(exc))
        if not isinstance(result, dict) or result.get("ok") is not False or not result.get("issues"):
            raise ValueError("invalid audit refusal") from exc
    overlay = ledger.get("current_receipt_overlay")
    if overlay is None:
        return result
    ids = handoff.finding_ids(ledger)

    def issue(code: str, item: str, detail: str) -> None:
        result["issues"].append({"code": code, "finding_id": item, "detail": detail})

    if overlay.get("schema") != "current-receipt-overlay/v1" or overlay.get("canonical_tracker") != handoff.TRACKER:
        raise ValueError("unsupported receipt overlay schema/tracker")
    rows = overlay["rows"]
    row_ids = [row["finding_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != ids:
        raise ValueError("receipt overlay must preserve the exact existing ID set")
    missing = overlay["missing_joins"]
    for key, code in (("owner_task_ids", "SOURCE_OWNER_JOIN_MISSING"), ("current_receipts", "CURRENT_RECEIPT_JOIN_MISSING")):
        members = missing[key]
        if not isinstance(members, list) or not set(members).issubset(ids):
            raise ValueError("missing-join IDs must belong to existing crosswalk")
        for item in members:
            issue(code, item, key + " remains unresolved; reconciliation ownership is not implementation ownership")
    for key, value in missing.items():
        if key not in {"owner_task_ids", "current_receipts"} and value:
            issue("ADDITIONAL_JOIN_MISSING", "", str(key) + " remains explicitly unresolved")
    for row in rows:
        task = row.get("source_owner_task_id")
        if task is None or not handoff.TASK.fullmatch(str(task)):
            issue("SOURCE_OWNER_JOIN_MISSING", row["finding_id"], "no exact source-owner task UUID")
        cycles = ledger.get("handoff", {}).get("cycles", [])
        item = cycles[-1]["items"].get(row["finding_id"], {}) if cycles else {}
        item_owner = item.get("owner")
        assigned_task = item_owner.get("task_id") if isinstance(item_owner, dict) else None
        if item.get("state") != "done" and (row.get("source_owner_observed_status") != "active"
                or assigned_task != task):
            issue("SOURCE_OWNER_NOT_ACTIVE", row["finding_id"], "metadata reconciliation owner cannot substitute for active source assignment")
    if overlay.get("report_status") != "RECONCILED":
        issue("OVERLAY_PENDING", "", "current overlay remains pending; do not infer closure from metadata joins")
    if overlay.get("unmapped_publications"):
        issue("UNMAPPED_PUBLICATION", "", "preserved unmapped publications require explicit reconciliation")
    result["ok"] = not result["issues"]
    return result


def render_report(ledger: dict, now: str) -> tuple[dict, dict]:
    """Audit before computing counts. Include every source ID on both branches."""
    if ledger.get("schema") != "backlog-assignment-crosswalk/v18":
        raise ValueError("expected existing canonical crosswalk schema")
    audit = require_handoff_audit(ledger, now)
    ids = [record["id"] for record in ledger["records"]]
    cycles = ledger.get("handoff", {}).get("cycles", [])
    items = cycles[-1]["items"] if cycles else {}
    unresolved = [item for item in ids if items.get(item, {}).get("state") != "done"
                  or any(i["finding_id"] in {item, ""} for i in audit["issues"])]
    report = {
        "schema": "backlog-report/v1", "canonical_tracker": handoff.TRACKER,
        "observed_at": now, "status": "CLEAN_DECLARED_HANDOFF" if audit["ok"] else "PENDING_AUDIT_FAILURE",
        "finding_ids": ids, "covered_source_denominator": len(ids), "unresolved_ids": unresolved,
        "declared_items": items, "current_receipt_overlay": ledger.get("current_receipt_overlay"),
        "handoff_audit": audit,
        "scope": "Existing source IDs only; no inferred finding closure, test credit, Atlas equivalence or execution authority",
        "enforcement_limit": "Only this local report consumer's clean seal is gated. Manual parent chat and hosted Atlas remain outside enforcement.",
    }
    return report, audit


def encoded(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def publish_report(ledger_path: Path, output_dir: Path, now: str, expected_sha256: str) -> dict:
    raw = handoff.read_bounded(ledger_path)
    if handoff.digest(raw) != expected_sha256:
        raise ValueError("input identity changed; capture and review the successor before reporting")
    ledger = json.loads(raw)
    report, audit = render_report(ledger, now)
    report_raw, audit_raw = encoded(report), encoded(audit)
    # An invalid audit produces a visibly pending report, never a clean seal.
    output_dir.mkdir(parents=True, exist_ok=False)
    write_new(output_dir / "ledger.json", raw)
    write_new(output_dir / "audit.json", audit_raw)
    report_name = "report.json" if audit["ok"] else "pending-report.json"
    write_new(output_dir / report_name, report_raw)
    result = {
        "status": report["status"], "audit_ok": audit["ok"], "clean_seal_created": False,
        "ledger_sha256": handoff.digest(raw), "report_sha256": handoff.digest(report_raw),
        "audit_sha256": handoff.digest(audit_raw), "finding_ids": report["finding_ids"],
        "unresolved_ids": report["unresolved_ids"], "report": report_name,
        "producer_sha256": handoff.digest(handoff.read_bounded(Path(__file__))),
        "audit_source_sha256": handoff.digest(handoff.read_bounded(Path(handoff.__file__))),
    }
    if audit["ok"]:
        seal = {**result, "clean_seal_created": True, "audit_gate": "require_handoff_audit and receipt-join checks passed"}
        write_new(output_dir / "seal.json", encoded(seal))
        result["clean_seal_created"] = True
    write_new(output_dir / "result.json", encoded(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--now", default=datetime.now(timezone.utc).isoformat())
    args = parser.parse_args(argv)
    try:
        result = publish_report(args.ledger, args.output, args.now, args.expected_sha256)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps({"status": "REPORT_FAILED", "error": str(exc)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["audit_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
