#!/usr/bin/env python3
"""Finite ledger audit. Records observations; never dispatches or grants authority.

The optional handoff member belongs to the existing finding ledger. Original
finding records and evidence remain intact. Call report_hook before emitting a
backlog report; a clean audit still requires the executor's normal action gates.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

TRACKER = "https://deepfin-review-atlas.new-gecko-0210.chatgpt.site/#queue"
TASK = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_BYTES = 4 * 1024 * 1024


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_bounded(path: Path) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("ledger/event exceeds size bound")
    return raw


def stamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamps require a timezone")
    return result.astimezone(timezone.utc)


def evidence(value: object) -> bool:
    return (isinstance(value, dict) and isinstance(value.get("ref"), str)
            and bool(value["ref"].strip()) and isinstance(value.get("sha256"), str)
            and SHA.fullmatch(value["sha256"]) is not None)


def assigned(value: object) -> bool:
    return (isinstance(value, dict) and isinstance(value.get("task_id"), str)
            and TASK.fullmatch(value["task_id"]) is not None
            and value.get("status") == "active" and evidence(value.get("evidence")))


def progress_signature(row: dict) -> str:
    """Artifact identities, independent of ordering, prose or refreshed polls."""
    material = {e["sha256"] for e in row["evidence"] if e.get("kind") in
                {"implementation", "result", "test", "source_review", "disposition", "settlement"}}
    resolution = row.get("resolution", {})
    credits = {json.dumps({"kind": c["kind"], "source": c.get("source_sha256"), "case": c.get("case_id"),
                           "evidence": c["evidence"]["sha256"], "review": c.get("review", {}).get("sha256")}, sort_keys=True)
               for c in row.get("credits", [])}
    return json.dumps({"material": sorted(material), "resolution": resolution.get("sha256"),
                       "credits": sorted(credits)}, sort_keys=True)


def finding_ids(ledger: dict) -> set[str]:
    """Consume existing IDs, never create IDs from case totals or role labels."""
    if ledger.get("schema") == "backlog-assignment-crosswalk/v18":
        if "findings" in ledger:
            raise ValueError("ambiguous crosswalk and finding inventory")
        rows, key = ledger["records"], "id"
    else:
        rows, key = ledger["findings"], "finding_id"
    if not isinstance(rows, list) or not rows:
        raise ValueError("existing findings must be a nonempty list")
    ids = [row[key] for row in rows]
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("finding_id must be a nonempty string")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source finding_id")
    return set(ids)


def validate_event(event: dict, ids: set[str]) -> None:
    if not isinstance(event, dict) or not isinstance(event.get("event_id"), str) or not event["event_id"]:
        raise ValueError("event_id required")
    stamp(event["at"])
    if not evidence(event.get("evidence")):
        raise ValueError("cycle observation evidence required")
    items = event["items"]
    if not isinstance(items, dict) or not set(items).issubset(ids):
        raise ValueError("items must refer only to existing finding IDs")
    for row in items.values():
        if not isinstance(row, dict):
            raise ValueError("handoff item must be an object")
        if row.get("state") not in {"open", "ready", "running", "blocked", "terminal", "done"}:
            raise ValueError("unknown item state")
        stamp(row["updated_at"])
        if not isinstance(row.get("evidence"), list) or not all(evidence(e) for e in row["evidence"]):
            raise ValueError("item evidence must be a list of pinned references")
        if not isinstance(row.get("depends_on", []), list) or not set(row.get("depends_on", [])).issubset(ids):
            raise ValueError("unknown dependency")
        if not isinstance(row.get("blockers", []), list) or not isinstance(row.get("credits", []), list):
            raise ValueError("blockers and credits must be lists")
    if not isinstance(event.get("selected"), list) or not set(event["selected"]).issubset(ids):
        raise ValueError("selected must contain existing finding IDs")
    if len(event["selected"]) != len(set(event["selected"])):
        raise ValueError("duplicate selected finding")
    if event.get("hold") is not None:
        hold = event["hold"]
        if not isinstance(hold, dict) or not evidence(hold.get("evidence")) or not hold.get("reason"):
            raise ValueError("hold requires reason and evidence")
        if not isinstance(hold.get("blocked_by"), list) or not set(hold["blocked_by"]).issubset(ids):
            raise ValueError("hold must name affected existing findings")


def append_cycle(ledger: dict, event: dict) -> dict:
    """Return successor; retain all source fields and immutable complete snapshots."""
    result = copy.deepcopy(ledger)
    ids = finding_ids(result)
    validate_event(event, ids)
    handoff = result.setdefault("handoff", {"schema": 1, "tracker": TRACKER, "cycles": []})
    if handoff.get("schema") != 1 or handoff.get("tracker") != TRACKER:
        raise ValueError("unsupported handoff schema/tracker")
    cycles = handoff["cycles"]
    for old in cycles:
        validate_event(old, ids)
        if old["event_id"] == event["event_id"]:
            if old == event:
                return result  # An exact replay has no progress or credit effect.
            raise ValueError("event_id reused with different content")
    if cycles and stamp(event["at"]) < stamp(cycles[-1]["at"]):
        raise ValueError("cycle time moved backwards")
    cycles.append(copy.deepcopy(event))
    return result


def audit(ledger: dict, now: str, stale_seconds: int = 3600, no_progress_cycles: int = 3) -> dict:
    """Audit declared evidence and assignments, not external truth or permission."""
    issues = []
    candidates = []

    def issue(code: str, item: str, detail: str) -> None:
        issues.append({"code": code, "finding_id": item, "detail": detail})

    try:
        current_time = stamp(now)
        if stale_seconds <= 0 or no_progress_cycles < 2:
            raise ValueError("positive stale limit and >=2 no-progress cycles required")
        ids = finding_ids(ledger)
        handoff = ledger["handoff"]
        if handoff["schema"] != 1 or handoff["tracker"] != TRACKER:
            raise ValueError("unsupported handoff schema/tracker")
        cycles = handoff["cycles"]
        if not isinstance(cycles, list) or not cycles:
            raise ValueError("no handoff cycles")
        event_ids = set()
        prior_time = None
        claims = {}
        units = {}
        pending_results = {}
        for event in cycles:
            validate_event(event, ids)
            event_time = stamp(event["at"])
            if event["event_id"] in event_ids or (prior_time is not None and event_time < prior_time):
                raise ValueError("duplicate cycle or backwards time")
            if event_time > current_time:
                raise ValueError("future observation")
            event_ids.add(event["event_id"])
            prior_time = event_time
            event_claims = set()
            event_units = set()
            for item, row in event["items"].items():
                if stamp(row["updated_at"]) > event_time:
                    raise ValueError("item update newer than observation")
                worker = row.get("worker")
                if row["state"] == "terminal" and worker is None:
                    issue("UNKNOWN_WORKER", item, "terminal item requires the observed worker result")
                if worker is not None:
                    if (not isinstance(worker, dict) or not TASK.fullmatch(str(worker.get("task_id", "")))
                            or worker.get("state") not in {"running", "completed", "failed", "timeout", "disconnected"}
                            or not evidence(worker.get("evidence"))):
                        issue("UNKNOWN_WORKER", item, "worker state/identity/result evidence invalid")
                    elif worker["state"] != "running":
                        pending_results.setdefault((item, worker["evidence"]["sha256"]), worker)
                    elif row["state"] != "running":
                        issue("WORKER_STATE_MISMATCH", item, "running worker cannot be selected again")
                continuation = row.get("continuation")
                if continuation is not None:
                    if (not isinstance(continuation, dict) or continuation.get("kind") not in {"review", "dispatch", "reconcile", "blocked"}
                            or not assigned(continuation.get("owner")) or not evidence(continuation.get("evidence"))
                            or not continuation.get("description") or not isinstance(continuation.get("worker_result"), str)):
                        issue("INVALID_CONTINUATION", item, "continuation requires active recipient and exact result binding")
                    else:
                        key = (item, continuation["worker_result"])
                        if key not in pending_results:
                            issue("INVALID_CONTINUATION", item, "continuation has no observed terminal result")
                        elif continuation["kind"] == "dispatch" and (row.get("permission", {}).get("state") != "granted" or row.get("stop") is not False or row.get("custody") != "settled"):
                            issue("UNSAFE_CONTINUATION", item, "denial, STOP or uncertain custody cannot be bypassed")
                        else:
                            pending_results[key] = None
                resolution = row.get("resolution")
                if row["state"] == "done" and evidence(resolution) and resolution.get("reason"):
                    if row.get("custody") != "settled":
                        issue("UNSETTLED_TERMINAL", item, "done cannot settle uncertain custody")
                    elif isinstance(resolution.get("worker_result"), str):
                        key = (item, resolution["worker_result"])
                        if key in pending_results:
                            pending_results[key] = None
                for credit in row.get("credits", []):
                    if not isinstance(credit, dict) or not credit.get("claim_id") or not evidence(credit.get("evidence")):
                        raise ValueError("credit requires claim_id and pinned evidence")
                    claim = credit["claim_id"]
                    if credit["kind"] == "test":
                        if not SHA.fullmatch(str(credit.get("source_sha256", ""))) or not credit.get("case_id"):
                            raise ValueError("test credit requires source_sha256 and stable case_id")
                        unit = ("test", credit["source_sha256"], credit["case_id"])
                    else:
                        unit = (credit["kind"], item)
                    if claim in event_claims or unit in event_units:
                        issue("DUPLICATE_CREDIT", item, "duplicate claim/unit inside one observed snapshot")
                    event_claims.add(claim)
                    event_units.add(unit)
                    identity = (item, credit)
                    if claim in claims and claims[claim] != identity:
                        issue("CREDIT_REWRITTEN", item, str(claim))
                    claims[claim] = identity
                    if unit in units and units[unit] != claim:
                        issue("DUPLICATE_CREDIT", item, str(unit))
                    units[unit] = claim
                    ev = credit["evidence"]
                    if (credit["kind"] not in {"test", "finding"}
                            or (credit["kind"] == "test" and (ev.get("kind") != "test" or ev.get("outcome") != "passed"))
                            or not evidence(credit.get("review"))):
                        issue("UNSUPPORTED_CREDIT", item, "source review is not test evidence; acceptance must be pinned")
        for (item, result_hash), worker in pending_results.items():
            if worker is not None:
                issue("DROPPED_HANDOFF", item, "unreconciled terminal result " + result_hash)
        latest = cycles[-1]
        items = latest["items"]
        for item in sorted(ids - set(items)):
            issue("MISSING_HANDOFF", item, "existing finding has no persistent owner/next action")
        for item, row in items.items():
            done = row["state"] == "done"
            if done:
                if not evidence(row.get("resolution")) or not row["resolution"].get("reason"):
                    issue("UNSUPPORTED_TERMINAL", item, "terminal disposition needs evidence and reason")
            else:
                if not assigned(row.get("owner")):
                    issue("UNOWNED", item, "requires evidenced active task UUID; role label is not an assignment")
                action = row.get("next_action", {})
                if not isinstance(action, dict) or not action.get("description") or not assigned(action.get("owner")):
                    issue("NO_NEXT_ACTION", item, "requires concrete action and active recipient")
                if (current_time - stamp(row["updated_at"])).total_seconds() > stale_seconds:
                    issue("STALE", item, "owner must reconcile stale evidence and next action")
            blockers = row.get("blockers", [])
            for blocker in blockers:
                if (not isinstance(blocker, dict) or not blocker.get("reason") or not blocker.get("resolution")
                        or not assigned(blocker.get("owner")) or not evidence(blocker.get("evidence"))):
                    issue("UNOWNED_BLOCKER", item, "exact blocker, recipient, evidence and resolving action required")
            permission = row.get("permission", {})
            if (not isinstance(permission, dict) or permission.get("state") not in {"granted", "pending", "denied", "uncertain"}
                    or not permission.get("reason") or not evidence(permission.get("evidence"))):
                issue("UNKNOWN_PERMISSION", item, "record approval/denial/pending/uncertain explicitly")
                permission = {}
            if type(row.get("stop")) is not bool or row.get("custody") not in {"settled", "unknown", "held"}:
                issue("UNKNOWN_CONTROL", item, "explicit STOP and custody observation required")
            deps = row.get("depends_on", [])
            if item in deps:
                issue("DEPENDENCY_CYCLE", item, "item depends on itself")
            ready = (row["state"] in {"open", "ready", "blocked"} and not blockers
                     and all(items.get(dep, {}).get("state") == "done" for dep in deps)
                     and permission.get("state") == "granted" and row.get("stop") is False
                     and row.get("custody") == "settled" and assigned(row.get("owner")))
            if ready:
                candidates.append(item)
                if row["state"] == "blocked":
                    issue("UNJUSTIFIED_BLOCK", item, "blocked label has no declared blocking prerequisite")
            if not done and len(cycles) >= no_progress_cycles:
                recent = [c["items"].get(item) for c in cycles[-no_progress_cycles:]]
                # Status churn and fresh zero-tally observations are not progress.
                signatures = [progress_signature(r) for r in recent if r is not None]
                if len(signatures) == no_progress_cycles and len(set(signatures)) == 1:
                    issue("NO_PROGRESS", item, "repeated cycles without new result/evidence; change bounded action or record exact hold")
        selected = set(latest["selected"])
        for item in selected - set(candidates):
            issue("SELECTION_NOT_READY", item, "selection has unmet prerequisites or controls")
        hold = latest.get("hold")
        if hold:
            for item in sorted(set(candidates) - selected - set(hold["blocked_by"])):
                issue("HIDDEN_RUNNABLE", item, "unrelated blocker cannot hide this independent candidate")
        for item in sorted(set(candidates) - selected):
            row = items[item]
            if not row.get("defer_reason") or not evidence(row.get("defer_evidence")):
                issue("UNSCHEDULED", item, "candidate needs an evidenced selection or specific deferral")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        issue("INVALID_LEDGER", "", str(exc))
    return {"schema": 1, "tracker": TRACKER, "ok": not issues, "issues": issues,
            "candidate_ids": sorted(candidates), "authority": "NONE; external action gates remain mandatory"}


def report_hook(ledger: dict, now: str) -> dict:
    """Existing report producers call this BEFORE serialization/sealing/counts."""
    result = audit(ledger, now)
    if not result["ok"]:
        raise ValueError(json.dumps(result, sort_keys=True))
    return result


def persist_cycle(path: Path, event: dict, expected_sha256: str) -> str:
    """Cooperating writers use exclusive lock + CAS + atomic replacement.

    Never steal a lock after timeout/restart. Reconcile its owner first. This is
    not protection against another writer ignoring the lock or an unsafe FS.
    """
    lock = path.with_name(path.name + ".handoff.lock")
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp = None
    try:
        raw = read_bounded(path)
        if digest(raw) != expected_sha256:
            raise ValueError("ledger changed; reread before update")
        successor = append_cycle(json.loads(raw), event)
        encoded = (json.dumps(successor, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        if len(encoded) > MAX_BYTES:
            raise ValueError("successor exceeds size bound")
        if successor == json.loads(raw):
            return digest(raw)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        temp = None
        return digest(encoded)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
        lock.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--now", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--event", type=Path, help="append observed cycle, does not execute it")
    parser.add_argument("--expected-sha256", help="required CAS identity for --event")
    args = parser.parse_args(argv)
    try:
        if args.event:
            if not args.expected_sha256:
                parser.error("--event requires --expected-sha256")
            persist_cycle(args.ledger, json.loads(read_bounded(args.event)), args.expected_sha256)
        raw = read_bounded(args.ledger)
        result = audit(json.loads(raw), args.now)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
