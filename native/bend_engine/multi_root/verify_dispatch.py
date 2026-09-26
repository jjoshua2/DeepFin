"""Opt-in actual native dispatch/control tests, not a performance experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import time
from typing import Any

from .verify_live import CONTROL, Client, assert_serial, serial, tree_snapshots

PREFIX = "info string live_dispatch fixed-package-v1 "
STEP = "info string live_dispatch_step "
BATCH = "info string cohort_batch "
POSITIONS = [
    "startpos",
    "startpos moves e2e4",
    "startpos moves d2d4",
    "startpos moves g1f3 g8f6 f3g1 f6g8",
    "startpos moves e2e4 a7a6 e4e5 d7d5",
    "startpos moves c2c4",
]


def observations(
    lines: list[str], caps: list[int] | None, batch: int = 4
) -> dict[str, Any]:
    headers = [x for x in lines if x.startswith(PREFIX)]
    if caps is None:
        if headers or any(x.startswith(STEP) for x in lines):
            raise ValueError("default path unexpectedly applied a policy")
    elif headers != [PREFIX + " ".join(map(str, [batch, *caps]))]:
        raise ValueError("missing or mismatched dispatch-policy acknowledgment")
    current = batch
    nsteps, max_candidates, sizes = 0, 0, []
    for line in lines:
        if line.startswith(STEP):
            n, limit, physical = map(int, line.removeprefix(STEP).split())
            if (
                caps is None
                or not 1 <= n <= 16
                or physical != batch
                or limit != caps[n - 1]
            ):
                raise ValueError("dispatch decision did not use configured table")
            nsteps += 1
            max_candidates = max(max_candidates, n)
            current = limit
        elif line.startswith(BATCH):
            seq, real, physical = map(int, line.removeprefix(BATCH).split())
            if seq != len(sizes) + 1 or physical != batch or not 1 <= real <= current:
                raise ValueError("physical dispatch ignored gather limit")
            sizes.append(real)
    return {
        "steps": nsteps,
        "max_candidates": max_candidates,
        "real_rows_per_call": sizes,
    }


def drive(
    gate: Path, caps: list[int] | None, *, mutate: bool = False
) -> tuple[dict[str, Any], list[str]]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("DEEPFIN_", "SERVICE_TEST_"))
    }
    if caps is not None:
        env.update(
            DEEPFIN_COHORT_DISPATCH=" ".join(map(str, [4, *caps])),
            DEEPFIN_COHORT_PROFILE_PACKAGE_SHA256="a" * 64,
        )
    c = Client(gate.resolve(), gate=True, simulations=8, environment=env)
    try:
        c.send("add " + POSITIONS[0])
        c.started()
        # Deterministic staged admissions: one lifecycle operation per held call.
        for i, position in enumerate(POSITIONS[1:], 2):
            c.send("add " + position)
            c.until(CONTROL + f"pending-install {i} {i}")
            c.release()
            c.until(CONTROL + f"admitted {i} {i}")
            c.started()
        c.send("isready")
        c.until("info string live_ready", 0.75)
        if mutate:
            c.send("cancel 2 2\ndeadline 3 3 0\nreplace 1 1 startpos moves b2b3")
            c.until(CONTROL + "pending-install 1 7", 0.75)
        assert c.proc.stdin
        c.proc.stdin.close()
        c.release()
        end = time.monotonic() + 30
        while c.proc.poll() is None:
            if time.monotonic() >= end:
                raise TimeoutError("profiled coordinator did not drain")
            if select.select([c.events], [], [], 0.02)[0]:
                value = os.read(c.events, 1)
                if value == b"S":
                    c.release()
                elif value:
                    raise ValueError("invalid callback notification")
        result = c.finish()
        result["dispatch"] = observations(c.lines, caps)
        return result, c.lines.copy()
    finally:
        c.close()


def verify(gate: Path, reference: Path) -> dict[str, Any]:
    controls = [serial(reference.resolve(), p, 8) for p in POSITIONS]
    replacement = serial(reference.resolve(), "startpos moves b2b3", 8)
    results = {}
    plans = {
        "greedy": None,
        "one": [1] * 16,
        "two": [1] + [2] * 15,
        "recorded": [1, 2, 3, 4, 3, 3, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4],
    }
    baseline = None
    for name, caps in plans.items():
        r, lines = drive(gate, caps)
        trees = tree_snapshots(lines)
        compared = 0
        for i, expected in enumerate(controls, 1):
            assert_serial(r["roots"][f"{i}:{i}"], trees[i, i], expected)
            compared += len(trees[i, i])
        if baseline is None:
            baseline = r["roots"]
        elif r["roots"] != baseline:
            raise ValueError("gather policy changed fixed-work results")
        if r["work"]["cancelled_rows"] or r["work"]["accepted_neural_rows"] != sum(
            c[0]["accepted_neural_rows"] for c in controls
        ):
            raise ValueError("fixed-work budget or acceptance mismatch")
        sizes = r["dispatch"]["real_rows_per_call"]
        if name == "one" and max(sizes) != 1:
            raise ValueError("singleton gather was not applied")
        if name == "two" and max(sizes) != 2:
            raise ValueError("two-row gather was not applied")
        if name in ("greedy", "recorded") and max(sizes) < 3:
            raise ValueError("shared batches not exercised")
        results[name] = {
            "work": r["work"],
            "dispatch": r["dispatch"],
            "complete_nodes_compared": compared,
        }
    r, lines = drive(gate, plans["two"], mutate=True)
    trees = tree_snapshots(lines)
    assert set(r["roots"]) == {"1:1", "1:7", "2:2", "3:3", "4:4", "5:5", "6:6"}
    for i in (4, 5, 6):
        assert_serial(r["roots"][f"{i}:{i}"], trees[i, i], controls[i - 1])
    assert_serial(r["roots"]["1:7"], trees[1, 7], replacement)
    assert r["roots"]["1:1"]["cancel_requested"]
    assert r["roots"]["2:2"]["cancel_requested"]
    assert r["roots"]["3:3"]["deadline_expired"]
    assert r["roots"]["1:7"]["deadline_offset_ms"] is None
    results["cancel_expire_replace"] = r["work"]
    return {
        "status": "passed",
        "scope": "actual Bend dispatch with held deterministic callback",
        "equal_work_tree_parity": True,
        "results": results,
        "neural_model_qualified": False,
        "speed_or_strength_qualified": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    r: dict[str, Any] = {"status": "failed"}
    try:
        r = verify(a.gate, a.reference)
    except Exception as error:
        r["error"] = str(error)
        raise
    finally:
        a.report.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
