"""Paired CI for the two promoted target-only knobs, on the STORED TARGET row.

Pairs on the audit-set FEN and bootstraps the per-position delta, because the two
arms score the SAME positions with the SAME net and seed -- an unpaired CI over
4000 noisy per-position regrets would be far wider than the contrast deserves.

Reports E[regret] AND top-1 AND entropy together, per the standing method rule:
E[regret] rewards sharpness, so a knob that only concentrates mass lowers it with
no ranking improvement. The E-top1 GAP says which regime the arms are in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROW = "train"  # row (d), the production training target
HERE = Path(__file__).parent


def load(tag: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(HERE / f"{tag}.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["key"]] = r["cand"][ROW]
    return out


def boot(d: np.ndarray, n: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(off_tag: str, on_tag: str, label: str) -> None:
    off, on = load(off_tag), load(on_tag)
    keys = sorted(set(off) & set(on))
    print(f"\n=== {label}   ({off_tag} -> {on_tag})   paired n={len(keys)} ===")
    if len(keys) != len(off) or len(keys) != len(on):
        print(f"  !! key mismatch: off={len(off)} on={len(on)} shared={len(keys)}")

    for field in ("exp", "top1", "entropy"):
        a = np.array([off[k][field] for k in keys], dtype=float)
        b = np.array([on[k][field] for k in keys], dtype=float)
        d = b - a
        lo, hi = boot(d)
        tied = float((d == 0).mean()) * 100.0
        sig = "" if (lo <= 0.0 <= hi) else "  SIGNIFICANT"
        # positive delta = MORE regret = WORSE (for exp/top1)
        print(
            f"  {field:8} {a.mean():8.3f} -> {b.mean():8.3f}   "
            f"delta {d.mean():+7.3f}  [{lo:+7.3f}, {hi:+7.3f}]  "
            f"tied {tied:5.1f}%{sig}"
        )

    for tag, src in (("OFF", off), ("ON ", on)):
        e = np.mean([src[k]["exp"] for k in keys])
        t = np.mean([src[k]["top1"] for k in keys])
        acc = np.mean([bool(src[k]["top1_agree"]) for k in keys]) * 100.0
        print(f"  {tag}: E-top1 gap {e - t:6.3f}   argmax==deep-SF-best {acc:5.2f}%")

    a_acc = np.array([bool(off[k]["top1_agree"]) for k in keys], dtype=float)
    b_acc = np.array([bool(on[k]["top1_agree"]) for k in keys], dtype=float)
    lo, hi = boot(b_acc - a_acc)
    print(
        f"  accuracy delta {100 * (b_acc - a_acc).mean():+.3f} pp "
        f"[{100 * lo:+.3f}, {100 * hi:+.3f}]"
    )


if __name__ == "__main__":
    have = {p.stem for p in HERE.glob("*.jsonl")}
    todo = [
        ("B_noknob", "B_prod", "checkpoint B (production ckpt), at production policy_temp 1.5"),
        ("A_noknob", "A_prod", "checkpoint A (anchor), at production policy_temp 1.5"),
        ("B_instr", "B_noknob", "checkpoint B: policy_temp 1.0 -> 1.5 ALONE (third drifted knob)"),
        ("A_instr", "A_noknob", "checkpoint A: policy_temp 1.0 -> 1.5 ALONE (third drifted knob)"),
        ("B_instr", "B_prod", "checkpoint B: what the INSTRUMENT reported vs production reality"),
        ("A_instr", "A_prod", "checkpoint A: what the INSTRUMENT reported vs production reality"),
    ]
    ran = 0
    for off, on, label in todo:
        if off in have and on in have:
            compare(off, on, label)
            ran += 1
        else:
            missing = [t for t in (off, on) if t not in have]
            print(f"\n=== {label}: SKIPPED, missing {missing} ===")
    if ran == 0:
        sys.exit("no complete pairs yet")
