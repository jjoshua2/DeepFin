"""#205: how much of the mate-bucket gap is AUDIT_REGRET_CAP_CP, not the net?

CPU-only re-analysis of already-banked dumps. No GPU, no net forward pass.

The claim under test (ledger `175598a36`): "mates are 8.1% of positions and carry
40% of the 22.93 cp top-1 gap between ours and BT4". `AUDIT_REGRET_CAP_CP = 1000`
clamps every per-move regret, and `mate_to_effective_cp` puts mates near +/-1e5,
so the HYPOTHESIS was that inside a mate position essentially every non-best move
saturates the cap, making the bucket a binary mate-finding indicator scaled by an
arbitrary constant.

RESULT (ledger `a50fb0c5b`) — the hypothesis is HALF wrong and the finding is
elsewhere. Section (1) refutes bimodality: 45.9% of listed moves in mate positions
land in [100, 900), and the bucket mean 163.06 is nowhere near 1000 x 0.4753.
Section (3) confirms the share IS cap-driven (27.8% at cap 250 -> 97.2% uncapped),
so quote the cap-free MISS RATE from (4), never the cp. Section (5) is the real
result: 60% of the gap lives in the 3676 NON-mate positions, where the cap can
bind on 0.8% and the capped and uncapped gaps agree to the cent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chess_anti_engine.eval.audit import load_audit_set

# Run from the repo root with PYTHONPATH=.
DUMPS = Path("scratchpad/argmax_20260813")
AUDIT = Path("data/audit_set_v1.jsonl")


def load_dump(name: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(DUMPS / name) as fh:
        for line in fh:
            r = json.loads(line)
            out[r["key"]] = r
    return out


def top1_regret(pos, move: str, cap: float) -> float:
    """Recompute the argmax move's regret under an arbitrary cap.

    Mirrors `move_regrets` exactly, including the worst-listed floor for moves
    the deep MultiPV never listed.
    """
    worst_listed = min(pos.move_cp.values())
    floor = pos.best_cp - worst_listed
    cp = pos.move_cp.get(move)
    raw = (pos.best_cp - cp) if cp is not None else floor
    return float(np.clip(raw, 0.0, cap))


def main() -> None:
    positions = {p.key: p for p in load_audit_set(AUDIT)}
    nets = {
        "ours": load_dump("full_ours-repeatfill.jsonl"),
        "BT4": load_dump("full_BT4.jsonl"),
        "C1-512-15": load_dump("full_C1-512-15.jsonl"),
    }
    keys = sorted(set.intersection(*(set(d) for d in nets.values())) & set(positions))
    print(f"positions joined on all three nets + audit set: {len(keys)}")

    # A mate position = the deep MultiPV listed at least one mate line.
    raw = {}
    with open(AUDIT) as fh:
        for line in fh:
            r = json.loads(line)
            raw[r["key"]] = r
    is_mate = {
        k: any(m.get("mate") for m in raw[k]["multipv"]) for k in keys
    }
    mate_keys = [k for k in keys if is_mate[k]]
    quiet_keys = [k for k in keys if not is_mate[k]]
    print(f"mate positions: {len(mate_keys)} ({100*len(mate_keys)/len(keys):.1f}%)"
          f"   non-mate: {len(quiet_keys)}")

    # ---- (1) STRUCTURE: is per-move regret inside a mate position bimodal? ----
    print("\n=== (1) per-LISTED-move regret distribution inside mate positions ===")
    bands = {"0 (best)": 0, "(0,100)": 0, "[100,900)": 0, "[900,1000)": 0, ">=1000 (capped)": 0}
    for k in mate_keys:
        p = positions[k]
        for cp in p.move_cp.values():
            r = p.best_cp - cp
            if r <= 0:
                bands["0 (best)"] += 1
            elif r < 100:
                bands["(0,100)"] += 1
            elif r < 900:
                bands["[100,900)"] += 1
            elif r < 1000:
                bands["[900,1000)"] += 1
            else:
                bands[">=1000 (capped)"] += 1
    tot = sum(bands.values())
    for b, n in bands.items():
        print(f"  {b:<18} {n:6d}  {100*n/tot:5.1f}%")
    print(f"  (total listed moves in mate positions: {tot})")

    same = bands["0 (best)"] + bands[">=1000 (capped)"]
    print(f"  ⇒ saturated-or-zero: {100*same/tot:.1f}% of listed moves")

    # ---- (2) miss rate vs cp: is mate-bucket cp just 1000 x miss rate? ----
    print("\n=== (2) mate bucket: miss rate vs measured mean top-1 regret ===")
    print(f"  {'net':<12} {'miss rate':>10} {'1000*miss':>10} {'measured':>10}")
    for name, d in nets.items():
        rs = np.array([top1_regret(positions[k], d[k]["best_move"], 1000.0) for k in mate_keys])
        miss = float(np.mean(rs > 0))
        print(f"  {name:<12} {miss:10.4f} {1000*miss:10.2f} {rs.mean():10.2f}")

    # ---- (3) CAP SENSITIVITY: how the headline gap moves with the constant ----
    print("\n=== (3) ours - BT4 top-1 gap (cp) as a function of AUDIT_REGRET_CAP_CP ===")
    print(f"  {'cap':>8} {'ours':>9} {'BT4':>9} {'gap':>9} {'mate share of gap':>19}")
    for cap in (250.0, 500.0, 1000.0, 2000.0, 5000.0, 1e9):
        row = {}
        for name in ("ours", "BT4"):
            d = nets[name]
            row[name] = {
                "all": np.mean([top1_regret(positions[k], d[k]["best_move"], cap) for k in keys]),
                "mate": np.mean([top1_regret(positions[k], d[k]["best_move"], cap) for k in mate_keys]),
            }
        gap = row["ours"]["all"] - row["BT4"]["all"]
        mate_contrib = (len(mate_keys) / len(keys)) * (row["ours"]["mate"] - row["BT4"]["mate"])
        share = 100 * mate_contrib / gap if gap else float("nan")
        label = "uncapped" if cap > 1e8 else f"{cap:.0f}"
        print(f"  {label:>8} {row['ours']['all']:9.2f} {row['BT4']['all']:9.2f} "
              f"{gap:9.2f} {share:18.1f}%")

    # ---- (4) the cap-free statement: mate-FINDING rates ----
    print("\n=== (4) cap-free readout — mate-position miss rate (binomial 95% CI) ===")
    n = len(mate_keys)
    for name, d in nets.items():
        misses = sum(1 for k in mate_keys if top1_regret(positions[k], d[k]["best_move"], 1000.0) > 0)
        p = misses / n
        se = (p * (1 - p) / n) ** 0.5
        print(f"  {name:<12} {misses:4d}/{n}  = {p:.4f}  [{p-1.96*se:.4f}, {p+1.96*se:.4f}]")

    # ---- (5) the non-mate bucket, which the cap barely touches ----
    print("\n=== (5) NON-mate positions: gap with and without the cap ===")
    for cap in (1000.0, 1e9):
        vals = {}
        for name in ("ours", "BT4", "C1-512-15"):
            d = nets[name]
            vals[name] = np.mean([top1_regret(positions[k], d[k]["best_move"], cap) for k in quiet_keys])
        label = "uncapped" if cap > 1e8 else f"cap {cap:.0f}"
        print(f"  {label:<10} ours {vals['ours']:7.2f}  BT4 {vals['BT4']:7.2f}  "
              f"C1-512-15 {vals['C1-512-15']:7.2f}  (ours-BT4 {vals['ours']-vals['BT4']:+.2f})")

    # how often does the cap actually bind outside mate positions?
    bind = 0
    for k in quiet_keys:
        p = positions[k]
        if max(p.best_cp - c for c in p.move_cp.values()) >= 1000.0:
            bind += 1
    print(f"  cap can bind on {bind}/{len(quiet_keys)} non-mate positions "
          f"({100*bind/len(quiet_keys):.1f}%)")


if __name__ == "__main__":
    main()
