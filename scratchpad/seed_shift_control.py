#!/usr/bin/env python3
"""Is the 08-11 mass retirement a GLOBAL pessimism shift, or real seed-specific learning?

Task #184.  The retire step fires on an ABSOLUTE bar, `net_q = W-L <= -0.4`.  Across the
resume the blind-spot seeds' median net_q fell -0.19 and 200 crossed the bar at once.
I called that a calibration artifact.  The objection -- and it is a good one -- is that
the seeds are OBJECTIVELY LOST positions, so a net that reads them as lost is RIGHT, and
a different model legitimately has different blind spots.

My first control was the median of the fixed 800 cumulative seeds.  That control is
WORTHLESS for this question: all 800 are blind-spot seeds, i.e. all lost.  Centring by it
controls for pessimism relative to other lost positions, not for GLOBAL pessimism.

The discriminator is whether the net DISCRIMINATES.  Score two populations with the SAME
two checkpoints that bracket the transition:

  SEEDS   -- the live blind-spot list (objectively lost)
  CONTROL -- audit-set FENs (general positions, NOT selected for being lost)

and compare the PAIRED shift within each.

  * both shift down by ~the same amount  => GLOBAL pessimism.  The bar is uncalibrated
    and the retirement is an artifact.
  * control flat, seeds shift down       => the net genuinely learned the seeds.
    Retirement is CORRECT and task #184 is a non-issue.

Comparing the shift WITHIN each population (never the levels ACROSS them) is deliberate:
the seeds carry real history planes and the control FENs do not, and that encoding
difference would otherwise masquerade as the effect.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def load_control_fens(dump: str, n: int) -> list[str]:
    """General-position FENs from a value_regret per-position dump."""
    out: list[str] = []
    with open(dump, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fen = json.loads(line).get("fen")
            if fen:
                out.append(str(fen))
            if len(out) >= n:
                break
    return out


def load_seed_lines(path: str, n: int) -> list[str]:
    out: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            body = line.partition("#")[0].strip()
            if body:
                out.append(body)
            if len(out) >= n:
                break
    return out


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(mean shift, median shift, 95% half-width) for b - a, paired."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    return float(d.mean()), float(np.median(d)), float(1.96 * d.std(ddof=1) / np.sqrt(d.size))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="checkpoint BEFORE the transition")
    ap.add_argument("--post", required=True, help="checkpoint AFTER the transition")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--control-dump", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.13)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    from scripts.blindspot_resolution import score_seeds

    seeds = load_seed_lines(args.seeds, args.n)
    control = load_control_fens(args.control_dump, args.n)
    print(f"seeds={len(seeds)}  control={len(control)}")

    res: dict[str, dict[str, np.ndarray]] = {}
    for label, lines in (("SEEDS", seeds), ("CONTROL", control)):
        res[label] = {}
        for arm, ck in (("pre", args.pre), ("post", args.post)):
            q = score_seeds(ck, lines, device=args.device,
                            gpu_mem_fraction=args.gpu_mem_fraction,
                            batch_size=args.batch_size)
            res[label][arm] = np.asarray(q, dtype=np.float64)
            print(f"  {label:8s} {arm:4s} n={q.size:4d} "
                  f"mean={q.mean():+.4f} median={float(np.median(q)):+.4f}")

    print()
    print(f"{'population':>10} {'mean shift':>12} {'median shift':>14} {'95% half-width':>16}")
    shifts = {}
    for label in ("SEEDS", "CONTROL"):
        m, md, hw = paired(res[label]["pre"], res[label]["post"])
        shifts[label] = (m, hw)
        print(f"{label:>10} {m:>+12.4f} {md:>+14.4f} {hw:>16.4f}")

    ms, hs = shifts["SEEDS"]
    mc, hc = shifts["CONTROL"]
    diff = ms - mc
    hw = float(np.sqrt(hs**2 + hc**2))
    print()
    print(f"DIFFERENTIAL (seeds - control): {diff:+.4f} +/- {hw:.4f} (95%)")
    if abs(diff) <= hw:
        print("  => NO differential. The seeds moved with everything else:")
        print("     a GLOBAL pessimism shift. The absolute bar is uncalibrated and the")
        print("     mass retirement is an ARTIFACT.")
    elif diff < 0:
        print("  => The seeds moved down MORE than general positions. The net learned")
        print("     something SEED-SPECIFIC; retirement is defensible.")
    else:
        print("  => The seeds moved down LESS than general positions -- retirement is")
        print("     firing on drift the seeds did not even share.")


if __name__ == "__main__":
    main()
