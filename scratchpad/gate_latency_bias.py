#!/usr/bin/env python3
"""Is the anchored gate's negative delta a COMPLETION-ORDER artifact? (task #187)

`gate_delta_elo` has drifted to -26 Elo with a CI excluding zero, and
`gate_sample_confound_elo` is small, so PID difficulty drift does not explain it.
The gate buckets each iteration's arriving games into `cur` (produced by the model
published this iteration) and `prev` (produced by the previous one), and scores the
difference.

⚑ That split is ALSO a LATENCY split.  A game that finishes inside one iteration
lands in `cur`; a game that takes longer lands in `prev`.  If score depends on how
long a game takes, the two buckets are not exchangeable and the gate reads a
game-length effect as a model regression.

This measures the dependence directly, on the ONE surface where provenance survives:
the server's `processed/` shards, whose `.attrs` still carry `model_step` and
`generated_at_unix` (ingest nulls both -- task #181).

    latency = generated_at_unix - (first generated_at_unix seen for that model_step)

and per shard the curriculum score W/D/L and mean plies/game.  Score plotted against
latency IS the confound, or its absence.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from collections import defaultdict

import numpy as np
import zarr


def curriculum_wdl(stats: dict[str, object]) -> tuple[int, int, int]:
    """Sum curriculum W/D/L over the composite outcome_stats keys.

    Keys look like `curriculum_book2_net_white_d`.  Only the LEAST specific family
    (`curriculum_<book>_<w|d|l>`) is summed -- the `net_white` / `net_black` variants
    are the SAME games split by colour, so adding both double-counts.
    """
    w = d = ell = 0
    for k, v in stats.items():
        if not k.startswith("curriculum_"):
            continue
        if "_net_white" in k or "_net_black" in k:
            continue
        n = int(v) if isinstance(v, (int, float)) else 0
        if k.endswith("_w"):
            w += n
        elif k.endswith("_d"):
            d += n
        elif k.endswith("_l"):
            ell += n
    return w, d, ell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="runs/pbt2_small/server/trials/5ce02_00000")
    ap.add_argument("--min-games", type=int, default=200)
    args = ap.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.trial, "processed", "**", "*.zarr"), recursive=True),
        key=os.path.getmtime,
    )
    if not paths:
        raise SystemExit(f"no processed shards under {args.trial}")

    rows: list[dict[str, float]] = []
    for p in paths:
        a = dict(zarr.open(p, mode="r").attrs)
        step = a.get("model_step")
        gen = a.get("generated_at_unix")
        if step is None or gen is None:
            continue
        w, d, ell = curriculum_wdl(dict(a.get("outcome_stats") or {}))
        games = w + d + ell
        if games <= 0:
            continue
        plies = sum(float(a.get(k, 0) or 0) for k in ("plies_win", "plies_loss", "plies_draw"))
        rows.append(
            {
                "step": float(step),
                "gen": float(gen),
                "w": float(w),
                "d": float(d),
                "l": float(ell),
                "games": float(games),
                "plies": plies,
            }
        )

    if not rows:
        raise SystemExit("no shard carried both model_step and curriculum outcomes")

    first_gen: dict[float, float] = {}
    for r in rows:
        s = r["step"]
        first_gen[s] = min(first_gen.get(s, math.inf), r["gen"])
    for r in rows:
        r["latency"] = r["gen"] - first_gen[r["step"]]

    lat = np.asarray([r["latency"] for r in rows])
    print(f"shards={len(rows)}  distinct model_steps={len(first_gen)}")
    print(f"latency s: median={np.median(lat):.0f} p90={np.percentile(lat, 90):.0f} max={lat.max():.0f}")

    # Bucket by latency and report score + game length.  Equal-count buckets, so a
    # skewed latency distribution cannot manufacture the trend.
    order = np.argsort(lat)
    nb = 5
    print()
    print("%10s %8s %8s %9s %9s %9s" % ("lat_bin_s", "shards", "games", "score", "elo", "plies/game"))
    per_bucket: list[tuple[float, float, float]] = []
    for b in range(nb):
        idx = order[b * len(order) // nb : (b + 1) * len(order) // nb]
        if idx.size == 0:
            continue
        w = sum(rows[i]["w"] for i in idx)
        d = sum(rows[i]["d"] for i in idx)
        ell = sum(rows[i]["l"] for i in idx)
        g = w + d + ell
        pl = sum(rows[i]["plies"] for i in idx)
        gm = sum(rows[i]["games"] for i in idx)
        if g < 1:
            continue
        score = (w + 0.5 * d) / g
        elo = (
            -400.0 * math.log10(1.0 / score - 1.0)
            if 0.0 < score < 1.0
            else float("nan")
        )
        lo = lat[idx].min()
        hi = lat[idx].max()
        # EXACT per-game variance for a score supported on {0, 0.5, 1}: no
        # sd<=0.5 bound, which is loose enough here to hide the effect.
        ex2 = (0.0 * ell + 0.25 * d + 1.0 * w) / g
        per_bucket.append((score, g, max(0.0, ex2 - score * score)))
        print(
            "%4.0f-%-5.0f %8d %8.0f %9.4f %9.1f %9.1f"
            % (lo, hi, idx.size, g, score, elo, pl / max(1.0, gm))
        )

    if len(per_bucket) >= 2:
        (p1, n1, v1), (p2, n2, v2) = per_bucket[0], per_bucket[-1]
        se = math.sqrt(v1 / n1 + v2 / n2)
        diff = p2 - p1
        elo = 0.0
        if 0.0 < p1 < 1.0 and 0.0 < p2 < 1.0:
            elo = -400.0 * math.log10(1.0 / p2 - 1.0) + 400.0 * math.log10(1.0 / p1 - 1.0)
        print()
        print(
            f"fastest vs slowest bucket: score {p1:.4f} -> {p2:.4f}  "
            f"diff {diff:+.4f} +/- {1.96 * se:.4f} (95%)  = {elo:+.1f} Elo"
        )
        print(
            "  ⇒ the cur/prev split is a LATENCY split.  A nonzero diff here biases\n"
            "    cur-minus-prev by the SAME amount with the opposite sign, and the gate\n"
            "    cannot see it: `gate_sample_confound_elo` tracks PID difficulty, not\n"
            "    completion order."
        )


if __name__ == "__main__":
    main()
