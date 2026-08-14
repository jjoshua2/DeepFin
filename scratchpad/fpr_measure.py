"""Measure the missingness guard's false-positive rate on TRUE-NULL data.

Calls the real `completion_bias_report`, not a reimplementation of it: a guard
must be measured through the same instrument the operator runs.

Null construction: pair outcomes are drawn independently of completion order,
so any flag is by definition a false positive.
"""
from __future__ import annotations

import contextlib
import io
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ordo_pooled_fit import (
    Game,
    Pair,
    completion_bias_report,
)

MATCHUPS = (("armA", "armB"), ("armB", "armC"), ("armA", "armC"))
COUNTS = {("armA", "armB"): 40, ("armB", "armC"): 30, ("armA", "armC"): 25}


def null_dataset(rng: random.Random) -> dict[tuple[str, str], list[Pair]]:
    by: dict[tuple[str, str], list[Pair]] = {}
    order = 0
    for a, b in MATCHUPS:
        blocks: list[Pair] = []
        for pid in range(COUNTS[(a, b)]):
            games: list[Game] = []
            for half in range(2):
                res = rng.choice(["1-0", "1/2-1/2", "0-1"])
                w, bl = (a, b) if half == 0 else (b, a)
                games.append(Game(white=w, black=bl, result=res,
                                  pair_id=str(pid), plies=rng.randint(40, 200),
                                  duration_s=1.0, order=order))
                order += 1
            blocks.append(Pair(matchup=(a, b), games=games))
        # completion order is assigned INDEPENDENTLY of outcome => true null
        rng.shuffle(blocks)
        for k, blk in enumerate(blocks):
            for j, g in enumerate(blk.games):
                g.order = 2 * k + j
        by[(a, b)] = sorted(blocks, key=lambda p: p.completion_order)
    return by


def main() -> int:
    n_sets = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    n_perm = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    rng = random.Random(20260813)

    corrected = 0
    uncorrected = 0
    for i in range(n_sets):
        by = null_dataset(rng)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            flagged = completion_bias_report(by, rng=rng, n_perm=n_perm)
        corrected += bool(flagged)
        # Recover the raw per-test decisions from the printed p-values to get
        # the UNCORRECTED rate through the very same statistics.
        ps: list[float] = [
            float(tok[2:])
            for line in buf.getvalue().splitlines()
            for tok in line.split()
            if tok.startswith("p=") and tok[2:].replace(".", "").isdigit()
        ]
        uncorrected += any(p < 0.05 for p in ps)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_sets}: corrected {corrected / (i + 1):.3f}  "
                  f"uncorrected {uncorrected / (i + 1):.3f}", flush=True)

    def ci(k: int) -> str:
        p = k / n_sets
        se = (p * (1 - p) / n_sets) ** 0.5
        return f"{p:.3f} +- {se:.3f}"

    print(f"\nn_sets={n_sets} n_perm={n_perm} (3 matchups, 6 tests per dataset)")
    print(f"  UNCORRECTED family false-positive rate: {ci(uncorrected)}")
    print(f"  HOLM-CORRECTED    false-positive rate: {ci(corrected)}"
          f"   (nominal 0.05)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
