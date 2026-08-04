#!/usr/bin/env python3
"""Paired, game-clustered comparison of two per-row eval dumps.

Companion to ``scripts/holdout_policy_screen.py``. Rows from one game are not
independent, so the CI resamples GAMES, not rows: per-game sums and counts are
resampled together, which keeps the estimand the row-mean while pricing the
within-game correlation. Comparing a dump against itself must give exactly
+0.00000 [+0.00000, +0.00000] -- a cheap self-test that the pairing is real.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def load(p: Path) -> dict[str, np.ndarray]:
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def paired(a: dict, b: dict, key: str, n_boot: int = 10000, seed: int = 0) -> dict:
    """b - a, averaged over rows, CI by resampling GAMES."""
    if not np.array_equal(a["game_id"], b["game_id"]):
        raise SystemExit("row order differs between dumps")
    m = (a["base"] > 0) & (b["base"] > 0)
    d = (b[key] - a[key])[m]
    g = a["game_id"][m]
    ug, inv = np.unique(g, return_inverse=True)
    # per-game sum and count so a bootstrap over games preserves the row-mean estimand
    s = np.bincount(inv, weights=d, minlength=ug.size)
    c = np.bincount(inv, minlength=ug.size).astype(np.float64)
    est = s.sum() / c.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ug.size, size=(n_boot, ug.size))
    boot = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta": float(est), "lo": float(lo), "hi": float(hi),
            "n_rows": int(m.sum()), "n_games": int(ug.size), "sd_boot": float(boot.std())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="reference npz (usually step 0)")
    ap.add_argument("--arm", required=True, help="npz to compare")
    ap.add_argument("--keys", default="ce,agree")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    A, B = load(Path(a.base)), load(Path(a.arm))
    out = {}
    for k in a.keys.split(","):
        out[k] = paired(A, B, k, seed=a.seed)
    for k, v in out.items():
        print(f"{k:8s} delta {v['delta']:+.5f}  95% CI [{v['lo']:+.5f}, {v['hi']:+.5f}]  "
              f"rows {v['n_rows']} games {v['n_games']}")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
