#!/usr/bin/env python3
"""E4: scan live replay shards for poisoned/malformed rows.

Two poisoning paths reached production before (broker zero-fill -> all-zero
policy/WDL, PR #9; malformed compact legal metadata, PR #10). Both were fixed,
and nothing since asserts a third is not open. This samples the newest shards of
the LIVE replay dir and checks the invariants a poisoned row would violate.

Read-only. CPU + disk only; safe alongside training and arenas.

  PYTHONPATH=. python3 scripts/audit_row_integrity.py --shards 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=12, help="newest N shards to scan")
    ap.add_argument("--replay-root", default=str(REPO / "runs/pbt2_small/replay"))
    args = ap.parse_args()

    trials = sorted(Path(args.replay_root).glob("*/replay_shards"),
                    key=lambda p: p.stat().st_mtime)
    if not trials:
        # Anchoring on the script's own repo is right for a live checkout, but a
        # git worktree has no runs/ dir -- say so instead of just "not found".
        print(f"no replay_shards dirs under {args.replay_root}\n"
              "  (a git worktree has no runs/; pass --replay-root pointing at the "
              "live checkout, e.g.\n"
              "   --replay-root /home/josh/projects/chess/runs/pbt2_small/replay)")
        return 1
    shard_dir = trials[-1]
    paths = iter_shard_paths(shard_dir)[-args.shards:]
    print(f"scanning {len(paths)} newest shards under {shard_dir}\n")

    tot = 0
    bad: dict[str, int] = {}

    def flag(name: str, mask: object) -> None:
        # `mask` is typed loosely on purpose: numpy's reductions are declared as
        # returning `np.bool_ | NDArray[np.bool_]` depending on `axis`, and which
        # branch a given call resolves to varies by numpy version. asarray here
        # is version-proof; a narrower annotation would need a suppression whose
        # validity is tied to the installed numpy.
        n = int(np.count_nonzero(np.asarray(mask)))
        if n:
            bad[name] = bad.get(name, 0) + n

    for p in paths:
        arrs, _ = load_shard_arrays(p, lazy=False)
        x = np.asarray(arrs["x"])
        pol = np.asarray(arrs["policy_target"], dtype=np.float32)
        wdl = np.asarray(arrs["wdl_target"])
        n = x.shape[0]
        if n == 0:
            continue
        tot += n

        # A poisoned board is identically zero across every plane.
        flag("x all-zero", ~np.any(x.reshape(n, -1) != 0, axis=1))
        flag("x has NaN", np.isnan(x.reshape(n, -1)).any(axis=1))

        # Policy must be a distribution: non-negative, finite, sums to ~1.
        psum = pol.reshape(n, -1).sum(axis=1)
        flag("policy all-zero", psum == 0)
        flag("policy NaN", np.isnan(pol.reshape(n, -1)).any(axis=1))
        flag("policy negative", (pol.reshape(n, -1) < 0).any(axis=1))
        nz = psum != 0
        flag("policy sum!=1 (>2% off)", nz & (np.abs(psum - 1.0) > 0.02))

        # wdl_target is an int8 class label: 0=W 1=D 2=L.
        w = np.asarray(wdl).reshape(n, -1)[:, 0] if wdl.ndim > 1 else np.asarray(wdl)
        flag("wdl out of range", (w < 0) | (w > 2))

        # sf_wdl, where present, is a soft label in [0,1].
        if "sf_wdl" in arrs and "has_sf_wdl" in arrs:
            sf = np.asarray(arrs["sf_wdl"], dtype=np.float32).reshape(n, -1)
            has = np.asarray(arrs["has_sf_wdl"]).reshape(n).astype(bool)
            if has.any():
                s = sf[has]
                flag("sf_wdl NaN", np.isnan(s).any(axis=1))
                flag("sf_wdl out of [0,1]", ((s < -0.001) | (s > 1.001)).any(axis=1))

    print(f"rows scanned: {tot:,}\n")
    if not bad:
        print("CLEAN — no poisoned or malformed rows found.")
        return 0
    print("FINDINGS:")
    for k, v in sorted(bad.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<28} {v:>8,}  ({100 * v / max(tot, 1):.4f}%)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
