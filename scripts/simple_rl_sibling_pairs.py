#!/usr/bin/env python3
"""Materialise sibling child pairs from a sealed simple_rl collect (CPU only).

    PYTHONPATH=. python scripts/simple_rl_sibling_pairs.py --collect RUN/gen_0001/collect \
        --out SIBLING_DIR/gen_0001 [--k 2] [--min-visits 2] [--seed 0] [--holdout-frac 0.05]
    PYTHONPATH=. python scripts/simple_rl_sibling_pairs.py --collect RUN/gen_0001/collect --self-check 20

The default mode writes one sidecar directory per shard plus ``sibling_manifest.json``.
The trainer reads it with ``--w-sibling W --sibling-dir SIBLING_DIR``. Every game is
rebuilt and every root's hash is checked before anything is written. A collect that
cannot be rebuilt is refused, and the exit code is 2.

``--self-check N`` rebuilds N games and compares each rebuilt root, and each played
child, byte for byte with the stored inputs. It exits 1 on any mismatch.

The rebuild rule, the pair choice and the training loss are documented in
``chess_anti_engine/simple_rl/sibling.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only: nothing here needs a GPU


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collect", required=True, help="a sealed collect directory (gen_<g>/collect)")
    p.add_argument("--out", default=None, help="sidecar directory to create (must not exist)")
    p.add_argument("--k", type=int, default=2, help="pairs per row (default 2; --unit pairs)")
    p.add_argument("--unit", choices=("pairs", "roots"), default="pairs",
                   help="pairs (default; version-2 pair sidecar) or roots (list sidecar: every eligible child of each "
                        "root with at least two; read with --sibling-unit roots)")
    p.add_argument("--min-visits", type=float, default=2.0, help="minimum visits of both children (default 2)")
    p.add_argument("--seed", type=int, default=0, help="pair-choice seed; the RNG is keyed by (seed, game, ply)")
    p.add_argument("--holdout-frac", type=float, default=0.05,
                   help="hold-out split flagged per pair (must equal the learner's --holdout-frac)")
    p.add_argument("--shards", default=None, help="comma-separated shard indices (a partial sidecar; the "
                                                   "trainer refuses it -- for measurement only)")
    p.add_argument("--self-check", type=int, default=None, metavar="N",
                   help="rebuild N games and compare roots and played children byte for byte; write nothing")
    a = p.parse_args(argv)
    from chess_anti_engine.simple_rl.sibling import CollectNotRebuildable, materialise, self_check

    try:
        if a.self_check is not None:
            res = self_check(Path(a.collect), games=int(a.self_check), seed=int(a.seed))
            print(json.dumps(res, indent=1))
            return 0 if res["ok"] else 1
        if not a.out:
            p.error("--out is required unless --self-check is given")
        shards = None if a.shards is None else [int(x) for x in a.shards.split(",") if x.strip()]
        man = materialise(Path(a.collect), Path(a.out), k=a.k, min_visits=a.min_visits, seed=a.seed,
                          holdout_frac=a.holdout_frac, shards=shards, log=lambda m: print(m, flush=True), unit=a.unit)
    except CollectNotRebuildable as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: man[k] for k in ("totals", "peak_rss_mib", "wall_seconds", "complete")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
