"""Does a preceding shallow query perturb a subsequent ``fresh=True`` deep query?

Pre-PR calibration for the targeted re-label arm (ledger `3e53ab051`).

ARMS, per frozen position, all on ONE engine (``num_workers=1``) so the shallow
search lands on the SAME process as the deep one. That is the WORST CASE: in
production the pool hands the re-query to whichever engine is free, so TT carry-
over is less likely there, not more.

    A1 = cold 500k              } the determinism control: A1 vs A2
    A2 = cold 500k              }
    s  = cold 175k              the contaminating shallow pass
    B  = cold 500k, run immediately after s on the same engine

⚑ PRE-REGISTERED PREDICTION: **A1 == A2 == B bit-for-bit.** `uci.py:405` pins
``Threads value 1`` and ``_new_game_locked`` does a synchronised ucinewgame /
isready / readyok, so a fixed-node search from a cleared TT is deterministic.
A non-zero A1-vs-A2 disagreement means a nondeterminism source I have not
identified, and is a finding in its own right.

Read-only, CPU-only, ONE nice'd worker (load average is already >32 on 32 CPUs).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/home/josh/projects/chess")

from chess_anti_engine.stockfish.pool import StockfishPool

SRC = "scratchpad/target_vs_bt4/tb4_rows.npz"
ENGINE = "/home/josh/projects/chess/e2e_server/publish/stockfish"
SYZYGY = (
    "/home/josh/projects/chess/data/syzygy_3-4-5"
    ":/home/josh/projects/chess/data/syzygy_6"
)
PROD_NODES, DEEP_NODES = 175_000, 500_000


def digest(r) -> dict:
    """Everything the efficacy result depends on, in a comparable form."""
    return {
        "bestmove": r.bestmove_uci,
        "cp": r.cp,
        "mate": r.mate,
        "depth": r.depth,
        "nodes": r.nodes,
        "pv_moves": [p.move_uci for p in r.pvs],
        "pv_cp": [p.cp for p in r.pvs],
    }


def diff(a: dict, b: dict) -> list[str]:
    out = []
    for k in ("bestmove", "cp", "mate", "pv_moves", "pv_cp"):
        if a[k] != b[k]:
            out.append(k)
    # depth/nodes are effort, not semantics: reported, never a semantic diff
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--multipv", type=int, default=40)
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--out", default="scratchpad/relabel_calibration.json")
    args = ap.parse_args()

    rows = np.load(SRC, allow_pickle=True)
    fens = [str(f) for f in rows["fens"]]
    agree = rows["agree"].astype(bool)
    # the population the arm is about: rows where target and SF DISAGREE
    idx = [i for i in range(len(fens)) if not agree[i]][: args.n]
    print(f"positions: {len(idx)} (target/SF disagreement rows)")

    pool = StockfishPool(
        path=ENGINE, nodes=DEEP_NODES, num_workers=1, multipv=args.multipv,
        hash_mb=17, syzygy_path=SYZYGY, nice=args.nice,
    )
    recs = []
    try:
        for n, i in enumerate(idx, 1):
            fen = fens[i]
            a1 = digest(pool.submit(fen, nodes=DEEP_NODES, fresh=True).result())
            a2 = digest(pool.submit(fen, nodes=DEEP_NODES, fresh=True).result())
            _ = pool.submit(fen, nodes=PROD_NODES, fresh=True).result()
            b = digest(pool.submit(fen, nodes=DEEP_NODES, fresh=True).result())
            recs.append({
                "row": int(i),
                "ctrl_diff": diff(a1, a2),   # A1 vs A2 -- expected []
                "test_diff": diff(a1, b),    # A1 vs B  -- expected []
                "a1": a1, "a2": a2, "b": b,
            })
            if n % 10 == 0:
                print(f"  {n}/{len(idx)}")
    finally:
        pool.close()

    ctrl = sum(1 for r in recs if r["ctrl_diff"])
    test = sum(1 for r in recs if r["test_diff"])
    print(f"\nA1 vs A2 (determinism control): {ctrl}/{len(recs)} positions differ")
    print(f"A1 vs B  (shallow-then-fresh) : {test}/{len(recs)} positions differ")
    for lab, key in (("control", "ctrl_diff"), ("test", "test_diff")):
        fields: dict[str, int] = {}
        for r in recs:
            for f in r[key]:
                fields[f] = fields.get(f, 0) + 1
        if fields:
            print(f"  {lab} differing fields: {fields}")
    nod = [r["a1"]["nodes"] for r in recs if r["a1"]["nodes"]]
    dep = [r["a1"]["depth"] for r in recs if r["a1"]["depth"]]
    if nod:
        print(f"\neffort at 500k/MPV{args.multipv}: nodes mean {np.mean(nod):.0f}  "
              f"depth mean {np.mean(dep):.1f}")
    print(f"\nVERDICT: {'PASS -- fresh=True isolates' if test == 0 and ctrl == 0 else 'INVESTIGATE'}")
    with open(args.out, "w") as fh:
        json.dump(recs, fh, indent=1)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
