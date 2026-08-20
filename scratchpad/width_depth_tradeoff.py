"""Is `500k @ MPV40` actually a STRONGER label than production's `175k @ MPV6`?

Josh's objection: MultiPV splits a fixed node budget across root moves, so
`500k/40 = 12.5k` per line against `175k/6 = 29.2k` — the "escalated" teacher
could be WEAKER per line than the one it replaces, not stronger.

Three arms on the same frozen positions:

    P    175k @ MPV6    production-equivalent label
    E6   500k @ MPV6    PURE node escalation, width held fixed
    E40  500k @ MPV40   what "escalate to the deep teacher" was taken to mean

If `depth(E40) <= depth(P)` the objection is confirmed and "escalation to
500k/MPV40" is not an upgrade on the value axis at all.

Read-only, CPU-only, ONE nice'd worker.
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
ARMS = (("P", 175_000, 6), ("E6", 500_000, 6), ("E40", 500_000, 40))


def wdl_of(r) -> np.ndarray | None:
    return None if r.wdl is None else np.asarray(r.wdl, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--out", default="scratchpad/width_depth_tradeoff.json")
    args = ap.parse_args()

    rows = np.load(SRC, allow_pickle=True)
    fens = [str(f) for f in rows["fens"]]
    agree = rows["agree"].astype(bool)
    idx = [i for i in range(len(fens)) if not agree[i]][: args.n]
    print(f"positions: {len(idx)}")

    res: dict[str, list[dict]] = {}
    for name, nodes, mpv in ARMS:
        pool = StockfishPool(
            path=ENGINE, nodes=nodes, num_workers=1, multipv=mpv,
            hash_mb=17, syzygy_path=SYZYGY, nice=args.nice,
        )
        got = []
        try:
            for i in idx:
                r = pool.submit(fens[i], nodes=nodes, fresh=True).result()
                w = wdl_of(r)
                got.append({
                    "row": int(i), "bestmove": r.bestmove_uci, "cp": r.cp,
                    "depth": r.depth, "nodes": r.nodes, "n_pv": len(r.pvs),
                    "wdl": None if w is None else w.tolist(),
                })
        finally:
            pool.close()
        res[name] = got
        d = [g["depth"] for g in got if g["depth"]]
        print(f"  {name:4s} {nodes//1000:>4}k MPV{mpv:<3d} depth mean {np.mean(d):5.2f} "
              f"median {np.median(d):5.1f}  lines {np.mean([g['n_pv'] for g in got]):.1f}")

    print(f"\n{'comparison':>14} {'depth delta':>12} {'best-move change':>18} "
          f"{'mean |dWDL|':>12} {'frac |dWDL|>=0.20':>18}")
    base = {g["row"]: g for g in res["P"]}
    for name in ("E6", "E40"):
        cur = {g["row"]: g for g in res[name]}
        rowsk = [k for k in base if k in cur]
        dd = np.mean([cur[k]["depth"] - base[k]["depth"] for k in rowsk
                      if cur[k]["depth"] and base[k]["depth"]])
        bm = np.mean([cur[k]["bestmove"] != base[k]["bestmove"] for k in rowsk])
        dw, big = [], []
        for k in rowsk:
            a, b = base[k]["wdl"], cur[k]["wdl"]
            if a is None or b is None:
                continue
            m = float(np.abs(np.asarray(a) - np.asarray(b)).max())
            dw.append(m)
            big.append(m >= 0.20)
        print(f"{'P -> ' + name:>14} {dd:+12.2f} {bm:18.3f} "
              f"{np.mean(dw) if dw else float('nan'):12.4f} "
              f"{np.mean(big) if big else float('nan'):18.3f}")

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
