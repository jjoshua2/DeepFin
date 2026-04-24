"""Sample ~30 positions (random-move walk from startpos), run SF at nodes=5k
multipv=20, measure gap between SF's chosen move (under regret_limit) and top.

Reports the distribution so we can tell whether regret=0.09 lets SF play
"trivial alternatives" or sometimes real blunders.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_anti_engine.stockfish import StockfishUCI  # noqa: E402

REGRET_LIMIT = 0.0925
SF_PATH = "/home/josh/projects/chess/e2e_server/publish/stockfish"
SF_NODES = 5000
SF_MULTIPV = 20
N_POSITIONS = 30
MIN_PLY = 6
MAX_PLY = 40
SEED = 42


def walk_positions(n: int, rng: random.Random) -> list[str]:
    fens: list[str] = []
    attempts = 0
    while len(fens) < n and attempts < n * 5:
        attempts += 1
        board = chess.Board()
        target_ply = rng.randint(MIN_PLY, MAX_PLY)
        ok = True
        for _ in range(target_ply):
            if board.is_game_over():
                ok = False
                break
            moves = list(board.legal_moves)
            board.push(rng.choice(moves))
        if ok and not board.is_game_over():
            fens.append(board.fen())
    return fens


def main() -> None:
    rng = random.Random(SEED)
    fens = walk_positions(N_POSITIONS, rng)
    print(f"generated {len(fens)} positions (plies {MIN_PLY}..{MAX_PLY})")

    sf = StockfishUCI(SF_PATH, nodes=SF_NODES, multipv=SF_MULTIPV)

    gaps: list[float] = []
    pool_sizes: list[int] = []
    blunders_over_05: int = 0
    played_top: int = 0

    for i, fen in enumerate(fens):
        try:
            res = sf.search(fen)
        except Exception as exc:
            print(f"[{i}] SF error: {exc}")
            continue

        scored = []
        for pv in res.pvs:
            if pv.wdl is None:
                continue
            w, d, _ = pv.wdl
            score = float(w) + 0.5 * float(d)
            scored.append((pv.move_uci, score))

        if len(scored) < 2:
            continue

        scored.sort(key=lambda x: -x[1])
        top_uci, top_score = scored[0]
        acceptable = [(u, s) for (u, s) in scored if (top_score - s) <= REGRET_LIMIT + 1e-12]
        chosen_uci, chosen_score = acceptable[rng.randrange(len(acceptable))]
        gap = top_score - chosen_score

        pool_sizes.append(len(acceptable))
        gaps.append(gap)
        if gap == 0.0:
            played_top += 1
        if gap > 0.05:
            blunders_over_05 += 1
            print(f"[{i}] BIG GAP gap={gap:.4f}  top={top_uci} (s={top_score:.3f})  chose={chosen_uci} (s={chosen_score:.3f})  pool_size={len(acceptable)}")

    sf.close()

    if not gaps:
        print("No data collected.")
        return

    gaps_arr = np.array(gaps)
    pool_arr = np.array(pool_sizes)

    print("\n=== summary ===")
    print(f"n positions scored:       {len(gaps)}")
    print(f"pool size  mean / median: {pool_arr.mean():.2f} / {int(np.median(pool_arr))}")
    print(f"pool size  max:           {int(pool_arr.max())}")
    print(f"positions where pool=1:   {int((pool_arr == 1).sum())}  ({100*(pool_arr==1).mean():.0f}%)  (SF forced to top)")
    print(f"pool >= 5:                {int((pool_arr >= 5).sum())}  ({100*(pool_arr>=5).mean():.0f}%)")
    print(f"chose top move:           {played_top}/{len(gaps)}  ({100*played_top/len(gaps):.0f}%)")
    print(f"gap  mean / median:       {gaps_arr.mean():.4f} / {np.median(gaps_arr):.4f}")
    print(f"gap  90th pct:            {np.percentile(gaps_arr, 90):.4f}")
    print(f"gap  max:                 {gaps_arr.max():.4f}")
    print(f"gap > 0.05 ('big'):       {blunders_over_05}/{len(gaps)}  ({100*blunders_over_05/len(gaps):.0f}%)")


if __name__ == "__main__":
    main()
