"""Same as blunder_check.py but parses both cp and WDL from raw UCI output.

For each position:
- top-move cp + wdl
- chosen-move (random pick inside regret_limit) cp + wdl
- gaps in both

Reports distribution so we can see how big the cp-equivalent handicap is.
"""
from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
            board.push(rng.choice(list(board.legal_moves)))
        if ok and not board.is_game_over():
            fens.append(board.fen())
    return fens


class SFRaw:
    def __init__(self) -> None:
        self.p = subprocess.Popen(  # pylint: disable=consider-using-with  # long-lived SF subprocess, explicit close() method
            [SF_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send("setoption name UCI_ShowWDL value true")
        self._send("setoption name Threads value 1")
        self._send(f"setoption name MultiPV value {SF_MULTIPV}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, s: str) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _wait_for(self, token: str) -> None:
        assert self.p.stdout is not None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("SF died")
            if token in line:
                return

    def search(self, fen: str) -> list[tuple[str, int, float]]:
        """Returns list of (move, cp, wdl_score) per multipv, sorted by mpv."""
        assert self.p.stdout is not None
        self._send(f"position fen {fen}")
        self._send(f"go nodes {SF_NODES}")

        # (mpv, move, cp, wdl_score)
        per_mpv: dict[int, tuple[str, int, float]] = {}
        while True:
            line = self.p.stdout.readline().strip()
            if not line:
                continue
            if line.startswith("info") and "pv" in line.split():
                parts = line.split()
                mpv = 1
                cp = None
                mate = None
                wdl = None
                move = None
                for i, tok in enumerate(parts):
                    if tok == "multipv" and i + 1 < len(parts):
                        try:
                            mpv = int(parts[i+1])
                        except ValueError:
                            pass
                    elif tok == "cp" and i + 1 < len(parts):
                        try:
                            cp = int(parts[i+1])
                        except ValueError:
                            pass
                    elif tok == "mate" and i + 1 < len(parts):
                        try:
                            mate = int(parts[i+1])
                        except ValueError:
                            pass
                    elif tok == "wdl" and i + 3 < len(parts):
                        try:
                            w = int(parts[i+1])
                            d = int(parts[i+2])
                            losses = int(parts[i+3])
                            s = w + d + losses
                            if s > 0:
                                wdl = (w + 0.5*d) / s
                        except ValueError:
                            pass
                    elif tok == "pv" and i + 1 < len(parts):
                        move = parts[i+1]
                if move is None:
                    continue
                if cp is None and mate is not None:
                    cp = 100_000 if mate > 0 else -100_000
                if cp is None or wdl is None:
                    continue
                per_mpv[mpv] = (move, cp, wdl)
            if line.startswith("bestmove"):
                break
        return [per_mpv[k] for k in sorted(per_mpv.keys())]

    def close(self) -> None:
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self.p.kill()
            self.p.wait(timeout=5)
        except Exception:
            pass


def main() -> None:
    rng = random.Random(SEED)
    fens = walk_positions(N_POSITIONS, rng)
    print(f"generated {len(fens)} positions (plies {MIN_PLY}..{MAX_PLY})")

    sf = SFRaw()
    wdl_gaps: list[float] = []
    cp_gaps: list[int] = []
    pool_sizes: list[int] = []
    big_cases: list[str] = []

    for i, fen in enumerate(fens):
        pvs = sf.search(fen)
        if len(pvs) < 2:
            continue
        # Sort by wdl descending so [0] is top
        pvs.sort(key=lambda x: -x[2])
        top_move, top_cp, top_wdl = pvs[0]
        acceptable = [(m, cp, w) for (m, cp, w) in pvs if top_wdl - w <= REGRET_LIMIT + 1e-12]
        chosen_move, chosen_cp, chosen_wdl = acceptable[rng.randrange(len(acceptable))]
        wdl_gap = top_wdl - chosen_wdl
        cp_gap = top_cp - chosen_cp
        wdl_gaps.append(wdl_gap)
        cp_gaps.append(cp_gap)
        pool_sizes.append(len(acceptable))

        if wdl_gap > 0.05 or cp_gap > 80:
            big_cases.append(
                f"[{i}] wdl_gap={wdl_gap:.4f} cp_gap={cp_gap:+d}  "
                f"top={top_move}(wdl={top_wdl:.3f} cp={top_cp:+d}) "
                f"chose={chosen_move}(wdl={chosen_wdl:.3f} cp={chosen_cp:+d}) "
                f"pool={len(acceptable)}"
            )

    sf.close()

    if not wdl_gaps:
        print("no data")
        return

    wa = np.array(wdl_gaps)
    ca = np.array(cp_gaps)
    pa = np.array(pool_sizes)
    print(f"\nn: {len(wa)}")
    print(f"pool size mean/median/max: {pa.mean():.2f} / {int(np.median(pa))} / {int(pa.max())}")
    print(f"wdl_gap mean/median/90p/max: {wa.mean():.4f} / {np.median(wa):.4f} / {np.percentile(wa,90):.4f} / {wa.max():.4f}")
    print(f"cp_gap  mean/median/90p/max: {ca.mean():+.1f} / {int(np.median(ca)):+d} / {int(np.percentile(ca,90)):+d} / {int(ca.max()):+d}")
    print("cases where wdl_gap>0.05 or cp_gap>80:")
    for s in big_cases:
        print(" ", s)


if __name__ == "__main__":
    main()
