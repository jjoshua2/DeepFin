"""In-process driver for profiling UCI single-game search.

Mirrors uci/__main__.py wiring (SearchWorker + DirectGPUEvaluator +
ThreadSafeGPUDispatcher) but skips the stdin protocol — runs N searches
at a fixed node budget so a sampler (py-spy) or cProfile can capture the
hot path without subprocess noise.

Usage:
  PYTHONPATH=. python3 scripts/profile_uci_search.py --checkpoint <path> [--nodes 1024] [--positions 4] [--cprofile out.pstats]
  PYTHONPATH=. py-spy record -o uci_search.svg -- python3 scripts/profile_uci_search.py --checkpoint <path> --nodes 1024 --positions 8
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import threading
import time

import chess
import numpy as np

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline


_TEST_FENS = [
    chess.STARTING_FEN,
  # Italian middlegame
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
  # Endgame: K+R vs K+P
    "8/8/8/4k3/8/4K3/4P3/4R3 w - - 0 1",
  # Sharp tactical position (Sicilian)
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 1",
  # Closed positional (French)
    "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 1",
  # KP endgame
    "8/8/8/8/4k3/8/4P3/4K3 w - - 0 1",
  # Complex middlegame (Ruy Lopez)
    "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
  # Pawn race endgame
    "8/8/2k5/8/8/4K3/8/8 w - - 0 1",
]


def build_worker(
    checkpoint: str, device: str, *,
    max_batch: int, compile_mode: str | None = None,
) -> SearchWorker:
    model = load_model_from_checkpoint(checkpoint, device=device)
    if compile_mode:
        import torch
        from typing import cast
        from pathlib import Path
        from chess_anti_engine.worker import _configure_shared_compile_cache
        _configure_shared_compile_cache(
            cache_dir=Path("~/.cache/deepfin/worker_cache").expanduser(),
        )
        model = cast("torch.nn.Module", torch.compile(model, mode=compile_mode))
    evaluator = DirectGPUEvaluator(model, device=device, max_batch=max_batch, n_slots=2)
    evaluator = ThreadSafeGPUDispatcher(evaluator)
  # Match UCI default warmup: bucket flush at 128, root eval at 1.
    for batch in (1, 128):
        xs = np.repeat(np.zeros((1, 146, 8, 8), dtype=np.float32), batch, axis=0)
        evaluator.evaluate_encoded(xs)
    return SearchWorker(
        evaluator,
        device=device,
        gumbel_cfg=GumbelConfig(simulations=128, topk=16, add_noise=False),
        chunk_sims=128,
        n_walkers=1,
        walker_gather=1,
    )


def run_searches(worker: SearchWorker, fens: list[str], nodes: int) -> tuple[float, int]:
    total_t = 0.0
    total_n = 0
    for fen in fens:
        worker.reset_tree()
        board = chess.Board(fen)
        stop = threading.Event()
  # 60s deadline so node budget is the binding constraint.
        deadline = Deadline(60_000)
        t0 = time.perf_counter()
        result = worker.run(board, stop_event=stop, deadline=deadline, max_nodes=nodes)
        total_t += time.perf_counter() - t0
        total_n += result.nodes
    return total_t, total_n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batch", type=int, default=512,
                   help="evaluator max batch (UCI default 512)")
    p.add_argument("--nodes", type=int, default=1024,
                   help="per-position node budget (search stops at this many sims)")
    p.add_argument("--positions", type=int, default=len(_TEST_FENS),
                   help="how many of the canned FENs to run")
    p.add_argument("--warmup", type=int, default=1,
                   help="warmup runs before the timed window (per position)")
    p.add_argument("--cprofile", default=None,
                   help="if set, run under cProfile and write stats to this path")
    p.add_argument("--top", type=int, default=40,
                   help="cProfile: top-N functions by cumulative time")
    p.add_argument("--compile", default=None,
                   help="torch.compile mode (e.g. reduce-overhead). Default: no compile.")
    args = p.parse_args()

    if args.positions <= 0:
        raise SystemExit("--positions must be > 0")
    if args.nodes <= 0:
        raise SystemExit("--nodes must be > 0")
    if args.max_batch <= 0:
        raise SystemExit("--max-batch must be > 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    fens = _TEST_FENS[: args.positions]
    print(f"[setup] checkpoint={args.checkpoint} device={args.device} "
          f"max_batch={args.max_batch} nodes={args.nodes} positions={len(fens)}")

    worker = build_worker(
        args.checkpoint, args.device, max_batch=args.max_batch,
        compile_mode=args.compile,
    )

  # Warmup: drives compile + first-call costs out of the timed window.
    for _ in range(args.warmup):
        run_searches(worker, fens[:1], args.nodes)
    print(f"[warmup] {args.warmup}× {fens[0][:30]!r} done")

    if args.cprofile:
        prof = cProfile.Profile()
        prof.enable()
        t, n = run_searches(worker, fens, args.nodes)
        prof.disable()
        prof.dump_stats(args.cprofile)
        print(f"[result] total={t:.2f}s nodes={n} → {n/t:.0f} nps")
        print(f"[cprofile] saved to {args.cprofile}")
        st = pstats.Stats(prof).sort_stats("cumulative")
        st.print_stats(args.top)
    else:
        t, n = run_searches(worker, fens, args.nodes)
        print(f"[result] total={t:.2f}s nodes={n} → {n/t:.0f} nps")
        print(f"[result] avg per position: {1000*t/len(fens):.1f}ms / "
              f"{n//len(fens)} nodes = {(n/t):.0f} nps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
