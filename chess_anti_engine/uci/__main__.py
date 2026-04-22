"""CLI entry point: ``python3 -m chess_anti_engine.uci --checkpoint PATH``.

Loads the checkpoint, constructs a DirectGPUEvaluator (CUDA if available,
CPU otherwise), and runs the UCI stdin loop until ``quit``.
"""
from __future__ import annotations

import argparse
import logging
import sys

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts.gumbel import GumbelConfig

from .engine import Engine
from .model_loader import load_model_from_checkpoint
from .protocol import parse_command
from .search import SearchWorker


def _build_engine(
    *,
    checkpoint: str,
    device: str,
    chunk_sims: int,
    topk: int,
    max_batch: int,
    thread_safe: bool,
    n_walkers: int,
    vloss_weight: int,
) -> Engine:
    model = load_model_from_checkpoint(checkpoint, device=device)
    evaluator: DirectGPUEvaluator | ThreadSafeGPUDispatcher = DirectGPUEvaluator(
        model, device=device, max_batch=max_batch,
    )
    # Walkers share the evaluator across threads; it must be wrapped.
    if thread_safe or n_walkers > 1:
        evaluator = ThreadSafeGPUDispatcher(evaluator)
    worker = SearchWorker(
        evaluator,
        device=device,
        gumbel_cfg=GumbelConfig(simulations=chunk_sims, topk=topk, add_noise=False),
        chunk_sims=chunk_sims,
        n_walkers=n_walkers,
        vloss_weight=vloss_weight,
    )
    return Engine(worker)


def _pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> int:
    p = argparse.ArgumentParser(prog="chess-anti-engine-uci")
    p.add_argument("--checkpoint", required=True, help="path to trainer.pt or checkpoint dir")
    p.add_argument("--device", default="auto", help="cpu|cuda|cuda:N (default: auto)")
    # Defaults from the 2026-04-21 bench sweep (bench_uci_engine.py --sweep):
    # chunk=512/topk=32/mb=1024 gave ~7.3x startpos nps vs 32/16/32. Chunk cap
    # of 512 (not the full node budget) keeps `stop` latency under ~400ms on
    # single-game CUDA searches.
    p.add_argument("--chunk-sims", type=int, default=512,
                   help="sims per start_gumbel_sims call (default: 512). Higher = fewer Python-C roundtrips, coarser stop latency.")
    p.add_argument("--topk", type=int, default=32, help="Gumbel root candidates (default: 32)")
    p.add_argument("--max-batch", type=int, default=1024,
                   help="DirectGPUEvaluator max batch (default: 1024). Must be >= expected leaf count per wavefront.")
    p.add_argument("--log-level", default="WARNING",
                   help="stderr log level (DEBUG|INFO|WARNING). DEBUG enables per-search gumbel profile with GPU-calls/avg-batch.")
    # Phase 1 of the walker-pool plan. Opt-in for single-walker searches;
    # forced on by --walkers > 1 since walkers share the evaluator.
    p.add_argument("--thread-safe-eval", action="store_true",
                   help="Wrap GPU evaluator in a thread-safe dispatcher (implied by --walkers > 1)")
    # Phase 5 of the walker-pool plan. --walkers > 1 switches from the
    # Gumbel-chunked path to a PUCT walker pool with virtual loss. Better
    # dispatch-bound throughput, no sequential-halving semantics.
    p.add_argument("--walkers", type=int, default=1,
                   help="number of PUCT walker threads (default: 1 = classic Gumbel path; >1 = pool)")
    p.add_argument("--vloss-weight", type=int, default=3,
                   help="virtual-loss weight in walker mode (default: 3, lc0 default)")
    args = p.parse_args()

    # Logs must go to stderr — stdout is reserved for UCI protocol.
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # UCI assumes line-buffered I/O. When a GUI pipes stdout, Python defaults
    # to block-buffered, which swallows our responses until the buffer fills.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except AttributeError:
        pass

    device = _pick_device(args.device)
    engine = _build_engine(
        checkpoint=args.checkpoint, device=device,
        chunk_sims=args.chunk_sims, topk=args.topk, max_batch=args.max_batch,
        thread_safe=args.thread_safe_eval,
        n_walkers=max(1, int(args.walkers)),
        vloss_weight=int(args.vloss_weight),
    )

    for raw in sys.stdin:
        engine.dispatch(parse_command(raw))
        if engine.quit_requested:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
