"""CLI entry point: ``python3 -m chess_anti_engine.uci --checkpoint PATH``.

Loads the checkpoint, constructs a DirectGPUEvaluator (CUDA if available,
CPU otherwise), and runs the UCI stdin loop until ``quit``. Model load +
evaluator construction run on a background thread so the ``uci`` handshake
can reply instantly; ``isready`` and later commands block until the
engine is actually ready.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading

import chess
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import (
    MultiGPUDispatcher,
    ThreadSafeGPUDispatcher,
)
from chess_anti_engine.mcts.gumbel import GumbelConfig

from .engine import Engine, EngineOptions, emit_handshake
from .model_loader import load_model_from_checkpoint
from .protocol import CmdQuit, CmdUci, parse_command
from .search import SearchWorker


def _warmup_evaluator(engine: Engine) -> None:
    """Trigger torch.compile + CUDA graph capture for the shapes the UCI
    search will hit, so the first `go` doesn't pay that cost. Runs one
    forward at batch=1 (root eval) and one at bucket size 128 (typical
    single-game leaf batch from gumbel_c._BUCKETS). Both are no-ops if
    the model path doesn't graph-capture, but the compile itself is
    shape-keyed under torch.compile reduce-overhead mode."""
    evaluator = engine._worker._evaluator  # pyright: ignore[reportPrivateUsage]
    cb = CBoard.from_board(chess.Board())
    encoded = cb.encode_146()
    for batch in (1, 128):
        xs = np.broadcast_to(encoded, (batch, 146, 8, 8)).astype(np.float32, copy=True)
        try:
            evaluator.evaluate_encoded(xs)
        except Exception:
            # Don't let a warmup failure block readyok — the real path would
            # raise the same error and the user would see it there.
            break


def _build_engine(
    *,
    checkpoint: str,
    devices: list[str],
    chunk_sims: int,
    topk: int,
    max_batch: int,
    thread_safe: bool,
    n_walkers: int,
    vloss_weight: int,
) -> Engine:
    # Multi-GPU (phase 7): one evaluator per device — each holds its own
    # compiled model + pinned buffers, no sharing. MultiGPUDispatcher does
    # least-loaded routing. For N=1 we keep the simpler code path (one
    # DirectGPUEvaluator, optional ThreadSafeGPUDispatcher).
    if len(devices) > 1:
        evaluators = [
            DirectGPUEvaluator(
                load_model_from_checkpoint(checkpoint, device=d),
                device=d, max_batch=max_batch,
            )
            for d in devices
        ]
        evaluator: DirectGPUEvaluator | ThreadSafeGPUDispatcher | MultiGPUDispatcher = (
            MultiGPUDispatcher(evaluators)
        )
        primary_device = devices[0]
    else:
        primary_device = devices[0]
        model = load_model_from_checkpoint(checkpoint, device=primary_device)
        evaluator = DirectGPUEvaluator(
            model, device=primary_device, max_batch=max_batch,
        )
        # Walkers share the evaluator across threads; it must be wrapped.
        if thread_safe or n_walkers > 1:
            evaluator = ThreadSafeGPUDispatcher(evaluator)
    worker = SearchWorker(
        evaluator,
        device=primary_device,
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
    p.add_argument("--devices", default=None,
                   help="comma-separated device list for multi-GPU (e.g. 'cuda:0,cuda:1'). Overrides --device.")
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

    # --devices wins over --device when set (explicit multi-GPU list).
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        devices = [_pick_device(args.device)]

    # Background-build so `uci` can be answered before model load finishes.
    # Any command other than uci/quit blocks on `engine_ready` below, which
    # gives us correct `readyok` semantics (readyok only fires once the
    # engine truly exists) for free.
    engine_ref: list[Engine | None] = [None]
    engine_error: list[BaseException | None] = [None]
    engine_ready = threading.Event()

    def _build() -> None:
        try:
            eng = _build_engine(
                checkpoint=args.checkpoint, devices=devices,
                chunk_sims=args.chunk_sims, topk=args.topk, max_batch=args.max_batch,
                thread_safe=args.thread_safe_eval,
                n_walkers=max(1, int(args.walkers)),
                vloss_weight=int(args.vloss_weight),
            )
            _warmup_evaluator(eng)
            engine_ref[0] = eng
        except BaseException as exc:  # pragma: no cover — surfaced via readyok
            engine_error[0] = exc
        finally:
            engine_ready.set()

    threading.Thread(target=_build, daemon=True, name="deepfin-build").start()

    for raw in sys.stdin:
        cmd = parse_command(raw)
        if isinstance(cmd, CmdUci):
            emit_handshake(EngineOptions())
            continue
        if isinstance(cmd, CmdQuit):
            break
        if not engine_ready.is_set():
            engine_ready.wait()
        if engine_error[0] is not None:
            print(f"info string engine load failed: {engine_error[0]!r}", flush=True)
            raise engine_error[0]
        engine = engine_ref[0]
        assert engine is not None
        engine.dispatch(cmd)
        if engine.quit_requested:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
