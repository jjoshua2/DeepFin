"""CLI entry point: ``python3 -m chess_anti_engine.uci --checkpoint PATH``.

Loads the checkpoint, constructs a DirectGPUEvaluator (CUDA if available,
CPU otherwise), and runs the UCI stdin loop until ``quit``. Model load +
evaluator construction run on a background thread so the ``uci`` handshake
can reply instantly; ``isready`` and later commands block until the
engine is actually ready.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import sys
import threading

import chess
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import (
    BatchCoalescingDispatcher,
    MultiGPUDispatcher,
    ThreadSafeGPUDispatcher,
)
from chess_anti_engine.mcts.gumbel import GumbelConfig

from .engine import Engine, EngineOptions, emit_handshake
from .model_loader import load_model_from_checkpoint
from .protocol import CmdQuit, CmdUci, parse_command
from .search import SearchWorker


def _warmup_evaluator(evaluator) -> None:
    """Trigger torch.compile + CUDA graph capture for the shapes the UCI
    search will hit, so the first `go` doesn't pay that cost. Runs one
    forward at batch=1 (root eval) and one at bucket size 128 (typical
    single-game leaf batch from gumbel_c._BUCKETS). Both are no-ops if
    the model path doesn't graph-capture, but the compile itself is
    shape-keyed under torch.compile reduce-overhead mode."""
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


def _load_models(checkpoint: str, devices: list[str]):
    """Load one model per device. Cached at startup and reused across
    evaluator rebuilds (e.g., when the UCI ``MaxBatch`` option changes)."""
    if len(devices) > 1:
        # Parallel load — each is hundreds of MB of weight copy + CUDA init.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
            return list(pool.map(
                lambda d: load_model_from_checkpoint(checkpoint, device=d),
                devices,
            ))
    return [load_model_from_checkpoint(checkpoint, device=devices[0])]


def _make_evaluator_factory(models, devices, coalesce):
    """Return a ``build(max_batch) -> evaluator`` closure. The models are
    captured once at startup; each call constructs fresh evaluator
    wrappers at the new max_batch and warms them so the next ``go``
    doesn't pay compile latency."""
    def build(max_batch: int):
        if len(devices) > 1:
            evaluators = [
                DirectGPUEvaluator(m, device=d, max_batch=max_batch)
                for m, d in zip(models, devices)
            ]
            evaluator = MultiGPUDispatcher(evaluators)
        else:
            evaluator = DirectGPUEvaluator(
                models[0], device=devices[0], max_batch=max_batch,
            )
            # Always wrap in ThreadSafeGPUDispatcher so the UCI `Threads`
            # option can bump walker count at runtime without a race. Lock
            # is uncontended at 1 thread — overhead is ~10ns.
            evaluator = ThreadSafeGPUDispatcher(evaluator)
        if coalesce:
            evaluator = BatchCoalescingDispatcher(evaluator, max_batch=max_batch)
        _warmup_evaluator(evaluator)
        return evaluator
    return build


def _build_engine(
    *,
    evaluator,
    primary_device: str,
    chunk_sims: int,
    topk: int,
    n_walkers: int,
    vloss_weight: int,
    rebuild_evaluator=None,
) -> Engine:
    worker = SearchWorker(
        evaluator,
        device=primary_device,
        gumbel_cfg=GumbelConfig(simulations=chunk_sims, topk=topk, add_noise=False),
        chunk_sims=chunk_sims,
        n_walkers=n_walkers,
        vloss_weight=vloss_weight,
    )
    return Engine(worker, rebuild_evaluator=rebuild_evaluator)


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
    # --checkpoint is required, but we accept DEEPFIN_CKPT as the default so
    # the `deepfin` console-script entry point (from pyproject.toml) can be
    # launched as a bare executable by chess GUIs that don't pass CLI args.
    p.add_argument("--checkpoint", default=os.environ.get("DEEPFIN_CKPT"),
                   help="path to trainer.pt or checkpoint dir (falls back to $DEEPFIN_CKPT)")
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
    # --walkers > 1 switches from the Gumbel-chunked path to a PUCT walker
    # pool with virtual loss. Better dispatch-bound throughput, no
    # sequential-halving semantics. walkers=2 is ~10x the baseline on CUDA
    # (bench_uci_engine --walker-sweep, 2026-04-22) — we default to 2 so the
    # ship path gets the win. --walkers 1 opts into the classic Gumbel path.
    p.add_argument("--walkers", type=int, default=2,
                   help="PUCT walker threads (default: 2; 1 = classic Gumbel; >2 = noisy scaling)")
    p.add_argument("--vloss-weight", type=int, default=3,
                   help="virtual-loss weight in walker mode (default: 3, lc0 default)")
    # Coalesce concurrent walker calls into batched submits. On by default
    # when walkers > 1 since batch=1 per walker wastes GPU.
    p.add_argument("--no-coalesce", dest="coalesce", action="store_false",
                   help="disable walker-call coalescing (debug / A-B bench only)")
    p.set_defaults(coalesce=True)
    args = p.parse_args()

    if not args.checkpoint:
        p.error(
            "--checkpoint is required (or set DEEPFIN_CKPT in the environment). "
            "GUI launchers typically set DEEPFIN_CKPT in the shell profile."
        )

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
            n_walkers = max(1, int(args.walkers))
            models = _load_models(args.checkpoint, devices)
            build_eval = _make_evaluator_factory(
                models, devices, coalesce=bool(args.coalesce),
            )
            # Initial build: warms the evaluator too (see factory body).
            evaluator = build_eval(args.max_batch)
            engine_ref[0] = _build_engine(
                evaluator=evaluator, primary_device=devices[0],
                chunk_sims=args.chunk_sims, topk=args.topk,
                n_walkers=n_walkers, vloss_weight=int(args.vloss_weight),
                rebuild_evaluator=build_eval,
            )
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
