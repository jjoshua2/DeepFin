"""CLI entry point: ``python3 -m chess_anti_engine.uci --checkpoint PATH``.

Loads the checkpoint, constructs a DirectGPUEvaluator (CUDA if available,
CPU otherwise), and runs the UCI stdin loop until ``quit``.
"""
from __future__ import annotations

import argparse
import sys

from chess_anti_engine.inference import DirectGPUEvaluator

from .engine import Engine
from .model_loader import load_model_from_checkpoint
from .protocol import parse_command
from .search import SearchWorker


def _build_engine(*, checkpoint: str, device: str) -> Engine:
    model = load_model_from_checkpoint(checkpoint, device=device)
    evaluator = DirectGPUEvaluator(model, device=device, max_batch=32)
    worker = SearchWorker(evaluator, device=device)
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
    args = p.parse_args()

    # UCI assumes line-buffered I/O. When a GUI pipes stdout, Python defaults
    # to block-buffered, which swallows our responses until the buffer fills.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except AttributeError:
        pass

    device = _pick_device(args.device)
    engine = _build_engine(checkpoint=args.checkpoint, device=device)

    for raw in sys.stdin:
        engine.dispatch(parse_command(raw))
        if engine.quit_requested:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
