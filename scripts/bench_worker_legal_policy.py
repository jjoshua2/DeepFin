#!/usr/bin/env python3
"""Benchmark the production worker's compiled dense vs legal-policy path.

Each invocation runs one path in an isolated process.  Run the two commands
back-to-back with identical arguments and compare ``positions_per_s`` and
``output_hash``::

    PYTHONPATH=. python3 scripts/bench_worker_legal_policy.py \
      --checkpoint CHECKPOINT --policy-path dense
    PYTHONPATH=. python3 scripts/bench_worker_legal_policy.py \
      --checkpoint CHECKPOINT --policy-path legal

The workload uses the production 175-plane checkpoint metadata, BF16-bit input
transport, compact legal-policy output, max-autotune compilation, batch 680,
and the same ``ThreadedDispatcher`` used by distributed workers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.inference_threaded import ThreadedDispatcher
from chess_anti_engine.model import infer_input_planes
from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--policy-path", choices=("dense", "legal"), default="dense")
    ap.add_argument(
        "--input-path",
        choices=("widened", "native-bf16"),
        default="widened",
    )
    ap.add_argument("--batch", type=int, default=680)
    ap.add_argument("--legal-per-position", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--compile-mode", default="max-autotune")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.batch <= 0 or args.legal_per_position <= 0:
        raise SystemExit("--batch and --legal-per-position must be > 0")

    if args.policy_path == "legal":
        os.environ["CAE_COMPILED_LEGAL_POLICY"] = "1"
    else:
        os.environ.pop("CAE_COMPILED_LEGAL_POLICY", None)

    model = load_model_from_checkpoint(args.checkpoint, device="cuda").eval()
    planes = infer_input_planes(getattr(model, "input_extra_features", None))
    rng = np.random.default_rng(20260711)
    x_f32 = rng.standard_normal((args.batch, planes, 8, 8), dtype=np.float32)
    x = torch.from_numpy(x_f32).to(torch.bfloat16).view(torch.uint16).numpy()

    counts = np.full((args.batch,), args.legal_per_position, dtype=np.int32)
    compact = rng.integers(
        0,
        len(COMPACT_TO_FULL_POLICY),
        size=int(counts.sum()),
        dtype=np.int32,
    )
    legal_flat = np.asarray(COMPACT_TO_FULL_POLICY[compact], dtype=np.int32)

    dispatcher = ThreadedDispatcher(
        model,
        device="cuda",
        max_batch=1190,
        target_batch=680,
        batch_wait_ms=5.0,
        compile_mode=args.compile_mode,
        input_bf16=args.input_path == "native-bf16",
    )
    first_pol: np.ndarray | None = None
    first_wdl: np.ndarray | None = None
    try:
        for _ in range(args.warmup):
            first_pol, first_wdl = dispatcher.evaluate_legal_bf16(
                x, legal_flat, counts,
            )

        times: list[float] = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            pol, wdl = dispatcher.evaluate_legal_bf16(x, legal_flat, counts)
            times.append(time.perf_counter() - t0)
            if first_pol is None:
                first_pol, first_wdl = pol, wdl
    finally:
        stats = dispatcher.stats
        dispatcher.shutdown()

    assert first_pol is not None
    assert first_wdl is not None
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(first_pol).tobytes())
    digest.update(np.ascontiguousarray(first_wdl).tobytes())
    median_s = float(np.median(np.asarray(times)))
    result = {
        "policy_path": args.policy_path,
        "input_path": args.input_path,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "batch": args.batch,
        "planes": planes,
        "legal_per_position": args.legal_per_position,
        "compile_mode": args.compile_mode,
        "median_ms": median_s * 1000.0,
        "positions_per_s": args.batch / median_s,
        "output_hash": digest.hexdigest(),
        "dispatcher": stats,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
