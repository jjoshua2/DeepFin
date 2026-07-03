#!/usr/bin/env python3
"""Gap-resolution readout: do high-SF-disagreement rows actually get fixed?

For full rows sampled from recent replay shards, scores the value error
|net_q - sf_q| under TWO checkpoints and reports the change per decile of the
stored ``priority_sf_search_gap``. This is the reproducible form of the
2026-07-02 finding that top-decile-gap rows did NOT resolve across ckpt466 →
ckpt478 (0.383 → 0.385) while bottom-half rows improved — and it is the
pre-committed SECONDARY readout for the #104 gap-priority activation: under
4-6x oversampling, the top-decile delta must turn positive, else the
bottleneck is capacity/targets, not sampling emphasis. See
docs/experiment_ledger.md (Analysis findings + LIVE table).

GPU-light, safe concurrent with training at the default memory fraction.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import torch

from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.replay.shard import load_shard_arrays
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

DEFAULT_SHARD_GLOB = (
    "runs/pbt2_small/server/trials/*/processed/_compacted/*.zarr"
)


def _q(wdl3: np.ndarray) -> np.ndarray:
    return wdl3[:, 0] - wdl3[:, 2]


def _net_q(ckpt: str, x: np.ndarray, *, device: str, batch: int) -> np.ndarray:
    model = load_model_from_checkpoint(ckpt, device=device)
    model.eval()
    ev = LocalModelEvaluator(model, device=device)
    out = np.empty(x.shape[0])
    for s in range(0, x.shape[0], batch):
        xs = np.asarray(x[s:s + batch], dtype=np.float32)
        with torch.no_grad():
            _, wdl = ev.evaluate_encoded(xs)
        wdl = np.asarray(wdl, dtype=np.float64)
        if not np.allclose(wdl.sum(axis=1), 1.0, atol=1e-3):
            e = np.exp(wdl - wdl.max(axis=1, keepdims=True))
            wdl = e / e.sum(axis=1, keepdims=True)
        out[s:s + batch] = wdl[:, 0] - wdl[:, 2]
    del model, ev
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-old", required=True)
    ap.add_argument("--checkpoint-new", required=True)
    ap.add_argument("--shard-glob", default=DEFAULT_SHARD_GLOB)
    ap.add_argument("--n-shards", type=int, default=30, help="newest N shards")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.15)
    args = ap.parse_args()

    if args.device.startswith("cuda") and args.gpu_mem_fraction:
        dev = torch.device(args.device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction), idx)

    paths = sorted(glob.glob(args.shard_glob))[-args.n_shards:]
    if not paths:
        raise SystemExit(f"no shards match {args.shard_glob}")
    xs, gaps, sf_qs = [], [], []
    for path in paths:
        arrs = load_shard_arrays(path)
        if isinstance(arrs, tuple):
            arrs = arrs[0]
        need = ("has_policy", "priority_sf_search_gap", "has_priority_sf_search_gap",
                "sf_wdl", "has_sf_wdl", "x")
        if any(k not in arrs for k in need):
            continue
        m = (np.asarray(arrs["has_policy"], dtype=bool)
             & np.asarray(arrs["has_priority_sf_search_gap"], dtype=bool)
             & np.asarray(arrs["has_sf_wdl"], dtype=bool))
        if not m.any():
            continue
        xs.append(np.asarray(arrs["x"])[m])
        gaps.append(np.asarray(arrs["priority_sf_search_gap"], dtype=np.float64)[m])
        sf_qs.append(_q(np.asarray(arrs["sf_wdl"], dtype=np.float64))[m])

    x = np.concatenate(xs)
    gap = np.concatenate(gaps)
    sf_q = np.concatenate(sf_qs)
    print(f"rows: {len(gap)} from {len(paths)} shards")

    verr_old = np.abs(_net_q(args.checkpoint_old, x, device=args.device,
                             batch=args.batch_size) - sf_q)
    verr_new = np.abs(_net_q(args.checkpoint_new, x, device=args.device,
                             batch=args.batch_size) - sf_q)

    print(f"\n{'gap bucket':>16} {'n':>7} {'verr_old':>9} {'verr_new':>9} {'delta':>8}")
    buckets = [("bottom half", 0, 50), ("p50-p90", 50, 90), ("top decile", 90, 100)]
    for name, lo, hi in buckets:
        m = (gap >= np.percentile(gap, lo)) & (gap <= np.percentile(gap, hi))
        d = float(np.mean(verr_old[m]) - np.mean(verr_new[m]))
        print(f"{name:>16} {int(m.sum()):>7} {np.mean(verr_old[m]):>9.4f} "
              f"{np.mean(verr_new[m]):>9.4f} {d:>+8.4f}")
    d_all = float(np.mean(verr_old) - np.mean(verr_new))
    print(f"{'ALL':>16} {len(gap):>7} {np.mean(verr_old):>9.4f} "
          f"{np.mean(verr_new):>9.4f} {d_all:>+8.4f}")
    print("\npositive delta = errors shrank (resolving); the #104 secondary "
          "readout requires the top-decile delta to turn positive.")


if __name__ == "__main__":
    main()
