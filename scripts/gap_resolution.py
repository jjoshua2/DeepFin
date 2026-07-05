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


def _net_q(
    ckpt: str, x: np.ndarray, rel: np.ndarray | None, *, device: str, batch: int,
) -> np.ndarray:
    model = load_model_from_checkpoint(ckpt, device=device)
    model.eval()
    # Dynamic-relation checkpoints apply their attention bias only when the
    # relation tensor is passed — scoring them without it would misread the
    # model (same contract as blindspot_panel / value_regret).
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    if use_rel and rel is None:
        raise SystemExit(
            f"{ckpt} was trained with dynamic relations but the sampled shards "
            "carry no relations arrays — refusing to score without them",
        )
    ev = LocalModelEvaluator(model, device=device)
    out = np.empty(x.shape[0])
    for s in range(0, x.shape[0], batch):
        xs = np.asarray(x[s:s + batch], dtype=np.float32)
        rs = np.asarray(rel[s:s + batch]) if use_rel and rel is not None else None
        with torch.no_grad():
            _, wdl = ev.evaluate_encoded(xs, relations=rs)
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
        try:
            idx = (int(args.device.split(":", 1)[1]) if ":" in args.device
                   else torch.cuda.current_device())
        except (ValueError, IndexError):
            raise SystemExit(
                f"invalid --device {args.device!r}; use 'cuda' or 'cuda:N'",
            ) from None
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction), idx)

    if args.n_shards < 1:
        # -args.n_shards with n_shards=0 is [-0:] == [0:] == ALL shards, a silent
        # unbounded load; negative slices from the wrong end (Codex review).
        raise SystemExit("--n-shards must be >= 1")
    paths = sorted(glob.glob(args.shard_glob))[-args.n_shards:]
    if not paths:
        raise SystemExit(f"no shards match {args.shard_glob}")
    xs, gaps, sf_qs, rels = [], [], [], []
    have_rel = True
    for path in paths:
        # lazy=True so only the ~6 keys read below are decompressed, not the
        # wide unused arrays (policy_target 1858, sf_policy, future_policy,
        # categorical) — keeps the tool's "GPU-light" footprint (Codex review).
        arrs, _meta = load_shard_arrays(path, lazy=True)
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
        # Relations ride along only if every selected row in every shard has
        # them; otherwise dynamic-relation checkpoints are rejected in _net_q.
        if (have_rel and "relations" in arrs and "has_relations" in arrs
                and bool(np.asarray(arrs["has_relations"], dtype=bool)[m].all())):
            rels.append(np.asarray(arrs["relations"])[m])
        else:
            have_rel = False

    if not xs:
        # Every sampled shard was skipped (no priority_sf_search_gap rows) — the
        # window likely predates #104 gap recording. Fail with a clear message
        # rather than an opaque "need at least one array to concatenate" (Codex).
        raise SystemExit(
            f"no rows carrying priority_sf_search_gap in the newest {len(paths)} "
            f"shard(s) matching {args.shard_glob}; the window may predate the "
            f"#104 gap recording — point --shard-glob at newer shards",
        )
    x = np.concatenate(xs)
    gap = np.concatenate(gaps)
    sf_q = np.concatenate(sf_qs)
    rel = np.concatenate(rels) if have_rel and rels else None
    print(f"rows: {len(gap)} from {len(paths)} shards"
          + ("" if rel is None else " (with relations)"))

    verr_old = np.abs(_net_q(args.checkpoint_old, x, rel, device=args.device,
                             batch=args.batch_size) - sf_q)
    verr_new = np.abs(_net_q(args.checkpoint_new, x, rel, device=args.device,
                             batch=args.batch_size) - sf_q)

    print(f"\n{'gap bucket':>16} {'n':>7} {'verr_old':>9} {'verr_new':>9} {'delta':>8}")
    buckets = [("bottom half", 0, 50), ("p50-p90", 50, 90), ("top decile", 90, 100)]
    for name, lo, hi in buckets:
        plo, phi = np.percentile(gap, lo), np.percentile(gap, hi)
        # Half-open [plo, phi) so a tied cluster at a percentile edge lands in
        # exactly one bucket; the top bucket is closed on the right. gap is
        # float16 (many ties), so inclusive-both bounds double-count a whole
        # edge cluster into two buckets and bias the top-decile delta — the
        # #104 secondary decision signal (Codex review).
        m = (gap >= plo) & ((gap <= phi) if hi == 100 else (gap < phi))
        n = int(m.sum())
        if n == 0:
            print(f"{name:>16} {0:>7} {'—':>9} {'—':>9} {'—':>8}")
            continue
        d = float(np.mean(verr_old[m]) - np.mean(verr_new[m]))
        print(f"{name:>16} {n:>7} {np.mean(verr_old[m]):>9.4f} "
              f"{np.mean(verr_new[m]):>9.4f} {d:>+8.4f}")
    d_all = float(np.mean(verr_old) - np.mean(verr_new))
    print(f"{'ALL':>16} {len(gap):>7} {np.mean(verr_old):>9.4f} "
          f"{np.mean(verr_new):>9.4f} {d_all:>+8.4f}")
    print("\npositive delta = errors shrank (resolving); the #104 secondary "
          "readout requires the top-decile delta to turn positive.")


if __name__ == "__main__":
    main()
