"""Analyze how well stored ``sf_wdl`` predicts actual game outcomes (``wdl_t``).

Existing shards contain ``sf_wdl`` already converted via SF's UCI_ShowWDL.
This script shows the calibration error across signal buckets so you can
see whether SF is over- or under-confident, and recommend a target
temperature / draw-width adjustment.

Usage:
    PYTHONPATH=. python3 scripts/analyze_sf_wdl_calibration.py [--n-shards N] [--shard-glob ...]

Output is text-only (printed to stdout). Bucketing is on the SF "signal"
score = P(W) - P(L) ∈ [-1, +1].
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import zarr


_DEFAULT_GLOB = (
    "/home/josh/projects/chess/runs/pbt2_small/replay/"
    "train_trial_d3156_00000_0_lr=0.0003_2026-04-29_10-58-04/replay_shards/shard_*.zarr"
)


def signal(p: np.ndarray) -> np.ndarray:
    return p[:, 0] - p[:, 2]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", default=_DEFAULT_GLOB)
    ap.add_argument("--n-shards", type=int, default=50, help="most-recent N shards")
    ap.add_argument("--bins", type=int, default=11)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.shard_glob))[-args.n_shards :]
    print(f"sampling {len(paths)} shards (most recent)")
    if not paths:
        raise SystemExit("no shards matched --shard-glob")

    sf_list, g_list, hsf_list = [], [], []
    for p in paths:
        z = zarr.open(p, mode="r")
        try:
            sf_wdl = np.asarray(z["sf_wdl"][:])  # (B, 3)
            wdl_t = np.asarray(z["wdl_target"][:])
            has_sf = np.asarray(z["has_sf_wdl"][:]).astype(bool)
        except KeyError:
            continue
        sf_list.append(sf_wdl)
        g_list.append(wdl_t)
        hsf_list.append(has_sf)
    if not sf_list:
        raise SystemExit("matched shards did not contain sf_wdl/wdl_target/has_sf_wdl arrays")

    sf = np.concatenate(sf_list)
    g = np.concatenate(g_list)
    hsf = np.concatenate(hsf_list)
    sf = sf[hsf]
    g = g[hsf]

  # Renormalise SF (some rows may have rounding drift).
    s = sf.sum(axis=-1, keepdims=True)
    sf = sf / np.clip(s, 1e-8, None)

    sig = signal(sf)
    n = len(sig)
    print(f"\ntotal positions with sf_wdl: {n:,}")
    one_hot = (sf.max(axis=-1) > 0.99).mean()
    print(f"  share with max(sf_wdl) > 0.99 : {one_hot:.3f}")
    print(f"  mean |signal| (SF confidence) : {np.abs(sig).mean():.3f}")

  # Bucketed calibration: for each bucket of SF signal, what's the empirical
  # outcome distribution + the mean SF prediction in that bucket?
    edges = np.linspace(-1.0, 1.0, args.bins + 1)
    print("\n--- Calibration table (signal bucketed) ---")
    print("  bucket          | n      | sf_pred (W,D,L)            | empirical (W,D,L)         | sf_signal | obs_signal")
    print("  ----------------|--------|----------------------------|---------------------------|-----------|----------")
    for i in range(args.bins):
        lo, hi = edges[i], edges[i + 1]
        m = (sig >= lo) & (sig < hi) if i < args.bins - 1 else (sig >= lo) & (sig <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        sf_mean = sf[m].mean(axis=0)
        obs_w = float((g[m] == 0).mean())
        obs_d = float((g[m] == 1).mean())
        obs_l = float((g[m] == 2).mean())
        sf_sig_mean = float(signal(sf[m]).mean())
        obs_sig = obs_w - obs_l
        print(
            f"  [{lo:+.2f},{hi:+.2f}) | {cnt:6d} | "
            f"({sf_mean[0]:.2f},{sf_mean[1]:.2f},{sf_mean[2]:.2f}) | "
            f"({obs_w:.2f},{obs_d:.2f},{obs_l:.2f}) | "
            f"{sf_sig_mean:+.3f}    | {obs_sig:+.3f}"
        )

  # Aggregate calibration: for the whole sample, how does mean predicted
  # signal compare to mean observed signal?
    sf_total_signal = sig.mean()
    obs_total_signal = ((g == 0).astype(np.float32) - (g == 2).astype(np.float32)).mean()
    print(f"\n  mean SF signal      : {sf_total_signal:+.4f}")
    print(f"  mean observed signal: {obs_total_signal:+.4f}")
    overconf = abs(sf_total_signal) - abs(obs_total_signal)
    print(f"  |SF| - |obs|        : {overconf:+.4f}  (positive = SF more confident than reality)")

  # Implied temperature: what T would shrink SF's confidence to match observed?
  # rough heuristic: |signal_softened| ≈ |signal| * f(T)
  # Use bisection on T ∈ [0.5, 5.0].
    if abs(sf_total_signal) > 1e-6 and abs(obs_total_signal) < abs(sf_total_signal):
        target = abs(obs_total_signal)
        lo, hi = 1.0, 8.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            p_softened = np.maximum(sf, 1e-6) ** (1.0 / mid)
            p_softened = p_softened / p_softened.sum(axis=-1, keepdims=True)
            sig_soft = abs((p_softened[:, 0] - p_softened[:, 2]).mean())
            if sig_soft > target:
                lo = mid
            else:
                hi = mid
        recommended_t = 0.5 * (lo + hi)
        print(f"\n  implied temperature to match observed |signal|: T ≈ {recommended_t:.2f}")
    else:
        print("\n  SF less confident than observed (no temperature softening helps).")


if __name__ == "__main__":
    main()
