#!/usr/bin/env python3
"""Measure how much ``policy_soft_target`` differs from ``policy_target``.

How the two targets are constructed (selfplay/finalize.py:finalize_game):

- ``policy_target`` is ``rec.policy_probs`` — the MCTS visit/improved
  distribution at the move-selection temperature, produced on NETWORK turns
  (selfplay/network_turn.py); Syzygy DTZ overrides may replace it for
  TB-covered positions (``tb_policy_overrides``). It is then mapped to the
  configured policy encoding.
- ``policy_soft_target`` is ``apply_policy_temperature(eff_probs,
  soft_policy_temp)`` applied to the SAME post-override distribution —
  i.e. ``p^(1/T)`` renormalized, with T = 3.0 in production
  (selfplay/temperature.py:apply_policy_temperature).

So the soft target is a deterministic retempering of the hard target, not an
independent signal. Construction does NOT differ between selfplay
(model-vs-model) and curriculum (SF-opponent) games: samples are only emitted
on network turns, and both game types flow through the same ``finalize_game``
path; the ``is_selfplay`` flag tags the game type. The only way the two
targets coincide exactly is when the visit distribution is (near-)one-hot —
``p^(1/3)`` fixes one-hot vectors — so divergence here directly measures how
often search output is still multi-modal.

Streams the most recent shards from a replay dir and reports, per source
type (selfplay vs curriculum) and game phase (piece-count thresholds 13/22,
matching the model's phase buckets):

  * KL(policy_target || policy_soft_target) and reverse KL
  * total-variation distance
  * fraction with TV < 0.01 ("effectively identical")
  * argmax agreement rate

Usage::

    PYTHONPATH=. python3 scripts/probe_policy_targets.py \\
        --replay-dir runs/pbt2_small/tune/<trial>/replay_shards \\
        --positions 200000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chess_anti_engine.utils.architecture import DEFAULT_PHASE_PIECE_THRESHOLDS
from chess_anti_engine.replay.shard import (
    iter_shard_paths,
    load_shard_arrays,
    shard_index,
)

_EPS = 1e-12
_IDENTICAL_TV = 0.01
# Imported, not re-typed. This was the FOURTH copy of the same two numbers
# (utils/architecture, eval/audit, train/losses, here); a probe that silently
# stopped agreeing with the trainer and the audit would be comparing buckets
# that no longer name the same positions.
_PHASE_THRESHOLDS = DEFAULT_PHASE_PIECE_THRESHOLDS
_PHASES = ("endgame", "middlegame", "opening")
_SOURCES = ("selfplay", "curriculum", "unknown")


def _phase_bucket(piece_counts: np.ndarray) -> np.ndarray:
    low, high = _PHASE_THRESHOLDS
    phase = np.ones(piece_counts.shape[0], dtype=np.int64)
    phase[piece_counts <= low] = 0
    phase[piece_counts > high] = 2
    return phase


def _row_metrics(p: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized per-row metrics for (n, P) float32 distributions."""
    p = np.maximum(p.astype(np.float64), 0.0)
    q = np.maximum(q.astype(np.float64), 0.0)
    p_sum = p.sum(axis=1, keepdims=True)
    q_sum = q.sum(axis=1, keepdims=True)
    valid = (p_sum[:, 0] > 0) & (q_sum[:, 0] > 0)
    p = p / np.maximum(p_sum, _EPS)
    q = q / np.maximum(q_sum, _EPS)

    tv = 0.5 * np.abs(p - q).sum(axis=1)
    kl_pq = np.where(p > 0, p * (np.log(p + _EPS) - np.log(q + _EPS)), 0.0).sum(axis=1)
    kl_qp = np.where(q > 0, q * (np.log(q + _EPS) - np.log(p + _EPS)), 0.0).sum(axis=1)
    argmax_agree = (p.argmax(axis=1) == q.argmax(axis=1)).astype(np.float64)
    return {
        "tv": tv, "kl_pq": kl_pq, "kl_qp": kl_qp,
        "argmax_agree": argmax_agree, "valid": valid,
    }


def _collect(
    replay_dir: Path, positions: int, chunk_rows: int = 4096,
) -> dict[str, np.ndarray]:
    paths = sorted(iter_shard_paths(replay_dir), key=shard_index, reverse=True)
    if not paths:
        raise SystemExit(f"no shards found under {replay_dir}")

    cols: dict[str, list[np.ndarray]] = {
        k: [] for k in ("tv", "kl_pq", "kl_qp", "argmax_agree", "source", "phase")
    }
    seen = 0
    used_shards = 0
    for path in paths:
        if seen >= positions:
            break
        arrs, _meta = load_shard_arrays(path, lazy=True)
        if "policy_soft_target" not in arrs or "has_policy_soft" not in arrs:
            continue
        has_soft = np.asarray(arrs["has_policy_soft"]).astype(bool)
        if not has_soft.any():
            continue
        used_shards += 1
        # .shape on the lazy zarr array is metadata-only; np.asarray here would
        # materialize the full x array just to read its length.
        n = int(arrs["x"].shape[0])
        for start in range(0, n, chunk_rows):
            stop = min(n, start + chunk_rows)
            sel = has_soft[start:stop]
            if not sel.any():
                continue
            p = np.asarray(arrs["policy_target"][start:stop], dtype=np.float32)[sel]
            q = np.asarray(arrs["policy_soft_target"][start:stop], dtype=np.float32)[sel]
            x = np.asarray(arrs["x"][start:stop], dtype=np.float32)[sel]
            metrics = _row_metrics(p, q)
            valid = metrics["valid"]
            if not valid.any():
                continue

            piece_counts = np.rint(x[:, :12].sum(axis=(1, 2, 3)))
            phase = _phase_bucket(piece_counts)[valid]

            source = np.full(p.shape[0], 2, dtype=np.int64)  # unknown
            if "is_selfplay" in arrs and "has_is_selfplay" in arrs:
                has_src = np.asarray(arrs["has_is_selfplay"][start:stop]).astype(bool)[sel]
                is_sp = np.asarray(arrs["is_selfplay"][start:stop]).astype(bool)[sel]
                source[has_src & is_sp] = 0       # selfplay
                source[has_src & ~is_sp] = 1      # curriculum (SF opponent)
            source = source[valid]

            for key in ("tv", "kl_pq", "kl_qp", "argmax_agree"):
                cols[key].append(metrics[key][valid])
            cols["source"].append(source)
            cols["phase"].append(phase)
            seen += int(valid.sum())
            if seen >= positions:
                break

    if seen == 0:
        raise SystemExit(
            f"no samples with policy_soft_target in the newest shards of {replay_dir}"
        )
    out = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    out["_used_shards"] = np.array(used_shards)
    return out


def _summarize(mask: np.ndarray, data: dict[str, np.ndarray]) -> dict | None:
    n = int(mask.sum())
    if n == 0:
        return None
    tv = data["tv"][mask]
    return {
        "n": n,
        "tv_mean": float(tv.mean()),
        "tv_median": float(np.median(tv)),
        "tv_p90": float(np.quantile(tv, 0.90)),
        "kl_mean": float(data["kl_pq"][mask].mean()),
        "rev_kl_mean": float(data["kl_qp"][mask].mean()),
        "frac_identical_tv_lt_0.01": float((tv < _IDENTICAL_TV).mean()),
        "frac_tv_gt_0.1": float((tv > 0.1).mean()),
        "argmax_agreement": float(data["argmax_agree"][mask].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--replay-dir", type=Path, required=True,
                    help="replay shard dir (e.g. runs/pbt2_small/tune/<trial>/replay_shards)")
    ap.add_argument("--positions", type=int, default=200_000,
                    help="most-recent positions to stream (default: 200k)")
    ap.add_argument("--out", type=Path, default=Path("runs/probe_policy_targets.json"))
    args = ap.parse_args()

    data = _collect(args.replay_dir, int(args.positions))
    n_total = int(data["tv"].shape[0])

    groups: dict[str, dict] = {}
    overall = _summarize(np.ones(n_total, dtype=bool), data)
    assert overall is not None
    groups["overall"] = overall
    for si, source in enumerate(_SOURCES):
        m = data["source"] == si
        got = _summarize(m, data)
        if got is not None:
            groups[source] = got
        for pi, phase in enumerate(_PHASES):
            got = _summarize(m & (data["phase"] == pi), data)
            if got is not None:
                groups[f"{source}/{phase}"] = got

    header = (
        f"{'group':28s} {'n':>8s} {'TV mean':>8s} {'TV med':>8s} {'TV p90':>8s} "
        f"{'KL':>8s} {'revKL':>8s} {'TV<.01':>7s} {'TV>.1':>7s} {'argmax=':>8s}"
    )
    print(f"[probe] replay dir: {args.replay_dir}  "
          f"(streamed {n_total} samples from {int(data['_used_shards'])} newest shards)")
    print(header)
    print("-" * len(header))
    for name, g in groups.items():
        print(
            f"{name:28s} {g['n']:>8d} {g['tv_mean']:>8.4f} {g['tv_median']:>8.4f} "
            f"{g['tv_p90']:>8.4f} {g['kl_mean']:>8.4f} {g['rev_kl_mean']:>8.4f} "
            f"{g['frac_identical_tv_lt_0.01']:>7.1%} {g['frac_tv_gt_0.1']:>7.1%} "
            f"{g['argmax_agreement']:>8.1%}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "replay_dir": str(args.replay_dir),
        "positions": n_total,
        "shards_used": int(data["_used_shards"]),
        "identical_tv_threshold": _IDENTICAL_TV,
        "groups": groups,
        "argv": sys.argv,
    }
    args.out.write_text(json.dumps(record, indent=2))
    print(f"\n[probe] JSON written to {args.out}")


if __name__ == "__main__":
    main()
