#!/usr/bin/env python3
"""Fit future-eval blend weights after game and eval-bucket averaging."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.replay.shard import load_shard_arrays, shard_positions
from scripts.diagnostic_replay_utils import (
    bool_field as _bool_field,
    final_q_from_wdl_target as _final_q_from_wdl_target,
    fit_simplex as _fit_simplex,
    latest_replay_dir as _latest_replay_dir,
    normalize_wdl as _normalize_wdl,
    record_skipped_shard as _record_skipped_shard,
    rmse as _rmse,
    select_shards as _select_shards,
)


DEFAULT_RUN_DIR = Path("runs/pbt2_small")
DEFAULT_EVAL_BINS = (-0.75, -0.50, -0.25, -0.10, -0.03, 0.03, 0.10, 0.25, 0.50, 0.75)
MAX_HORIZONS = 12
MAX_EVAL_BINS = 25


@dataclass(frozen=True, slots=True)
class Sample:
    game_id: int
    ply: int
    sf_q: float
    search_q: float
    final_q: float
    sf_played_regret: float
    has_sf_played_regret: bool
    is_selfplay: bool

    @property
    def avg_q(self) -> float:
        return 0.5 * (self.sf_q + self.search_q)


@dataclass(frozen=True, slots=True)
class FitRow:
    horizon: int
    mode: str
    n_train: int
    n_test: int
    w_sf: float
    w_search: float
    w_final: float
    rmse_test: float
    weighted_rmse_test: float
    target_mean_test: float


def _load_samples(
    replay_dir: Path,
    *,
    max_shards: int,
) -> tuple[dict[int, list[Sample]], dict[str, Any], float]:
    games: dict[int, list[Sample]] = defaultdict(list)
    observed_regrets: list[float] = []
    scan: dict[str, Any] = {
        "selected_shards": 0,
        "scanned_positions": 0,
        "valid_samples": 0,
        "samples_with_regret": 0,
        "selfplay_samples": 0,
        "selfplay_missing_regret": 0,
        "nonselfplay_missing_regret": 0,
        "skipped_shards": [],
        "skipped_shards_omitted": 0,
    }
    for shard in _select_shards(replay_dir, max_shards):
        n = int(shard_positions(shard))
        if n <= 0:
            continue
        scan["selected_shards"] += 1
        scan["scanned_positions"] += n
        try:
            arrs, _meta = load_shard_arrays(shard, lazy=True)
        except Exception as exc:  # noqa: BLE001
            # Replay scans should continue past partially written or corrupt live shards.
            _record_skipped_shard(scan, shard, exc)
            continue
        required = ("game_id", "ply_index", "sf_wdl", "search_wdl", "wdl_target")
        flags = ("has_game_id", "has_ply_index", "has_sf_wdl", "has_search_wdl")
        if any(name not in arrs for name in (*required, *flags)):
            continue

        sf_wdl, valid_sf = _normalize_wdl(np.asarray(arrs["sf_wdl"], dtype=np.float64))
        search_wdl, valid_search = _normalize_wdl(np.asarray(arrs["search_wdl"], dtype=np.float64))
        valid = (
            _bool_field(arrs, "has_game_id", n)
            & _bool_field(arrs, "has_ply_index", n)
            & _bool_field(arrs, "has_sf_wdl", n)
            & _bool_field(arrs, "has_search_wdl", n)
            & valid_sf
            & valid_search
        )
        rows = np.flatnonzero(valid)
        if rows.size == 0:
            continue

        game_ids = np.asarray(arrs["game_id"], dtype=np.int64)
        plies = np.asarray(arrs["ply_index"], dtype=np.int64)
        sf_q = sf_wdl[:, 0] - sf_wdl[:, 2]
        search_q = search_wdl[:, 0] - search_wdl[:, 2]
        final_q = _final_q_from_wdl_target(np.asarray(arrs["wdl_target"]))
        has_regret = _bool_field(arrs, "has_sf_played_regret", n)
        regret = np.asarray(arrs.get("sf_played_regret", np.full((n,), np.nan)), dtype=np.float64)
        is_selfplay = _bool_field(arrs, "has_is_selfplay", n) & _bool_field(arrs, "is_selfplay", n)

        scan["valid_samples"] += int(rows.size)
        scan["samples_with_regret"] += int(has_regret[rows].sum())
        scan["selfplay_samples"] += int(is_selfplay[rows].sum())
        scan["selfplay_missing_regret"] += int((is_selfplay[rows] & ~has_regret[rows]).sum())
        scan["nonselfplay_missing_regret"] += int((~is_selfplay[rows] & ~has_regret[rows]).sum())
        observed_regrets.extend(np.maximum(0.0, regret[rows][has_regret[rows]]).tolist())

        for row_i in rows:
            row = int(row_i)
            games[int(game_ids[row])].append(
                Sample(
                    game_id=int(game_ids[row]),
                    ply=int(plies[row]),
                    sf_q=float(sf_q[row]),
                    search_q=float(search_q[row]),
                    final_q=float(final_q[row]),
                    sf_played_regret=float(regret[row]) if bool(has_regret[row]) else math.nan,
                    has_sf_played_regret=bool(has_regret[row]),
                    is_selfplay=bool(is_selfplay[row]),
                ),
            )
    for samples in games.values():
        samples.sort(key=lambda s: s.ply)
    scan["games"] = len(games)
    mean_observed_regret = float(np.mean(observed_regrets)) if observed_regrets else 0.0
    scan["mean_observed_regret"] = mean_observed_regret
    scan["p90_observed_regret"] = float(np.percentile(observed_regrets, 90)) if observed_regrets else 0.0
    return games, scan, mean_observed_regret


def _bucket_ids(mid_q: np.ndarray, bins: list[float]) -> np.ndarray:
    bins_arr = np.asarray(bins, dtype=np.float64)
    bins_arr = bins_arr[np.isfinite(bins_arr)]
    return np.digitize(mid_q, bins_arr).astype(np.int16)


def _equal_bucket_weights(bucket: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(bucket, return_counts=True)
    by_bucket = {int(b): int(c) for b, c in zip(unique, counts, strict=True)}
    return np.asarray([1.0 / float(by_bucket[int(b)]) for b in bucket], dtype=np.float64)


def _group_mean(
    features: np.ndarray,
    target: np.ndarray,
    game_ids: np.ndarray,
    bucket: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if bucket is None:
        keys = game_ids.astype(np.int64, copy=False)
        unique, inv = np.unique(keys, return_inverse=True)
        out_bucket = np.zeros((unique.size,), dtype=np.int16)
        out_game = unique.astype(np.int64, copy=False)
    else:
        key = np.empty((game_ids.shape[0],), dtype=[("game", np.int64), ("bucket", np.int16)])
        key["game"] = game_ids
        key["bucket"] = bucket
        unique, inv = np.unique(key, return_inverse=True)
        out_game = unique["game"].astype(np.int64, copy=False)
        out_bucket = unique["bucket"].astype(np.int16, copy=False)
    n = int(inv.max()) + 1 if inv.size else 0
    counts = np.zeros((n,), dtype=np.float64)
    out_features = np.zeros((n, features.shape[1]), dtype=np.float64)
    out_target = np.zeros((n,), dtype=np.float64)
    np.add.at(counts, inv, 1.0)
    for col in range(features.shape[1]):
        np.add.at(out_features[:, col], inv, features[:, col])
    np.add.at(out_target, inv, target)
    out_features /= counts[:, None]
    out_target /= counts
    return out_features, out_target, out_game, out_bucket


def _fit_row(
    *,
    horizon: int,
    mode: str,
    features: np.ndarray,
    target: np.ndarray,
    game_ids: np.ndarray,
    fit_weights: np.ndarray | None = None,
    eval_weights: np.ndarray | None = None,
) -> FitRow | None:
    if target.size < 20:
        return None
    test = (np.asarray(game_ids, dtype=np.int64) % 5) == 0
    train = ~test
    if int(test.sum()) < 20 or int(train.sum()) < 20:
        return None
    train_weights = None if fit_weights is None else fit_weights[train]
    weights = _fit_simplex(features[train], target[train], train_weights)
    pred = features @ weights
    test_eval_weights = None if eval_weights is None else eval_weights[test]
    return FitRow(
        horizon=int(horizon),
        mode=mode,
        n_train=int(train.sum()),
        n_test=int(test.sum()),
        w_sf=float(weights[0]),
        w_search=float(weights[1]),
        w_final=float(weights[2]),
        rmse_test=_rmse(pred[test], target[test]),
        weighted_rmse_test=_rmse(pred[test], target[test], test_eval_weights),
        target_mean_test=float(np.mean(target[test])),
    )


def _pairs_for_horizon(
    games: dict[int, list[Sample]],
    *,
    horizon: int,
    missing_regret_impute: float,
    eval_bins: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    game_ids: list[int] = []
    features: list[tuple[float, float, float]] = []
    targets: list[float] = []
    mid_q: list[float] = []
    path_regret: list[float] = []
    for samples in games.values():
        by_ply = {s.ply: s for s in samples}
        game_selfplay = sum(1 for s in samples if s.is_selfplay) > (len(samples) // 2)
        for current in samples:
            future = by_ply.get(current.ply + horizon)
            if future is None:
                continue
            regret_sum = 0.0
            for offset in range(0, int(horizon), 2):
                path_sample = by_ply.get(current.ply + offset)
                if path_sample is None:
                    if not game_selfplay:
                        regret_sum += missing_regret_impute
                    continue
                if path_sample.has_sf_played_regret:
                    regret_sum += max(0.0, float(path_sample.sf_played_regret))
                elif not path_sample.is_selfplay:
                    regret_sum += missing_regret_impute
            raw_target = future.avg_q
            target = float(np.clip(raw_target - 2.0 * regret_sum, -1.0, 1.0))
            game_ids.append(current.game_id)
            features.append((current.sf_q, current.search_q, current.final_q))
            targets.append(target)
            mid_q.append(current.avg_q)
            path_regret.append(regret_sum)
    return (
        np.asarray(game_ids, dtype=np.int64),
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        _bucket_ids(np.asarray(mid_q, dtype=np.float64), eval_bins),
        np.asarray(path_regret, dtype=np.float64),
    )


def _fit_modes(
    *,
    horizon: int,
    game_ids: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    bucket: np.ndarray,
    near_even: float,
) -> list[FitRow]:
    rows: list[FitRow] = []
    candidates: list[FitRow | None] = []
    candidates.append(
        _fit_row(
            horizon=horizon,
            mode="row_all",
            features=features,
            target=target,
            game_ids=game_ids,
        ),
    )
    bucket_weights = _equal_bucket_weights(bucket)
    candidates.append(
        _fit_row(
            horizon=horizon,
            mode="row_equal_eval_bucket",
            features=features,
            target=target,
            game_ids=game_ids,
            fit_weights=bucket_weights,
            eval_weights=bucket_weights,
        ),
    )
    game_features, game_target, game_ids2, _game_bucket = _group_mean(features, target, game_ids)
    candidates.append(
        _fit_row(
            horizon=horizon,
            mode="game_mean",
            features=game_features,
            target=game_target,
            game_ids=game_ids2,
        ),
    )
    gb_features, gb_target, gb_game_ids, gb_bucket = _group_mean(features, target, game_ids, bucket)
    gb_weights = _equal_bucket_weights(gb_bucket)
    candidates.append(
        _fit_row(
            horizon=horizon,
            mode="game_eval_bucket_mean",
            features=gb_features,
            target=gb_target,
            game_ids=gb_game_ids,
        ),
    )
    candidates.append(
        _fit_row(
            horizon=horizon,
            mode="game_eval_bucket_equal",
            features=gb_features,
            target=gb_target,
            game_ids=gb_game_ids,
            fit_weights=gb_weights,
            eval_weights=gb_weights,
        ),
    )
    mid = 0.5 * (features[:, 0] + features[:, 1])
    near = np.abs(mid) <= float(near_even)
    if int(near.sum()) >= 20:
        candidates.append(
            _fit_row(
                horizon=horizon,
                mode=f"near_even_abs<={near_even:.2f}",
                features=features[near],
                target=target[near],
                game_ids=game_ids[near],
            ),
        )
    rows.extend(row for row in candidates if row is not None)
    return rows


def _parse_horizons(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("horizons must be positive comma-separated integers")
    if any(value % 2 != 0 for value in values):
        raise argparse.ArgumentTypeError("horizons must be even ply offsets")
    if len(values) > MAX_HORIZONS:
        raise argparse.ArgumentTypeError(f"at most {MAX_HORIZONS} horizons are allowed")
    return values


def _parse_eval_bins(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) > MAX_EVAL_BINS:
        raise argparse.ArgumentTypeError(f"at most {MAX_EVAL_BINS} eval bins are allowed")
    if any(not math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("eval bins must be finite numbers")
    if any(right <= left for left, right in zip(values, values[1:], strict=False)):
        raise argparse.ArgumentTypeError("eval bins must be strictly increasing")
    return values


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:.4f}"
    return str(value)


def _print_table(rows: list[FitRow]) -> None:
    cols = ["horizon", "mode", "n_test", "w_sf", "w_search", "w_final", "rmse_test", "weighted_rmse_test"]
    widths: dict[str, int] = {}
    for col in cols:
        width_candidates = [len(col)]
        width_candidates.extend(len(_format_cell(getattr(row, col))) for row in rows)
        widths[col] = max(width_candidates)
    print(" ".join(col.rjust(widths[col]) for col in cols))
    print(" ".join("-" * widths[col] for col in cols))
    for row in rows:
        print(" ".join(_format_cell(getattr(row, col)).rjust(widths[col]) for col in cols))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit future-eval weights after game/eval-bucket averaging.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--max-shards", type=int, default=0, help="newest shards to scan; 0 scans all")
    parser.add_argument("--horizons", type=_parse_horizons, default="2,4,6,8,10,12,16,20,24")
    parser.add_argument(
        "--eval-bins",
        type=_parse_eval_bins,
        default=",".join(str(value) for value in DEFAULT_EVAL_BINS),
        help=f"strictly increasing comma-separated eval buckets, capped at {MAX_EVAL_BINS} cut points",
    )
    parser.add_argument("--near-even", type=float, default=0.10)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    replay_dir = args.replay_dir or _latest_replay_dir(args.run_dir)
    games, scan, mean_observed_regret = _load_samples(replay_dir, max_shards=int(args.max_shards))
    rows: list[FitRow] = []
    pair_counts: dict[int, int] = {}
    path_regret_mean: dict[int, float] = {}
    for horizon in args.horizons:
        game_ids, features, target, bucket, path_regret = _pairs_for_horizon(
            games,
            horizon=int(horizon),
            missing_regret_impute=mean_observed_regret,
            eval_bins=[float(value) for value in args.eval_bins],
        )
        pair_counts[int(horizon)] = int(target.size)
        path_regret_mean[int(horizon)] = float(np.mean(path_regret)) if path_regret.size else math.nan
        if target.size == 0:
            continue
        rows.extend(
            _fit_modes(
                horizon=int(horizon),
                game_ids=game_ids,
                features=features,
                target=target,
                bucket=bucket,
                near_even=float(args.near_even),
            ),
        )

    payload: dict[str, Any] = {
        "replay_dir": str(replay_dir),
        "scan": scan,
        "horizons": [int(h) for h in args.horizons],
        "eval_bins": [float(value) for value in args.eval_bins],
        "pair_counts": pair_counts,
        "path_regret_mean": path_regret_mean,
        "rows": [asdict(row) for row in rows],
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"replay_dir={replay_dir}")
    print(f"scan={json.dumps(scan, sort_keys=True)}")
    print(f"pair_counts={json.dumps(pair_counts, sort_keys=True)}")
    print(f"path_regret_mean={json.dumps(path_regret_mean, sort_keys=True)}")
    _print_table(rows)


if __name__ == "__main__":
    main()
