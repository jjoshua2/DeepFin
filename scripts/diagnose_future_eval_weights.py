#!/usr/bin/env python3
"""Fit future-eval blend weights from replay SF/search labels."""
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
    corr as _corr,
    latest_replay_dir as _latest_replay_dir,
    normalize_wdl as _normalize_wdl,
    optional_float_array_with_mask as _optional_float_field,
    record_skipped_shard as _record_skipped_shard,
    rmse as _rmse,
    select_shards as _select_shards,
)


DEFAULT_RUN_DIR = Path("runs/pbt2_small")
MAX_HORIZONS = 12


@dataclass(frozen=True, slots=True)
class Sample:
    game_id: int
    ply: int
    sf_q: float
    search_q: float
    sf_played_regret: float
    has_sf_played_regret: bool

    @property
    def avg_q(self) -> float:
        return 0.5 * (self.sf_q + self.search_q)

    @property
    def gap(self) -> float:
        return self.search_q - self.sf_q


@dataclass(frozen=True, slots=True)
class FitRow:
    subset: str
    target: str
    horizon: int
    n: int
    w_unclamped: float
    w_clamped: float
    rmse_sf: float
    rmse_search: float
    rmse_half: float
    rmse_opt: float
    mae_opt: float
    target_mean: float
    sf_corr: float
    search_corr: float
    path_regret_mean: float = math.nan
    path_regret_p90: float = math.nan


def _load_samples(replay_dir: Path, max_shards: int) -> tuple[dict[int, list[Sample]], dict[str, Any]]:
    games: dict[int, list[Sample]] = defaultdict(list)
    scan: dict[str, Any] = {
        "selected_shards": 0,
        "scanned_positions": 0,
        "valid_samples": 0,
        "samples_with_regret": 0,
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
            # Replay diagnostics should summarize bad live shards and continue scanning.
            _record_skipped_shard(scan, shard, exc)
            continue
        required = ("game_id", "ply_index", "sf_wdl", "search_wdl")
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
        sf_played_regret, has_sf_played_regret = _optional_float_field(
            arrs,
            "sf_played_regret",
            "has_sf_played_regret",
            n,
        )
        scan["valid_samples"] += int(rows.size)
        scan["samples_with_regret"] += int(has_sf_played_regret[rows].sum())
        for row in rows:
            games[int(game_ids[row])].append(
                Sample(
                    game_id=int(game_ids[row]),
                    ply=int(plies[row]),
                    sf_q=float(sf_q[row]),
                    search_q=float(search_q[row]),
                    sf_played_regret=float(sf_played_regret[row])
                    if bool(has_sf_played_regret[row])
                    else math.nan,
                    has_sf_played_regret=bool(has_sf_played_regret[row]),
                ),
            )
    for samples in games.values():
        samples.sort(key=lambda s: s.ply)
    scan["games"] = len(games)
    return games, scan


def _future_pairs(
    games: dict[int, list[Sample]],
    *,
    horizon: int,
    require_known_path_regret: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sf_now: list[float] = []
    search_now: list[float] = []
    sf_future: list[float] = []
    search_future: list[float] = []
    path_regret: list[float] = []
    for samples in games.values():
        by_ply = {s.ply: s for s in samples}
        for current in samples:
            future = by_ply.get(current.ply + horizon)
            if future is None:
                continue
            regret_sum = 0.0
            if require_known_path_regret:
                path_known = True
                for offset in range(0, int(horizon), 2):
                    path_sample = by_ply.get(current.ply + offset)
                    if path_sample is None or not path_sample.has_sf_played_regret:
                        path_known = False
                        break
                    regret_sum += max(0.0, float(path_sample.sf_played_regret))
                if not path_known:
                    continue
            sf_now.append(current.sf_q)
            search_now.append(current.search_q)
            sf_future.append(future.sf_q)
            search_future.append(future.search_q)
            path_regret.append(regret_sum if require_known_path_regret else math.nan)
    return (
        np.asarray(sf_now, dtype=np.float64),
        np.asarray(search_now, dtype=np.float64),
        np.asarray(sf_future, dtype=np.float64),
        np.asarray(search_future, dtype=np.float64),
        np.asarray(path_regret, dtype=np.float64),
    )


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    if target.size == 0:
        return math.nan
    return float(np.mean(np.abs(pred - target)))


def _fit_weight(
    *,
    subset: str,
    target_name: str,
    horizon: int,
    sf: np.ndarray,
    search: np.ndarray,
    target: np.ndarray,
    path_regret: np.ndarray | None = None,
) -> FitRow:
    delta = search - sf
    denom = float(np.dot(delta, delta))
    if denom <= 1e-12:
        w_unclamped = 0.0
    else:
        w_unclamped = float(np.dot(target - sf, delta) / denom)
    w_clamped = min(1.0, max(0.0, w_unclamped))
    pred_half = 0.5 * (sf + search)
    pred_opt = sf + w_clamped * delta
    return FitRow(
        subset=subset,
        target=target_name,
        horizon=int(horizon),
        n=int(target.size),
        w_unclamped=w_unclamped,
        w_clamped=w_clamped,
        rmse_sf=_rmse(sf, target),
        rmse_search=_rmse(search, target),
        rmse_half=_rmse(pred_half, target),
        rmse_opt=_rmse(pred_opt, target),
        mae_opt=_mae(pred_opt, target),
        target_mean=float(np.mean(target)) if target.size else math.nan,
        sf_corr=_corr(sf, target),
        search_corr=_corr(search, target),
        path_regret_mean=(
            math.nan if path_regret is None or path_regret.size == 0 else float(np.mean(path_regret))
        ),
        path_regret_p90=(
            math.nan if path_regret is None or path_regret.size == 0 else float(np.quantile(path_regret, 0.90))
        ),
    )


def _rows_for_horizon(
    games: dict[int, list[Sample]],
    *,
    horizon: int,
    gap_threshold: float,
    include_regret_adjusted: bool,
) -> list[FitRow]:
    sf, search, future_sf, future_search, _path_regret = _future_pairs(games, horizon=horizon)
    future_avg = 0.5 * (future_sf + future_search)
    targets = {
        "future_avg_q": future_avg,
        "future_sf_q": future_sf,
        "future_search_q": future_search,
    }
    rows: list[FitRow] = []
    subsets = {
        "all": np.ones((sf.size,), dtype=bool),
        f"gap>={gap_threshold:.3f}": np.abs(search - sf) >= float(gap_threshold),
    }
    for subset_name, mask in subsets.items():
        if not mask.any():
            continue
        for target_name, target in targets.items():
            rows.append(
                _fit_weight(
                    subset=subset_name,
                    target_name=target_name,
                    horizon=horizon,
                    sf=sf[mask],
                    search=search[mask],
                    target=target[mask],
                ),
            )
    if include_regret_adjusted:
        adj_sf, adj_search, adj_future_sf, adj_future_search, path_regret = _future_pairs(
            games,
            horizon=horizon,
            require_known_path_regret=True,
        )
        if path_regret.size > 0:
            adj_future_avg = 0.5 * (adj_future_sf + adj_future_search)
            adjusted = np.clip(adj_future_avg - 2.0 * path_regret, -1.0, 1.0)
            adj_targets = {
                "future_avg_q_known_regret_raw": adj_future_avg,
                "future_avg_q_regret_adjusted": adjusted,
            }
            adj_subsets = {
                "all_known_regret": np.ones((adj_sf.size,), dtype=bool),
                f"known_regret_gap>={gap_threshold:.3f}": (
                    np.abs(adj_search - adj_sf) >= float(gap_threshold)
                ),
            }
            for subset_name, mask in adj_subsets.items():
                if not mask.any():
                    continue
                for target_name, target in adj_targets.items():
                    rows.append(
                        _fit_weight(
                            subset=subset_name,
                            target_name=target_name,
                            horizon=horizon,
                            sf=adj_sf[mask],
                            search=adj_search[mask],
                            target=target[mask],
                            path_regret=path_regret[mask],
                        ),
                    )
    return rows


def _fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _print_report(report: dict[str, Any], rows: list[FitRow]) -> None:
    scan = report["scan"]
    print("# Future Eval Blend Weights")
    print(f"- replay dir: `{report['replay_dir']}`")
    print(
        "- scanned: "
        f"{scan['selected_shards']} shards, {scan['scanned_positions']} rows, "
        f"{scan['valid_samples']} valid samples, {scan['games']} games, "
        f"{scan['samples_with_regret']} samples with regret"
    )
    print("- model: `pred = (1-w)*current_sf_q + w*current_search_q`")
    print("- adjusted target: `future_avg_q - 2 * sum(sf_played_regret before target)`")
    print(f"- horizons are ply offsets: `{report['horizons']}`")
    print()
    print(
        "| subset | target | horizon | n | w* | RMSE sf/search/50/opt | "
        "corr sf/search | target mean | path regret mean/p90 |"
    )
    print("|---|---|---:|---:|---:|---|---|---:|---:|")
    for row in rows:
        print(
            f"| {row.subset} | {row.target} | {row.horizon} | {row.n} | "
            f"{_fmt(row.w_clamped, 3)} | "
            f"`{_fmt(row.rmse_sf)}/{_fmt(row.rmse_search)}/{_fmt(row.rmse_half)}/{_fmt(row.rmse_opt)}` | "
            f"`{_fmt(row.sf_corr)}/{_fmt(row.search_corr)}` | {_fmt(row.target_mean)} | "
            f"`{_fmt(row.path_regret_mean)}/{_fmt(row.path_regret_p90)}` |"
        )


def _parse_horizons(raw: str) -> list[int]:
    out = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("at least one horizon is required")
    if any(h <= 0 for h in out):
        raise argparse.ArgumentTypeError("horizons must be positive")
    if any(h % 2 != 0 for h in out):
        raise argparse.ArgumentTypeError("horizons must be even ply offsets")
    if len(out) > MAX_HORIZONS:
        raise argparse.ArgumentTypeError(f"at most {MAX_HORIZONS} horizons are allowed")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--max-shards", type=int, default=256, help="newest shards to scan; 0 scans all")
    parser.add_argument("--horizons", type=_parse_horizons, default="2,4,6")
    parser.add_argument("--gap-threshold", type=float, default=0.15)
    parser.add_argument(
        "--include-regret-adjusted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also fit future_avg_q with cumulative Stockfish regret subtracted",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    replay_dir = args.replay_dir or _latest_replay_dir(args.run_dir)
    games, scan = _load_samples(replay_dir, int(args.max_shards))
    rows: list[FitRow] = []
    for horizon in args.horizons:
        rows.extend(
            _rows_for_horizon(
                games,
                horizon=int(horizon),
                gap_threshold=float(args.gap_threshold),
                include_regret_adjusted=bool(args.include_regret_adjusted),
            ),
        )
    report: dict[str, Any] = {
        "replay_dir": str(replay_dir),
        "max_shards": int(args.max_shards),
        "horizons": [int(h) for h in args.horizons],
        "gap_threshold": float(args.gap_threshold),
        "include_regret_adjusted": bool(args.include_regret_adjusted),
        "scan": scan,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(report)
        payload["rows"] = [asdict(row) for row in rows]
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    _print_report(report, rows)


if __name__ == "__main__":
    main()
