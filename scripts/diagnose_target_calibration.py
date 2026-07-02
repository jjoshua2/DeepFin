#!/usr/bin/env python3
"""Report replay target calibration and policy sharpness over recent positions."""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.replay.shard import load_shard_arrays, shard_positions
from scripts.diagnostic_replay_utils import (
    FUTURE_REGRET_FIELDS,
    adjusted_wdl_game_target as _adjusted_wdl_game_target,
    future_regret_field_names as _future_regret_field_names,
    record_skipped_shard as _record_skipped_shard,
    wdl_one_hot as _wdl_one_hot,
)
import contextlib


DEFAULT_RUN = Path("runs/pbt2_small")
MAX_BUCKETS = 20
WDL_BLEND_MODES = ("interpolate", "renormalize")
WDL_BLEND_FALLBACKS = ("adjusted_game", "raw_game")
WDL_BLEND_COUNTERFACTUALS = (
    ("interpolate_adjusted", "interpolate", "adjusted_game"),
    ("interpolate_raw", "interpolate", "raw_game"),
    ("renormalize", "renormalize", "raw_game"),
)


@dataclass(frozen=True)
class ShardSlice:
    path: Path
    start: int
    take: int


def _latest_trial_dir(run_dir: Path) -> Path:
    trials = sorted((run_dir / "tune").glob("train_trial_*"), key=lambda p: p.stat().st_mtime)
    if not trials:
        raise FileNotFoundError(f"No Ray trial directories under {run_dir / 'tune'}")
    return trials[-1]


def _try_latest_trial_dir(run_dir: Path) -> Path | None:
    try:
        return _latest_trial_dir(run_dir)
    except FileNotFoundError:
        return None


def _replay_dir_for_trial(run_dir: Path, trial_dir: Path) -> Path:
    replay_dir = run_dir / "replay" / trial_dir.name / "replay_shards"
    if not replay_dir.exists():
        raise FileNotFoundError(f"Replay shard directory does not exist: {replay_dir}")
    return replay_dir


def _latest_result(trial_dir: Path) -> dict[str, Any]:
    result_path = trial_dir / "result.json"
    latest: dict[str, Any] = {}
    if not result_path.exists():
        return latest
    with result_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            with contextlib.suppress(json.JSONDecodeError):
                latest = json.loads(line)
    return latest


def _shard_num(path: Path) -> int:
    match = re.search(r"shard_(\d+)\.zarr$", path.name)
    return int(match.group(1)) if match else -1


def _newest_window_slices(replay_dir: Path, max_positions: int, max_shards: int = 0) -> list[ShardSlice]:
    total = 0
    out: list[ShardSlice] = []
    for shard in sorted(replay_dir.glob("shard_*.zarr"), key=_shard_num, reverse=True):
        if int(max_shards) > 0 and len(out) >= int(max_shards):
            break
        n = int(shard_positions(shard))
        if n <= 0:
            continue
        take = min(n, max(0, int(max_positions) - total))
        if take <= 0:
            break
        start = n - take if take < n else 0
        out.append(ShardSlice(shard, start, take))
        total += take
        if total >= int(max_positions):
            break
    return list(reversed(out))


def _normalize(p: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(p, dtype=np.float64)
    arr = np.clip(arr, 1e-9, None)
    if abs(float(temperature) - 1.0) > 1e-12:
        arr = arr ** (1.0 / max(float(temperature), 1e-6))
    return arr / np.maximum(arr.sum(axis=-1, keepdims=True), 1e-12)


def _one_hot(outcome: np.ndarray) -> np.ndarray:
    return _wdl_one_hot(outcome)


def _ce_to_outcome(p: np.ndarray, outcome: np.ndarray) -> float:
    if outcome.size == 0:
        return math.nan
    return float((-np.log(np.clip(p[np.arange(outcome.size), outcome], 1e-9, 1.0))).mean())


def _brier_to_outcome(p: np.ndarray, outcome: np.ndarray) -> float:
    if outcome.size == 0:
        return math.nan
    oh = _one_hot(outcome)
    return float(((p - oh) ** 2).sum(axis=1).mean())


def _signal(p: np.ndarray) -> np.ndarray:
    return p[:, 0] - p[:, 2]


def _blend_wdl(
    game_target: np.ndarray,
    sf: np.ndarray,
    search: np.ndarray,
    *,
    raw_game_target: np.ndarray | None = None,
    fallback_target: np.ndarray | None = None,
    sf_available: np.ndarray | None = None,
    search_available: np.ndarray | None = None,
    sf_frac: float,
    search_frac: float,
    dampen_sf_low: float,
    dampen_sf_high: float,
    blend_mode: str = "interpolate",
) -> np.ndarray:
    mode = str(blend_mode)
    if mode not in WDL_BLEND_MODES:
        raise ValueError(f"unknown WDL blend mode {blend_mode!r}; expected one of {WDL_BLEND_MODES}")
    sf_weight = max(0.0, float(sf_frac))
    search_weight = max(0.0, float(search_frac))
    total = sf_weight + search_weight
    if total > 1.0:
        sf_weight /= total
        search_weight /= total
        game_weight = 0.0
    else:
        game_weight = 1.0 - total
    game = np.asarray(game_target, dtype=np.float64)
    game = game / np.maximum(game.sum(axis=1, keepdims=True), 1e-12)
    raw_game = game if raw_game_target is None else np.asarray(raw_game_target, dtype=np.float64)
    raw_game = raw_game / np.maximum(raw_game.sum(axis=1, keepdims=True), 1e-12)
    fallback = game if fallback_target is None else np.asarray(fallback_target, dtype=np.float64)
    fallback = fallback / np.maximum(fallback.sum(axis=1, keepdims=True), 1e-12)
    sf_av = (
        np.ones((game.shape[0],), dtype=bool)
        if sf_available is None
        else np.asarray(sf_available, dtype=bool)
    )
    search_av = (
        np.ones((game.shape[0],), dtype=bool)
        if search_available is None
        else np.asarray(search_available, dtype=bool)
    )
    sf_sig = _signal(sf)
    search_sig = _signal(search)
    joint = sf_av & search_av
    sf_low = joint & (sf_sig < 0.0) & (search_sig > 0.0)
    sf_high = joint & (sf_sig > 0.0) & (search_sig < 0.0)
    keep = (
        1.0
        - float(dampen_sf_low) * sf_low.astype(np.float64)
        - float(dampen_sf_high) * sf_high.astype(np.float64)
    )
    keep = np.clip(keep, 0.0, 1.0)[:, None]
    sf_keep = sf_av.astype(np.float64)[:, None] * keep
    search_keep = search_av.astype(np.float64)[:, None]
    if mode == "renormalize":
        game_w = np.full((game.shape[0], 1), game_weight, dtype=np.float64)
        sf_w = sf_weight * sf_keep
        search_w = search_weight * search_keep
        denom = game_w + sf_w + search_w
        weighted = game_w * game + sf_w * sf + search_w * search
        target = np.where(denom > 1e-12, weighted / np.maximum(denom, 1e-12), raw_game)
        return _normalize(target)

    target = game_weight * game
    target += sf_weight * (sf_keep * sf + (1.0 - sf_keep) * fallback)
    target += search_weight * (search_keep * search + (1.0 - search_keep) * fallback)
    return _normalize(target)


def _entropy_stats(p: np.ndarray, legal: np.ndarray | None = None) -> dict[str, float]:
    if p.size == 0:
        return {"n": 0.0}
    arr = np.asarray(p, dtype=np.float64)
    if legal is not None:
        arr = np.where(np.asarray(legal, dtype=bool), arr, 0.0)
    arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-12)
    nz = arr > 0.0
    ent = -(np.where(nz, arr * np.log(np.clip(arr, 1e-12, None)), 0.0)).sum(axis=1)
    top1 = arr.max(axis=1)
    top5 = np.sort(arr, axis=1)[:, -5:].sum(axis=1)
    eff = np.exp(ent)
    return {
        "n": float(arr.shape[0]),
        "entropy": float(ent.mean()),
        "effective_moves": float(eff.mean()),
        "top1": float(top1.mean()),
        "top5": float(top5.mean()),
    }


def _empty_wdl_stats() -> dict[str, Any]:
    return {
        "n": 0,
        "outcome": np.zeros((3,), dtype=np.int64),
        "game_target_sum": np.zeros((3,), dtype=np.float64),
        "sf_sum": np.zeros((3,), dtype=np.float64),
        "search_sum": np.zeros((3,), dtype=np.float64),
        "blend_sum": np.zeros((3,), dtype=np.float64),
        "sf_ce": 0.0,
        "search_ce": 0.0,
        "blend_ce": 0.0,
        "sf_brier": 0.0,
        "search_brier": 0.0,
        "blend_brier": 0.0,
        "agree": 0,
        "sf_low_search_high": 0,
        "sf_high_search_low": 0,
    }


def _add_wdl_stats(
    dst: dict[str, Any],
    *,
    outcome: np.ndarray,
    game_target: np.ndarray,
    sf: np.ndarray,
    search: np.ndarray,
    blend: np.ndarray,
) -> None:
    n = int(outcome.size)
    if n <= 0:
        return
    sf_sig = _signal(sf)
    search_sig = _signal(search)
    dst["n"] += n
    dst["outcome"] += np.bincount(outcome, minlength=3).astype(np.int64)
    dst["game_target_sum"] += game_target.sum(axis=0)
    dst["sf_sum"] += sf.sum(axis=0)
    dst["search_sum"] += search.sum(axis=0)
    dst["blend_sum"] += blend.sum(axis=0)
    dst["sf_ce"] += _ce_to_outcome(sf, outcome) * n
    dst["search_ce"] += _ce_to_outcome(search, outcome) * n
    dst["blend_ce"] += _ce_to_outcome(blend, outcome) * n
    dst["sf_brier"] += _brier_to_outcome(sf, outcome) * n
    dst["search_brier"] += _brier_to_outcome(search, outcome) * n
    dst["blend_brier"] += _brier_to_outcome(blend, outcome) * n
    dst["agree"] += int(((sf_sig == 0.0) | (search_sig == 0.0) | (np.sign(sf_sig) == np.sign(search_sig))).sum())
    dst["sf_low_search_high"] += int(((sf_sig < 0.0) & (search_sig > 0.0)).sum())
    dst["sf_high_search_low"] += int(((sf_sig > 0.0) & (search_sig < 0.0)).sum())


def _finish_wdl_stats(raw: dict[str, Any]) -> dict[str, Any]:
    n = int(raw["n"])
    if n <= 0:
        return {"n": 0}
    return {
        "n": n,
        "outcome_wdl": (raw["outcome"] / n).tolist(),
        "mean_game_target_wdl": (raw["game_target_sum"] / n).tolist(),
        "mean_sf_wdl": (raw["sf_sum"] / n).tolist(),
        "mean_search_wdl": (raw["search_sum"] / n).tolist(),
        "mean_blend_wdl": (raw["blend_sum"] / n).tolist(),
        "sf_ce": float(raw["sf_ce"] / n),
        "search_ce": float(raw["search_ce"] / n),
        "blend_ce": float(raw["blend_ce"] / n),
        "sf_brier": float(raw["sf_brier"] / n),
        "search_brier": float(raw["search_brier"] / n),
        "blend_brier": float(raw["blend_brier"] / n),
        "agree_frac": float(raw["agree"] / n),
        "sf_low_search_high_frac": float(raw["sf_low_search_high"] / n),
        "sf_high_search_low_frac": float(raw["sf_high_search_low"] / n),
    }


def _load_loss_config(latest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    raw_cfg = latest.get("config")
    cfg: dict[str, Any] = raw_cfg if isinstance(raw_cfg, dict) else {}

    def pick(name: str, default: float) -> float:
        override = getattr(args, name, None)
        value = override if override is not None else latest.get(name, cfg.get(name, default))
        return float(value)

    def pick_bool(name: str, default: bool) -> bool:
        override = getattr(args, name, None)
        value = override if override is not None else latest.get(name, cfg.get(name, default))
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def pick_str(name: str, default: str) -> str:
        override = getattr(args, name, None)
        value = override if override is not None else latest.get(name, cfg.get(name, default))
        return str(value)

    return {
        "sf_wdl_frac": pick("sf_wdl_frac", 0.5),
        "search_wdl_frac": pick("search_wdl_frac", 0.5),
        "sf_wdl_temperature": pick("sf_wdl_temperature", 1.0),
        "sf_search_dampen_sf_low": pick("sf_search_dampen_sf_low", 0.0),
        "sf_search_dampen_sf_high": pick("sf_search_dampen_sf_high", 0.0),
        "use_adjusted_wdl_target": pick_bool("use_adjusted_wdl_target", False),
        "adjusted_wdl_regret_source": pick_str("adjusted_wdl_regret_source", "sum"),
        "adjusted_wdl_regret_scale": pick("adjusted_wdl_regret_scale", 1.0),
        "adjusted_wdl_regret_cap": pick("adjusted_wdl_regret_cap", 0.0),
        "wdl_blend_mode": pick_str("wdl_blend_mode", "interpolate"),
        "wdl_blend_fallback": pick_str("wdl_blend_fallback", "adjusted_game"),
    }


def _future_regret_slice(
    arrs: dict[str, Any],
    row_slice: slice,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(row_slice.stop - row_slice.start)
    value_name, has_name = _future_regret_field_names(source)
    values = np.zeros((n,), dtype=np.float64)
    if value_name not in arrs:
        return values, np.zeros((n,), dtype=bool)
    values = np.asarray(arrs[value_name][row_slice], dtype=np.float64)
    if has_name in arrs:
        has = np.asarray(arrs[has_name][row_slice], dtype=bool)
    else:
        has = np.isfinite(values)
    return values, has & np.isfinite(values)


def _fmt_wdl(vals: list[float] | np.ndarray) -> str:
    x = [float(v) for v in vals]
    return f"{x[0]:.3f}/{x[1]:.3f}/{x[2]:.3f}"


def _print_wdl_table(title: str, stats: list[dict[str, Any]]) -> None:
    print(f"## {title}")
    print()
    print(
        "| Bucket | n | outcome W/D/L | game target WDL | SF WDL | search WDL | blend WDL | "
        "CE sf/search/blend | Brier sf/search/blend | disagree low/high |"
    )
    print("|---:|---:|---|---|---|---|---|---|---|---|")
    for i, raw in enumerate(stats, start=1):
        s = _finish_wdl_stats(raw)
        if int(s.get("n", 0)) <= 0:
            continue
        print(
            f"| {i} | {s['n']} | `{_fmt_wdl(s['outcome_wdl'])}` | "
            f"`{_fmt_wdl(s['mean_game_target_wdl'])}` | `{_fmt_wdl(s['mean_sf_wdl'])}` | "
            f"`{_fmt_wdl(s['mean_search_wdl'])}` | `{_fmt_wdl(s['mean_blend_wdl'])}` | "
            f"`{s['sf_ce']:.3f}/{s['search_ce']:.3f}/{s['blend_ce']:.3f}` | "
            f"`{s['sf_brier']:.3f}/{s['search_brier']:.3f}/{s['blend_brier']:.3f}` | "
            f"`{100*s['sf_low_search_high_frac']:.1f}%/{100*s['sf_high_search_low_frac']:.1f}%` |"
        )
    print()


def _print_policy_table(title: str, stats: list[dict[str, dict[str, float]]]) -> None:
    print(f"## {title}")
    print()
    print("| Bucket | target | n | effective moves | top1 | top5 | entropy |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    for i, bucket in enumerate(stats, start=1):
        for name in ("policy", "soft_policy", "sf_policy"):
            s = bucket.get(name, {"n": 0.0})
            if int(s.get("n", 0.0)) <= 0:
                continue
            print(
                f"| {i} | {name} | {int(s['n'])} | {s['effective_moves']:.2f} | "
                f"{s['top1']:.3f} | {s['top5']:.3f} | {s['entropy']:.3f} |"
            )
    print()


def _print_blend_mode_table(title: str, stats: dict[str, dict[str, Any]]) -> None:
    print(f"## {title}")
    print()
    print("| mode | n | blend CE to outcome | blend Brier to outcome | blend WDL |")
    print("|---|---:|---:|---:|---|")
    rows = []
    for name, raw in stats.items():
        s = _finish_wdl_stats(raw)
        if int(s.get("n", 0)) <= 0:
            continue
        rows.append((s["blend_ce"], s["blend_brier"], name, s))
    for _ce, _brier, name, s in sorted(rows):
        print(
            f"| {name} | {s['n']} | {s['blend_ce']:.4f} | "
            f"{s['blend_brier']:.4f} | `{_fmt_wdl(s['mean_blend_wdl'])}` |"
        )
    print()


def _fallback_for_mode(name: str, *, raw_game: np.ndarray, game_target: np.ndarray) -> np.ndarray:
    if name == "raw_game":
        return raw_game
    if name == "adjusted_game":
        return game_target
    raise ValueError(f"unknown WDL blend fallback {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--trial-dir", type=Path, default=None)
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--max-shards", type=int, default=0, help="newest shards to scan; 0 scans all")
    parser.add_argument("--window-positions", type=int, default=2_000_000)
    parser.add_argument("--buckets", type=int, default=5, help="oldest-to-newest age buckets")
    parser.add_argument(
        "--policy-sample-per-bucket",
        type=int,
        default=20_000,
        help="sample at most this many rows per age bucket/target for wide policy entropy stats",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sf-wdl-frac", type=float, default=None)
    parser.add_argument("--search-wdl-frac", type=float, default=None)
    parser.add_argument("--sf-wdl-temperature", type=float, default=None)
    parser.add_argument("--sf-search-dampen-sf-low", type=float, default=None)
    parser.add_argument("--sf-search-dampen-sf-high", type=float, default=None)
    parser.add_argument("--use-adjusted-wdl-target", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--adjusted-wdl-regret-source",
        choices=sorted(FUTURE_REGRET_FIELDS),
        default=None,
    )
    parser.add_argument("--adjusted-wdl-regret-scale", type=float, default=None)
    parser.add_argument("--adjusted-wdl-regret-cap", type=float, default=None)
    parser.add_argument(
        "--wdl-blend-mode",
        choices=WDL_BLEND_MODES,
        default=None,
        help="How SF/search dampening changes blended WDL targets.",
    )
    parser.add_argument(
        "--wdl-blend-fallback",
        choices=WDL_BLEND_FALLBACKS,
        default=None,
        help="Fallback target for interpolate mode when labels are missing or dampened.",
    )
    args = parser.parse_args()
    if int(args.buckets) < 1 or int(args.buckets) > MAX_BUCKETS:
        parser.error(f"--buckets must be between 1 and {MAX_BUCKETS}")

    trial_dir = args.trial_dir or _try_latest_trial_dir(args.run_dir)
    if args.replay_dir is None and trial_dir is None:
        raise FileNotFoundError(
            f"No trial directories under {args.run_dir / 'tune'}; pass --replay-dir explicitly",
        )
    if args.replay_dir is not None:
        replay_dir = args.replay_dir
    else:
        if trial_dir is None:
            raise AssertionError("trial_dir unexpectedly missing")
        replay_dir = _replay_dir_for_trial(args.run_dir, trial_dir)
    latest = _latest_result(trial_dir) if trial_dir is not None else {}
    cfg = _load_loss_config(latest, args)
    if cfg["wdl_blend_mode"] not in WDL_BLEND_MODES:
        parser.error(f"--wdl-blend-mode must be one of {', '.join(WDL_BLEND_MODES)}")
    if cfg["wdl_blend_fallback"] not in WDL_BLEND_FALLBACKS:
        parser.error(f"--wdl-blend-fallback must be one of {', '.join(WDL_BLEND_FALLBACKS)}")
    if int(args.max_shards) < 0:
        parser.error("--max-shards must be non-negative")
    slices = _newest_window_slices(replay_dir, int(args.window_positions), int(args.max_shards))
    total_positions = sum(s.take for s in slices)
    if total_positions <= 0:
        raise SystemExit("no replay positions found")

    bucket_count = int(args.buckets)
    rng = np.random.default_rng(int(args.seed))
    wdl_by_bucket = [_empty_wdl_stats() for _ in range(bucket_count)]
    policy_batches: list[dict[str, list[dict[str, float]]]] = [
        {"policy": [], "soft_policy": [], "sf_policy": []} for _ in range(bucket_count)
    ]
    policy_seen: list[dict[str, int]] = [
        {"policy": 0, "soft_policy": 0, "sf_policy": 0} for _ in range(bucket_count)
    ]
    sf_grid = {
        (sf_frac, damp_low, damp_high): _empty_wdl_stats()
        for sf_frac in (0.0, 0.1, 0.15, 0.25, 0.5)
        for damp_low in (0.0, 0.25, 0.5, 1.0)
        for damp_high in (0.0, 0.25, 0.5, 1.0)
    }
    blend_mode_grid = {
        name: _empty_wdl_stats()
        for name, _mode, _fallback in WDL_BLEND_COUNTERFACTUALS
    }
    scan: dict[str, Any] = {
        "skipped_shards": [],
        "skipped_shards_omitted": 0,
        "adjusted_wdl_target_rows": 0,
        "adjusted_wdl_target_missing_rows": 0,
    }

    pos = 0
    for spec in slices:
        try:
            arrs, _meta = load_shard_arrays(spec.path, lazy=True)
        except Exception as exc:
            # Keep age-bucket diagnostics usable while live replay shards are being written.
            _record_skipped_shard(scan, spec.path, exc)
            pos += spec.take
            continue
        row0 = spec.start
        local_pos = 0
        while local_pos < spec.take:
            global_start = pos + local_pos
            bucket = min(bucket_count - 1, int(global_start * bucket_count / total_positions))
            bucket_end_global = math.ceil((bucket + 1) * total_positions / bucket_count)
            n = min(spec.take - local_pos, max(1, bucket_end_global - global_start))
            sl = slice(row0 + local_pos, row0 + local_pos + n)

            wdl_fields = ("has_sf_wdl", "has_search_wdl", "wdl_target", "sf_wdl", "search_wdl")
            if all(name in arrs for name in wdl_fields):
                has_sf = np.asarray(arrs["has_sf_wdl"][sl], dtype=bool)
                has_search = np.asarray(arrs["has_search_wdl"][sl], dtype=bool)
                outcome = np.asarray(arrs["wdl_target"][sl], dtype=np.int64)
                both = has_sf & has_search
                if both.any():
                    sf = _normalize(
                        np.asarray(arrs["sf_wdl"][sl], dtype=np.float64)[both],
                        temperature=cfg["sf_wdl_temperature"],
                    )
                    search = _normalize(np.asarray(arrs["search_wdl"][sl], dtype=np.float64)[both])
                    y = outcome[both]
                    raw_game_target = _one_hot(y)
                    game_target = raw_game_target
                    if bool(cfg["use_adjusted_wdl_target"]):
                        future_regret, has_future_regret = _future_regret_slice(
                            arrs,
                            sl,
                            str(cfg["adjusted_wdl_regret_source"]),
                        )
                        future_regret = future_regret[both]
                        has_future_regret = has_future_regret[both]
                        scan["adjusted_wdl_target_rows"] += int(has_future_regret.sum())
                        scan["adjusted_wdl_target_missing_rows"] += int((~has_future_regret).sum())
                        game_target = _adjusted_wdl_game_target(
                            y,
                            future_regret,
                            has_future_regret,
                            regret_scale=float(cfg["adjusted_wdl_regret_scale"]),
                            regret_cap=float(cfg["adjusted_wdl_regret_cap"]),
                        )
                    fallback_target = _fallback_for_mode(
                        str(cfg["wdl_blend_fallback"]),
                        raw_game=raw_game_target,
                        game_target=game_target,
                    )
                    blend = _blend_wdl(
                        game_target,
                        sf,
                        search,
                        raw_game_target=raw_game_target,
                        fallback_target=fallback_target,
                        sf_frac=cfg["sf_wdl_frac"],
                        search_frac=cfg["search_wdl_frac"],
                        dampen_sf_low=cfg["sf_search_dampen_sf_low"],
                        dampen_sf_high=cfg["sf_search_dampen_sf_high"],
                        blend_mode=str(cfg["wdl_blend_mode"]),
                    )
                    _add_wdl_stats(
                        wdl_by_bucket[bucket],
                        outcome=y,
                        game_target=game_target,
                        sf=sf,
                        search=search,
                        blend=blend,
                    )
                    for name, mode, fallback_name in WDL_BLEND_COUNTERFACTUALS:
                        mode_fallback = _fallback_for_mode(
                            fallback_name,
                            raw_game=raw_game_target,
                            game_target=game_target,
                        )
                        mode_blend = _blend_wdl(
                            game_target,
                            sf,
                            search,
                            raw_game_target=raw_game_target,
                            fallback_target=mode_fallback,
                            sf_frac=cfg["sf_wdl_frac"],
                            search_frac=cfg["search_wdl_frac"],
                            dampen_sf_low=cfg["sf_search_dampen_sf_low"],
                            dampen_sf_high=cfg["sf_search_dampen_sf_high"],
                            blend_mode=mode,
                        )
                        _add_wdl_stats(
                            blend_mode_grid[name],
                            outcome=y,
                            game_target=game_target,
                            sf=sf,
                            search=search,
                            blend=mode_blend,
                        )
                    for (sf_frac, damp_low, damp_high), dst in sf_grid.items():
                        grid_blend = _blend_wdl(
                            game_target,
                            sf,
                            search,
                            raw_game_target=raw_game_target,
                            fallback_target=fallback_target,
                            sf_frac=sf_frac,
                            search_frac=cfg["search_wdl_frac"],
                            dampen_sf_low=damp_low,
                            dampen_sf_high=damp_high,
                            blend_mode=str(cfg["wdl_blend_mode"]),
                        )
                        _add_wdl_stats(
                            dst,
                            outcome=y,
                            game_target=game_target,
                            sf=sf,
                            search=search,
                            blend=grid_blend,
                        )

            for out_name, target_name, has_name, legal_name in (
                ("policy", "policy_target", "has_policy", "legal_mask"),
                ("soft_policy", "policy_soft_target", "has_policy_soft", "legal_mask"),
                ("sf_policy", "sf_policy_target", "has_sf_policy", "sf_legal_mask"),
            ):
                if target_name not in arrs or has_name not in arrs:
                    continue
                has = np.asarray(arrs[has_name][sl], dtype=bool)
                if not has.any():
                    continue
                local_rows = np.flatnonzero(has)
                remaining = int(args.policy_sample_per_bucket) - int(policy_seen[bucket][out_name])
                if remaining <= 0:
                    continue
                if local_rows.size > remaining:
                    local_rows = np.sort(rng.choice(local_rows, size=remaining, replace=False))
                policy_seen[bucket][out_name] += int(local_rows.size)
                abs_rows = row0 + local_pos + local_rows
                target = np.asarray(arrs[target_name][abs_rows], dtype=np.float64)
                legal = None
                if legal_name in arrs:
                    legal = np.asarray(arrs[legal_name][abs_rows], dtype=bool)
                policy_batches[bucket][out_name].append(_entropy_stats(target, legal))

            local_pos += n
        pos += spec.take

    policy_by_bucket: list[dict[str, dict[str, float]]] = []
    for bucket in policy_batches:
        dst: dict[str, dict[str, float]] = {}
        for name, parts in bucket.items():
            n = sum(p.get("n", 0.0) for p in parts)
            if n <= 0:
                dst[name] = {"n": 0.0}
                continue
            dst[name] = {"n": n}
            for key in ("entropy", "effective_moves", "top1", "top5"):
                dst[name][key] = sum(p[key] * p["n"] for p in parts) / n
        policy_by_bucket.append(dst)

    print("# Target Calibration Diagnostics")
    print()
    print(f"- trial: `{None if trial_dir is None else trial_dir.name}`")
    print(f"- replay: `{replay_dir}`")
    print(f"- scanned newest positions: `{total_positions}`")
    if scan["skipped_shards"]:
        skipped = len(scan["skipped_shards"]) + int(scan["skipped_shards_omitted"])
        print(f"- skipped shards: `{skipped}` ({len(scan['skipped_shards'])} shown)")
    print(
        "- current blend: "
        f"sf_wdl_frac={cfg['sf_wdl_frac']:.3f}, search_wdl_frac={cfg['search_wdl_frac']:.3f}, "
        f"sf_wdl_temperature={cfg['sf_wdl_temperature']:.3f}, "
        f"dampen_sf_low={cfg['sf_search_dampen_sf_low']:.3f}, "
        f"dampen_sf_high={cfg['sf_search_dampen_sf_high']:.3f}, "
        f"wdl_blend_mode={cfg['wdl_blend_mode']}, "
        f"wdl_blend_fallback={cfg['wdl_blend_fallback']}"
    )
    if bool(cfg["use_adjusted_wdl_target"]):
        print(
            "- adjusted game target: "
            f"source={cfg['adjusted_wdl_regret_source']}, "
            f"scale={cfg['adjusted_wdl_regret_scale']:.3f}, "
            f"cap={cfg['adjusted_wdl_regret_cap']:.3f}, "
            f"rows={scan['adjusted_wdl_target_rows']}, "
            f"missing={scan['adjusted_wdl_target_missing_rows']}"
        )
    else:
        print("- adjusted game target: disabled")
    print()
    _print_wdl_table("WDL Calibration By Age, Oldest To Newest", wdl_by_bucket)
    _print_blend_mode_table("WDL Blend Mode Counterfactuals", blend_mode_grid)
    _print_policy_table("Policy Target Sharpness By Age, Oldest To Newest", policy_by_bucket)

    print("## SF Fraction / Dampening Counterfactuals")
    print()
    print("| sf_frac | search_frac | game_frac | dampen_sf_low | dampen_sf_high | blend CE to outcome | blend Brier to outcome | blend WDL |")
    print("|---:|---:|---:|---:|---:|---:|---:|---|")
    rows = []
    for (sf_frac, damp_low, damp_high), raw in sf_grid.items():
        stats = _finish_wdl_stats(raw)
        if int(stats.get("n", 0)) <= 0:
            continue
        rows.append((stats["blend_ce"], stats["blend_brier"], sf_frac, damp_low, damp_high, stats))
    for row in sorted(rows):
        sf_frac = row[2]
        damp_low = row[3]
        damp_high = row[4]
        stats = row[5]
        marker = (
            " *"
            if abs(sf_frac - cfg["sf_wdl_frac"]) < 1e-9
            and abs(damp_low - cfg["sf_search_dampen_sf_low"]) < 1e-9
            and abs(damp_high - cfg["sf_search_dampen_sf_high"]) < 1e-9
            else ""
        )
        game_frac = max(0.0, 1.0 - sf_frac - cfg["search_wdl_frac"])
        print(
            f"| {sf_frac:.2f}{marker} | {cfg['search_wdl_frac']:.2f} | {game_frac:.2f} | "
            f"{damp_low:.2f} | {damp_high:.2f} | "
            f"{stats['blend_ce']:.4f} | {stats['blend_brier']:.4f} | `{_fmt_wdl(stats['mean_blend_wdl'])}` |"
        )


if __name__ == "__main__":
    main()
