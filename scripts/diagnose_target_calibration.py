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
from scripts.diagnostic_replay_utils import record_skipped_shard as _record_skipped_shard


DEFAULT_RUN = Path("runs/pbt2_small")
MAX_BUCKETS = 20


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
            try:
                latest = json.loads(line)
            except json.JSONDecodeError:
                pass
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
    y = np.asarray(outcome, dtype=np.int64)
    oh = np.zeros((y.size, 3), dtype=np.float64)
    oh[np.arange(y.size), y] = 1.0
    return oh


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
    outcome: np.ndarray,
    sf: np.ndarray,
    search: np.ndarray,
    *,
    sf_frac: float,
    search_frac: float,
    dampen_sf_low: float,
    dampen_sf_high: float,
) -> np.ndarray:
    sf_weight = max(0.0, float(sf_frac))
    search_weight = max(0.0, float(search_frac))
    total = sf_weight + search_weight
    if total > 1.0:
        sf_weight /= total
        search_weight /= total
        game_weight = 0.0
    else:
        game_weight = 1.0 - total
    game = _one_hot(outcome)
    sf_sig = _signal(sf)
    search_sig = _signal(search)
    sf_low = (sf_sig < 0.0) & (search_sig > 0.0)
    sf_high = (sf_sig > 0.0) & (search_sig < 0.0)
    keep = (
        1.0
        - float(dampen_sf_low) * sf_low.astype(np.float64)
        - float(dampen_sf_high) * sf_high.astype(np.float64)
    )
    keep = np.clip(keep, 0.0, 1.0)[:, None]
    target = game_weight * game
    target += sf_weight * (keep * sf + (1.0 - keep) * game)
    target += search_weight * search
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


def _soft_ce(target: np.ndarray, pred: np.ndarray) -> float:
    if target.size == 0:
        return math.nan
    t = _normalize(target)
    p = _normalize(pred)
    return float((-(t * np.log(np.clip(p, 1e-9, 1.0))).sum(axis=1)).mean())


def _empty_wdl_stats() -> dict[str, Any]:
    return {
        "n": 0,
        "outcome": np.zeros((3,), dtype=np.int64),
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


def _add_wdl_stats(dst: dict[str, Any], *, outcome: np.ndarray, sf: np.ndarray, search: np.ndarray, blend: np.ndarray) -> None:
    n = int(outcome.size)
    if n <= 0:
        return
    sf_sig = _signal(sf)
    search_sig = _signal(search)
    dst["n"] += n
    dst["outcome"] += np.bincount(outcome, minlength=3).astype(np.int64)
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


def _load_loss_config(latest: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    raw_cfg = latest.get("config")
    cfg: dict[str, Any] = raw_cfg if isinstance(raw_cfg, dict) else {}

    def pick(name: str, default: float) -> float:
        override = getattr(args, name, None)
        value = override if override is not None else latest.get(name, cfg.get(name, default))
        return float(value)

    return {
        "sf_wdl_frac": pick("sf_wdl_frac", 0.5),
        "search_wdl_frac": pick("search_wdl_frac", 0.5),
        "sf_wdl_temperature": pick("sf_wdl_temperature", 1.0),
        "sf_search_dampen_sf_low": pick("sf_search_dampen_sf_low", 0.0),
        "sf_search_dampen_sf_high": pick("sf_search_dampen_sf_high", 0.0),
    }


def _fmt_wdl(vals: list[float] | np.ndarray) -> str:
    x = [float(v) for v in vals]
    return f"{x[0]:.3f}/{x[1]:.3f}/{x[2]:.3f}"


def _print_wdl_table(title: str, stats: list[dict[str, Any]]) -> None:
    print(f"## {title}")
    print()
    print("| Bucket | n | outcome W/D/L | SF WDL | search WDL | blend WDL | CE sf/search/blend | Brier sf/search/blend | disagree low/high |")
    print("|---:|---:|---|---|---|---|---|---|---|")
    for i, raw in enumerate(stats, start=1):
        s = _finish_wdl_stats(raw)
        if int(s.get("n", 0)) <= 0:
            continue
        print(
            f"| {i} | {s['n']} | `{_fmt_wdl(s['outcome_wdl'])}` | `{_fmt_wdl(s['mean_sf_wdl'])}` | "
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
    scan: dict[str, Any] = {
        "skipped_shards": [],
        "skipped_shards_omitted": 0,
    }

    pos = 0
    for spec in slices:
        try:
            arrs, _meta = load_shard_arrays(spec.path, lazy=True)
        except Exception as exc:  # noqa: BLE001
            # Keep age-bucket diagnostics usable while live replay shards are being written.
            _record_skipped_shard(scan, spec.path, exc)
            pos += spec.take
            continue
        row0 = spec.start
        local_pos = 0
        while local_pos < spec.take:
            global_start = pos + local_pos
            bucket = min(bucket_count - 1, int(global_start * bucket_count / total_positions))
            bucket_end_global = int(math.ceil((bucket + 1) * total_positions / bucket_count))
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
                    blend = _blend_wdl(
                        y,
                        sf,
                        search,
                        sf_frac=cfg["sf_wdl_frac"],
                        search_frac=cfg["search_wdl_frac"],
                        dampen_sf_low=cfg["sf_search_dampen_sf_low"],
                        dampen_sf_high=cfg["sf_search_dampen_sf_high"],
                    )
                    _add_wdl_stats(wdl_by_bucket[bucket], outcome=y, sf=sf, search=search, blend=blend)
                    for (sf_frac, damp_low, damp_high), dst in sf_grid.items():
                        grid_blend = _blend_wdl(
                            y,
                            sf,
                            search,
                            sf_frac=sf_frac,
                            search_frac=cfg["search_wdl_frac"],
                            dampen_sf_low=damp_low,
                            dampen_sf_high=damp_high,
                        )
                        _add_wdl_stats(dst, outcome=y, sf=sf, search=search, blend=grid_blend)

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
        f"dampen_sf_high={cfg['sf_search_dampen_sf_high']:.3f}"
    )
    print()
    _print_wdl_table("WDL Calibration By Age, Oldest To Newest", wdl_by_bucket)
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
