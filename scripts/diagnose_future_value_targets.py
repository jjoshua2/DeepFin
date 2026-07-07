#!/usr/bin/env python3
"""Compare live WDL target mixes against multi-horizon future-value targets."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.diagnostic_replay_utils import (
    REGRET_TO_Q_SCALE,
    fit_simplex as _fit_simplex,
    game_fold_masks as _game_fold_masks,
    latest_replay_dir as _latest_replay_dir,
    rmse as _rmse,
)
from scripts.diagnose_future_eval_bucketed import (
    DEFAULT_EVAL_BINS,
    Sample,
    _load_samples,
    _pairs_for_horizon,
)
from scripts.trial_paths import latest_result, latest_trial_dir


DEFAULT_RUN_DIR = Path("runs/pbt2_small")
DEFAULT_CONFIG = Path("configs/pbt2_small.yaml")
MAX_HORIZONS = 12
MAX_TAUS = 8
MAX_SOURCES = 3
MAX_OFFSET = 200
MIN_SPLIT_ROWS = 20
SuffixKey = tuple[str, float | None, bool]


@dataclass(frozen=True, slots=True)
class MixCandidate:
    name: str
    w_sf: float
    w_search: float
    w_final: float

    @property
    def weights(self) -> np.ndarray:
        return np.asarray([self.w_sf, self.w_search, self.w_final], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class HorizonScore:
    candidate: str
    w_sf: float
    w_search: float
    w_final: float
    h_rmse: dict[int, float]
    avg_rmse: float
    worst_rmse: float


@dataclass(frozen=True, slots=True)
class SuffixFitRow:
    source: str
    decay: str
    regret_adjusted: bool
    n_train: int
    n_test: int
    w_sf: float
    w_search: float
    w_final: float
    best_rmse: float
    current_rmse: float
    sf_only_rmse: float
    search_only_rmse: float
    final_only_rmse: float


def _parse_ints(raw: str, *, label: str, max_count: int, require_even: bool = False) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{label} must be positive comma-separated integers")
    if require_even and any(value % 2 != 0 for value in values):
        raise argparse.ArgumentTypeError(f"{label} must be even ply offsets")
    if len(values) > max_count:
        raise argparse.ArgumentTypeError(f"at most {max_count} {label} values are allowed")
    return values


def _parse_taus(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any((not math.isfinite(value)) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("taus must be positive finite comma-separated numbers")
    if len(values) > MAX_TAUS:
        raise argparse.ArgumentTypeError(f"at most {MAX_TAUS} tau values are allowed")
    return values


def _parse_sources(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = {"avg", "sf", "search"}
    if not values or any(value not in allowed for value in values):
        raise argparse.ArgumentTypeError("sources must be comma-separated values from: avg,sf,search")
    if len(values) > MAX_SOURCES:
        raise argparse.ArgumentTypeError(f"at most {MAX_SOURCES} sources are allowed")
    return values


def _optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _normalize_mix(name: str, sf: float, search: float, final: float) -> MixCandidate:
    weights = np.maximum(np.asarray([sf, search, final], dtype=np.float64), 0.0)
    total = float(weights.sum())
    if total <= 1e-12:
        weights = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    elif total > 1.0:
        weights /= total
    return MixCandidate(name, float(weights[0]), float(weights[1]), float(weights[2]))


def _trial_dir_for_replay(run_dir: Path, replay_dir: Path) -> Path | None:
    try:
        rel = replay_dir.resolve().relative_to((run_dir / "replay").resolve())
    except ValueError:
        return None
    if len(rel.parts) < 2 or rel.parts[-1] != "replay_shards":
        return None
    trial_dir = run_dir / "tune" / rel.parts[-2]
    return trial_dir if trial_dir.is_dir() else None


def _config_values(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

    return flatten_run_config_defaults(load_yaml_file(config_path))


def _current_candidates(run_dir: Path, config_path: Path, *, trial_dir: Path | None = None) -> list[MixCandidate]:
    cfg = _config_values(config_path)
    latest = latest_result(trial_dir or latest_trial_dir(run_dir))
    rows: list[MixCandidate] = []
    yaml_search = _optional_float(cfg.get("search_wdl_frac"), 0.0) if cfg else None
    reported_sf = latest.get("sf_wdl_frac")
    latest_cfg = latest.get("config", {}) if isinstance(latest.get("config"), dict) else {}
    reported_search = latest.get("search_wdl_frac", latest_cfg.get("search_wdl_frac", yaml_search))
    sf = _optional_float(reported_sf)
    search = _optional_float(reported_search)
    if sf is not None and search is not None:
        rows.append(_normalize_mix("reported_live", sf, search, 1.0 - sf - search))

    if cfg:
        sf = _optional_float(cfg.get("sf_wdl_frac_floor"), _optional_float(cfg.get("sf_wdl_frac"), 0.0))
        search = yaml_search
        if sf is not None and search is not None:
            rows.append(_normalize_mix("yaml_low_regret", sf, search, 1.0 - sf - search))

    if not rows:
        rows.append(MixCandidate("default", 0.0, 0.0, 1.0))
    return rows


def _fit_portfolio_mix(
    pairs_by_horizon: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> MixCandidate | None:
    features_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for game_ids, features, target in pairs_by_horizon.values():
        train, _test = _game_fold_masks(game_ids)
        if int(train.sum()) < MIN_SPLIT_ROWS:
            continue
        features_parts.append(features[train])
        target_parts.append(target[train])
        weight_parts.append(np.full((int(train.sum()),), 1.0 / float(train.sum()), dtype=np.float64))
    if not features_parts:
        return None
    weights = _fit_simplex(np.vstack(features_parts), np.concatenate(target_parts), np.concatenate(weight_parts))
    return MixCandidate("portfolio_fit", float(weights[0]), float(weights[1]), float(weights[2]))


def _score_mix(
    candidate: MixCandidate,
    pairs_by_horizon: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    split: str = "test",
) -> HorizonScore:
    if split not in {"train", "test"}:
        raise ValueError(f"unknown split {split!r}; expected 'train' or 'test'")
    scores: dict[int, float] = {}
    for horizon, (game_ids, features, target) in pairs_by_horizon.items():
        train, test = _game_fold_masks(game_ids)
        mask = train if split == "train" else test
        if int(mask.sum()) < MIN_SPLIT_ROWS:
            scores[int(horizon)] = math.nan
            continue
        scores[int(horizon)] = _rmse(features[mask] @ candidate.weights, target[mask])
    finite = [value for value in scores.values() if math.isfinite(value)]
    return HorizonScore(
        candidate=candidate.name,
        w_sf=candidate.w_sf,
        w_search=candidate.w_search,
        w_final=candidate.w_final,
        h_rmse=scores,
        avg_rmse=float(np.mean(finite)) if finite else math.nan,
        worst_rmse=float(np.max(finite)) if finite else math.nan,
    )


def _grid_candidates(
    pairs_by_horizon: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[MixCandidate]:
    weights = np.asarray(
        [
            (sf_i / 100.0, search_i / 100.0, 1.0 - (sf_i + search_i) / 100.0)
            for sf_i in range(101)
            for search_i in range(101 - sf_i)
        ],
        dtype=np.float64,
    )
    per_horizon_rmse: list[np.ndarray] = []
    for game_ids, features, target in pairs_by_horizon.values():
        train, _test = _game_fold_masks(game_ids)
        if int(train.sum()) < MIN_SPLIT_ROWS:
            continue
        pred = features[train] @ weights.T
        err = pred - target[train, None]
        per_horizon_rmse.append(np.sqrt(np.mean(err * err, axis=0)))
    if not per_horizon_rmse:
        return []
    stacked = np.stack(per_horizon_rmse, axis=0)
    avg = np.mean(stacked, axis=0)
    worst = np.max(stacked, axis=0)
    choices = (
        ("grid_best_avg", int(np.argmin(avg))),
        ("grid_balanced", int(np.argmin(avg + 0.5 * worst))),
        ("grid_best_worst", int(np.argmin(worst))),
    )
    out = []
    for name, idx in choices:
        weight = weights[idx]
        out.append(MixCandidate(name, float(weight[0]), float(weight[1]), float(weight[2])))
    return out


def _future_q(sample: Sample, source: str) -> float:
    if source == "sf":
        return float(sample.sf_q)
    if source == "search":
        return float(sample.search_q)
    return float(sample.avg_q)


def _regret_for_path_ply(
    by_ply: dict[int, Sample],
    *,
    ply: int,
    missing_regret_impute: float,
    game_known_nonselfplay: bool,
) -> float:
    sample = by_ply.get(int(ply))
    if sample is None:
        return float(missing_regret_impute) if game_known_nonselfplay else 0.0
    if sample.has_sf_played_regret:
        return max(0.0, float(sample.sf_played_regret))
    if sample.is_selfplay is False:
        return float(missing_regret_impute)
    return 0.0


def _suffix_rows_by_spec(
    games: dict[int, list[Sample]],
    *,
    specs: list[SuffixKey],
    min_offset: int,
    max_offset: int,
    missing_regret_impute: float,
) -> dict[SuffixKey, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    game_ids_by_spec: dict[SuffixKey, list[int]] = {spec: [] for spec in specs}
    features_by_spec: dict[SuffixKey, list[tuple[float, float, float]]] = {spec: [] for spec in specs}
    targets_by_spec: dict[SuffixKey, list[float]] = {spec: [] for spec in specs}
    tau_values = sorted({tau for _source, tau, _adjusted in specs if tau is not None})
    sources = sorted({source for source, _tau, _adjusted in specs})
    weights_by_offset = {
        offset: {None: 1.0, **{tau: math.exp(-float(offset) / float(tau)) for tau in tau_values}}
        for offset in range(2, int(max_offset) + 1, 2)
    }
    for game_id, samples in games.items():
        by_ply = {s.ply: s for s in samples}
        known_nonselfplay = sum(1 for s in samples if s.is_selfplay is False)
        known_selfplay = sum(1 for s in samples if s.is_selfplay is True)
        game_known_nonselfplay = known_nonselfplay > known_selfplay
        for current_index, current in enumerate(samples):
            weighted_sum = dict.fromkeys(specs, 0.0)
            weight_sum = dict.fromkeys(specs, 0.0)
            path_regret = 0.0
            next_path_ply = int(current.ply)
            for future in samples[current_index + 1:]:
                offset = int(future.ply) - int(current.ply)
                if offset <= 0 or offset % 2 != 0:
                    continue
                if offset > int(max_offset):
                    break
                while next_path_ply < int(future.ply):
                    path_regret += _regret_for_path_ply(
                        by_ply,
                        ply=next_path_ply,
                        missing_regret_impute=float(missing_regret_impute),
                        game_known_nonselfplay=game_known_nonselfplay,
                    )
                    next_path_ply += 2
                if offset < int(min_offset) or offset > int(max_offset):
                    continue
                future_values = {source: _future_q(future, source) for source in sources}
                weights_by_tau = weights_by_offset[int(offset)]
                for source in sources:
                    raw_value = future_values[source]
                    adjusted_value = float(np.clip(raw_value - REGRET_TO_Q_SCALE * path_regret, -1.0, 1.0))
                    for tau, weight in weights_by_tau.items():
                        raw_spec = (source, tau, False)
                        if raw_spec in weighted_sum:
                            weighted_sum[raw_spec] += weight * raw_value
                            weight_sum[raw_spec] += weight
                        adjusted_spec = (source, tau, True)
                        if adjusted_spec in weighted_sum:
                            weighted_sum[adjusted_spec] += weight * adjusted_value
                            weight_sum[adjusted_spec] += weight
            for spec in specs:
                denom = weight_sum[spec]
                if denom <= 1e-12:
                    continue
                game_ids_by_spec[spec].append(int(game_id))
                features_by_spec[spec].append((current.sf_q, current.search_q, current.final_q))
                targets_by_spec[spec].append(float(weighted_sum[spec] / denom))
    return {
        spec: (
            np.asarray(game_ids_by_spec[spec], dtype=np.int64),
            np.asarray(features_by_spec[spec], dtype=np.float64),
            np.asarray(targets_by_spec[spec], dtype=np.float64),
        )
        for spec in specs
    }


def _fit_suffix_arrays(
    *,
    source: str,
    tau: float | None,
    regret_adjusted: bool,
    current: MixCandidate,
    game_ids: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
) -> SuffixFitRow | None:
    if target.size < 40:
        return None
    train, test = _game_fold_masks(game_ids)
    if int(test.sum()) < MIN_SPLIT_ROWS or int(train.sum()) < MIN_SPLIT_ROWS:
        return None
    weights = _fit_simplex(features[train], target[train])
    decay = "uniform" if tau is None else f"ema_tau{tau:g}"
    return SuffixFitRow(
        source=source,
        decay=decay,
        regret_adjusted=bool(regret_adjusted),
        n_train=int(train.sum()),
        n_test=int(test.sum()),
        w_sf=float(weights[0]),
        w_search=float(weights[1]),
        w_final=float(weights[2]),
        best_rmse=_rmse(features[test] @ weights, target[test]),
        current_rmse=_rmse(features[test] @ current.weights, target[test]),
        sf_only_rmse=_rmse(features[test, 0], target[test]),
        search_only_rmse=_rmse(features[test, 1], target[test]),
        final_only_rmse=_rmse(features[test, 2], target[test]),
    )


def _fmt(value: float) -> str:
    if not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.4f}"


def _fmt_mix(row: MixCandidate | HorizonScore | SuffixFitRow) -> str:
    return f"{row.w_sf:.3f}/{row.w_search:.3f}/{row.w_final:.3f}"


def _print_horizon_scores(horizons: list[int], rows: list[HorizonScore]) -> None:
    print("## Discrete Horizon Candidate Scores")
    print()
    print("- target: future same-side SF/search avg_q with path regret correction")
    print()
    header = "| candidate | weights sf/search/final | " + " | ".join(f"h{h}" for h in horizons)
    print(header + " | avg | worst |")
    print("|---|---|" + "---:|" * (len(horizons) + 2))
    for row in sorted(rows, key=lambda item: (item.avg_rmse, item.worst_rmse, item.candidate)):
        cells = " | ".join(_fmt(row.h_rmse.get(int(h), math.nan)) for h in horizons)
        print(f"| {row.candidate} | `{_fmt_mix(row)}` | {cells} | {_fmt(row.avg_rmse)} | {_fmt(row.worst_rmse)} |")
    print()


def _print_suffix_rows(rows: list[SuffixFitRow], current_name: str) -> None:
    print("## Whole-Suffix Objective Fits")
    print()
    print(
        "| source | decay | regret adjusted | n_test | best weights sf/search/final | "
        "best RMSE | " + current_name + " RMSE | sf/search/final only |"
    )
    print("|---|---|---:|---:|---|---:|---:|---|")
    for row in sorted(rows, key=lambda item: (item.source, item.regret_adjusted, item.decay)):
        print(
            f"| {row.source} | {row.decay} | {int(row.regret_adjusted)} | {row.n_test} | "
            f"`{_fmt_mix(row)}` | {_fmt(row.best_rmse)} | {_fmt(row.current_rmse)} | "
            f"`{_fmt(row.sf_only_rmse)}/{_fmt(row.search_only_rmse)}/{_fmt(row.final_only_rmse)}` |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare live WDL mixes to future-value target families.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--max-shards", type=int, default=20, help="newest shards to scan; 0 scans all")
    parser.add_argument(
        "--horizons",
        type=lambda raw: _parse_ints(raw, label="horizons", max_count=MAX_HORIZONS, require_even=True),
        default=_parse_ints("4,8,12,24,50", label="horizons", max_count=MAX_HORIZONS, require_even=True),
    )
    parser.add_argument("--taus", type=_parse_taus, default=_parse_taus("8,24"))
    parser.add_argument("--sources", type=_parse_sources, default=_parse_sources("avg"))
    parser.add_argument(
        "--min-offset",
        type=int,
        default=2,
        help="minimum even ply offset included in whole-suffix targets",
    )
    parser.add_argument(
        "--max-offset",
        type=int,
        default=50,
        help=f"maximum even ply offset included in whole-suffix targets, capped at {MAX_OFFSET}",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    min_offset = int(args.min_offset)
    max_offset = int(args.max_offset)
    if min_offset <= 0 or max_offset < min_offset or min_offset % 2 != 0 or max_offset % 2 != 0:
        parser.error("--min-offset and --max-offset must be positive even ply offsets with max >= min")
    if max_offset > MAX_OFFSET:
        parser.error(f"--max-offset must be <= {MAX_OFFSET}")

    replay_dir = args.replay_dir or _latest_replay_dir(args.run_dir)
    games, scan, mean_observed_regret = _load_samples(replay_dir, max_shards=int(args.max_shards))
    selected_trial_dir = _trial_dir_for_replay(args.run_dir, replay_dir)
    currents = _current_candidates(args.run_dir, args.config, trial_dir=selected_trial_dir)
    current = currents[0]

    pairs_by_horizon: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for horizon in args.horizons:
        game_ids, features, target, _bucket, _path_regret = _pairs_for_horizon(
            games,
            horizon=int(horizon),
            missing_regret_impute=mean_observed_regret,
            eval_bins=list(DEFAULT_EVAL_BINS),
        )
        if target.size:
            pairs_by_horizon[int(horizon)] = (game_ids, features, target)

    candidates: list[MixCandidate] = []
    seen: set[str] = set()
    for candidate in [
        *currents,
        MixCandidate("sf_only", 1.0, 0.0, 0.0),
        MixCandidate("search_only", 0.0, 1.0, 0.0),
        MixCandidate("final_only", 0.0, 0.0, 1.0),
        MixCandidate("sf_search_half", 0.5, 0.5, 0.0),
        *([fit] if (fit := _fit_portfolio_mix(pairs_by_horizon)) is not None else []),
        *_grid_candidates(pairs_by_horizon),
    ]:
        if candidate.name in seen:
            continue
        seen.add(candidate.name)
        candidates.append(candidate)
    horizon_scores = [_score_mix(candidate, pairs_by_horizon) for candidate in candidates]

    suffix_rows: list[SuffixFitRow] = []
    decay_values: list[float | None] = [None, *[float(tau) for tau in args.taus]]
    specs = [
        (str(source), tau, regret_adjusted)
        for source in args.sources
        for tau in decay_values
        for regret_adjusted in (False, True)
    ]
    suffix_arrays = _suffix_rows_by_spec(
        games,
        specs=specs,
        min_offset=min_offset,
        max_offset=max_offset,
        missing_regret_impute=mean_observed_regret,
    )
    for source, tau, regret_adjusted in specs:
        game_ids, features, target = suffix_arrays[(source, tau, regret_adjusted)]
        row = _fit_suffix_arrays(
            source=source,
            tau=tau,
            regret_adjusted=regret_adjusted,
            current=current,
            game_ids=game_ids,
            features=features,
            target=target,
        )
        if row is not None:
            suffix_rows.append(row)

    payload: dict[str, Any] = {
        "replay_dir": str(replay_dir),
        "scan": scan,
        "horizons": [int(h) for h in args.horizons],
        "taus": [float(tau) for tau in args.taus],
        "sources": [str(source) for source in args.sources],
        "min_offset": min_offset,
        "max_offset": max_offset,
        "mean_observed_regret": mean_observed_regret,
        "current_candidate": asdict(current),
        "horizon_scores": [asdict(row) for row in horizon_scores],
        "suffix_rows": [asdict(row) for row in suffix_rows],
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("# Future Value Target Diagnostics")
    print()
    print(f"- replay dir: `{replay_dir}`")
    print(
        "- scanned: "
        f"{scan['selected_shards']} shards, {scan['scanned_positions']} rows, "
        f"{scan['valid_samples']} valid samples, {scan['games']} games"
    )
    print(f"- current comparator: `{current.name}` = `{_fmt_mix(current)}`")
    print(f"- whole-suffix offsets: even plies `{min_offset}..{max_offset}`")
    print(f"- mean observed regret impute: `{mean_observed_regret:.6f}`")
    print()
    _print_horizon_scores([int(h) for h in args.horizons if int(h) in pairs_by_horizon], horizon_scores)
    _print_suffix_rows(suffix_rows, current.name)


if __name__ == "__main__":
    main()
