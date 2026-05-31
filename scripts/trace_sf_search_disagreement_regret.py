#!/usr/bin/env python3
"""Trace Stockfish/search WDL disagreements through later regretful replies."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.replay.shard import load_shard_arrays, shard_index, shard_positions


DEFAULT_RUN_DIR = Path("runs/pbt2_small")
DEFAULT_MIN_GAP_REDUCTION = 0.02
MAX_JSON_TRACES = 50
MAX_SKIPPED_SHARDS = 20
MAX_TOP = 50


@dataclass(frozen=True, slots=True)
class Sample:
    game_id: int
    ply: int
    sf_q: float
    search_q: float
    regret: float
    rank: int
    priority: float
    priority_q_delta: float
    priority_sf_search_gap: float
    shard: str
    row: int

    @property
    def gap(self) -> float:
        return self.search_q - self.sf_q

    @property
    def abs_gap(self) -> float:
        return abs(self.gap)


@dataclass(frozen=True, slots=True)
class Trace:
    game_id: int
    trigger_ply: int
    event_ply: int
    after_ply: int | None
    direction: str
    verdict: str
    initial_gap: float
    event_gap: float
    after_gap: float | None
    gap_reduction: float | None
    search_toward_sf: float | None
    sf_toward_search: float | None
    trigger_sf_q: float
    trigger_search_q: float
    event_sf_q: float
    event_search_q: float
    after_sf_q: float | None
    after_search_q: float | None
    regret: float
    rank: int
    priority: float
    priority_q_delta: float
    priority_sf_search_gap: float
    shard: str
    row: int


def _latest_replay_dir(run_dir: Path) -> Path:
    replay_dirs = [p for p in (run_dir / "replay").glob("train_trial_*/replay_shards") if p.is_dir()]
    if not replay_dirs:
        raise FileNotFoundError(f"No replay shard directories under {run_dir / 'replay'}")
    return max(replay_dirs, key=lambda p: p.stat().st_mtime)


def _select_shards(replay_dir: Path, max_shards: int) -> list[Path]:
    shards = sorted(replay_dir.glob("shard_*.zarr"), key=shard_index)
    if max_shards > 0:
        return shards[-max_shards:]
    return shards


def _record_skipped_shard(stats: dict[str, Any], shard: Path, exc: Exception) -> None:
    if len(stats["skipped_shards"]) < MAX_SKIPPED_SHARDS:
        stats["skipped_shards"].append({"shard": str(shard), "reason": repr(exc)})
    else:
        stats["skipped_shards_omitted"] += 1


def _bool_field(arrs: dict[str, Any], name: str, n: int) -> np.ndarray:
    if name not in arrs:
        return np.zeros((n,), dtype=bool)
    return np.asarray(arrs[name], dtype=bool)


def _float_field(arrs: dict[str, Any], name: str, n: int, default: float = math.nan) -> np.ndarray:
    if name not in arrs:
        return np.full((n,), default, dtype=np.float64)
    return np.asarray(arrs[name], dtype=np.float64)


def _int_field(arrs: dict[str, Any], name: str, n: int, default: int = -1) -> np.ndarray:
    if name not in arrs:
        return np.full((n,), default, dtype=np.int64)
    return np.asarray(arrs[name], dtype=np.int64)


def _optional_float_field(
    arrs: dict[str, Any],
    value_name: str,
    has_name: str,
    n: int,
) -> np.ndarray:
    out = np.full((n,), math.nan, dtype=np.float64)
    if value_name not in arrs:
        return out
    values = np.asarray(arrs[value_name], dtype=np.float64)
    mask = _bool_field(arrs, has_name, n) if has_name in arrs else np.isfinite(values)
    out[mask] = values[mask]
    return out


def _optional_int_field(
    arrs: dict[str, Any],
    value_name: str,
    has_name: str,
    n: int,
) -> np.ndarray:
    out = np.full((n,), -1, dtype=np.int64)
    if value_name not in arrs:
        return out
    values = np.asarray(arrs[value_name], dtype=np.int64)
    mask = _bool_field(arrs, has_name, n) if has_name in arrs else values >= 0
    out[mask] = values[mask]
    return out


def _normalize_wdl(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wdl = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(wdl).all(axis=1)
    non_negative = (wdl >= 0.0).all(axis=1)
    sums = wdl.sum(axis=1)
    valid = finite & non_negative & (sums > 1e-12)
    out = np.zeros_like(wdl, dtype=np.float64)
    out[valid] = wdl[valid] / sums[valid, None]
    return out, valid


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


def _classify(
    trigger: Sample,
    after: Sample | None,
    *,
    min_gap_reduction: float,
) -> tuple[str, float | None, float | None, float | None]:
    if after is None:
        return "no_after_sample", None, None, None
    direction = 1.0 if trigger.gap > 0.0 else -1.0
    search_toward_sf = -direction * (after.search_q - trigger.search_q)
    sf_toward_search = direction * (after.sf_q - trigger.sf_q)
    gap_reduction = trigger.abs_gap - after.abs_gap
    if gap_reduction <= float(min_gap_reduction):
        return "still_split", gap_reduction, search_toward_sf, sf_toward_search
    search_pos = max(0.0, search_toward_sf)
    sf_pos = max(0.0, sf_toward_search)
    if search_pos <= 1e-9 and sf_pos <= 1e-9:
        return "gap_reduced_by_crossing", gap_reduction, search_toward_sf, sf_toward_search
    if search_pos > sf_pos * 1.25:
        return "search_moved_toward_sf", gap_reduction, search_toward_sf, sf_toward_search
    if sf_pos > search_pos * 1.25:
        return "sf_moved_toward_search", gap_reduction, search_toward_sf, sf_toward_search
    return "both_moved", gap_reduction, search_toward_sf, sf_toward_search


def _load_samples(replay_dir: Path, max_shards: int) -> tuple[dict[int, list[Sample]], dict[str, Any]]:
    shards = _select_shards(replay_dir, max_shards)
    games: dict[int, list[Sample]] = defaultdict(list)
    stats: dict[str, Any] = {
        "selected_shards": len(shards),
        "scanned_positions": 0,
        "valid_samples": 0,
        "samples_with_regret": 0,
        "skipped_shards": [],
        "skipped_shards_omitted": 0,
    }
    for shard in shards:
        n = int(shard_positions(shard))
        if n <= 0:
            continue
        stats["scanned_positions"] += n
        try:
            arrs, _meta = load_shard_arrays(shard, lazy=True)
        except Exception as exc:  # noqa: BLE001
            # Live diagnostics should keep scanning around partially written or corrupt shards.
            _record_skipped_shard(stats, shard, exc)
            continue

        required = ("game_id", "ply_index", "sf_wdl", "search_wdl")
        has_required = ("has_game_id", "has_ply_index", "has_sf_wdl", "has_search_wdl")
        if any(name not in arrs for name in (*required, *has_required)):
            continue

        has_game = _bool_field(arrs, "has_game_id", n)
        has_ply = _bool_field(arrs, "has_ply_index", n)
        has_sf = _bool_field(arrs, "has_sf_wdl", n)
        has_search = _bool_field(arrs, "has_search_wdl", n)
        sf_wdl, valid_sf = _normalize_wdl(np.asarray(arrs["sf_wdl"], dtype=np.float64))
        search_wdl, valid_search = _normalize_wdl(np.asarray(arrs["search_wdl"], dtype=np.float64))
        valid = has_game & has_ply & has_sf & has_search & valid_sf & valid_search
        rows = np.flatnonzero(valid)
        if rows.size == 0:
            continue

        game_ids = _int_field(arrs, "game_id", n)
        plies = _int_field(arrs, "ply_index", n)
        priority = _float_field(arrs, "priority", n, default=1.0)
        regret = _optional_float_field(arrs, "sf_played_regret", "has_sf_played_regret", n)
        rank = _optional_int_field(arrs, "sf_played_rank", "has_sf_played_rank", n)
        priority_q_delta = _optional_float_field(arrs, "priority_q_delta", "has_priority_q_delta", n)
        priority_sf_search_gap = _optional_float_field(
            arrs,
            "priority_sf_search_gap",
            "has_priority_sf_search_gap",
            n,
        )
        sf_q = sf_wdl[:, 0] - sf_wdl[:, 2]
        search_q = search_wdl[:, 0] - search_wdl[:, 2]
        stats["valid_samples"] += int(rows.size)
        stats["samples_with_regret"] += int(np.isfinite(regret[rows]).sum())
        shard_label = shard.name
        for row in rows:
            games[int(game_ids[row])].append(
                Sample(
                    game_id=int(game_ids[row]),
                    ply=int(plies[row]),
                    sf_q=float(sf_q[row]),
                    search_q=float(search_q[row]),
                    regret=float(regret[row]),
                    rank=int(rank[row]),
                    priority=float(priority[row]),
                    priority_q_delta=float(priority_q_delta[row]),
                    priority_sf_search_gap=float(priority_sf_search_gap[row]),
                    shard=shard_label,
                    row=int(row),
                ),
            )
    for samples in games.values():
        samples.sort(key=lambda s: (s.ply, s.shard, s.row))
    stats["games"] = len(games)
    return games, stats


def _build_trace(
    trigger: Sample,
    event: Sample,
    after: Sample | None,
    *,
    min_gap_reduction: float,
) -> Trace:
    verdict, gap_reduction, search_toward_sf, sf_toward_search = _classify(
        trigger,
        after,
        min_gap_reduction=min_gap_reduction,
    )
    direction = "search_high" if trigger.gap > 0.0 else "search_low"
    return Trace(
        game_id=trigger.game_id,
        trigger_ply=trigger.ply,
        event_ply=event.ply,
        after_ply=None if after is None else after.ply,
        direction=direction,
        verdict=verdict,
        initial_gap=trigger.gap,
        event_gap=event.gap,
        after_gap=None if after is None else after.gap,
        gap_reduction=gap_reduction,
        search_toward_sf=search_toward_sf,
        sf_toward_search=sf_toward_search,
        trigger_sf_q=trigger.sf_q,
        trigger_search_q=trigger.search_q,
        event_sf_q=event.sf_q,
        event_search_q=event.search_q,
        after_sf_q=None if after is None else after.sf_q,
        after_search_q=None if after is None else after.search_q,
        regret=event.regret,
        rank=event.rank,
        priority=trigger.priority,
        priority_q_delta=trigger.priority_q_delta,
        priority_sf_search_gap=trigger.priority_sf_search_gap,
        shard=trigger.shard,
        row=trigger.row,
    )


def _trace_games(
    games: dict[int, list[Sample]],
    *,
    gap_threshold: float,
    regret_threshold: float,
    max_trace_plies: int,
    include_trigger_regret: bool,
    min_gap_reduction: float,
) -> tuple[list[Trace], dict[str, Any]]:
    traces: list[Trace] = []
    trigger_gaps: list[float] = []
    direction_counts: Counter[str] = Counter()
    for samples in games.values():
        for i, trigger in enumerate(samples):
            if trigger.abs_gap < gap_threshold:
                continue
            trigger_gaps.append(trigger.abs_gap)
            direction_counts["search_high" if trigger.gap > 0.0 else "search_low"] += 1
            start = i if include_trigger_regret else i + 1
            event: Sample | None = None
            event_idx = -1
            for j in range(start, len(samples)):
                candidate = samples[j]
                if candidate.ply - trigger.ply > max_trace_plies:
                    break
                if math.isfinite(candidate.regret) and candidate.regret >= regret_threshold:
                    event = candidate
                    event_idx = j
                    break
            if event is None:
                continue
            after = samples[event_idx + 1] if event_idx + 1 < len(samples) else None
            if after is not None and after.ply - trigger.ply > max_trace_plies:
                after = None
            traces.append(_build_trace(trigger, event, after, min_gap_reduction=min_gap_reduction))
    summary: dict[str, Any] = {
        "triggers": len(trigger_gaps),
        "trigger_gap_quantiles": _quantiles(trigger_gaps),
        "trigger_directions": dict(direction_counts),
    }
    return traces, summary


def _finite_mean(values: list[float | None]) -> float | None:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def _summarize_traces(traces: list[Trace]) -> dict[str, Any]:
    verdicts = Counter(t.verdict for t in traces)
    with_after = [t for t in traces if t.after_ply is not None]
    return {
        "traces_with_regret_event": len(traces),
        "traces_with_after_sample": len(with_after),
        "verdicts": dict(verdicts),
        "mean_abs_initial_gap": _finite_mean([abs(t.initial_gap) for t in traces]),
        "mean_regret": _finite_mean([t.regret for t in traces]),
        "mean_gap_reduction_after": _finite_mean([t.gap_reduction for t in with_after]),
        "regret_quantiles": _quantiles([t.regret for t in traces]),
        "gap_reduction_quantiles": _quantiles([t.gap_reduction for t in with_after if t.gap_reduction is not None]),
    }


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _rank_traces(traces: list[Trace]) -> list[Trace]:
    return sorted(
        traces,
        key=lambda t: (
            t.after_ply is not None,
            -1.0 if t.gap_reduction is None else t.gap_reduction,
            abs(t.initial_gap),
        ),
        reverse=True,
    )


def _print_report(report: dict[str, Any], traces: list[Trace], top: int) -> None:
    scan = report["scan"]
    trigger = report["trigger_summary"]
    trace_summary = report["trace_summary"]
    print("# SF/search disagreement -> regret trace")
    print(f"- replay dir: `{report['replay_dir']}`")
    print(
        "- scanned: "
        f"{scan['selected_shards']} shards, {scan['scanned_positions']} rows, "
        f"{scan['valid_samples']} valid samples, {scan['games']} games"
    )
    print(
        "- thresholds: "
        f"abs(search_q - sf_q) >= {report['gap_threshold']:.3f}, "
        f"sf_played_regret >= {report['regret_threshold']:.4f}, "
        f"within {report['max_trace_plies']} plies"
    )
    print("- note: regret is recorded on the network-turn sample before the SF reply; verdict uses the next sample.")
    print(
        "- found: "
        f"{trigger['triggers']} disagreement triggers, "
        f"{trace_summary['traces_with_regret_event']} reached a regret event, "
        f"{trace_summary['traces_with_after_sample']} had an after-sample"
    )
    print(f"- trigger directions: `{trigger['trigger_directions']}`")
    print(f"- trigger gap quantiles: `{trigger['trigger_gap_quantiles']}`")
    print(f"- verdicts: `{trace_summary['verdicts']}`")
    print(
        "- means: "
        f"abs_initial_gap={_fmt_float(trace_summary['mean_abs_initial_gap'])}, "
        f"regret={_fmt_float(trace_summary['mean_regret'])}, "
        f"gap_reduction_after={_fmt_float(trace_summary['mean_gap_reduction_after'])}"
    )
    if scan["skipped_shards"]:
        skipped = len(scan["skipped_shards"]) + int(scan.get("skipped_shards_omitted", 0))
        print(f"- skipped shards: {skipped} ({len(scan['skipped_shards'])} shown)")
    if top <= 0 or not traces:
        return
    sorted_traces = _rank_traces(traces)[:top]
    print()
    print("Top traces:")
    print(
        "| game | trig | regret ply | after | verdict | gap0 | gap_after | "
        "reduced | regret | rank | sf0/search0 -> sf_after/search_after |"
    )
    print("|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|")
    for t in sorted_traces:
        after = "-" if t.after_ply is None else str(t.after_ply)
        print(
            f"| {t.game_id} | {t.trigger_ply} | {t.event_ply} | {after} | {t.verdict} | "
            f"{_fmt_float(t.initial_gap)} | {_fmt_float(t.after_gap)} | "
            f"{_fmt_float(t.gap_reduction)} | {_fmt_float(t.regret)} | {t.rank} | "
            f"{_fmt_float(t.trigger_sf_q)}/{_fmt_float(t.trigger_search_q)} -> "
            f"{_fmt_float(t.after_sf_q)}/{_fmt_float(t.after_search_q)} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trace positions where Stockfish WDL and search WDL disagree, then follow "
            "the same game to the next sufficiently regretful Stockfish reply."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--max-shards", type=int, default=256, help="newest shards to scan; 0 scans all")
    parser.add_argument("--gap-threshold", type=float, default=0.15)
    parser.add_argument("--regret-threshold", type=float, default=0.005)
    parser.add_argument("--max-trace-plies", type=int, default=80)
    parser.add_argument(
        "--min-gap-reduction",
        type=float,
        default=DEFAULT_MIN_GAP_REDUCTION,
        help="minimum after-sample absolute-gap reduction to count as progress",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--json-trace-limit", type=int, default=MAX_JSON_TRACES)
    parser.add_argument(
        "--exclude-trigger-regret",
        action="store_true",
        help="start looking after the trigger sample instead of allowing its attached SF reply to count",
    )
    args = parser.parse_args()
    if int(args.top) < 0 or int(args.top) > MAX_TOP:
        parser.error(f"--top must be between 0 and {MAX_TOP}")
    if int(args.json_trace_limit) < 0 or int(args.json_trace_limit) > MAX_JSON_TRACES:
        parser.error(f"--json-trace-limit must be between 0 and {MAX_JSON_TRACES}")

    replay_dir = args.replay_dir or _latest_replay_dir(args.run_dir)
    games, scan = _load_samples(replay_dir, int(args.max_shards))
    traces, trigger_summary = _trace_games(
        games,
        gap_threshold=float(args.gap_threshold),
        regret_threshold=float(args.regret_threshold),
        max_trace_plies=int(args.max_trace_plies),
        include_trigger_regret=not bool(args.exclude_trigger_regret),
        min_gap_reduction=float(args.min_gap_reduction),
    )
    report: dict[str, Any] = {
        "replay_dir": str(replay_dir),
        "max_shards": int(args.max_shards),
        "gap_threshold": float(args.gap_threshold),
        "regret_threshold": float(args.regret_threshold),
        "max_trace_plies": int(args.max_trace_plies),
        "min_gap_reduction": float(args.min_gap_reduction),
        "scan": scan,
        "trigger_summary": trigger_summary,
        "trace_summary": _summarize_traces(traces),
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        json_traces = _rank_traces(traces)[: int(args.json_trace_limit)]
        payload = dict(report)
        payload["traces_total"] = len(traces)
        payload["traces_included"] = len(json_traces)
        payload["traces_omitted"] = max(0, len(traces) - len(json_traces))
        payload["traces"] = [asdict(t) for t in json_traces]
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    _print_report(report, traces, top=int(args.top))


if __name__ == "__main__":
    main()
