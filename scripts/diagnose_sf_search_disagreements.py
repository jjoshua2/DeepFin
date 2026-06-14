#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.encoding.lc0 import LC0_HISTORY_ROOT_LEGACY_META
from chess_anti_engine.moves import index_to_move_for_encoding
from chess_anti_engine.replay.shard import load_shard_arrays, shard_positions


DEFAULT_RUN = Path("runs/pbt2_small")
_OUTCOME = ("W", "D", "L")
_DAMPEN_VALUES = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class ExampleRef:
    gap: float
    shard: Path
    row: int
    bucket: str
    is_selfplay: bool
    outcome: int
    sf_wdl: tuple[float, float, float]
    search_wdl: tuple[float, float, float]


_EXAMPLE_SEQ = 0


def _latest_trial_dir(run_dir: Path) -> Path:
    trials = sorted((run_dir / "tune").glob("train_trial_*"), key=lambda p: p.stat().st_mtime)
    if not trials:
        raise FileNotFoundError(f"No Ray trial directories under {run_dir / 'tune'}")
    return trials[-1]


def _replay_dir_for_trial(run_dir: Path, trial_dir: Path) -> Path:
    replay = run_dir / "replay" / trial_dir.name / "replay_shards"
    if not replay.exists():
        raise FileNotFoundError(f"Replay shard directory does not exist: {replay}")
    return replay


def _shard_num(path: Path) -> int:
    match = re.search(r"shard_(\d+)\.zarr$", path.name)
    return int(match.group(1)) if match else -1


def _newest_window_shards(replay_dir: Path, max_positions: int) -> list[tuple[Path, int, int]]:
    total = 0
    out: list[tuple[Path, int, int]] = []
    for shard in sorted(replay_dir.glob("shard_*.zarr"), key=_shard_num, reverse=True):
        n = int(shard_positions(shard))
        if n <= 0:
            continue
        take = min(n, max(0, int(max_positions) - total))
        if take <= 0:
            break
        start = n - take if take < n else 0
        out.append((shard, start, take))
        total += take
        if total >= int(max_positions):
            break
    return out


def _normalize(p: np.ndarray) -> np.ndarray:
    arr = np.asarray(p, dtype=np.float64)
    denom = np.maximum(arr.sum(axis=-1, keepdims=True), 1e-12)
    return arr / denom


def _ce_to_outcome(p: np.ndarray, outcome: np.ndarray) -> float:
    if int(outcome.size) == 0:
        return math.nan
    probs = np.clip(p[np.arange(outcome.size), outcome], 1e-9, 1.0)
    return float((-np.log(probs)).mean())


def _brier_to_outcome(p: np.ndarray, outcome: np.ndarray) -> float:
    if int(outcome.size) == 0:
        return math.nan
    target = np.zeros_like(p)
    target[np.arange(outcome.size), outcome] = 1.0
    return float(((p - target) ** 2).sum(axis=1).mean())


def _soft_ce(target: np.ndarray, pred: np.ndarray) -> float:
    if int(target.shape[0]) == 0:
        return math.nan
    target = np.clip(target, 1e-9, 1.0)
    pred = np.clip(pred, 1e-9, 1.0)
    return float((-(target * np.log(pred)).sum(axis=1)).mean())


def _blend_targets(
    game_oh: np.ndarray,
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

    sf_sig = sf[:, 0] - sf[:, 2]
    search_sig = search[:, 0] - search[:, 2]
    sf_low = (sf_sig < 0.0) & (search_sig > 0.0)
    sf_high = (sf_sig > 0.0) & (search_sig < 0.0)
    keep = (
        1.0
        - float(dampen_sf_low) * sf_low.astype(np.float64)
        - float(dampen_sf_high) * sf_high.astype(np.float64)
    )
    keep = np.clip(keep, 0.0, 1.0)[:, None]
    target = game_weight * game_oh
    target += sf_weight * (keep * sf + (1.0 - keep) * game_oh)
    target += search_weight * search
    return _normalize(target)


def _empty_stats() -> dict[str, Any]:
    return {
        "n": 0,
        "outcome": np.zeros((3,), dtype=np.int64),
        "sf_sum": np.zeros((3,), dtype=np.float64),
        "search_sum": np.zeros((3,), dtype=np.float64),
        "sf_ce_z_sum": 0.0,
        "search_ce_z_sum": 0.0,
        "sf_brier_z_sum": 0.0,
        "search_brier_z_sum": 0.0,
        "sf_ce_search_sum": 0.0,
        "selfplay_n": 0,
        "dampen": {float(v): {"ce_z_sum": 0.0, "brier_z_sum": 0.0, "ce_search_sum": 0.0} for v in _DAMPEN_VALUES},
    }


def _add_stats(
    stats: dict[str, Any],
    *,
    sf: np.ndarray,
    search: np.ndarray,
    outcome: np.ndarray,
    is_selfplay: np.ndarray,
    sf_frac: float,
    search_frac: float,
    dampen_sf_high: float,
) -> None:
    n = int(outcome.size)
    if n <= 0:
        return
    game_oh = np.zeros((n, 3), dtype=np.float64)
    game_oh[np.arange(n), outcome] = 1.0
    stats["n"] += n
    stats["outcome"] += np.bincount(outcome, minlength=3).astype(np.int64)
    stats["sf_sum"] += sf.sum(axis=0)
    stats["search_sum"] += search.sum(axis=0)
    stats["sf_ce_z_sum"] += _ce_to_outcome(sf, outcome) * n
    stats["search_ce_z_sum"] += _ce_to_outcome(search, outcome) * n
    stats["sf_brier_z_sum"] += _brier_to_outcome(sf, outcome) * n
    stats["search_brier_z_sum"] += _brier_to_outcome(search, outcome) * n
    stats["sf_ce_search_sum"] += _soft_ce(search, sf) * n
    stats["selfplay_n"] += int(np.asarray(is_selfplay, dtype=bool).sum())
    for damp in _DAMPEN_VALUES:
        blended = _blend_targets(
            game_oh,
            sf,
            search,
            sf_frac=sf_frac,
            search_frac=search_frac,
            dampen_sf_low=damp,
            dampen_sf_high=dampen_sf_high,
        )
        dst = stats["dampen"][float(damp)]
        dst["ce_z_sum"] += _ce_to_outcome(blended, outcome) * n
        dst["brier_z_sum"] += _brier_to_outcome(blended, outcome) * n
        dst["ce_search_sum"] += _soft_ce(search, blended) * n


def _fmt_wdl(p: Iterable[float]) -> str:
    vals = [float(x) for x in p]
    return f"{vals[0]:.3f}/{vals[1]:.3f}/{vals[2]:.3f}"


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    n = int(stats["n"])
    if n <= 0:
        return {"n": 0}
    dampen: dict[str, dict[str, float]] = {}
    out: dict[str, Any] = {
        "n": n,
        "selfplay_frac": float(stats["selfplay_n"] / n),
        "outcome_wdl": (stats["outcome"] / n).tolist(),
        "mean_sf_wdl": (stats["sf_sum"] / n).tolist(),
        "mean_search_wdl": (stats["search_sum"] / n).tolist(),
        "sf_ce_to_outcome": float(stats["sf_ce_z_sum"] / n),
        "search_ce_to_outcome": float(stats["search_ce_z_sum"] / n),
        "sf_brier_to_outcome": float(stats["sf_brier_z_sum"] / n),
        "search_brier_to_outcome": float(stats["search_brier_z_sum"] / n),
        "sf_ce_to_search": float(stats["sf_ce_search_sum"] / n),
        "dampen": dampen,
    }
    for damp, vals in stats["dampen"].items():
        dampen[f"{damp:.2f}"] = {
            "blend_ce_to_outcome": float(vals["ce_z_sum"] / n),
            "blend_brier_to_outcome": float(vals["brier_z_sum"] / n),
            "blend_ce_to_search": float(vals["ce_search_sum"] / n),
        }
    return out


def _push_example(heap: list[tuple[float, int, ExampleRef]], ref: ExampleRef, *, limit: int) -> None:
    if int(limit) <= 0:
        return
    global _EXAMPLE_SEQ  # noqa: PLW0603
    _EXAMPLE_SEQ += 1
    item = (float(ref.gap), int(_EXAMPLE_SEQ), ref)
    if len(heap) < int(limit):
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


_INV_MAP: dict[chess.Color, dict[tuple[int, int], chess.Square]] = {}


def _inverse_coord_map(turn: chess.Color) -> dict[tuple[int, int], chess.Square]:
    cached = _INV_MAP.get(turn)
    if cached is not None:
        return cached
    inv: dict[tuple[int, int], chess.Square] = {}
    for sq in chess.SQUARES:
        board = chess.Board(None)
        board.turn = turn
        board.set_piece_at(sq, chess.Piece(chess.PAWN, turn))
        x = encode_position(
            board,
            add_features=False,
            input_history_encoding=LC0_HISTORY_ROOT_LEGACY_META,
        )
        coords = np.argwhere(x[0] > 0.5)
        if coords.shape != (1, 2):
            raise RuntimeError(f"unexpected encoded coords for {sq}: {coords!r}")
        inv[(int(coords[0, 0]), int(coords[0, 1]))] = sq
    _INV_MAP[turn] = inv
    return inv


def _decode_current_board(x: np.ndarray) -> chess.Board:
    x = np.asarray(x, dtype=np.float32)
    turn = chess.BLACK if float(x[108].mean()) > 0.5 else chess.WHITE
    inv = _inverse_coord_map(turn)
    board = chess.Board(None)
    board.turn = turn
    for base, color in ((0, turn), (6, not turn)):
        for off, piece_type in enumerate(chess.PIECE_TYPES):
            for y, col in np.argwhere(x[base + off] > 0.5):
                board.set_piece_at(inv[(int(y), int(col))], chess.Piece(piece_type, color))

    rights = 0
    us, them = turn, not turn
    flags = [float(x[i].mean()) > 0.5 for i in range(104, 108)]
    for color, q_side, k_side in ((us, flags[0], flags[1]), (them, flags[2], flags[3])):
        if q_side:
            rights |= chess.BB_A1 if color == chess.WHITE else chess.BB_A8
        if k_side:
            rights |= chess.BB_H1 if color == chess.WHITE else chess.BB_H8
    board.castling_rights = rights
    board.ep_square = None
    board.halfmove_clock = int(round(float(x[109].mean()) * 100.0))
    board.fullmove_number = 1
    return board


def _decode_example(ref: ExampleRef) -> dict[str, Any]:
    arrs, meta = load_shard_arrays(ref.shard, lazy=True)
    policy_encoding = "lc0_1858" if int(meta.get("policy_size") or 4672) == 1858 else None
    x = np.asarray(arrs["x"][ref.row], dtype=np.float32)
    pre = _decode_current_board(x)
    policy = np.asarray(arrs["policy_target"][ref.row], dtype=np.float32)
    top_idx = int(np.argmax(policy))
    top_prob = float(policy[top_idx])
    net_move = None
    post = pre.copy(stack=False)
    net_legal = False
    try:
        net_move = index_to_move_for_encoding(top_idx, pre, policy_encoding=policy_encoding)
        net_legal = bool(net_move in pre.legal_moves)
        if net_legal:
            post.push(net_move)
    except Exception:
        net_move = None

    sf_idx = int(arrs["sf_move_index"][ref.row]) if "sf_move_index" in arrs else -1
    sf_move = None
    sf_legal = False
    if sf_idx >= 0:
        try:
            sf_move = index_to_move_for_encoding(sf_idx, post, policy_encoding=policy_encoding)
            sf_legal = bool(sf_move in post.legal_moves)
        except Exception:
            sf_move = None

    return {
        "bucket": ref.bucket,
        "gap": ref.gap,
        "shard": ref.shard.name,
        "row": ref.row,
        "is_selfplay": ref.is_selfplay,
        "outcome": _OUTCOME[int(ref.outcome)],
        "pre_fen": pre.fen(),
        "net_move": None if net_move is None else net_move.uci(),
        "net_target_prob": top_prob,
        "net_move_legal": net_legal,
        "post_fen": post.fen() if net_legal else None,
        "sf_reply": None if sf_move is None else sf_move.uci(),
        "sf_reply_legal_on_post": sf_legal,
        "sf_wdl_net_pov": list(ref.sf_wdl),
        "search_wdl_net_pov": list(ref.search_wdl),
    }


def _load_loss_config(trial_dir: Path, args: argparse.Namespace) -> dict[str, float]:
    latest: dict[str, Any] = {}
    result_path = trial_dir / "result.json"
    if result_path.exists():
        with result_path.open() as fh:
            for line in fh:
                if line.strip():
                    latest = json.loads(line)
    latest_cfg = latest.get("config")
    cfg = latest_cfg if isinstance(latest_cfg, dict) else {}

    def pick(key: str, default: float) -> float:
        override = getattr(args, key, None)
        if override is not None:
            return float(override)
        value = latest.get(key, None)
        if value is None:
            value = cfg.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    return {
        "sf_wdl_frac": pick("sf_wdl_frac", 0.1),
        "search_wdl_frac": pick("search_wdl_frac", 0.9),
        "sf_search_dampen_sf_high": pick("sf_search_dampen_sf_high", 0.0),
        "current_dampen_sf_low": pick("sf_search_dampen_sf_low", 0.0),
    }


def _print_report(
    *,
    trial_dir: Path,
    replay_dir: Path,
    scanned_positions: int,
    stats_by_bucket: dict[str, dict[str, Any]],
    examples: list[dict[str, Any]],
    loss_cfg: dict[str, float],
) -> None:
    print("# SF/Search disagreement diagnostics")
    print()
    print(f"- trial: `{trial_dir.name}`")
    print(f"- replay: `{replay_dir}`")
    print(f"- scanned newest positions: `{scanned_positions}`")
    print(
        f"- blend config used for counterfactuals: sf_wdl_frac={loss_cfg['sf_wdl_frac']:.3f}, "
        f"search_wdl_frac={loss_cfg['search_wdl_frac']:.3f}, "
        f"sf_high_dampen={loss_cfg['sf_search_dampen_sf_high']:.3f}"
    )
    print()
    print("## Buckets")
    print()
    for name, raw in stats_by_bucket.items():
        stats = _finalize_stats(raw)
        if int(stats.get("n", 0)) <= 0:
            continue
        print(f"### {name}")
        print()
        print(f"- n: `{stats['n']}`; selfplay: `{stats['selfplay_frac']:.1%}`")
        print(f"- outcome W/D/L: `{_fmt_wdl(stats['outcome_wdl'])}`")
        print(f"- mean SF WDL: `{_fmt_wdl(stats['mean_sf_wdl'])}`")
        print(f"- mean search WDL: `{_fmt_wdl(stats['mean_search_wdl'])}`")
        print(
            f"- CE to final outcome: SF `{stats['sf_ce_to_outcome']:.4f}`, "
            f"search `{stats['search_ce_to_outcome']:.4f}`"
        )
        print(
            f"- Brier to final outcome: SF `{stats['sf_brier_to_outcome']:.4f}`, "
            f"search `{stats['search_brier_to_outcome']:.4f}`"
        )
        print(f"- SF CE to search target: `{stats['sf_ce_to_search']:.4f}`")
        print("- dampen-sf-low counterfactuals:")
        print()
        print("| dampen | blend CE to outcome | blend Brier to outcome | blend CE to search |")
        print("|---:|---:|---:|---:|")
        for damp, vals in stats["dampen"].items():
            marker = " (current)" if abs(float(damp) - loss_cfg["current_dampen_sf_low"]) < 1e-9 else ""
            print(
                f"| {damp}{marker} | {vals['blend_ce_to_outcome']:.4f} | "
                f"{vals['blend_brier_to_outcome']:.4f} | {vals['blend_ce_to_search']:.4f} |"
            )
        print()
    if examples:
        print("## Largest-Gap Examples")
        print()
        for ex in examples:
            print(f"### {ex['bucket']} gap={ex['gap']:.3f} `{ex['shard']}:{ex['row']}`")
            print()
            print(f"- selfplay: `{ex['is_selfplay']}`; outcome: `{ex['outcome']}`")
            print(f"- pre: `{ex['pre_fen']}`")
            print(
                f"- net/search move: `{ex['net_move']}` "
                f"(target p={ex['net_target_prob']:.3f}, legal={ex['net_move_legal']})"
            )
            if ex["post_fen"] is not None:
                print(f"- post: `{ex['post_fen']}`")
            print(f"- SF reply on post: `{ex['sf_reply']}` (legal={ex['sf_reply_legal_on_post']})")
            print(f"- SF WDL net POV: `{_fmt_wdl(ex['sf_wdl_net_pov'])}`")
            print(f"- search WDL net POV: `{_fmt_wdl(ex['search_wdl_net_pov'])}`")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--trial-dir", type=Path, default=None)
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--window-positions", type=int, default=2_000_000)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--sf-wdl-frac", dest="sf_wdl_frac", type=float, default=None)
    parser.add_argument("--search-wdl-frac", dest="search_wdl_frac", type=float, default=None)
    parser.add_argument("--sf-search-dampen-sf-low", dest="sf_search_dampen_sf_low", type=float, default=None)
    parser.add_argument("--sf-search-dampen-sf-high", dest="sf_search_dampen_sf_high", type=float, default=None)
    args = parser.parse_args()

    trial_dir = args.trial_dir or _latest_trial_dir(args.run_dir)
    replay_dir = args.replay_dir or _replay_dir_for_trial(args.run_dir, trial_dir)
    loss_cfg = _load_loss_config(trial_dir, args)
    shards = _newest_window_shards(replay_dir, int(args.window_positions))
    stats_by_bucket = {
        "all": _empty_stats(),
        "sf_low_search_high": _empty_stats(),
        "sf_high_search_low": _empty_stats(),
        "agree": _empty_stats(),
    }
    heaps: dict[str, list[tuple[float, int, ExampleRef]]] = {
        "sf_low_search_high": [],
        "sf_high_search_low": [],
    }
    scanned = 0
    for shard, start, take in shards:
        arrs, _meta = load_shard_arrays(shard, lazy=True)
        n = int(take)
        scanned += n
        shard_slice = slice(int(start), int(start) + n)
        has_sf = np.asarray(arrs["has_sf_wdl"][shard_slice], dtype=bool)
        has_search = np.asarray(arrs["has_search_wdl"][shard_slice], dtype=bool)
        mask = has_sf & has_search
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        rows = idx + int(start)
        sf = _normalize(np.asarray(arrs["sf_wdl"][rows], dtype=np.float64))
        search = _normalize(np.asarray(arrs["search_wdl"][rows], dtype=np.float64))
        outcome = np.asarray(arrs["wdl_target"][rows], dtype=np.int64)
        is_selfplay = np.asarray(
            arrs.get("is_selfplay", np.zeros((int(shard_positions(shard)),), dtype=np.uint8))[shard_slice],
            dtype=bool,
        )[idx]
        sf_sig = sf[:, 0] - sf[:, 2]
        search_sig = search[:, 0] - search[:, 2]
        bucket_masks = {
            "all": np.ones((idx.size,), dtype=bool),
            "sf_low_search_high": (sf_sig < 0.0) & (search_sig > 0.0),
            "sf_high_search_low": (sf_sig > 0.0) & (search_sig < 0.0),
            "agree": (sf_sig == 0.0) | (search_sig == 0.0) | (np.sign(sf_sig) == np.sign(search_sig)),
        }
        for name, bm in bucket_masks.items():
            if not bm.any():
                continue
            _add_stats(
                stats_by_bucket[name],
                sf=sf[bm],
                search=search[bm],
                outcome=outcome[bm],
                is_selfplay=is_selfplay[bm],
                sf_frac=loss_cfg["sf_wdl_frac"],
                search_frac=loss_cfg["search_wdl_frac"],
                dampen_sf_high=loss_cfg["sf_search_dampen_sf_high"],
            )
        for name in ("sf_low_search_high", "sf_high_search_low"):
            bm = bucket_masks[name]
            if not bm.any():
                continue
            local = np.flatnonzero(bm)
            gaps = np.abs(sf_sig[local] - search_sig[local])
            for j in local[np.argsort(gaps)[-max(1, int(args.examples)):]]:
                ref = ExampleRef(
                    gap=float(abs(sf_sig[j] - search_sig[j])),
                    shard=shard,
                    row=int(rows[j]),
                    bucket=name,
                    is_selfplay=bool(is_selfplay[j]),
                    outcome=int(outcome[j]),
                    sf_wdl=(float(sf[j, 0]), float(sf[j, 1]), float(sf[j, 2])),
                    search_wdl=(float(search[j, 0]), float(search[j, 1]), float(search[j, 2])),
                )
                _push_example(heaps[name], ref, limit=max(0, int(args.examples)))

    refs = []
    for heap in heaps.values():
        refs.extend([item[2] for item in heap])
    refs.sort(key=lambda r: r.gap, reverse=True)
    examples = [_decode_example(ref) for ref in refs[: max(0, int(args.examples))]]
    report = {
        "trial": trial_dir.name,
        "replay_dir": str(replay_dir),
        "scanned_positions": scanned,
        "loss_config": loss_cfg,
        "buckets": {name: _finalize_stats(stats) for name, stats in stats_by_bucket.items()},
        "examples": examples,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    _print_report(
        trial_dir=trial_dir,
        replay_dir=replay_dir,
        scanned_positions=scanned,
        stats_by_bucket=stats_by_bucket,
        examples=examples,
        loss_cfg=loss_cfg,
    )


if __name__ == "__main__":
    main()
