#!/usr/bin/env python3
"""Fit future-eval blend weights using replay labels and model eval heads."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.replay.shard import (
    INPUT_HISTORY_ENCODING_ARRAY_KEY,
    load_shard_arrays,
    shard_index,
    shard_positions,
)
from scripts.diagnostic_replay_utils import (
    REGRET_TO_Q_SCALE as _REGRET_TO_Q_SCALE,
    bool_field as _bool_field,
    corr as _corr,
    final_q_from_wdl_target as _final_q_from_wdl_target,
    fit_simplex as _fit_simplex,
    latest_replay_dir as _latest_replay_dir,
    normalize_wdl as _normalize_wdl,
    record_skipped_shard as _record_skipped_shard,
    rmse as _rmse,
    select_shards as _select_shards,
)


DEFAULT_RUN_DIR = Path("runs/pbt2_small")
MAX_HORIZONS = 12


@dataclass(frozen=True, slots=True)
class SampleRef:
    shard: Path
    row: int


@dataclass(frozen=True, slots=True)
class Sample:
    ref: SampleRef
    game_id: int
    ply: int
    sf_q: float
    search_q: float
    final_q: float
    sf_played_regret: float
    has_sf_played_regret: bool

    @property
    def avg_q(self) -> float:
        return 0.5 * (self.sf_q + self.search_q)


@dataclass(frozen=True, slots=True)
class PairRow:
    horizon: int
    game_id: int
    ref: SampleRef
    sf_now: float
    search_now: float
    final_q: float
    target_raw: float
    target_adjusted: float
    path_regret: float


def _latest_trial_dir(run_dir: Path) -> Path:
    trials = sorted((run_dir / "tune").glob("train_trial_*"), key=lambda p: p.stat().st_mtime)
    if not trials:
        raise FileNotFoundError(f"No trial directories under {run_dir / 'tune'}")
    return trials[-1]


def _latest_checkpoint(trial_dir: Path) -> Path:
    direct = trial_dir / "ckpt" / "trainer.pt"
    if direct.is_file():
        return direct
    checkpoints = [
        candidate
        for checkpoint_dir in trial_dir.glob("checkpoint_*")
        if (candidate := checkpoint_dir / "trainer.pt").is_file()
    ]
    if checkpoints:
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"No checkpoint trainer.pt under {trial_dir}")


def _load_samples(
    replay_dir: Path,
    *,
    max_shards: int,
) -> tuple[dict[int, list[Sample]], dict[str, Any]]:
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
            # A diagnostic run should report and skip bad live replay shards instead of aborting.
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

        scan["valid_samples"] += int(rows.size)
        scan["samples_with_regret"] += int(has_regret[rows].sum())
        for row_i in rows:
            row = int(row_i)
            games[int(game_ids[row])].append(
                Sample(
                    ref=SampleRef(shard=shard, row=row),
                    game_id=int(game_ids[row]),
                    ply=int(plies[row]),
                    sf_q=float(sf_q[row]),
                    search_q=float(search_q[row]),
                    final_q=float(final_q[row]),
                    sf_played_regret=float(regret[row]) if bool(has_regret[row]) else math.nan,
                    has_sf_played_regret=bool(has_regret[row]),
                ),
            )
    for samples in games.values():
        samples.sort(key=lambda s: s.ply)
    scan["games"] = len(games)
    return games, scan


def _build_pairs(games: dict[int, list[Sample]], horizons: list[int]) -> list[PairRow]:
    out: list[PairRow] = []
    for samples in games.values():
        by_ply = {s.ply: s for s in samples}
        for current in samples:
            for horizon in horizons:
                future = by_ply.get(current.ply + horizon)
                if future is None:
                    continue
                path_regret = 0.0
                path_known = True
                for offset in range(0, int(horizon), 2):
                    path_sample = by_ply.get(current.ply + offset)
                    if path_sample is None or not path_sample.has_sf_played_regret:
                        path_known = False
                        break
                    path_regret += max(0.0, float(path_sample.sf_played_regret))
                if not path_known:
                    continue
                target_raw = future.avg_q
                target_adjusted = float(np.clip(target_raw - _REGRET_TO_Q_SCALE * path_regret, -1.0, 1.0))
                out.append(
                    PairRow(
                        horizon=int(horizon),
                        game_id=int(current.game_id),
                        ref=current.ref,
                        sf_now=current.sf_q,
                        search_now=current.search_q,
                        final_q=current.final_q,
                        target_raw=float(target_raw),
                        target_adjusted=target_adjusted,
                        path_regret=float(path_regret),
                    ),
                )
    return out


def _predict_heads(
    refs: list[SampleRef],
    *,
    checkpoint: Path,
    device: str,
    batch_size: int,
) -> dict[tuple[str, int], tuple[float, float]]:
    if not refs:
        return {}
    import torch

    from chess_anti_engine.train.trainer import select_input_history_arrays
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    model = load_model_from_checkpoint(checkpoint, device=device)
    input_history_encoding = str(getattr(model, "input_history_encoding", "legacy"))
    by_shard: dict[Path, list[int]] = defaultdict(list)
    for i, ref in enumerate(refs):
        by_shard[ref.shard].append(i)

    preds: dict[tuple[str, int], tuple[float, float]] = {}
    with torch.inference_mode():
        for shard, ref_indices in sorted(by_shard.items(), key=lambda item: shard_index(item[0])):
            arrs, _meta = load_shard_arrays(shard, lazy=True)
            rows = np.asarray([refs[i].row for i in ref_indices], dtype=np.int64)
            order = np.argsort(rows)
            rows = rows[order]
            sorted_indices = [ref_indices[int(i)] for i in order]
            for start in range(0, int(rows.size), batch_size):
                row_batch = rows[start:start + batch_size]
                ref_batch = sorted_indices[start:start + batch_size]
                batch_arrs: dict[str, np.ndarray] = {"x": np.asarray(arrs["x"][row_batch])}
                if INPUT_HISTORY_ENCODING_ARRAY_KEY in arrs:
                    stored_history = np.asarray(arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY])
                    batch_arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY] = (
                        np.asarray(stored_history[row_batch])
                        if stored_history.ndim > 0 else stored_history
                    )
                if "x_lc0_root" in arrs:
                    batch_arrs["x_lc0_root"] = np.asarray(arrs["x_lc0_root"][row_batch])
                if "has_x_lc0_root" in arrs:
                    batch_arrs["has_x_lc0_root"] = np.asarray(arrs["has_x_lc0_root"][row_batch])
                selected = select_input_history_arrays(
                    batch_arrs,
                    input_history_encoding=input_history_encoding,
                )
                x = torch.from_numpy(np.asarray(selected["x"])).to(device=device, dtype=torch.float32)
                outputs = model(x)
                wdl_probs = torch.softmax(outputs["wdl"], dim=-1).detach().cpu().numpy()
                sf_probs = torch.softmax(outputs["sf_eval"], dim=-1).detach().cpu().numpy()
                wdl_q = wdl_probs[:, 0] - wdl_probs[:, 2]
                sf_eval_q = sf_probs[:, 0] - sf_probs[:, 2]
                for local_i, ref_index in enumerate(ref_batch):
                    ref = refs[ref_index]
                    preds[(str(ref.shard), int(ref.row))] = (
                        float(wdl_q[local_i]),
                        float(sf_eval_q[local_i]),
                    )
    return preds


def _summarize_fit(
    *,
    name: str,
    feature_names: list[str],
    features: np.ndarray,
    target: np.ndarray,
    game_ids: np.ndarray,
    fold: int,
) -> dict[str, Any]:
    test = (np.asarray(game_ids, dtype=np.int64) % 5) == int(fold)
    train = ~test
    if int(test.sum()) < max(20, len(feature_names) * 4) or int(train.sum()) < max(20, len(feature_names) * 4):
        train = np.ones_like(test, dtype=bool)
        test = np.ones_like(test, dtype=bool)
    weights = _fit_simplex(features[train], target[train])
    pred = features @ weights
    payload: dict[str, Any] = {
        "fit": name,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "rmse_train": _rmse(pred[train], target[train]),
        "rmse_test": _rmse(pred[test], target[test]),
        "corr_test": _corr(pred[test], target[test]),
    }
    for feat, weight in zip(feature_names, weights, strict=True):
        payload[f"w_{feat}"] = float(weight)
    return payload


def _rows_for_horizon(
    pairs: list[PairRow],
    preds: dict[tuple[str, int], tuple[float, float]],
    *,
    horizon: int,
    target_name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    selected = [p for p in pairs if p.horizon == horizon and (str(p.ref.shard), p.ref.row) in preds]
    n = len(selected)
    data: dict[str, np.ndarray] = {
        "sf_label": np.empty((n,), dtype=np.float64),
        "search_label": np.empty((n,), dtype=np.float64),
        "final": np.empty((n,), dtype=np.float64),
        "model_wdl": np.empty((n,), dtype=np.float64),
        "model_sf_eval": np.empty((n,), dtype=np.float64),
        "path_regret": np.empty((n,), dtype=np.float64),
    }
    target = np.empty((n,), dtype=np.float64)
    game_ids = np.empty((n,), dtype=np.int64)
    for i, pair in enumerate(selected):
        wdl_q, sf_eval_q = preds[(str(pair.ref.shard), pair.ref.row)]
        data["sf_label"][i] = pair.sf_now
        data["search_label"][i] = pair.search_now
        data["final"][i] = pair.final_q
        data["model_wdl"][i] = wdl_q
        data["model_sf_eval"][i] = sf_eval_q
        data["path_regret"][i] = pair.path_regret
        target[i] = pair.target_adjusted if target_name == "adjusted" else pair.target_raw
        game_ids[i] = pair.game_id
    meta = {
        "horizon": int(horizon),
        "target": target_name,
        "n": int(n),
        "target_mean": float(np.mean(target)) if n else math.nan,
        "path_regret_mean": float(np.mean(data["path_regret"])) if n else math.nan,
        "path_regret_p90": float(np.percentile(data["path_regret"], 90)) if n else math.nan,
    }
    return data, target, game_ids, meta


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = [
        "target",
        "horizon",
        "fit",
        "n_test",
        "rmse_test",
        "corr_test",
        "w_sf_label",
        "w_search_label",
        "w_final",
        "w_model_wdl",
        "w_model_sf_eval",
    ]
    cols = [c for c in preferred if any(c in row for row in rows)]
    widths: dict[str, int] = {}
    for col in cols:
        widths[col] = max(len(col), *(len(_format_cell(row.get(col))) for row in rows))
    print(" ".join(col.rjust(widths[col]) for col in cols))
    print(" ".join("-" * widths[col] for col in cols))
    for row in rows:
        print(" ".join(_format_cell(row.get(col)).rjust(widths[col]) for col in cols))


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def _try_latest_trial_dir(run_dir: Path) -> Path | None:
    try:
        return _latest_trial_dir(run_dir)
    except FileNotFoundError:
        return None


def _resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_horizons(values: list[int]) -> list[int]:
    out = [int(value) for value in values]
    if not out:
        raise argparse.ArgumentTypeError("at least one horizon is required")
    if any(value <= 0 for value in out):
        raise argparse.ArgumentTypeError("horizons must be positive")
    if any(value % 2 != 0 for value in out):
        raise argparse.ArgumentTypeError("horizons must be even ply offsets")
    if len(out) > MAX_HORIZONS:
        raise argparse.ArgumentTypeError(f"at most {MAX_HORIZONS} horizons are allowed")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit future-eval blend weights with the model's predicted sf_eval head.",
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--max-shards", type=int, default=256)
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        horizons = _normalize_horizons(args.horizons)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    device = _resolve_device(str(args.device))

    trial_dir = _try_latest_trial_dir(args.run_dir)
    if args.checkpoint is None and trial_dir is None:
        raise FileNotFoundError(
            f"No trial directories under {args.run_dir / 'tune'}; pass --checkpoint explicitly",
        )
    replay_dir = args.replay_dir if args.replay_dir is not None else _latest_replay_dir(args.run_dir, trial_dir)
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        if trial_dir is None:
            raise AssertionError("trial_dir unexpectedly missing")
        checkpoint = _latest_checkpoint(trial_dir)

    games, scan = _load_samples(replay_dir, max_shards=int(args.max_shards))
    pairs = _build_pairs(games, horizons=horizons)
    unique_refs = sorted(
        {pair.ref for pair in pairs},
        key=lambda ref: (str(ref.shard), int(ref.row)),
    )
    preds = _predict_heads(
        unique_refs,
        checkpoint=checkpoint,
        device=str(device),
        batch_size=int(args.batch_size),
    )

    rows: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    fits = [
        ("labels+final", ["sf_label", "search_label", "final"]),
        ("labels+final+sf_head", ["sf_label", "search_label", "final", "model_sf_eval"]),
        ("heads_only", ["model_wdl", "model_sf_eval"]),
        ("heads+labels", ["sf_label", "search_label", "model_wdl", "model_sf_eval"]),
        ("all", ["sf_label", "search_label", "final", "model_wdl", "model_sf_eval"]),
    ]
    for target_name in ("raw", "adjusted"):
        for horizon in horizons:
            data, target, game_ids, meta = _rows_for_horizon(
                pairs,
                preds,
                horizon=horizon,
                target_name=target_name,
            )
            metas.append(meta)
            if target.size == 0:
                continue
            for fit_name, feature_names in fits:
                features = np.stack([data[name] for name in feature_names], axis=1)
                row = _summarize_fit(
                    name=fit_name,
                    feature_names=feature_names,
                    features=features,
                    target=target,
                    game_ids=game_ids,
                    fold=int(args.fold),
                )
                row["target"] = target_name
                row["horizon"] = int(horizon)
                rows.append(row)

    payload = {
        "trial": None if trial_dir is None else trial_dir.name,
        "checkpoint": str(checkpoint),
        "replay_dir": str(replay_dir),
        "device": str(device),
        "scan": scan,
        "pairs": len(pairs),
        "unique_inference_rows": len(unique_refs),
        "target_meta": metas,
        "fits": rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"trial={None if trial_dir is None else trial_dir.name}")
        print(f"checkpoint={checkpoint}")
        print(f"replay_dir={replay_dir}")
        print(f"device={device} max_shards={args.max_shards}")
        print(f"scan={json.dumps(scan, sort_keys=True)}")
        print(f"pairs={len(pairs)} unique_inference_rows={len(unique_refs)}")
        for meta in metas:
            print(
                "target_meta "
                f"target={meta['target']} h={meta['horizon']} n={meta['n']} "
                f"mean={meta['target_mean']:.4f} "
                f"path_regret_mean={meta['path_regret_mean']:.4f} "
                f"path_regret_p90={meta['path_regret_p90']:.4f}"
            )
        _print_table(rows)


if __name__ == "__main__":
    main()
