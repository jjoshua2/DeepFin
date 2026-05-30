#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chess_anti_engine.replay.shard import load_shard_arrays, shard_index, shard_positions
from chess_anti_engine.train.trainer import select_input_history_arrays
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


DEFAULT_RUN_DIR = Path("runs/pbt2_small")


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


def _latest_replay_dir(run_dir: Path, trial_dir: Path | None = None) -> Path:
    if trial_dir is not None:
        replay_dir = run_dir / "replay" / trial_dir.name / "replay_shards"
        if replay_dir.is_dir():
            return replay_dir
    replay_dirs = [p for p in (run_dir / "replay").glob("train_trial_*/replay_shards") if p.is_dir()]
    if not replay_dirs:
        raise FileNotFoundError(f"No replay shard directories under {run_dir / 'replay'}")
    return max(replay_dirs, key=lambda p: p.stat().st_mtime)


def _latest_checkpoint(trial_dir: Path) -> Path:
    checkpoints = sorted(trial_dir.glob("checkpoint_*"), key=lambda p: p.stat().st_mtime)
    if checkpoints:
        return checkpoints[-1] / "trainer.pt"
    direct = trial_dir / "ckpt" / "trainer.pt"
    if direct.is_file():
        return direct
    raise FileNotFoundError(f"No checkpoint trainer.pt under {trial_dir}")


def _select_shards(replay_dir: Path, max_shards: int) -> list[Path]:
    shards = sorted(replay_dir.glob("shard_*.zarr"), key=shard_index)
    return shards if max_shards <= 0 else shards[-max_shards:]


def _normalize_wdl(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wdl = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(wdl).all(axis=1)
    non_negative = (wdl >= 0.0).all(axis=1)
    sums = wdl.sum(axis=1)
    valid = finite & non_negative & (sums > 1e-12)
    out = np.zeros_like(wdl, dtype=np.float64)
    out[valid] = wdl[valid] / sums[valid, None]
    return out, valid


def _bool_field(arrs: dict[str, Any], name: str, n: int) -> np.ndarray:
    if name not in arrs:
        return np.zeros((n,), dtype=bool)
    return np.asarray(arrs[name], dtype=bool)


def _final_q_from_wdl_target(wdl_target: np.ndarray) -> np.ndarray:
    target = np.asarray(wdl_target, dtype=np.int64)
    return np.where(target == 0, 1.0, np.where(target == 1, 0.0, -1.0)).astype(np.float64)


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
            scan["skipped_shards"].append({"shard": str(shard), "reason": repr(exc)})
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
                target_adjusted = float(np.clip(target_raw - 2.0 * path_regret, -1.0, 1.0))
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


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    if target.size == 0:
        return math.nan
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return math.nan
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    denom = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
    if denom <= 1e-12:
        return math.nan
    return float(np.dot(aa, bb) / denom)


def _fit_simplex(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    n_features = int(features.shape[1])
    best_w = np.full((n_features,), 1.0 / max(1, n_features), dtype=np.float64)
    best_err = math.inf
    for mask in range(1, 1 << n_features):
        active = [i for i in range(n_features) if (mask >> i) & 1]
        x = features[:, active]
        if len(active) == 1:
            w_active = np.ones((1,), dtype=np.float64)
        else:
            gram = x.T @ x
            rhs = x.T @ target
            ones = np.ones((len(active), 1), dtype=np.float64)
            kkt = np.block([[gram, ones], [ones.T, np.zeros((1, 1), dtype=np.float64)]])
            vec = np.concatenate([rhs, np.ones((1,), dtype=np.float64)])
            try:
                sol = np.linalg.solve(kkt, vec)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(kkt, vec, rcond=None)[0]
            w_active = sol[: len(active)]
        if np.any(w_active < -1e-9):
            continue
        w = np.zeros((n_features,), dtype=np.float64)
        w[np.asarray(active, dtype=np.int64)] = np.maximum(0.0, w_active)
        total = float(w.sum())
        if total <= 1e-12:
            continue
        w /= total
        err = float(np.mean((features @ w - target) ** 2))
        if err < best_err:
            best_err = err
            best_w = w
    return best_w


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

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    trial_dir = _latest_trial_dir(args.run_dir)
    replay_dir = args.replay_dir if args.replay_dir is not None else _latest_replay_dir(args.run_dir, trial_dir)
    checkpoint = args.checkpoint if args.checkpoint is not None else _latest_checkpoint(trial_dir)

    games, scan = _load_samples(replay_dir, max_shards=int(args.max_shards))
    pairs = _build_pairs(games, horizons=[int(h) for h in args.horizons])
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
        for horizon in [int(h) for h in args.horizons]:
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
        "trial": trial_dir.name,
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
        print(f"trial={trial_dir.name}")
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
