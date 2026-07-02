#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from chess_anti_engine.replay.dataset import collate_arrays
from chess_anti_engine.replay.shard import (
    INPUT_HISTORY_ENCODING_ARRAY_KEY,
    iter_shard_paths,
    load_shard_arrays,
    shard_positions,
)
from chess_anti_engine.train.losses import _normalize_sf_wdl_probs, soft_cross_entropy
from chess_anti_engine.train.trainer import select_input_history_arrays
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


DEFAULT_RUN = Path("runs/pbt2_small")


@dataclass(frozen=True)
class BandSpec:
    name: str
    start_frac: float
    end_frac: float


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


def _latest_checkpoint(trial_dir: Path) -> Path:
    checkpoints = sorted(trial_dir.glob("checkpoint_*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_* directories under {trial_dir}")
    return checkpoints[-1]


def _latest_result(trial_dir: Path) -> dict[str, Any]:
    result_path = trial_dir / "result.json"
    last: dict[str, Any] = {}
    if not result_path.exists():
        return last
    with result_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def _loss_config(latest: dict[str, Any]) -> dict[str, float]:
    cfg_raw = latest.get("config")
    cfg: dict[str, Any] = cfg_raw if isinstance(cfg_raw, dict) else {}

    def get_float(key: str, default: float) -> float:
        value = latest.get(key)
        if value is None:
            value = cfg.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    return {
        "sf_wdl_frac": get_float("sf_wdl_frac", 0.0),
        "search_wdl_frac": get_float("search_wdl_frac", 0.0),
        "sf_wdl_temperature": get_float("sf_wdl_temperature", 1.0),
        "sf_search_dampen_sf_low": get_float("sf_search_dampen_sf_low", 0.0),
        "sf_search_dampen_sf_high": get_float("sf_search_dampen_sf_high", 0.0),
    }


def _prefix_counts(shards: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([shard_positions(p) for p in shards], dtype=np.int64)
    if int(counts.sum()) <= 0:
        raise RuntimeError("Replay directory has no readable positions")
    return counts, np.concatenate([[0], np.cumsum(counts)])


def _sample_global_offsets(
    rng: np.random.Generator,
    *,
    lo: int,
    hi: int,
    n: int,
) -> np.ndarray:
    span = hi - lo
    if span <= 0:
        return np.empty((0,), dtype=np.int64)
    n = min(int(n), span)
    return np.sort(rng.choice(np.arange(lo, hi, dtype=np.int64), size=n, replace=False))


def _group_offsets_by_shard(
    offsets: np.ndarray,
    prefix: np.ndarray,
) -> dict[int, np.ndarray]:
    shard_ids = np.searchsorted(prefix, offsets, side="right") - 1
    grouped: dict[int, list[int]] = defaultdict(list)
    for shard_id, offset in zip(shard_ids.tolist(), offsets.tolist(), strict=True):
        grouped[int(shard_id)].append(int(offset - prefix[shard_id]))
    return {k: np.asarray(v, dtype=np.int64) for k, v in grouped.items()}


def _take_rows(arrs: dict[str, Any], rows: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in arrs.items():
        shape = getattr(value, "shape", None)
        if key == INPUT_HISTORY_ENCODING_ARRAY_KEY:
            arr = np.asarray(value)
            if arr.ndim == 0:
                out[key] = np.full((int(rows.shape[0]),), arr.item(), dtype=arr.dtype)
            else:
                out[key] = np.asarray(arr[rows])
            continue
        if key.startswith("_"):
            out[key] = np.asarray(value)
            continue
        if shape is None or len(shape) == 0:
            continue
        if int(shape[0]) <= int(rows.max(initial=0)):
            continue
        out[key] = np.asarray(value[rows])
    return out


def _concat_batches(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = set.intersection(*(set(p) for p in parts))
    if any(INPUT_HISTORY_ENCODING_ARRAY_KEY in p for p in parts):
        keys.add(INPUT_HISTORY_ENCODING_ARRAY_KEY)
    out: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        if key == INPUT_HISTORY_ENCODING_ARRAY_KEY:
            values = []
            for part in parts:
                n = int(np.asarray(part["x"]).shape[0])
                if key not in part:
                    values.append(np.full((n,), "", dtype=object))
                    continue
                arr = np.asarray(part[key])
                if arr.ndim == 0:
                    values.append(np.full((n,), arr.item(), dtype=object))
                else:
                    values.append(arr)
            out[key] = np.concatenate(values, axis=0)
        elif key.startswith("_"):
            out[key] = parts[0][key]
        else:
            out[key] = np.concatenate([p[key] for p in parts], axis=0)
    return out


def _blended_wdl_ce(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    sf_wdl_frac: float,
    search_wdl_frac: float,
    sf_wdl_temperature: float,
    sf_search_dampen_sf_low: float,
    sf_search_dampen_sf_high: float,
) -> torch.Tensor:
    wdl_t = batch["wdl_t"].to(torch.int64)
    game_oh = F.one_hot(wdl_t, 3).float()
    sf_frac = max(0.0, float(sf_wdl_frac))
    search_frac = max(0.0, float(search_wdl_frac))
    blend_sum = sf_frac + search_frac
    if blend_sum > 1.0:
        sf_frac /= blend_sum
        search_frac /= blend_sum
        game_frac = 0.0
    else:
        game_frac = 1.0 - blend_sum

    target = game_frac * game_oh
    has_sf = batch.get("has_sf_wdl", torch.zeros_like(wdl_t, dtype=torch.float32)).float()
    has_search = batch.get("has_search_wdl", torch.zeros_like(wdl_t, dtype=torch.float32)).float()
    sf_probs = _normalize_sf_wdl_probs(batch.get("sf_wdl"), temperature=sf_wdl_temperature)
    search_probs = _normalize_sf_wdl_probs(batch.get("search_wdl"))

    if sf_probs is not None and search_probs is not None:
        sf_sig = sf_probs[:, 0] - sf_probs[:, 2]
        sr_sig = search_probs[:, 0] - search_probs[:, 2]
        joint = has_sf * has_search
        dis_sf_low = ((sf_sig < 0) & (sr_sig > 0)).float() * joint
        dis_sf_high = ((sf_sig > 0) & (sr_sig < 0)).float() * joint
    else:
        dis_sf_low = torch.zeros_like(has_sf)
        dis_sf_high = torch.zeros_like(has_sf)
    keep = 1.0 - float(sf_search_dampen_sf_low) * dis_sf_low - float(sf_search_dampen_sf_high) * dis_sf_high

    if sf_probs is not None:
        sf_eff = (has_sf * keep).unsqueeze(1)
        target = target + sf_frac * (sf_eff * sf_probs + (1.0 - sf_eff) * game_oh)
    else:
        target = target + sf_frac * game_oh
    if search_probs is not None:
        search_eff = has_search.unsqueeze(1)
        target = target + search_frac * (search_eff * search_probs + (1.0 - search_eff) * game_oh)
    else:
        target = target + search_frac * game_oh
    return soft_cross_entropy(logits, target.detach())


def _empty_stats() -> dict[str, float]:
    return {
        "n": 0.0,
        "wdl_ce_sum": 0.0,
        "blend_ce_sum": 0.0,
        "acc_sum": 0.0,
        "brier_sum": 0.0,
        "target_w_sum": 0.0,
        "target_d_sum": 0.0,
        "target_l_sum": 0.0,
        "prob_w_sum": 0.0,
        "prob_d_sum": 0.0,
        "prob_l_sum": 0.0,
        "has_sf_sum": 0.0,
        "has_search_sum": 0.0,
        "selfplay_sum": 0.0,
        "selfplay_tag_sum": 0.0,
    }


def _accumulate(
    stats: dict[str, float],
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    loss_cfg: dict[str, float],
) -> None:
    n = int(logits.shape[0])
    probs = torch.softmax(logits, dim=-1)
    target = batch["wdl_t"].to(torch.int64)
    one_hot = F.one_hot(target, 3).float()
    wdl_ce = F.cross_entropy(logits, target, reduction="none")
    blend_ce = _blended_wdl_ce(logits, batch, **loss_cfg)
    pred = probs.argmax(dim=-1)
    brier = ((probs - one_hot) ** 2).sum(dim=-1)

    stats["n"] += n
    stats["wdl_ce_sum"] += float(wdl_ce.sum().detach().cpu())
    stats["blend_ce_sum"] += float(blend_ce.sum().detach().cpu())
    stats["acc_sum"] += float((pred == target).float().sum().detach().cpu())
    stats["brier_sum"] += float(brier.sum().detach().cpu())
    sums = one_hot.sum(dim=0).detach().cpu().tolist()
    psums = probs.sum(dim=0).detach().cpu().tolist()
    stats["target_w_sum"] += float(sums[0])
    stats["target_d_sum"] += float(sums[1])
    stats["target_l_sum"] += float(sums[2])
    stats["prob_w_sum"] += float(psums[0])
    stats["prob_d_sum"] += float(psums[1])
    stats["prob_l_sum"] += float(psums[2])
    if "has_sf_wdl" in batch:
        stats["has_sf_sum"] += float(batch["has_sf_wdl"].float().sum().detach().cpu())
    if "has_search_wdl" in batch:
        stats["has_search_sum"] += float(batch["has_search_wdl"].float().sum().detach().cpu())
    if "has_is_selfplay" in batch:
        tagged = batch["has_is_selfplay"].float()
        stats["selfplay_tag_sum"] += float(tagged.sum().detach().cpu())
        if "is_selfplay" in batch:
            stats["selfplay_sum"] += float((batch["is_selfplay"].float() * tagged).sum().detach().cpu())


def _summarize(stats: dict[str, float]) -> dict[str, float]:
    n = max(1.0, stats["n"])
    tagged = max(1.0, stats["selfplay_tag_sum"])
    return {
        "n": stats["n"],
        "wdl_ce": stats["wdl_ce_sum"] / n,
        "blended_wdl_ce": stats["blend_ce_sum"] / n,
        "wdl_acc": stats["acc_sum"] / n,
        "brier": stats["brier_sum"] / n,
        "target_w": stats["target_w_sum"] / n,
        "target_d": stats["target_d_sum"] / n,
        "target_l": stats["target_l_sum"] / n,
        "prob_w": stats["prob_w_sum"] / n,
        "prob_d": stats["prob_d_sum"] / n,
        "prob_l": stats["prob_l_sum"] / n,
        "has_sf": stats["has_sf_sum"] / n,
        "has_search": stats["has_search_sum"] / n,
        "selfplay_frac": stats["selfplay_sum"] / tagged,
        "selfplay_tagged": stats["selfplay_tag_sum"] / n,
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = (
        "band",
        "samples",
        "shards",
        "shard_idx",
        "wdl_ce",
        "blend_ce",
        "acc",
        "brier",
        "target W/D/L",
        "model W/D/L",
        "sf/search",
        "selfplay",
    )
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["band"]),
                    str(int(row["n"])),
                    str(row["shards"]),
                    str(row["shard_idx"]),
                    f"{row['wdl_ce']:.5f}",
                    f"{row['blended_wdl_ce']:.5f}",
                    f"{row['wdl_acc']:.3f}",
                    f"{row['brier']:.5f}",
                    f"{row['target_w']:.3f}/{row['target_d']:.3f}/{row['target_l']:.3f}",
                    f"{row['prob_w']:.3f}/{row['prob_d']:.3f}/{row['prob_l']:.3f}",
                    f"{row['has_sf']:.2f}/{row['has_search']:.2f}",
                    f"{row['selfplay_frac']:.2f} ({row['selfplay_tagged']:.2f} tagged)",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate current WDL head by replay shard age band.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--trial-dir", type=Path)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples-per-band", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    trial_dir = args.trial_dir or _latest_trial_dir(args.run_dir)
    replay_dir = args.replay_dir or _replay_dir_for_trial(args.run_dir, trial_dir)
    checkpoint = args.checkpoint or _latest_checkpoint(trial_dir)
    latest = _latest_result(trial_dir)
    loss_cfg = _loss_config(latest)

    shards = iter_shard_paths(replay_dir)
    if not shards:
        raise FileNotFoundError(f"No replay shards under {replay_dir}")
    _counts, prefix = _prefix_counts(shards)
    total_positions = int(prefix[-1])
    bands = (
        BandSpec("oldest20", 0.00, 0.20),
        BandSpec("middle20", 0.40, 0.60),
        BandSpec("newest20", 0.80, 1.00),
    )

    print(f"trial={trial_dir.name}")
    print(f"checkpoint={checkpoint}")
    print(f"replay_dir={replay_dir}")
    print(f"replay_shards={len(shards)} replay_positions={total_positions}")
    print(f"loss_cfg={json.dumps(loss_cfg, sort_keys=True)}")

    model = load_model_from_checkpoint(checkpoint, device=args.device)
    input_history_encoding = str(getattr(model, "input_history_encoding", "legacy"))
    rng = np.random.default_rng(args.seed)
    rows_out: list[dict[str, Any]] = []
    with torch.no_grad():
        for band in bands:
            lo = round(total_positions * band.start_frac)
            hi = round(total_positions * band.end_frac)
            offsets = _sample_global_offsets(rng, lo=lo, hi=hi, n=args.samples_per_band)
            grouped = _group_offsets_by_shard(offsets, prefix)
            shard_ids = sorted(grouped)
            stats = _empty_stats()
            pending: list[dict[str, np.ndarray]] = []
            pending_n = 0
            for shard_id in shard_ids:
                arrs, _meta = load_shard_arrays(shards[shard_id], lazy=True)
                part = _take_rows(arrs, grouped[shard_id])
                if not part:
                    continue
                pending.append(part)
                pending_n += int(part["x"].shape[0])
                if pending_n >= args.batch_size:
                    merged = _concat_batches(pending)
                    merged = select_input_history_arrays(
                        merged,
                        input_history_encoding=input_history_encoding,
                    )
                    batch = collate_arrays(merged, device=args.device)
                    outputs = model(batch["x"])
                    _accumulate(stats, outputs["wdl"], batch, loss_cfg)
                    pending.clear()
                    pending_n = 0
            if pending:
                merged = _concat_batches(pending)
                merged = select_input_history_arrays(
                    merged,
                    input_history_encoding=input_history_encoding,
                )
                batch = collate_arrays(merged, device=args.device)
                outputs = model(batch["x"])
                _accumulate(stats, outputs["wdl"], batch, loss_cfg)
            summary = _summarize(stats)
            row: dict[str, Any] = dict(summary)
            row.update(
                {
                    "band": band.name,
                    "shards": len(shard_ids),
                    "shard_idx": f"{shard_ids[0]}-{shard_ids[-1]}" if shard_ids else "-",
                }
            )
            rows_out.append(row)

    _print_table(rows_out)


if __name__ == "__main__":
    main()
