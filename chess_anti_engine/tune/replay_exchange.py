from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from chess_anti_engine.replay import DiskReplayBuffer
from chess_anti_engine.replay.shard import (
    copy_or_link_shard,
    delete_shard_path,
    find_shard_path,
    iter_shard_paths,
    load_shard_arrays,
)
from chess_anti_engine.tune._utils import (
    resolve_local_override_root,
    slice_array_batch,
    to_nonnegative_int,
)
from chess_anti_engine.utils.atomic import atomic_write_text

log = logging.getLogger(__name__)


def _trial_replay_shard_dir(*, config: dict, trial_dir: Path) -> Path:
    """Return replay shard storage for a trial, optionally outside Ray artifacts."""
    raw_root = str(config.get("tune_replay_root_override", "") or "").strip()
    if raw_root:
        replay_root = resolve_local_override_root(
            raw_root=raw_root,
            tune_work_dir=config.get("work_dir", trial_dir),
            suffix="replay",
        )
        return replay_root / Path(trial_dir).name / "replay_shards"
    return Path(trial_dir) / "replay_shards"


def _read_jsonl_rows(path: Path) -> list[dict]:
    """Read all valid JSON dict rows from a JSONL file.

    ``errors="replace"`` defends against invalid UTF-8 at a truncated EOF
    that would otherwise raise UnicodeDecodeError during iteration (before
    the inner try catches), losing every row after the bad one. Replacement
    chars then fail json.loads cleanly per-line. (Codex adversarial review.)
    """
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue  # partially flushed line — skip and keep reading
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []  # file vanished or unreadable mid-iteration — caller retries
    return rows


def _read_last_jsonl_row(path: Path) -> dict | None:
    """Read the last valid JSON dict row from a JSONL file (or None if empty)."""
    rows = _read_jsonl_rows(path)
    return rows[-1] if rows else None


def _latest_trial_result_row(trial_dir: Path) -> dict | None:
    """Read the latest result row from Ray Tune's JSONL stream."""
    return _read_last_jsonl_row(Path(trial_dir) / "result.json")


def _all_trial_result_rows(trial_dir: Path) -> list[dict]:
    """Read all result rows from Ray Tune's JSONL stream."""
    return _read_jsonl_rows(Path(trial_dir) / "result.json")


def _metric_from_result_row(row: dict) -> float | None:
  # Prefer EMA-smoothed metric — less noisy for GPBT exploit/explore
  # decisions.  Falls back to raw opponent_strength for older rows.
    v = row.get("opponent_strength_ema", row.get("opponent_strength"))
    if isinstance(v, (int, float)):
        fv = float(v)
        if math.isfinite(fv):
            return fv
    return None


def _latest_trial_snapshot(trial_dir: Path) -> dict | None:
    """Best-effort latest trial snapshot with metric + generation size."""
    row = _latest_trial_result_row(trial_dir)
    if row is None:
        return None
    metric = _metric_from_result_row(row)
    if metric is None:
        return None
    iter_idx = to_nonnegative_int(row.get("training_iteration", row.get("iter", -1)), default=-1)
    positions_added = to_nonnegative_int(row.get("positions_added", 0), default=0)
    return {
        "trial_dir": Path(trial_dir),
        "metric": float(metric),
        "iter": int(iter_idx),
        "positions_added": int(positions_added),
    }


def _all_trial_snapshots(trial_dir: Path) -> list[dict]:
    """Best-effort per-iteration snapshots with metric + generation size."""
    snapshots: list[dict] = []
    seen_iters: set[int] = set()
    for row in _all_trial_result_rows(trial_dir):
        metric = _metric_from_result_row(row)
        if metric is None:
            continue
        iter_idx = to_nonnegative_int(row.get("training_iteration", row.get("iter", -1)), default=-1)
        if iter_idx < 0 or iter_idx in seen_iters:
            continue
        seen_iters.add(iter_idx)
        positions_added = to_nonnegative_int(row.get("positions_added", 0), default=0)
        snapshots.append(
            {
                "trial_dir": Path(trial_dir),
                "metric": float(metric),
                "iter": int(iter_idx),
                "positions_added": int(positions_added),
            }
        )
    snapshots.sort(key=lambda s: int(s["iter"]))
    return snapshots


def _estimate_recent_shard_count(
    *,
    positions_added: int,
    shard_size: int,
    holdout_fraction: float,
) -> int:
    """Estimate shard count for the latest generation from positions_added."""
    pa = max(0, int(positions_added))
    if pa <= 0:
        return 0
    ss = max(1, int(shard_size))
  # Positions entering train replay are roughly (1-holdout_fraction) of total.
    hf = max(0.0, min(0.99, float(holdout_fraction)))
    train_positions = max(1, int(math.ceil(float(pa) * (1.0 - hf))))
    return max(1, int(math.ceil(float(train_positions) / float(ss))))


def _load_shard_arrays_with_retry(
    shard_path: Path,
    *,
    retries: int = 4,
    sleep_s: float = 0.15,
) -> tuple[dict[str, np.ndarray], dict] | None:
    """Best-effort shard read with short retries for in-flight writes."""
    attempts = max(1, int(retries))
    for i in range(attempts):
        try:
            return load_shard_arrays(shard_path, lazy=True)
        except Exception:
            if i + 1 < attempts and sleep_s > 0:
                time.sleep(float(sleep_s))
    return None


def _select_top_trial_snapshots(
    *,
    recipient_trial_dir: Path,
    top_k_trials: int,
    within_best_frac: float,
    min_metric: float,
) -> list[dict]:
    """Pick top sibling trials constrained by best-relative threshold."""
    parent = Path(recipient_trial_dir).parent
    if not parent.is_dir():
        return []

    try:
        recipient_resolved = recipient_trial_dir.resolve()
    except Exception:
        recipient_resolved = recipient_trial_dir

    snapshots: list[dict] = []
    for td in sorted(parent.glob("train_trial_*")):
        if not td.is_dir():
            continue
        try:
            td_resolved = td.resolve()
        except Exception:
            td_resolved = td
        if td_resolved == recipient_resolved:
            continue
        snap = _latest_trial_snapshot(td)
        if snap is None:
            continue
        snapshots.append(snap)

    if not snapshots:
        return []

    best_metric = max(float(s["metric"]) for s in snapshots)
    frac = max(0.0, min(1.0, float(within_best_frac)))
    cutoff = max(float(min_metric), best_metric * (1.0 - frac))
    eligible = [s for s in snapshots if float(s["metric"]) >= cutoff]
    eligible.sort(key=lambda s: float(s["metric"]), reverse=True)

    if top_k_trials > 0:
        eligible = eligible[:top_k_trials]
    return eligible


def _refresh_replay_shards_on_exploit(
    *,
    config: dict,
    replay_shard_dir: Path,
    recipient_trial_dir: Path,
    donor_trial_dir: Path | None,
    keep_recent_fraction: float,
    keep_older_fraction: float,
    donor_shards: int,
    donor_skip_newest: int,
    shard_size: int,
    holdout_fraction: float,
) -> dict[str, int]:
    """Refresh recipient replay after PB2 exploit.

    Strategy:
    - Keep a small fraction of recipient's *most recent* local generation
      (likely poor just before exploit) while retaining more of older history.
    - Copy a configurable slice of the donor's recent replay shards so exploit
      recipients actually inherit some stronger data instead of only weights.
    """
    replay_shard_dir = Path(replay_shard_dir)
    replay_shard_dir.mkdir(parents=True, exist_ok=True)

    keep_recent_fraction = max(0.0, min(1.0, float(keep_recent_fraction)))
    keep_older_fraction = max(0.0, min(1.0, float(keep_older_fraction)))
    donor_shards = int(donor_shards)  # -1 = copy all, 0 = disabled, N = last N
    donor_skip_newest = max(0, int(donor_skip_newest))
    shard_size = max(1, int(shard_size))
    holdout_fraction = max(0.0, min(0.99, float(holdout_fraction)))

    summary = {
        "local_before": 0,
        "local_deleted": 0,
        "local_recent_deleted": 0,
        "local_older_deleted": 0,
        "local_after_keep": 0,
        "local_final": 0,
        "donor_available": 0,
        "donor_selected": 0,
        "donor_copied": 0,
    }

    local_shards = iter_shard_paths(replay_shard_dir)
    summary["local_before"] = len(local_shards)

    recipient_snap = _latest_trial_snapshot(recipient_trial_dir)
    recipient_recent_shards = 0
    if recipient_snap is not None:
        recipient_recent_shards = _estimate_recent_shard_count(
            positions_added=int(recipient_snap.get("positions_added", 0)),
            shard_size=shard_size,
            holdout_fraction=holdout_fraction,
        )
    recipient_recent_shards = max(0, min(len(local_shards), int(recipient_recent_shards)))

    if local_shards:
        if recipient_recent_shards > 0:
            local_recent = local_shards[-recipient_recent_shards:]
            local_older = local_shards[:-recipient_recent_shards]
        else:
            local_recent = []
            local_older = local_shards

        keep_recent_n = int(math.ceil(len(local_recent) * keep_recent_fraction))
        keep_older_n = int(math.ceil(len(local_older) * keep_older_fraction))
        keep_recent = set(local_recent[-keep_recent_n:]) if keep_recent_n > 0 else set()
        keep_older = set(local_older[-keep_older_n:]) if keep_older_n > 0 else set()
        keep_set = keep_recent | keep_older

  # Keep at least one local shard to avoid an empty replay on corner cases.
        if not keep_set and local_shards:
            keep_set.add(local_shards[-1])

        for sp in local_shards:
            if sp in keep_set:
                continue
            is_recent = sp in local_recent
            try:
                delete_shard_path(sp)
                summary["local_deleted"] += 1
                if is_recent:
                    summary["local_recent_deleted"] += 1
                else:
                    summary["local_older_deleted"] += 1
            except Exception:
                pass

    local_after_keep = iter_shard_paths(replay_shard_dir)
    summary["local_after_keep"] = len(local_after_keep)

    donor_dir = Path(donor_trial_dir) if donor_trial_dir is not None else None
    donor_replay_dir = (
        _trial_replay_shard_dir(config=config, trial_dir=donor_dir)
        if donor_dir is not None
        else None
    )
    log.info(
        "exploit replay donor lookup: donor_replay_dir=%s is_dir=%s donor_shards=%d",
        donor_replay_dir,
        donor_replay_dir.is_dir() if donor_replay_dir else None,
        donor_shards,
    )
    if donor_shards != 0 and donor_replay_dir is not None and donor_replay_dir.is_dir():
        from chess_anti_engine.replay.shard import shard_index

        donor_files = iter_shard_paths(donor_replay_dir)
        summary["donor_available"] = len(donor_files)
        if donor_skip_newest > 0 and len(donor_files) > donor_skip_newest:
            donor_files = donor_files[:-donor_skip_newest]
        elif donor_skip_newest > 0:
            donor_files = []
        if donor_files:
            if donor_shards > 0:
                donor_files = donor_files[-donor_shards:]
            summary["donor_selected"] = len(donor_files)

            existing = iter_shard_paths(replay_shard_dir)
            next_idx = 0
            for sp in existing:
                idx = shard_index(sp)
                if idx >= 0:
                    next_idx = max(next_idx, idx + 1)

            for src in donor_files:
                dst = replay_shard_dir / f"shard_{next_idx:06d}{src.suffix}"
                next_idx += 1
                try:
                    copy_or_link_shard(src, dst)
                    summary["donor_copied"] += 1
                except Exception:
                    pass

    summary["local_final"] = len(iter_shard_paths(replay_shard_dir))
    return summary


def _load_import_state(path: Path) -> dict[str, int]:
    """Read the import-state JSON sidecar; return {} on any error."""
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    return {str(k): int(v) for k, v in obj.items() if isinstance(v, (int, float))}


def _resolve_source_shard_dir(
    td: Path, *, config: dict
) -> tuple[Path, bool] | None:
    """Pick the source's shard dir. Returns ``(dir, use_selfplay_export)`` or None.

    Prefer the clean selfplay-only export; fall back to the trial's replay_shards
    dir for trials predating selfplay_shards/.
    """
    src_dir = td / "selfplay_shards"
    use_selfplay_export = src_dir.is_dir() and bool(iter_shard_paths(src_dir))
    if not use_selfplay_export:
        src_dir = _trial_replay_shard_dir(config=config, trial_dir=td)
    return (src_dir, use_selfplay_export) if src_dir.is_dir() else None


def _collect_iter_shards(
    *,
    src_dir: Path,
    snap_iter: int,
    unseen_snap: dict,
    use_selfplay_export: bool,
    source_skip_newest: int,
    shard_size: int,
    holdout_fraction: float,
    max_shards_per_source: int,
    shards_loaded_this_source: int,
) -> list[Path]:
    if use_selfplay_export:
        maybe = find_shard_path(src_dir, snap_iter)
        return [maybe] if maybe is not None else []
  # Legacy: estimate which trailing shards belong to this iter.
    shards = iter_shard_paths(src_dir)
    if source_skip_newest > 0:
        shards = shards[:-source_skip_newest] if len(shards) > source_skip_newest else []
    n_recent = _estimate_recent_shard_count(
        positions_added=int(unseen_snap.get("positions_added", 0)),
        shard_size=shard_size,
        holdout_fraction=holdout_fraction,
    )
    iter_shards = shards[max(0, len(shards) - n_recent):] if n_recent > 0 else []
    if max_shards_per_source > 0:
        remaining = max_shards_per_source - shards_loaded_this_source
        iter_shards = iter_shards[-remaining:]
    return iter_shards


def _ingest_one_shard(
    sp: Path, *, buf: DiskReplayBuffer, share_fraction: float, summary: dict[str, int],
) -> bool:
    """Load one shard and add it to ``buf``. Returns True on success."""
    loaded = _load_shard_arrays_with_retry(sp, retries=5, sleep_s=0.12)
    if loaded is None:
        return False
    shard_arrs, _meta = loaded
    n_samples = int(np.asarray(shard_arrs["x"]).shape[0])
    if n_samples <= 0:
        return False
    if 0.0 < share_fraction < 1.0:
        k = max(1, int(round(n_samples * share_fraction)))
        chosen = buf.rng.choice(n_samples, size=k, replace=False)
        shard_arrs = slice_array_batch(shard_arrs, chosen)
        n_samples = int(np.asarray(shard_arrs["x"]).shape[0])
    buf.add_many_arrays(shard_arrs)
    summary["source_shards_loaded"] += 1
    summary["source_samples_ingested"] += int(n_samples)
    return True


def _share_top_replay_each_iteration(
    *,
    config: dict,
    recipient_trial_dir: Path,
    replay_shard_dir: Path,
    buf: DiskReplayBuffer,
    top_k_trials: int,
    within_best_frac: float,
    min_metric: float,
    source_skip_newest: int,
    shard_size: int,
    holdout_fraction: float,
    max_unseen_iters_per_source: int,
    max_shards_per_source: int = 0,
    share_fraction: float = 1.0,
) -> dict[str, int]:
    """Ingest recent unseen top-trial generations into the live replay buffer."""
    source_skip_newest = max(0, int(source_skip_newest))
    shard_size = max(1, int(shard_size))
    holdout_fraction = max(0.0, min(0.99, float(holdout_fraction)))
    max_unseen_iters_per_source = max(1, int(max_unseen_iters_per_source))
    max_shards_per_source = max(0, int(max_shards_per_source))

    source_snaps = _select_top_trial_snapshots(
        recipient_trial_dir=recipient_trial_dir,
        top_k_trials=int(top_k_trials),
        within_best_frac=float(within_best_frac),
        min_metric=float(min_metric),
    )

    summary = {
        "source_trials_selected": len(source_snaps),
        "source_trials_ingested": 0,
        "source_trials_skipped_repeat": 0,
        "source_shards_loaded": 0,
        "source_samples_ingested": 0,
    }
    if not source_snaps:
        return summary

    import_state_path = Path(replay_shard_dir) / "_import_state.json"
    import_state = _load_import_state(import_state_path)

    for snap in source_snaps:
        td = Path(snap["trial_dir"])
        source_key = str(td.resolve()) if td.exists() else str(td)
        last_imported_iter = int(import_state.get(source_key, -1))

        unseen_snaps = [
            s for s in _all_trial_snapshots(td)
            if int(s.get("iter", -1)) > last_imported_iter
        ]
        if not unseen_snaps:
            summary["source_trials_skipped_repeat"] += 1
            continue
        if len(unseen_snaps) > max_unseen_iters_per_source:
            unseen_snaps = unseen_snaps[-max_unseen_iters_per_source:]

        resolved = _resolve_source_shard_dir(td, config=config)
        if resolved is None:
            continue
        src_dir, use_selfplay_export = resolved

        ingested_any = False
        imported_iters: list[int] = []
        shards_loaded_this_source = 0

        for unseen_snap in reversed(unseen_snaps):
            if max_shards_per_source > 0 and shards_loaded_this_source >= max_shards_per_source:
                break
            snap_iter = int(unseen_snap.get("iter", -1))
            if snap_iter < 0:
                continue

            iter_shards = _collect_iter_shards(
                src_dir=src_dir,
                snap_iter=snap_iter,
                unseen_snap=unseen_snap,
                use_selfplay_export=use_selfplay_export,
                source_skip_newest=source_skip_newest,
                shard_size=shard_size,
                holdout_fraction=holdout_fraction,
                max_shards_per_source=max_shards_per_source,
                shards_loaded_this_source=shards_loaded_this_source,
            )
            iter_had_ingest = False
            for sp in iter_shards:
                if _ingest_one_shard(sp, buf=buf, share_fraction=share_fraction, summary=summary):
                    shards_loaded_this_source += 1
                    iter_had_ingest = True
                    ingested_any = True
            if iter_had_ingest:
                imported_iters.append(snap_iter)

        if ingested_any:
            summary["source_trials_ingested"] += 1
            if imported_iters:
                import_state[source_key] = max(imported_iters)

    try:
        atomic_write_text(
            import_state_path,
            json.dumps(import_state, sort_keys=True, indent=2),
        )
    except Exception:
        pass

    return summary
