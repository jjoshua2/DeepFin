"""Export top-N trial seeds (checkpoint + replay shards) for fresh restart.

A "salvage pool" is the input to ``salvage-restart``: a manifest plus
per-slot checkpoints (and optionally replay shards) extracted from past
trial results, so a new run can warm-start from the best historical state
rather than re-discovering it. See ``CLAUDE.md`` for the larger workflow.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from chess_anti_engine.replay.shard import iter_shard_paths
from chess_anti_engine.tune.replay_exchange import _read_jsonl_rows


def build_pool_manifest_dict(
    *,
    metric: str,
    entries: list[dict],
    label: str | None = None,
    source_tune_dir: Path | None = None,
    source_run_id: str | None = None,
) -> dict:
    """Salvage-pool ``manifest.json`` schema. Both ``export_seed_pool`` (run
    salvage from a tune dir) and ``trainable_report._update_best_regret_checkpoints``
    (auto-save best-regret slots during training) write through this so the
    on-disk format stays consistent — ``salvage-restart`` consumes either.
    """
    out: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metric": str(metric),
        "top_n": int(len(entries)),
        "entries": list(entries),
    }
    if label is not None:
        out["label"] = str(label)
    if source_tune_dir is not None:
        out["source_tune_dir"] = str(source_tune_dir.resolve())
    if source_run_id is not None:
        out["source_run_id"] = str(source_run_id)
    return out


def _trial_run_id_from_name(name: str) -> str | None:
    m = re.match(r"^train_trial_(?P<rid>[^_]+)_", name)
    if not m:
        return None
    return str(m.group("rid"))


def _latest_tune_run_id(tune_dir: Path) -> str | None:
  # Prefer latest PB2 policy log naming (pbt_policy_<runid>_<trial>.txt).
    policy_files = sorted(
        tune_dir.glob("pbt_policy_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in policy_files:
        m = re.match(r"^pbt_policy_(?P<rid>[^_]+)_\d+\.txt$", p.name)
        if m:
            return str(m.group("rid"))

  # Fallback: latest trial directory by mtime.
    best: tuple[float, str] | None = None
    for d in tune_dir.glob("train_trial_*"):
        if not d.is_dir():
            continue
        rid = _trial_run_id_from_name(d.name)
        if not rid:
            continue
        mt = float(d.stat().st_mtime)
        if best is None or mt > best[0]:
            best = (mt, rid)
    return best[1] if best is not None else None


def _pick_best_row(
    rows: list[dict], metric_key: str, td: Path,
) -> tuple[float, int, dict] | None:
    """Return ``(metric, iter, row)`` for the trial's best row.

    Prefers rows whose ``checkpoint_dir_name`` actually exists on disk;
    falls back to the absolute best-metric row if none has a present
    checkpoint (caller then falls back to ``ckpt/`` for trainer state).
    """
    best_any: tuple[float, int, dict] | None = None
    best_with_ckpt: tuple[float, int, dict] | None = None
    for row in rows:
        mv = row.get(metric_key)
        if not isinstance(mv, (int, float)):
            continue
        metric = float(mv)
        if not np.isfinite(metric):
            continue
        itv = row.get("training_iteration", row.get("iter", -1))
        it = int(itv) if isinstance(itv, (int, float)) else -1
        cand = (metric, it, row)
        if best_any is None or (metric, it) > (best_any[0], best_any[1]):
            best_any = cand
        ckname = row.get("checkpoint_dir_name")
        has_ckpt = (
            isinstance(ckname, str)
            and bool(ckname.strip())
            and (td / ckname / "trainer.pt").exists()
        )
        if has_ckpt and (best_with_ckpt is None or (metric, it) > (best_with_ckpt[0], best_with_ckpt[1])):
            best_with_ckpt = cand
    return best_with_ckpt or best_any


def _align_pid_state(pid_seed_path: Path, row: dict) -> list[str]:
    """Overwrite ``nodes`` and ``ema_winrate`` in pid_state.json with the
    salvaged row's values so difficulty doesn't carry stale state across
    a phase mismatch between checkpoint and result-row writes.
    """
    if not pid_seed_path.exists():
        return []
    try:
  # ValueError covers both UnicodeDecodeError (read_text decode failure)
  # and JSONDecodeError (its subclass). Without the broader catch a
  # corrupted/non-UTF-8 pid_state.json would propagate and abort the
  # salvage mid-loop, leaving an unusable pool with partial replay-shard
  # copies on disk.
        pid_obj = json.loads(pid_seed_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(pid_obj, dict):
        return []
    overrides: list[str] = []

    def _set_pid(dst_key: str, src_keys: tuple[str, ...], *, as_int: bool = False) -> None:
        for sk in src_keys:
            v = row.get(sk)
            if not isinstance(v, (int, float)):
                continue
            fv = float(v)
            if not np.isfinite(fv):
                continue
            pid_obj[dst_key] = int(fv) if as_int else fv
            overrides.append(f"{dst_key}<-{sk}")
            return

    _set_pid("nodes", ("sf_nodes_next", "sf_nodes"), as_int=True)
    _set_pid("ema_winrate", ("pid_ema_winrate",))

    if overrides:
        try:
            pid_seed_path.write_text(
                json.dumps(pid_obj, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            pass  # best-effort; caller continues with the in-memory state
    return overrides


def _copy_replay_shards(src_replay: Path, dst_replay: Path) -> int:
    """Copy every shard under ``src_replay`` into ``dst_replay``."""
    dst_replay.mkdir(parents=True, exist_ok=True)
    copied = 0
    for sp in iter_shard_paths(src_replay):
        dst = dst_replay / sp.name
        if sp.is_dir():
            shutil.copytree(str(sp), str(dst))
        else:
            shutil.copy2(str(sp), str(dst))
        copied += 1
    return copied


def export_seed_pool(args: argparse.Namespace) -> None:
    """Export top-N trial seeds (checkpoint + replay shards) for fresh restart."""
    tune_dir = Path(args.work_dir) / "tune"
    if not tune_dir.exists():
        raise SystemExit(f"Tune directory not found: {tune_dir}")

    run_id = (
        str(args.salvage_source_run_id)
        if args.salvage_source_run_id
        else _latest_tune_run_id(tune_dir)
    )
    if not run_id:
        raise SystemExit(f"Could not infer run id from {tune_dir}")

    metric_key = str(getattr(args, "salvage_metric", "opponent_strength"))
    replay_root_override = str(getattr(args, "tune_replay_root_override", "") or "").strip()
    top_n = int(getattr(args, "salvage_top_n", 0))
    if top_n <= 0:
        top_n = int(getattr(args, "num_samples", 1))
    top_n = max(1, top_n)

    trial_dirs = sorted([d for d in tune_dir.glob(f"train_trial_{run_id}_*") if d.is_dir()])
    scored: list[tuple[float, int, Path, dict]] = []
    for td in trial_dirs:
        rows = _read_jsonl_rows(td / "result.json")
        if not rows:
            continue
        picked = _pick_best_row(rows, metric_key, td)
        if picked is None:
            continue
        metric, it, row = picked
        scored.append((float(metric), int(it), td, row))

    if not scored:
        raise SystemExit(
            f"No trials with metric '{metric_key}' found under run_id={run_id} in {tune_dir}"
        )

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = scored[: min(top_n, len(scored))]

    out_dir_raw = getattr(args, "salvage_out_dir", None)
    if out_dir_raw:
        out_dir = Path(str(out_dir_raw))
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.work_dir) / "salvage" / f"{run_id}_{stamp}"
    seeds_dir = out_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    copy_replay = bool(getattr(args, "salvage_copy_replay", True))
    entries: list[dict] = []
    failed_slots: list[tuple[int, str]] = []
    for slot, (metric, it, td, row) in enumerate(selected):
  # Each slot is best-effort independently. A single slot's OSError
  # (disk full, permission, source vanishes) or unanticipated failure
  # must not abort the run after we've already done expensive
  # replay-shard copies for prior slots — we still want manifest.json
  # to land covering the slots that succeeded.
        try:
            seed_dir = seeds_dir / f"slot_{slot:03d}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            row_ckpt_name = row.get("checkpoint_dir_name")
            row_ckpt_dir = (
                td / str(row_ckpt_name)
                if isinstance(row_ckpt_name, str) and row_ckpt_name.strip()
                else None
            )
            ckpt_dir = (
                row_ckpt_dir
                if (row_ckpt_dir is not None and (row_ckpt_dir / "trainer.pt").exists())
                else (td / "ckpt")
            )
            ckpt_source = (
                "result_row_checkpoint" if ckpt_dir is row_ckpt_dir else "mutable_ckpt_fallback"
            )
            if ckpt_source != "result_row_checkpoint":
                print(
                    f"[salvage] WARNING: using fallback ckpt for {td.name} "
                    f"(row checkpoint missing: {row_ckpt_name})"
                )
            for fn in ("trainer.pt", "pid_state.json", "trial_meta.json", "rng_state.json"):
                src = ckpt_dir / fn
                if src.exists():
                    shutil.copy2(str(src), str(seed_dir / fn))

            pid_state_overrides = _align_pid_state(seed_dir / "pid_state.json", row)

            copied_shards = 0
            if copy_replay:
                src_replay = td / "replay_shards"
                if (not src_replay.is_dir()) and replay_root_override:
                    src_replay = Path(replay_root_override).expanduser() / td.name / "replay_shards"
                if src_replay.is_dir():
                    copied_shards = _copy_replay_shards(src_replay, seed_dir / "replay_shards")

            entries.append(
                {
                    "slot": int(slot),
                    "metric": float(metric),
                    "training_iteration": int(it),
                    "source_trial_dir": str(td.resolve()),
                    "checkpoint_source": str(ckpt_source),
                    "checkpoint_dir_name": str(row_ckpt_name) if isinstance(row_ckpt_name, str) else "",
                    "seed_dir": str(seed_dir.relative_to(out_dir)),
                    "copied_replay_shards": int(copied_shards),
                    "pid_state_overrides": list(pid_state_overrides),
                    "result_row": row,
                }
            )
        except Exception as exc:  # pylint: disable=broad-except  # per-slot isolation, see comment above
            failed_slots.append((int(slot), f"{type(exc).__name__}: {exc}"))
            print(
                f"[salvage] ERROR: slot {slot:03d} from {td.name} failed "
                f"({type(exc).__name__}: {exc}); manifest will skip this slot",
                flush=True,
            )

    manifest = build_pool_manifest_dict(
        metric=metric_key,
        entries=entries,
        source_tune_dir=tune_dir,
        source_run_id=run_id,
    )
    if failed_slots:
        manifest["failed_slots"] = [
            {"slot": s, "error": e} for s, e in failed_slots
        ]
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        f"[salvage] wrote {len(entries)} seeds from run_id={run_id} "
        f"metric={metric_key} to {out_dir}"
    )
    for e in entries:
        print(
            f"[salvage] slot={e['slot']:02d} metric={e['metric']:.3f} "
            f"iter={e['training_iteration']} shards={e['copied_replay_shards']} "
            f"src={Path(e['source_trial_dir']).name}"
        )
