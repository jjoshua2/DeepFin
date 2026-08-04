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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from chess_anti_engine.replay.shard import iter_shard_paths
from chess_anti_engine.tune._utils import (
    DURABLE_BEST_STATE,
    DURABLE_GATE_STATE,
    SIDECAR_HOLDOUT_ROWS,
)
from chess_anti_engine.tune.replay_exchange import _read_jsonl_rows

# How far (in checkpoint-dir index units) the newest result row may lag the
# newest on-disk ``checkpoint_*`` dir before we treat result.json as stale.
# Compared in checkpoint-index units, NOT training_iteration units: the two
# drift apart across Ray resumes (live trial 2026-07-08: row iter 691 ->
# checkpoint_000686), so an iteration-based comparison would false-positive.
_STALE_CHECKPOINT_TOLERANCE = 2


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
        "top_n": len(entries),
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


def _row_iter(row: dict) -> int:
    """Training iteration of a result row (-1 when absent/malformed)."""
    itv = row.get("training_iteration", row.get("iter", -1))
    return int(itv) if isinstance(itv, (int, float)) else -1


def _metric_value(row: dict | None, metric_key: str) -> float | None:
    """Finite numeric ``metric_key`` value of ``row`` (None when unavailable)."""
    if not isinstance(row, dict):
        return None
    v = row.get(metric_key)
    if isinstance(v, (int, float)) and np.isfinite(float(v)):
        return float(v)
    return None


def _checkpoint_index(name: str) -> int | None:
    """Numeric index of a Ray ``checkpoint_NNNNNN`` dir name (None if not one)."""
    m = re.fullmatch(r"checkpoint_(\d+)", name.strip())
    return int(m.group(1)) if m else None


def _newest_disk_checkpoint(td: Path) -> tuple[int, Path] | None:
    """Newest ``checkpoint_*`` dir (with ``trainer.pt``) in a trial dir.

    Checkpoint dirs are written directly by the trainable at report time, so
    unlike ``result.json`` (a Ray sync copy, see ``_plan_seed_export``) they
    are trustworthy ground truth for what state actually exists on disk.
    """
    best: tuple[int, Path] | None = None
    for d in td.glob("checkpoint_*"):
        if not d.is_dir() or not (d / "trainer.pt").exists():
            continue
        idx = _checkpoint_index(d.name)
        if idx is None:
            continue
        if best is None or idx > best[0]:
            best = (idx, d)
    return best


def _row_for_checkpoint(rows: list[dict], ckpt_name: str) -> dict | None:
    """Highest-iteration row whose ``checkpoint_dir_name`` is ``ckpt_name``."""
    best: dict | None = None
    for r in rows:
        if str(r.get("checkpoint_dir_name") or "").strip() != ckpt_name:
            continue
        if best is None or _row_iter(r) > _row_iter(best):
            best = r
    return best


def _last_row_checkpoint_index(rows: list[dict]) -> tuple[dict, int | None]:
    last_row = max(rows, key=_row_iter)
    last_idx = _checkpoint_index(str(last_row.get("checkpoint_dir_name") or "").strip())
    return last_row, last_idx


def _rows_fresh_against_newest(rows: list[dict], newest: tuple[int, Path] | None) -> bool:
    if newest is None:
        return True
    _, last_idx = _last_row_checkpoint_index(rows)
    return last_idx is not None and last_idx >= newest[0] - _STALE_CHECKPOINT_TOLERANCE


def _training_iteration_selection_key(
    metric: float, it: int, td: Path, rows: list[dict],
) -> tuple[float, int]:
    newest = _newest_disk_checkpoint(td)
    if newest is None or _rows_fresh_against_newest(rows, newest):
        return float(metric), int(it)
    last_row, last_idx = _last_row_checkpoint_index(rows)
    estimated_iter = float(newest[0])
    if last_idx is not None:
        estimated_iter += float(_row_iter(last_row) - last_idx)
    return estimated_iter, int(newest[0])


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
        it = _row_iter(row)
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


@dataclass(frozen=True)
class _ReplayPlan:
    """Where a slot's replay shards will come from, decided BEFORE anything is
    written. ``source is None`` means no candidate held any shards."""
    source: Path | None
    tried: tuple[Path, ...]


def _resolve_replay_shard_source(
    *, td: Path, replay_root_override: str, work_dir: object,
) -> _ReplayPlan:
    """Resolve a trial's replay-shard directory, preferring PRODUCTION's answer.

    `replay_exchange._trial_replay_shard_dir` is the authority on where the
    trainable actually writes shards, and it CHOOSES on
    ``tune_replay_root_override`` rather than falling through. Salvage asks it
    first — including `resolve_local_override_root`, which rewrites a
    `/mnt/c/chess_active/` root to `<run_root>_replay` and which a plain
    `.expanduser()` here would silently get wrong.

    The other location is kept only as a fallback for when production's answer
    is empty. Selecting on "contains shards" rather than `is_dir()` is the
    actual 2026-07-30 fix: the trial dir holds an EMPTY `replay_shards/`
    (created 07-27), so an `is_dir()` test on it succeeded vacuously, the
    override was never tried, and SEVEN revert points were banked weights-only.
    Preferring production's answer also means a stale non-empty decoy cannot
    outrank the live window — "non-empty wins" alone would have picked it and
    reported a plausible non-zero count.
    """
    from chess_anti_engine.tune.replay_exchange import (
        _trial_replay_shard_dir,
    )

    prod = _trial_replay_shard_dir(
        config={"tune_replay_root_override": replay_root_override,
                "work_dir": work_dir},
        trial_dir=td,
    )
    other = td / "replay_shards"
    candidates = [prod] if prod == other else [prod, other]
    for cand in candidates:
        if cand.is_dir() and iter_shard_paths(cand):
            return _ReplayPlan(source=cand, tried=tuple(candidates))
    return _ReplayPlan(source=None, tried=tuple(candidates))


def _copy_replay_shards(src_replay: Path, dst_replay: Path) -> int:
    """Copy every shard under ``src_replay`` into ``dst_replay``.

    Idempotent: an existing destination entry is removed first. Without this a
    re-run into the same ``--out`` dies with FileExistsError from copytree and
    the slot is silently dropped from the manifest — and the ledger protocol
    pins a fixed ``--out <label>``, so re-running into the same directory is
    the normal case, not an edge one. `replay/shard.py`'s `copy_or_link_shard`
    already pre-cleans; these two helpers must not disagree.
    """
    dst_replay.mkdir(parents=True, exist_ok=True)
    copied = 0
    for sp in iter_shard_paths(src_replay):
        dst = dst_replay / sp.name
        if dst.is_dir():
            shutil.rmtree(str(dst))
        elif dst.exists():
            dst.unlink()
        if sp.is_dir():
            shutil.copytree(str(sp), str(dst))
        else:
            shutil.copy2(str(sp), str(dst))
        copied += 1
    return copied


def _score_trials(
    tune_dir: Path, run_id: str, metric_key: str,
) -> list[tuple[float, int, Path, dict, list[dict]]]:
    """Score every trial under ``run_id`` by ``metric_key`` (best row per trial).

    Returns ``(metric, iter, trial_dir, best_row, all_rows)`` per trial —
    ``all_rows`` is carried through so ``_plan_seed_export`` can cross-check
    the pick against the on-disk checkpoints from the same read (re-reading
    result.json later could observe a different mid-sync state).
    """
    scored: list[tuple[float, int, Path, dict, list[dict]]] = []
    for td in sorted(d for d in tune_dir.glob(f"train_trial_{run_id}_*") if d.is_dir()):
        rows = _read_jsonl_rows(td / "result.json")
        if not rows:
            continue
        picked = _pick_best_row(rows, metric_key, td)
        if picked is None:
            continue
        metric, it, row = picked
        scored.append((float(metric), int(it), td, row, rows))
    return scored


def _resolve_seed_pool_out_dir(args: argparse.Namespace, run_id: str) -> Path:
    raw = getattr(args, "salvage_out_dir", None)
    if raw:
        return Path(str(raw))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(args.work_dir) / "salvage" / f"{run_id}_{stamp}"


@dataclass
class _SeedExportPlan:
    """Which on-disk state to export for a picked result row, and its truth."""

    ckpt_dir: Path
    ckpt_source: str  # result_row_checkpoint | newest_disk_checkpoint | mutable_ckpt_fallback
    row: dict | None  # result row matching the EXPORTED state (None if unknown)
    training_iteration: int | None  # true iteration of the exported state
    stale_result_rows: bool  # result.json lagged the on-disk checkpoints
    newest_ckpt_name: str | None
    row_ckpt_name: str


def _plan_seed_export(*, td: Path, rows: list[dict], row: dict, metric_key: str) -> _SeedExportPlan:
    """Decide which checkpoint dir to export for the picked ``row``.

    Ray's storage-side ``result.json`` is a wholesale sync COPY of the driver
    staging file (``/tmp/ray/session_*/artifacts/.../driver_artifacts/...``) —
    a read racing that truncate-and-rewrite, or a driver killed mid-sync,
    observes a truncated prefix, so the "newest" row can silently lag the
    on-disk checkpoints by hundreds of iterations (2026-07-08 incident: rows
    ended at 449 while checkpoint_000682 existed on disk). Checkpoint dirs are
    written directly by the trainable, so they are the ground truth:

    - ``metric_key == "training_iteration"`` means "snapshot the current
      state"; when the rows lag the newest on-disk checkpoint beyond
      ``_STALE_CHECKPOINT_TOLERANCE`` we export the newest on-disk checkpoint
      instead of the stale row (never silently export older than disk).
    - Other metrics legitimately pick older rows; we keep the pick but warn
      that a stale row scan may have missed newer rows, and when falling back
      to the mutable ``ckpt/`` dir we record its TRUE iteration (near-live)
      in the manifest rather than mislabelling it with the picked row's.
    """
    newest = _newest_disk_checkpoint(td)
    newest_name = newest[1].name if newest is not None else None
    row_ckpt_name = str(row.get("checkpoint_dir_name") or "").strip()
    last_row, _ = _last_row_checkpoint_index(rows)
    rows_fresh = _rows_fresh_against_newest(rows, newest)

    if not rows_fresh and newest is not None:
        print(
            f"[salvage] WARNING: result.json for {td.name} is STALE — its newest row "
            f"(iter {_row_iter(last_row)}, checkpoint {last_row.get('checkpoint_dir_name')!r}) "
            f"lags {newest_name} on disk. result.json in the tune dir is a Ray sync copy "
            f"and can be a truncated prefix after a killed or in-flight sync."
        )
        if metric_key == "training_iteration":
            match = _row_for_checkpoint(rows, str(newest_name))
            print(
                f"[salvage] WARNING: exporting the newest on-disk checkpoint "
                f"{newest_name} instead of the stale row."
            )
            return _SeedExportPlan(
                ckpt_dir=newest[1],
                ckpt_source="newest_disk_checkpoint",
                row=match,
                training_iteration=_row_iter(match) if match is not None else None,
                stale_result_rows=True,
                newest_ckpt_name=newest_name,
                row_ckpt_name=row_ckpt_name,
            )
        print(
            f"[salvage] WARNING: the best-{metric_key} scan may have missed newer rows."
        )

    if row_ckpt_name and (td / row_ckpt_name / "trainer.pt").exists():
        return _SeedExportPlan(
            ckpt_dir=td / row_ckpt_name,
            ckpt_source="result_row_checkpoint",
            row=row,
            training_iteration=_row_iter(row),
            stale_result_rows=not rows_fresh,
            newest_ckpt_name=newest_name,
            row_ckpt_name=row_ckpt_name,
        )

    # Mutable ckpt/ fallback: holds NEAR-LIVE state (rewritten every report),
    # not the picked row's state. Its true iteration is the newest row's when
    # the rows are fresh, and unknown when they are stale.
    fallback_row = last_row if rows_fresh else None
    fallback_iter = _row_iter(last_row) if rows_fresh else None
    print(
        f"[salvage] WARNING: using mutable ckpt/ fallback for {td.name} "
        f"(row checkpoint missing: {row_ckpt_name!r}); ckpt/ holds near-live state "
        f"(iteration {'unknown' if fallback_iter is None else fallback_iter}), "
        f"NOT the picked row's (iter {_row_iter(row)})."
    )
    return _SeedExportPlan(
        ckpt_dir=td / "ckpt",
        ckpt_source="mutable_ckpt_fallback",
        row=fallback_row,
        training_iteration=fallback_iter,
        stale_result_rows=not rows_fresh,
        newest_ckpt_name=newest_name,
        row_ckpt_name=row_ckpt_name,
    )


def _fmt_metric(v: float | None) -> str:
    return "n/a" if v is None else f"{float(v):.3f}"


def _fmt_iter(v: int | None) -> str:
    return "unknown" if v is None else str(int(v))


def _print_dry_run_plan(
    *,
    slot: int,
    picked_metric: float,
    picked_it: int,
    td: Path,
    plan: _SeedExportPlan,
    metric_key: str,
) -> None:
    newest = plan.newest_ckpt_name or "-"
    row_iter = _row_iter(plan.row) if plan.row is not None else None
    print(
        f"[salvage] DRY-RUN slot={slot:02d} "
        f"picked_metric={picked_metric:.3f} picked_iter={picked_it} "
        f"export_metric={_fmt_metric(_metric_value(plan.row, metric_key))} "
        f"export_iter={_fmt_iter(plan.training_iteration)} "
        f"src={td.name} ckpt={plan.ckpt_dir.name} "
        f"checkpoint_source={plan.ckpt_source} newest_on_disk={newest} "
        f"stale_result_rows={bool(plan.stale_result_rows)} "
        f"row_checkpoint={plan.row_ckpt_name or '-'} "
        f"matching_row_iter={_fmt_iter(row_iter)}"
    )


# The durable files a pool deliberately does NOT carry, and why — copied into
# every manifest entry so the omission is a recorded decision rather than a
# discovery made after a restart. `best/` and `best_regret/` are whole model
# exports (~250 MB each, plus optimizer state) and `gate_promoted_model.pt` is
# the same again: a pool is already ~2.3 GB with its replay window, and none of
# the three is needed to CONTINUE — the loop re-creates each on its next
# accept/promote. What is needed to continue is the WINDOW behind those
# artifacts, which is exactly what `gate_state.json` holds.
_DURABLE_STATE_NOT_EXPORTED: tuple[str, ...] = (
    "best/", "best_regret/", "gate_promoted_model.pt",
)


def _export_durable_trial_state(*, td: Path, seed_dir: Path) -> list[str]:
    """Copy the trial's durable-dir state files into the seed slot (audit T10).

    ``salvage-export`` copied the Ray checkpoint directory and nothing else, so
    a salvage restart silently began with an EMPTY promotion-gate window while
    an ordinary ``--resume`` kept it — the two restart modes disagreed about
    state that decides what the fleet is served, and nothing said so.

    ``gate_state.json`` is copied AND consumed: ``trainable`` seeds a trial that
    has no gate state of its own from the pool's, which is safe because the
    anchor (``gate_promoted_model.pt``) is not exported and
    ``GateHoldController.create`` RELEASES a restored hold whose anchor is
    absent rather than trusting it.

    ``best.json`` is copied and deliberately NOT consumed. It records a loss
    earned by the weights in ``best/``, which the pool does not carry, so
    restoring it as an active record would leave the new trial defending a
    number whose model is gone — and locking its own best-model tracking out
    until it beat that number. Banked for the operator and the manifest; the
    one field inside it a restart needs (``opp_strength_ema``) now rides in the
    checkpoint's ``trial_meta.json``, which IS consumed.
    """
    exported: list[str] = []
    for fn in (DURABLE_GATE_STATE, DURABLE_BEST_STATE):
        src = td / fn
        if src.is_file():
            shutil.copy2(str(src), str(seed_dir / fn))
            exported.append(fn)
    return exported


def _export_one_seed(
    *,
    slot: int,
    picked_metric: float,
    picked_it: int,
    td: Path,
    plan: _SeedExportPlan,
    metric_key: str,
    seeds_dir: Path,
    out_dir: Path,
    copy_replay: bool,
    replay_plan: _ReplayPlan,
) -> dict:
    seed_dir = seeds_dir / f"slot_{slot:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

  # SIDECAR_HOLDOUT_ROWS carries the holdout ruler so a salvage restart
  # resumes measuring `test_loss` against the same rows instead of a fresh
  # post-restart sample; trial_meta.json carries its frozen/generation flags
  # and (audit T1) the opponent-strength EMA.
    for fn in ("trainer.pt", "pid_state.json", "trial_meta.json", "rng_state.json",
               SIDECAR_HOLDOUT_ROWS):
        src = plan.ckpt_dir / fn
        if src.exists():
            shutil.copy2(str(src), str(seed_dir / fn))

    durable_exported = _export_durable_trial_state(td=td, seed_dir=seed_dir)

    # Align the exported pid_state only to the row that actually matches the
    # exported checkpoint. Aligning to a mismatched (stale) row stamps old
    # difficulty state onto newer weights — part of the 2026-07-08 incident.
    pid_state_overrides = (
        _align_pid_state(seed_dir / "pid_state.json", plan.row)
        if plan.row is not None
        else []
    )

    copied_shards = 0
    replay_source = ""
    # Only paths actually considered FOR COPYING are recorded; with
    # --no-copy-replay nothing was looked at on the seed's behalf, so an empty
    # list is the honest record rather than a list implying a search happened.
    replay_tried = [str(p) for p in replay_plan.tried] if copy_replay else []
    if copy_replay and replay_plan.source is not None:
        copied_shards = _copy_replay_shards(
            replay_plan.source, seed_dir / "replay_shards")
        replay_source = str(replay_plan.source)

    return {
        "slot": int(slot),
  # metric/training_iteration describe the EXPORTED state (None = unknown);
  # picked_row_* keep the metric-pick provenance.
        "metric": _metric_value(plan.row, metric_key),
        "training_iteration": plan.training_iteration,
        "picked_row_metric": float(picked_metric),
        "picked_row_training_iteration": int(picked_it),
        "source_trial_dir": str(td.resolve()),
        "checkpoint_source": str(plan.ckpt_source),
        "checkpoint_dir_name": plan.ckpt_dir.name,
        "row_checkpoint_dir_name": plan.row_ckpt_name,
        "newest_disk_checkpoint": plan.newest_ckpt_name,
        "stale_result_rows": bool(plan.stale_result_rows),
        "seed_dir": str(seed_dir.relative_to(out_dir)),
        "copied_replay_shards": int(copied_shards),
        # Which path the shards actually came from, and every path considered —
        # so a banked pool is self-describing and a zero count can be diagnosed
        # from the manifest alone, months later, without the source tree.
        "replay_shard_source": replay_source,
        "replay_shard_paths_tried": list(replay_tried),
        "pid_state_overrides": list(pid_state_overrides),
  # Which of the trial's DURABLE files (one level above the checkpoint) came
  # along, so a pool is self-describing about the gate/best state a restart
  # from it will and will not have. See `_export_durable_trial_state`.
        "durable_state_exported": list(durable_exported),
        "durable_state_not_exported": list(_DURABLE_STATE_NOT_EXPORTED),
        "result_row": plan.row,
    }


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
    top_n = int(getattr(args, "salvage_top_n", 0)) or int(getattr(args, "num_samples", 1))
    top_n = max(1, top_n)

    scored = _score_trials(tune_dir, run_id, metric_key)
    if not scored:
        raise SystemExit(
            f"No trials with metric '{metric_key}' found under run_id={run_id} in {tune_dir}"
        )

    if metric_key == "training_iteration":
        scored.sort(
            key=lambda x: _training_iteration_selection_key(x[0], x[1], x[2], x[4]),
            reverse=True,
        )
    else:
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = scored[: min(top_n, len(scored))]

    copy_replay = bool(getattr(args, "salvage_copy_replay", True))

    # ── PRE-FLIGHT: resolve every slot's replay source BEFORE writing anything.
    # This must run ahead of out_dir creation. Failing mid-loop instead would
    # abort AFTER earlier slots had already paid multi-GB copies, and — because
    # SystemExit is a BaseException and so slips past the per-slot
    # `except Exception` isolation below — would leave orphan seed dirs with NO
    # manifest.json. The operator's natural retry into the same --out (the
    # ledger protocol pins a fixed label) would then write a manifest saying
    # copied_replay_shards=0 while the leftover shards were still on disk, and
    # trainable_init._seed_replay_from_warmstart gates on the DIRECTORY, not on
    # the manifest — so a "weights-only" pool would silently seed the replay
    # window, possibly from a different trial if the ranking had shifted.
    # Deciding up front costs ~5ms of globbing and makes that unreachable.
    replay_plans: list[_ReplayPlan] = [
        _resolve_replay_shard_source(
            td=td, replay_root_override=replay_root_override,
            work_dir=getattr(args, "work_dir", tune_dir),
        )
        for (_m, _it, td, _row, _rows) in selected
    ]
    if copy_replay:
        missing = [
            (slot, selected[slot][2].name, replay_plans[slot])
            for slot in range(len(selected))
            if replay_plans[slot].source is None
        ]
        if missing:
            detail = "\n".join(
                f"  slot {slot:03d} ({name}) tried:\n"
                + "\n".join(f"    {p}" for p in plan.tried)
                for slot, name, plan in missing
            )
            raise SystemExit(
                "salvage-export: --salvage-copy-replay was requested but ZERO "
                f"replay shards were found for {len(missing)} of "
                f"{len(selected)} slot(s):\n" + detail +
                "\nA pool without a replay window cannot roll back a "
                "data-affecting change — the window holds ~a day of data made "
                "under the old settings. Nothing has been written. Pass "
                "--no-copy-replay to export weights+optimizer only on purpose, "
                "or set tune_replay_root_override.")

    dry_run = bool(getattr(args, "salvage_dry_run", False))
    if dry_run:
        print(
            f"[salvage] DRY-RUN: planning {len(selected)} seeds from run_id={run_id} "
            f"metric={metric_key}; no files will be written"
        )
        for slot, (metric, it, td, row, rows) in enumerate(selected):
            plan = _plan_seed_export(td=td, rows=rows, row=row, metric_key=metric_key)
            _print_dry_run_plan(
                slot=slot, picked_metric=metric, picked_it=it, td=td,
                plan=plan, metric_key=metric_key,
            )
            src = replay_plans[slot].source
            print(f"[salvage]   replay: "
                  f"{src if src is not None else 'NONE (would abort)'}"
                  + ("" if copy_replay else "  [--no-copy-replay]"))
        print("[salvage] DRY-RUN: no files written")
        return

    out_dir = _resolve_seed_pool_out_dir(args, run_id)
    seeds_dir = out_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    failed_slots: list[tuple[int, str]] = []
    for slot, (metric, it, td, row, rows) in enumerate(selected):
  # Each slot is best-effort independently. A single slot's OSError
  # (disk full, permission, source vanishes) or unanticipated failure
  # must not abort the run after we've done expensive replay-shard
  # copies for prior slots — we still want manifest.json to land
  # covering the slots that succeeded.
        try:
            plan = _plan_seed_export(td=td, rows=rows, row=row, metric_key=metric_key)
            entries.append(_export_one_seed(
                slot=slot, picked_metric=metric, picked_it=it, td=td,
                plan=plan, metric_key=metric_key,
                seeds_dir=seeds_dir, out_dir=out_dir,
                copy_replay=copy_replay, replay_plan=replay_plans[slot],
            ))
        except Exception as exc:  # per-slot isolation
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
        manifest["failed_slots"] = [{"slot": s, "error": e} for s, e in failed_slots]
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )

    print(
        f"[salvage] wrote {len(entries)} seeds from run_id={run_id} "
        f"metric={metric_key} to {out_dir}"
    )
    for e in entries:
        newest = e["newest_disk_checkpoint"] or "-"
        print(
            f"[salvage] slot={e['slot']:02d} metric={_fmt_metric(e['metric'])} "
            f"iter={_fmt_iter(e['training_iteration'])} shards={e['copied_replay_shards']} "
            f"src={Path(e['source_trial_dir']).name} "
            f"ckpt={e['checkpoint_dir_name']} (newest on disk: {newest})"
        )
