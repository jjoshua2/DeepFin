"""Per-iteration reporting / persistence helpers for the Ray Tune trainable.

Writers and builders: CSV rows, TensorBoard scalars, best-model tracking,
Ray checkpoint directories, and the per-iteration Ray report dict.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import shutil
import time
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from chess_anti_engine.eval.era_probe import probe_metric_defaults
from chess_anti_engine.tune._utils import (
    SIDECAR_PID_STATE,
    SIDECAR_RNG_STATE,
    SIDECAR_TRIAL_META,
    load_optional_json,
)
from chess_anti_engine.tune.holdout_state import save_holdout_rows
from chess_anti_engine.tune.promotion_gate import gate_metrics
from chess_anti_engine.tune.trial_config import (
    DriftMetrics,
    PidResult,
    RestoreResult,
    SelfplayResult,
    TrainingResult,
    TrialConfig,
)
from chess_anti_engine.utils.atomic import atomic_write_text
import contextlib

log = logging.getLogger(__name__)


def _checkpoint_index(path: Path) -> int | None:
    """Index N from a ``checkpoint_NNNNNN`` directory name, or None."""
    suffix = path.name.removeprefix("checkpoint_")
    return int(suffix) if suffix.isdigit() else None


def _existing_checkpoint_dirs(trial_dir: Path) -> list[tuple[int, Path]]:
    """Indexed ``checkpoint_NNNNNN`` dirs under ``trial_dir``, oldest first.

    Sorted by parsed index rather than by name so the ordering stays correct if
    the run ever crosses the six-digit width Ray pads to.
    """
    out: list[tuple[int, Path]] = []
    for p in trial_dir.glob("checkpoint_*"):
        if not p.is_dir():
            continue
        idx = _checkpoint_index(p)
        if idx is not None:
            out.append((idx, p))
    out.sort()
    return out


def _prune_trial_checkpoints(*, trial_dir: Path, keep_last: int) -> None:
    """Best-effort deletion of old checkpoint_* dirs inside a Tune trial.

    This complements Ray's `CheckpointConfig(num_to_keep=...)`.
    In particular, it helps when resuming an older experiment whose RunConfig did
    not have checkpoint retention enabled.

    Deletions are logged. This runs in the trainable actor and trims by disk
    glob, so it cannot see Ray's checkpoint-manager list, which lives in the
    driver -- meaning it will remove directories Ray still tracks. That is
    tolerated (`_make_checkpoint_eviction_idempotent` keeps the resulting
    eviction from raising), but an unlogged deleter of ~650MB directories is
    impossible to attribute after the fact, and attributing one is exactly what
    the 2026-07-25 checkpoint-index investigation needed.
    """

    keep_last = int(keep_last)
    if keep_last <= 0:
        return

    ckpts = [p for _, p in _existing_checkpoint_dirs(trial_dir)]
    if len(ckpts) <= keep_last:
        return

  # `ignore_errors=True` means rmtree cannot tell us whether it worked, so ask
  # the filesystem afterwards. Logging the *selected* set as "pruned" would
  # recreate the very failure this log exists to prevent: an operator reading a
  # confident retention message while ~650MB dirs quietly accumulate.
    removed: list[str] = []
    survived: list[str] = []
    for p in ckpts[:-keep_last]:
        shutil.rmtree(p, ignore_errors=True)
        (survived if p.exists() else removed).append(p.name)

    if removed:
        log.info(
            "pruned %d trial checkpoint(s) to keep_last=%d: %s",
            len(removed), keep_last, ", ".join(removed),
        )
    if survived:
        log.warning(
            "failed to prune %d trial checkpoint(s); they still exist on disk "
            "and retention is NOT being realized: %s",
            len(survived), ", ".join(survived),
        )


def _guard_checkpoint_index(*, trial_dir: Path) -> int | None:
    """Refuse to resume onto a checkpoint index that would overwrite one on disk.

    Ray derives the checkpoint directory name purely from
    `StorageContext.current_checkpoint_index`, and that counter is advanced in
    two places that are meant to stay in lockstep: the actor's own copy in
    `Trainable.save`, and the driver's copy in `Trial.on_checkpoint`. Only the
    driver's copy is persisted, and only the driver's copy is skipped when
    `register_checkpoint` raises -- so a failed eviction leaves the persisted
    index behind the directories that actually exist.

    On the next restart the actor is seeded from that stale index and starts
    writing over live checkpoints. Observed 2026-07-25: restored from
    `checkpoint_000250`, then wrote 246 through 251, silently replacing five
    existing checkpoints including the restore source itself. Nothing in Ray
    objects to this -- `persist_current_checkpoint` overwrites whatever is at
    the path.

    `_make_checkpoint_eviction_idempotent` removes the cause; this removes the
    consequence, and is what repairs an index that has ALREADY drifted. Returns
    the index it advanced to, or None if it left things alone.
    """
    existing = _existing_checkpoint_dirs(trial_dir)
    if not existing:
        return None
    highest = existing[-1][0]

    try:
        from ray.train._internal.session import get_session

        storage = get_session().storage  # pyright: ignore[reportOptionalMemberAccess]
        current = int(storage.current_checkpoint_index)
    except Exception as exc:
        log.warning("Could not check the checkpoint index (non-fatal): %s", exc)
        return None

  # `_update_checkpoint_index` increments BEFORE persisting, so the next
  # directory written is `current + 1`. A collision means current < highest.
    if current >= highest:
        return None

    storage.current_checkpoint_index = highest
    print(
        f"[trial] checkpoint index had drifted to {current} while "
        f"checkpoint_{highest:06d} exists on disk; advanced to {highest} so the "
        f"next checkpoint does not overwrite a live one",
        flush=True,
    )
    return highest


_STATUS_COLS = (
    "iter", "global_iter", "opp", "opp_ema", "sf_nodes", "regret",
    "ingest_s", "train_s", "iter_s", "steps", "replay", "pos_added",
    # `lr_mean` (was `lr`): the iteration-mean LR of the matrix/trunk group.
    # The old column sampled the optimizer after the train phase, i.e. at the
    # trough of the sqrt_release ramp (docs/rl_loop_audit.md I19).
    "stale", "train_loss", "best_loss", "win", "draw", "loss", "lr_mean", "startup",
)


def _init_status_csv(trial_dir: Path) -> Path:
    """Write fresh status.csv header and return its path.

    Truncates any prior file — TensorBoard is the durable cross-restart
    record; status.csv is a per-segment compact view.
    """
    path = trial_dir / "status.csv"
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(_STATUS_COLS)
    return path


def _write_status_csv_row(
    path: Path,
    *,
    iteration_idx: int,
    global_iter: int,
    opp_strength: float,
    opp_strength_ema: float,
    sf_nodes: int,
    wdl_regret: float,
    ingest_ms: float,
    train_ms: float,
    total_iter_ms: float,
    steps: int,
    replay_size: int,
    positions_ingested: int,
    stale_games: int,
    train_loss: float | None,
    best_loss: float,
    total_w: int,
    total_d: int,
    total_l: int,
    opt_lr_mean: float,
    startup_source: str,
) -> None:
    """Append a compact status CSV row (best-effort)."""
    try:
        with path.open("a", newline="") as f:
            csv.writer(f).writerow([
                int(iteration_idx),
                int(global_iter),
                f"{float(opp_strength):.1f}",
                f"{float(opp_strength_ema):.1f}",
                int(sf_nodes),
                f"{float(wdl_regret):.4f}",
                f"{float(ingest_ms)/1000:.1f}",
                f"{float(train_ms)/1000:.1f}",
                f"{float(total_iter_ms)/1000:.1f}",
                int(steps),
                int(replay_size),
                int(positions_ingested),
                int(stale_games),
                f"{float(train_loss):.4f}" if train_loss is not None else "",
                f"{float(best_loss):.4f}",
                int(total_w),
                int(total_d),
                int(total_l),
                f"{float(opt_lr_mean):.2e}",
                str(startup_source),
            ])
    except Exception:
        pass


def _write_rng_state_sidecar(*, ckpt_dir: Path, rng) -> None:
    try:
        atomic_write_text(
            ckpt_dir / SIDECAR_RNG_STATE,
            json.dumps(rng.bit_generator.state, sort_keys=True),
        )
    except (OSError, TypeError, ValueError) as exc:
        # Silent loss here breaks deterministic resume — log once, don't crash.
        log.warning("[trial] failed to write rng_state.json: %s", exc)


def _save_trial_checkpoint(
    *,
    trainer,
    buf,
    ckpt_dir: Path,
    rng,
    trial_id: str,
    trial_dir: Path,
    config: dict,
    base_seed: int,
    restore: RestoreResult,
    iteration_idx: int,
    current_window: int,
    holdout_buf,
    holdout_frozen: bool,
    holdout_generation: int,
    holdout_ruler: str,
    opp_strength_ema: float,
    yaml_keys: Iterable[str],
    Checkpoint,
):
    """Flush replay buffer and save a lightweight checkpoint.

    The holdout rides along: its rows as a sidecar npz, its scalars in
    trial_meta.json. That is what makes `test_loss` comparable across a
    restart instead of being re-measured against whatever the next few
    iterations of ingest happen to donate. See ``tune/holdout_state``.

    `holdout_ruler` is the other half of that comparability: the set can come
    back unchanged and still be read by a different instrument, which is how a
    promotion across two rulers happened at iter 165. Storing it here — beside
    the generation and the rows, in the file both restore paths already read —
    is what lets the next restart notice.

    `opp_strength_ema` rides here for the same reason and none other: it is the
    scheduler's own objective metric (`GPBTPairwiseScheduler._metric`), it is
    what salvage and the sibling-share ranking sort on, and until this it died
    at every process start (audit T1). `best.json` also carries it, but that
    file is only rewritten when a best-model candidate is ACCEPTED, so it can be
    hundreds of iterations stale; trial_meta.json is rewritten with every
    checkpoint. The value stored is the EMA in effect when the checkpoint was
    written — this iteration's PID update runs after the save — so a resume
    continues the series one update behind instead of restarting it.

    `yaml_keys` is the flat key set of the yaml this process last loaded
    successfully. It is banked for one purpose: the next process compares the
    yaml it finds against it, so a key DELETED while the trial was down is
    reported instead of silently keeping its restored value (review B2). Key
    NAMES only, never values — the values are the yaml's job, and copying them
    here would create a second source of truth for the live config.
    """
    buf.flush()
    trainer.save(ckpt_dir / "trainer.pt")
    _write_rng_state_sidecar(ckpt_dir=ckpt_dir, rng=rng)
    save_holdout_rows(ckpt_dir=ckpt_dir, holdout_buf=holdout_buf)
    try:
        atomic_write_text(
            ckpt_dir / SIDECAR_TRIAL_META,
            json.dumps({
                "owner_trial_id": str(trial_id),
                "owner_trial_dir": str(trial_dir.resolve()),
                "optimizer": str(config.get("optimizer", "nadamw")).lower(),
                "base_seed": int(base_seed),
                "active_seed": int(restore.active_seed),
                "startup_source": str(restore.startup_source),
                "salvage_origin_used": bool(restore.salvage_origin_used),
                "salvage_origin_slot": int(restore.salvage_origin_slot),
                "salvage_origin_slots_total": int(restore.salvage_origin_slots_total),
                "salvage_origin_dir": str(restore.salvage_origin_dir),
                "global_iter": int(iteration_idx),
                "current_window": int(current_window),
                "holdout_frozen": bool(holdout_frozen),
                "holdout_generation": int(holdout_generation),
                "holdout_ruler": str(holdout_ruler),
                "opp_strength_ema": float(opp_strength_ema),
                "yaml_keys": sorted(yaml_keys),
            }, sort_keys=True, indent=2),
        )
    except (OSError, TypeError, ValueError) as exc:
        # trial_meta drives exploit-clone owner tracking; loud failure is the right call.
        log.warning("[trial] failed to write trial_meta.json: %s", exc)
    return Checkpoint.from_directory(str(ckpt_dir))


def _reset_best_on_cross_trial_restore(
    *, restore, best_loss: float, best_source: str,
    best_dir: Path, best_state_path: Path,
) -> tuple[float, str]:
    """Clear best-model tracking when the weights came from another trial.

    A PB2 exploit replaces the recipient's trainer with a donor checkpoint, but
    `best.json` and `best/` are the recipient's own and are not carried in the
    checkpoint. Keeping them would leave the record describing a lineage that
    no longer exists -- and if that record is on the holdout scale, the donor
    would have to beat an unrelated recipient number, using a holdout that is
    itself empty at that moment, before the best model could ever be replaced.

    The on-disk record is deleted, not merely superseded in memory. Resetting
    only the in-memory tuple leaves `best.json` and `best/` describing the
    discarded lineage until some later iteration happens to accept a metric --
    and if none does, an ordinary resume after this one reloads the stale
    record, because `cross_trial_restore` is false by then. An operator reading
    `best/` in the meantime would see the wrong model with no indication.
    """
    if not getattr(restore, "cross_trial_restore", False):
        return best_loss, best_source

    print(
        f"[trial] cross-trial restore ({getattr(restore, 'startup_source', '?')}): "
        f"resetting best-model tracking, which held {best_loss} ({best_source}) "
        f"from this trial's own discarded lineage",
        flush=True,
    )
    best_state_path.unlink(missing_ok=True)
    shutil.rmtree(best_dir, ignore_errors=True)
    best_dir.mkdir(parents=True, exist_ok=True)
  # `rmtree(ignore_errors=True)` cannot report failure, so ask the filesystem.
    leftovers = sorted(p.name for p in best_dir.iterdir())
    if leftovers or best_state_path.exists():
        print(
            f"[trial] WARN: could not fully clear the superseded best record; "
            f"best.json exists={best_state_path.exists()} leftovers={leftovers}",
            flush=True,
        )
    return float("inf"), "train_loss"


def _best_record_generation(best_state_path: Path) -> int | None:
    """Which holdout generation the loss in `best.json` was measured on.

    ``None`` when there is no record, or one written before the field existed
    -- unknown, not zero. Zero is a real generation (every run starts there),
    so defaulting to it would claim a match with the current holdout and
    suppress the handover exactly when the record is oldest.
    """
    d = load_optional_json(best_state_path) or {}
    gen = d.get("holdout_generation")
    return int(gen) if isinstance(gen, (int, float)) and not isinstance(gen, bool) else None


def _update_best_model(
    *,
    trainer,
    test_metrics,
    train_metrics,
    test_metrics_source_iter: int,
    holdout_generation: int,
    holdout_ruler: str,
    best_loss: float,
    best_source: str,
    best_dir: Path,
    best_state_path: Path,
    iteration_idx: int,
    opp_strength_ema: float,
) -> tuple[float, str]:
    """Update best model if current loss improved. Returns (best_loss, source).

    Holdout loss and training loss are DIFFERENT RULERS and must never be
    compared to each other. Live on 2026-07-25 they sat ~0.3 nats apart --
    train 4.88-4.94, holdout 5.14-5.25 -- because training loss is measured on
    batches the model has just fitted.

    The holdout buffer is rebuilt empty on every process start, so for the
    first several iterations after a restart there is no holdout and
    `test_metrics` is None. The old code then compared the LOWER training loss
    against a `best_loss` earned on the holdout, won automatically, and pinned
    `best_loss` to the training scale -- after which no holdout evaluation
    could ever beat it again. Best-model tracking silently stopped responding
    to holdout quality. `best.json` recorded `"source": "train_loss"` the whole
    time, so the code knew the two differed; only the comparison did not.

    Rules, in order:
      * Same ruler -> ordinary improvement test.
      * Holdout available while the record is on the training scale -> ADOPT
        the holdout as the ruler. Not a claimed improvement, just a handover to
        the ruler we actually trust; without it the stale training-scale number
        locks holdout evaluation out forever.
      * Only training loss available while the record is on the holdout scale
        -> do nothing. The fallback ruler must never overwrite the real one.

    A non-finite candidate is always rejected. The same-ruler test rejects NaN
    on its own (`nan < x` is False), but the handover does not, and a single
    NaN written to `best.json` would poison every later comparison against it
    permanently, across restarts -- the record would be unbeatable by any
    finite loss.

    `holdout_generation` IS now compared on, as a ruler identity. Two
    `test_loss` values from different generations are different rulers for the
    same reason train and holdout loss are: a drift reset
    (`_maybe_reset_holdout_on_drift`) rebuilds the holdout from a distribution
    that changed on purpose, and a restart that could not restore the sidecar
    starts from a different sample. A generation change therefore triggers the
    same handover as a train->holdout change: adopt, don't compare.

    This only became safe once the counter was durable. While it was
    in-memory-only -- reset to 0 at every process start -- comparing on it
    would have forced a handover on every restart. It now rides in the
    checkpoint's trial_meta.json next to the rows themselves
    (``tune/holdout_state``), so an ordinary resume keeps both the ruler and
    its identity, and the handover fires only when the ruler really changed.
    An older `best.json` with no recorded generation reads as unknown and is
    left to the ordinary rules -- the conservative direction, matching how a
    missing `source` is treated.

    A generation now moves for a change of MEASUREMENT as well as a change of
    SET (``trainable._maybe_bump_generation_on_ruler_change``), which is what
    the comparison always meant by "ruler" and never enforced: iter 165's
    4.85326 beat iter 162's 4.90535 with both sides at generation 1 and a
    different instrument on each. `holdout_ruler` is written into the record
    for the audit trail only -- the generation remains the single compared
    identity, because it is derived from the ruler and two comparison keys
    would need a rule for what to do when they disagree.

    `eval_source_iter` records WHICH model the holdout number describes. With
    `distributed_async_test_eval` (production), the holdout result collected
    during iteration N was computed on the snapshot taken at the end of
    iteration N-1, while `trainer` here holds N. Adoption is still correct --
    one iteration of drift is far smaller than the ~0.3-nat ruler gap this
    function exists to stop -- but the exported weights and the recorded loss
    are one step apart, and that has to be legible in the artifact rather than
    inferred later.
    """
    if test_metrics is not None:
        cur_loss, cur_source = float(test_metrics.loss), "test_loss"
    elif train_metrics is not None:
        cur_loss, cur_source = float(train_metrics.loss), "train_loss"
    else:
        return best_loss, best_source

    if not math.isfinite(cur_loss):
        print(
            f"[trial] best-model candidate from {cur_source} is {cur_loss}, "
            f"not a finite loss; leaving the record at {best_loss:.4f} "
            f"({best_source})",
            flush=True,
        )
        return best_loss, best_source

    record_generation = _best_record_generation(best_state_path)
    stale_ruler = (
        cur_source == "test_loss"
        and best_source == "test_loss"
        and record_generation is not None
        and record_generation != int(holdout_generation)
    )
    if stale_ruler:
        adopt = True
        print(
            f"[trial] best-model ruler handover: the record {best_loss:.4f} was "
            f"measured on holdout generation {record_generation}, the current "
            f"holdout is generation {holdout_generation}; adopting "
            f"{cur_loss:.4f} rather than comparing across rulers",
            flush=True,
        )
    elif cur_source == best_source:
        adopt = cur_loss < best_loss - 1e-12
    elif cur_source == "test_loss":
        adopt = True
        print(
            f"[trial] best-model ruler handover: adopting holdout loss "
            f"{cur_loss:.4f} over the training-loss record {best_loss:.4f} "
            f"(the two are not comparable)",
            flush=True,
        )
    else:
        adopt = False

    if not adopt:
        return best_loss, best_source

    trainer.save(best_dir / "trainer.pt")
    trainer.export_swa(best_dir / "best_model.pt")
    atomic_write_text(
        best_state_path,
        json.dumps({
            "best_loss": float(cur_loss),
            "iter": int(iteration_idx),
            "trainer_step": int(getattr(trainer, "step", 0)),
            "source": cur_source,
            "eval_source_iter": (
                int(test_metrics_source_iter) if cur_source == "test_loss"
                else int(iteration_idx)
            ),
            "holdout_generation": int(holdout_generation),
            "holdout_ruler": str(holdout_ruler),
            "opp_strength_ema": float(opp_strength_ema),
        }, indent=2, sort_keys=True),
    )
    return cur_loss, cur_source


_BEST_REGRET_KEEP = 3  # Keep top-N checkpoints by lowest regret


def _update_best_regret_checkpoints(
    *,
    trainer,
    pid,
    best_regret_dir: Path,
    iteration_idx: int,
    opp_strength_ema: float,
    best_loss: float,
) -> None:
    """Save checkpoint if current regret is in the top-N lowest seen."""
    if pid is None:
        return
    regret = float(pid.wdl_regret)
    if regret < 0:
        return
    ema_wr = float(pid.ema_winrate)
    step = int(getattr(trainer, "step", 0))

    best_regret_dir.mkdir(parents=True, exist_ok=True)

  # Read existing entries
    index_path = best_regret_dir / "index.json"
    entries: list[dict] = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            entries = []

  # Check if this regret qualifies
    if len(entries) >= _BEST_REGRET_KEEP:
        worst = max(entries, key=lambda e: e["regret"])
        if regret >= worst["regret"]:
            return  # Not in top-N

  # Save checkpoint
    tag = f"regret_{regret:.4f}_step{step}_iter{iteration_idx}"
    slot_dir = best_regret_dir / tag
    slot_dir.mkdir(parents=True, exist_ok=True)
    try:
        trainer.save(slot_dir / "trainer.pt")
        pid_state = pid.state_dict()
        atomic_write_text(
            slot_dir / SIDECAR_PID_STATE,
            json.dumps(pid_state, sort_keys=True, indent=2),
        )
        atomic_write_text(
            slot_dir / "meta.json",
            json.dumps({
                "regret": regret, "step": step, "iter": iteration_idx,
                "ema_winrate": ema_wr, "best_loss": best_loss,
                "opp_strength_ema": opp_strength_ema,
            }, indent=2),
        )
    except Exception:
        print(
            f"[best_regret] WARN: failed to save checkpoint tag={tag} "
            f"regret={regret:.4f} iter={iteration_idx}; skipping entry",
            flush=True,
        )
        traceback.print_exc()
        with contextlib.suppress(Exception):
            shutil.rmtree(slot_dir, ignore_errors=True)
        return

    entries.append({
        "regret": regret, "step": step, "iter": iteration_idx, "tag": tag,
        "ema_winrate": ema_wr, "opp_strength_ema": opp_strength_ema,
    })

  # Prune to top-N (lowest regret)
    entries.sort(key=lambda e: e["regret"])
    evicted = entries[_BEST_REGRET_KEEP:]
    entries = entries[:_BEST_REGRET_KEEP]

    for ev in evicted:
        ev_dir = best_regret_dir / ev["tag"]
        if ev_dir.exists():
            with contextlib.suppress(Exception):
                shutil.rmtree(ev_dir)

    atomic_write_text(index_path, json.dumps(entries, indent=2))

  # Also emit a salvage-pool-compatible manifest.json so this directory can be
  # consumed directly by `train.sh salvage-restart` without further packaging.
    try:
        from chess_anti_engine.tune.salvage import build_pool_manifest_dict
        manifest = build_pool_manifest_dict(
            metric="wdl_regret",
            label="auto_best_regret",
            entries=[
                {
                    "slot": i,
                    "metric": float(e["regret"]),
                    "training_iteration": int(e["iter"]),
                    "seed_dir": e["tag"],
                    "copied_replay_shards": 0,
                    "result_row": {
                        "wdl_regret": float(e["regret"]),
                        "pid_ema_winrate": float(e.get("ema_winrate", -1)),
                        "opponent_strength": float(e.get("opp_strength_ema", -1)),
                    },
                }
                for i, e in enumerate(entries)
            ],
        )
        atomic_write_text(
            best_regret_dir / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True),
        )
    except Exception:
        print(
            f"[best_regret] WARN: failed to emit manifest at "
            f"{best_regret_dir}/manifest.json (index still updated)",
            flush=True,
        )
        traceback.print_exc()


_PID_REASON_CODES = {
    "not_active": 0,
    "deadband": 1,
    "airbag": 2,
    "fit": 3,
    "fit_capped": 4,
    "degenerate": 5,
    "raw_override": 6,
}
# Note: `tighten_gain` and `crash_ease` are intentionally absent — they are
# modifiers applied on top of a magnitude-deciding branch, reported via the
# dedicated `*_tighten_gain_applied` / `*_crash_ease_applied` fields rather
# than as a `reason`. Any unmapped reason resolves to code -1 below.


_PID_STEP_DIAG_DEFAULTS: dict[str, float | int] = {
    "reason_code": _PID_REASON_CODES["not_active"],
    "changed": 0, "value_before": 0.0, "value_after": 0.0, "delta": 0.0,
    "raw_delta": 0.0, "cap": 0.0,
    # observation_se is a MEASUREMENT, not a state — NaN (not 0.0) when the lever
    # didn't run, so dashboards don't read a skipped step as a perfectly-certain one.
    "observation_se": float("nan"),
    "raw_deadband": 0.0, "ema_deadband": 0.0, "history_len": 0,
    "tighten_gain_applied": 1.0, "crash_ease_applied": 0,
    "predicted_value": float("nan"), "fit_slope": float("nan"),
}


def _pid_step_diag_dict(prefix: str, diag: Any | None) -> dict:
    # Always emit the SAME key set so the CSV schema is stable across iterations.
    # Ray's CSVLoggerCallback fixes the header from the first row and, on resume,
    # appends without re-heading — so a key set that varies by iteration/segment
    # silently misaligns every later segment's columns. Defaults stand in for the
    # lever that didn't run this iteration.
    if diag is None:
        return {
            f"{prefix}_reason": "not_active",
            **{f"{prefix}_{k}": v for k, v in _PID_STEP_DIAG_DEFAULTS.items()},
        }

    reason = str(getattr(diag, "reason", "not_active"))
    out = {
        f"{prefix}_reason": reason,
        f"{prefix}_reason_code": int(_PID_REASON_CODES.get(reason, -1)),
        f"{prefix}_changed": (1 if bool(getattr(diag, "changed", False)) else 0),
        f"{prefix}_value_before": float(getattr(diag, "value_before", 0.0)),
        f"{prefix}_value_after": float(getattr(diag, "value_after", 0.0)),
        f"{prefix}_delta": float(getattr(diag, "applied_delta", 0.0)),
        f"{prefix}_raw_delta": float(getattr(diag, "raw_delta", 0.0)),
        f"{prefix}_cap": float(getattr(diag, "cap", 0.0)),
        f"{prefix}_observation_se": float(getattr(diag, "observation_se", 0.0)),
        f"{prefix}_raw_deadband": float(getattr(diag, "raw_deadband", 0.0)),
        f"{prefix}_ema_deadband": float(getattr(diag, "ema_deadband", 0.0)),
        f"{prefix}_history_len": int(getattr(diag, "history_len", 0)),
        f"{prefix}_tighten_gain_applied": float(getattr(diag, "tighten_gain_applied", 1.0)),
        f"{prefix}_crash_ease_applied": (1 if bool(getattr(diag, "crash_ease_applied", False)) else 0),
    }
    # Always emit predicted_value/fit_slope so active-step rows share a stable
    # schema; NaN marks the steps where no fit ran (airbag/degenerate). The
    # TensorBoard scalars stay guarded against None separately, since NaN
    # pollutes those plots.
    predicted_value = getattr(diag, "predicted_value", None)
    out[f"{prefix}_predicted_value"] = (
        float(predicted_value) if predicted_value is not None else float("nan")
    )
    fit_slope = getattr(diag, "fit_slope", None)
    out[f"{prefix}_fit_slope"] = (
        float(fit_slope) if fit_slope is not None else float("nan")
    )
    return out


def _pid_report_dict(pr: PidResult) -> dict:
    update = pr.pid_update
    if update is None:
        # Stable schema even when no PID update ran this iteration (e.g. zero
        # finished games) — emit the full key set with defaults, not {}. The two
        # OBSERVATION fields are NaN (not 0.0): there was no observation, so a real
        # "0% winrate, zero SE" would mislead any dashboard charting the stream.
        return {
            "pid_active_levers": "none",
            "pid_raw_winrate": float("nan"),
            "pid_observation_se": float("nan"),
            "pid_regret_frozen": 0,
            "pid_nodes_active": 0,
            **_pid_step_diag_dict("pid_regret", None),
            **_pid_step_diag_dict("pid_nodes", None),
        }

    regret_diag = getattr(update, "regret_diag", None)
    nodes_diag = getattr(update, "nodes_diag", None)
    active = []
    if regret_diag is not None:
        active.append("regret")
    if nodes_diag is not None:
        active.append("nodes")

    return {
        "pid_active_levers": "+".join(active) if active else "none",
        "pid_raw_winrate": float(getattr(update, "raw_winrate", 0.0)),
        "pid_observation_se": float(getattr(update, "observation_se", 0.0)),
        "pid_regret_frozen": (1 if bool(getattr(update, "regret_frozen", False)) else 0),
        "pid_nodes_active": (1 if bool(getattr(update, "nodes_active", False)) else 0),
        **_pid_step_diag_dict("pid_regret", regret_diag),
        **_pid_step_diag_dict("pid_nodes", nodes_diag),
    }


def _log_pid_step_scalars(writer: Any, pr: PidResult, iteration_step: int) -> None:
    update = pr.pid_update
    if update is None:
        return

    writer.add_scalar("difficulty/pid_observation_se", float(getattr(update, "observation_se", 0.0)), iteration_step)
    writer.add_scalar("difficulty/pid_regret_frozen", float(1 if getattr(update, "regret_frozen", False) else 0), iteration_step)
    writer.add_scalar("difficulty/pid_nodes_active", float(1 if getattr(update, "nodes_active", False) else 0), iteration_step)
    for name, diag in (
        ("pid_regret", getattr(update, "regret_diag", None)),
        ("pid_nodes", getattr(update, "nodes_diag", None)),
    ):
        if diag is None:
            continue
        reason = str(getattr(diag, "reason", "not_active"))
        tag = f"difficulty/{name}"
        writer.add_scalar(f"{tag}_reason_code", float(_PID_REASON_CODES.get(reason, -1)), iteration_step)
        writer.add_scalar(f"{tag}_changed", float(1 if getattr(diag, "changed", False) else 0), iteration_step)
        writer.add_scalar(f"{tag}_value_before", float(getattr(diag, "value_before", 0.0)), iteration_step)
        writer.add_scalar(f"{tag}_value_after", float(getattr(diag, "value_after", 0.0)), iteration_step)
        writer.add_scalar(f"{tag}_delta", float(getattr(diag, "applied_delta", 0.0)), iteration_step)
        writer.add_scalar(f"{tag}_raw_delta", float(getattr(diag, "raw_delta", 0.0)), iteration_step)
        writer.add_scalar(f"{tag}_cap", float(getattr(diag, "cap", 0.0)), iteration_step)
        writer.add_scalar(f"{tag}_raw_deadband", float(getattr(diag, "raw_deadband", 0.0)), iteration_step)
        writer.add_scalar(f"{tag}_ema_deadband", float(getattr(diag, "ema_deadband", 0.0)), iteration_step)
        writer.add_scalar(f"{tag}_history_len", float(getattr(diag, "history_len", 0)), iteration_step)
        predicted_value = getattr(diag, "predicted_value", None)
        if predicted_value is not None:
            writer.add_scalar(f"{tag}_predicted_value", float(predicted_value), iteration_step)
        fit_slope = getattr(diag, "fit_slope", None)
        if fit_slope is not None:
            writer.add_scalar(f"{tag}_fit_slope", float(fit_slope), iteration_step)


def _log_iteration_scalars(
    *,
    writer: Any,
    pid_result: PidResult,
    wdl_regret_used: float,
    pause_metrics: dict,
    restore: RestoreResult,
    iteration_step: int,
) -> None:
    """Write per-iteration TensorBoard scalars (best-effort)."""
    try:
        pr = pid_result
        writer.add_scalar("difficulty/opponent_strength", float(pr.opp_strength), iteration_step)
        writer.add_scalar("difficulty/opponent_strength_ema", float(pr.opp_strength_ema), iteration_step)
        writer.add_scalar("difficulty/pid_ema_winrate", float(pr.pid_ema_wr), iteration_step)
        writer.add_scalar("difficulty/wdl_regret", float(wdl_regret_used), iteration_step)
        writer.add_scalar("difficulty/wdl_regret_next", float(pr.wdl_regret_next), iteration_step)
        _log_pid_step_scalars(writer, pr, iteration_step)
        if pr.curriculum_winrate_raw is not None:
            writer.add_scalar("difficulty/curriculum_winrate_raw", float(pr.curriculum_winrate_raw), iteration_step)
            writer.add_scalar("selfplay/avg_game_plies", float(pr.avg_game_plies), iteration_step)
            if pr.avg_plies_win > 0:
                writer.add_scalar("selfplay/avg_plies_win", float(pr.avg_plies_win), iteration_step)
            if pr.avg_plies_draw > 0:
                writer.add_scalar("selfplay/avg_plies_draw", float(pr.avg_plies_draw), iteration_step)
            if pr.avg_plies_loss > 0:
                writer.add_scalar("selfplay/avg_plies_loss", float(pr.avg_plies_loss), iteration_step)
            writer.add_scalar("selfplay/adjudication_rate", float(pr.adjudication_rate), iteration_step)
            writer.add_scalar("selfplay/tb_adjudication_rate", float(pr.tb_adjudication_rate), iteration_step)
            writer.add_scalar("selfplay/draw_rate", float(pr.draw_rate), iteration_step)
            writer.add_scalar("selfplay/selfplay_adjudication_rate", float(pr.selfplay_adjudication_rate), iteration_step)
            writer.add_scalar("selfplay/selfplay_draw_rate", float(pr.selfplay_draw_rate), iteration_step)
        writer.add_scalar("selfplay/curriculum_adjudication_rate", float(pr.curriculum_adjudication_rate), iteration_step)
        writer.add_scalar("selfplay/curriculum_draw_rate", float(pr.curriculum_draw_rate), iteration_step)
        writer.add_scalar("backpressure/paused_seconds", float(pause_metrics["paused_seconds"]), iteration_step)
        writer.add_scalar("backpressure/paused_fraction", float(pause_metrics["paused_fraction"]), iteration_step)
        writer.add_scalar("backpressure/paused_percent", float(pause_metrics["paused_percent"]), iteration_step)
        writer.add_scalar("meta/salvage_warmstart_used", float(1 if restore.seed_warmstart_used else 0), iteration_step)
        writer.add_scalar("meta/salvage_warmstart_slot", float(restore.seed_warmstart_slot), iteration_step)
    except Exception:
        pass


_TRAIN_METRIC_DEFAULTS: dict[str, float | int] = {
    "train_loss": 999.0, "train_time_s": 0.0, "optimizer_step_time_s": 0.0,
    "trainer_steps_done": 0, "train_samples_seen": 0,
    "trainer_steps_per_s": 0.0, "trainer_samples_per_s": 0.0, "optimizer_steps_per_s": 0.0,
    "policy_loss": 0.0, "soft_policy_loss": 0.0, "future_policy_loss": 0.0,
  # `wdl_loss` == `blended_wdl_loss` == the trained value loss;
  # `wdl_onehot_loss` is the hard-label diagnostic (no gradient).
    "wdl_loss": 0.0, "blended_wdl_loss": 0.0, "wdl_onehot_loss": 0.0,
    "sf_move_loss": 0.0, "sf_move_acc": 0.0, "sf_eval_loss": 0.0,
  # ⚑ `sf_move_acc` scores `policy_sf` -- the OPPONENT-REPLY head, whose
  # weight `w_sf_move` the live branch parks at 0.0 (origin/main still says
  # 0.02; grep the yaml rather than trusting either number here). At 0.0 its
  # drift is an untrained head under a moving trunk, NOT a progress signal.
  # The four below score heads that ARE trained; they were computed every
  # iteration since the accuracy stats landed and reached nothing, so "is
  # ranking improving?" -- the question the 2026-08-08 fixed-point finding
  # turns on -- was answered off E[regret], which rewards sharpness.
  #
  # ⚑ THE TWO ARE NOT EQUALLY TRUSTWORTHY, and neither is a fixed-point test.
  # `policy_own` is the clean one: same-position target, same head MCTS reads.
  # `policy_future` inherits `rl_loop_audit` D18 -- it is masked by
  # `future_legal_mask`, the legal moves of a position TWO plies ahead, which
  # is not derivable from this row's input, so part of what it scores is that
  # leak rather than the head. And `policy_own`'s TRAIN row measures agreement
  # with search(net) on rows the net was fitted to: at a self-referential fixed
  # point that saturates by construction. The generalisation reading is the
  # `test_` twin.
    "policy_own_acc_top1": 0.0, "policy_own_acc_top5": 0.0,
    "policy_own_acc_rows": 0.0,
    "policy_future_acc_top1": 0.0, "policy_future_acc_top5": 0.0,
    "policy_future_acc_rows": 0.0,
    "sf_search_agree_frac": 0.0,
    "sf_search_disagree_sf_low_frac": 0.0,
    "sf_search_disagree_sf_high_frac": 0.0,
    "categorical_loss": 0.0, "volatility_loss": 0.0, "sf_volatility_loss": 0.0,
    "moves_left_loss": 0.0,
  # sf_p0 policy teacher: masked losses over eligible rows + the eligible
  # fractions. The fractions are the outage detector (see TrainMetrics).
    "m_sf_own": 0.0, "m_sf_own_regret": 0.0,
    "has_sf_p0_frac": 0.0, "has_sf_p0_regret_frac": 0.0,
  # SF-approved-move floor. Read `sf_policy_floor_binds_frac` FIRST: it is the
  # only one of the pair that separates "the term selected nothing" from "the
  # net already clears the floor", and it is live at `w_sf_policy_floor: 0.0`.
    "m_sf_policy_floor": 0.0, "sf_policy_floor_binds_frac": 0.0,
  # ALWAYS-ON SF-label contamination detector (see TrainMetrics). Healthy is
  # EXACTLY 0.000000, so any non-zero value is an incident, not a threshold
  # call. Never read it without `sf_multipv_checked_frac`: 0.0 there means
  # nothing was inspected, in either of two operationally identical cases —
  # the batch carried no `has_sf_multipv_raw` field, or it held no SF-labelled
  # rows at all. `scripts/loop_health.py` alerts on that; the RATE alert is
  # deferred until a TRAIN row reads 0.0 (the `test_` twin cannot, see below).
    "sf_labelled_no_multipv_frac": 0.0, "sf_multipv_checked_frac": 0.0,
  # VALUE half of the same always-on detector (see TrainMetrics).
  # `sf_eval_pv_orphan_frac` is the one with power: it inspects the rows the
  # policy column PASSES, floor exactly 0.000000 (474,278 pre-episode labelled
  # rows), 0.117386 on the 122 quarantined shards. Read it with
  # `sf_eval_pv_checked_frac` — zero there
  # means nothing was inspected. `sf_wdl_degenerate_frac` reads 0.0 on
  # poisoned data too and is a producer tripwire, not a desync one;
  # `sf_wdl_orphaned_frac` is `sf_labelled_no_multipv_frac`'s twin by design
  # and must never be summed with it.
    "sf_wdl_degenerate_frac": 0.0, "sf_wdl_orphaned_frac": 0.0,
    "sf_eval_pv_orphan_frac": 0.0, "sf_eval_pv_checked_frac": 0.0,
  # Terminal-proximal outcome share of the value target. Exactly 0.0 while
  # `wdl_terminal_outcome_plies` is 0 (the default), so a non-zero value is the
  # proof the knob reached the trained target -- the echoed config value is
  # not. Read with `wdl_terminal_outcome_rows`: at 0.0 the frac cannot tell the
  # knob being off from a batch with no near-terminal rows.
    "wdl_terminal_outcome_frac": 0.0, "wdl_terminal_outcome_rows": 0.0,
  # SF target rebuild coverage (train.rebuild_sf_targets). 0.0 with the flag
  # off; non-zero is the proof the flip reached the batch pipeline. The
  # masked_p0/_volatility pair are PRE-mask presence fractions — the outage
  # detector while a rebuild pins `has_sf_p0_frac` at 0.0 (see TrainMetrics).
    "sf_rebuild_policy_frac": 0.0, "sf_rebuild_wdl_frac": 0.0,
    "sf_rebuild_masked_frac": 0.0,
    "sf_rebuild_masked_p0_frac": 0.0, "sf_rebuild_masked_volatility_frac": 0.0,
    "policy_loss_selfplay": 0.0, "policy_loss_curriculum": 0.0,
    "wdl_loss_selfplay": 0.0, "wdl_loss_curriculum": 0.0,
    "frac_is_selfplay_batch": 0.0, "frac_tagged_batch": 0.0,
    "policy_loss_phase_open": 0.0, "policy_loss_phase_mid": 0.0, "policy_loss_phase_end": 0.0,
    "wdl_loss_phase_open": 0.0, "wdl_loss_phase_mid": 0.0, "wdl_loss_phase_end": 0.0,
  # Denominators of the three columns above. `masked_mean` clamps its
  # denominator to 1.0, so an empty bucket reports a loss of 0.0 and nothing
  # else can say so — see train/losses.py.
    "wdl_loss_phase_n_open": 0.0, "wdl_loss_phase_n_mid": 0.0,
    "wdl_loss_phase_n_end": 0.0,
  # The policy head's own denominators — a different mask (`has_policy` on top
  # of `net_mask`), so these can be empty where the wdl counts are not.
    "policy_loss_phase_n_open": 0.0, "policy_loss_phase_n_mid": 0.0,
    "policy_loss_phase_n_end": 0.0,
    "grad_norm_mean": 0.0, "grad_norm_median": 0.0, "grad_norm_p95": 0.0,
    "grad_norm_max": 0.0, "grad_clip_rate": 0.0, "grad_adaptive_clip_rate": 0.0,
    "grad_hard_clip_rate": 0.0, "grad_norm_samples": 0,
    "grad_norm_aurora": 0.0, "grad_norm_adamw": 0.0,
    "grad_nonfinite_skip_rate": 0.0,
    "opt_lr_mean": 0.0, "opt_lr_max": 0.0,
    "aurora_uw_count": 0.0, "aurora_uw_lr": 0.0,
    "aurora_uw_ratio_median": 0.0,
    "aurora_uw_effective_ratio_min": 0.0,
    "aurora_uw_effective_ratio_median": 0.0,
    "aurora_polar_steps_configured": 0.0, "aurora_polar_sv_samples": 0.0,
    "aurora_polar_sv_errors": 0.0,
    "aurora_polar_sv_ratio_square": 0.0, "aurora_polar_sv_ratio_rect": 0.0,
    "aurora_polar_orth_err_square": 0.0, "aurora_polar_orth_err_rect": 0.0,
  # Draw-sequence provenance. See the block in `_train_metrics_dict` below.
    "batches_drawn": 0.0, "transient_cuda_retry_batches": 0.0,
}


def _train_metrics_dict(metrics) -> dict:
    """Trainer per-iter metrics (uniform 0/999 fallback when the train phase ran no steps)."""
    if metrics is None:
        return dict(_TRAIN_METRIC_DEFAULTS)
    train_t = max(metrics.train_time_s, 1e-9)
    opt_t = max(metrics.opt_step_time_s, 1e-9)
    return {
        "train_loss": float(metrics.loss),
        "train_time_s": float(metrics.train_time_s),
        "optimizer_step_time_s": float(metrics.opt_step_time_s),
        "trainer_steps_done": int(metrics.train_steps_done),
        "train_samples_seen": int(metrics.train_samples_seen),
        "trainer_steps_per_s": float(metrics.train_steps_done / train_t) if metrics.train_time_s > 0.0 else 0.0,
        "trainer_samples_per_s": float(metrics.train_samples_seen / train_t) if metrics.train_time_s > 0.0 else 0.0,
        "optimizer_steps_per_s": float(metrics.train_steps_done / opt_t) if metrics.opt_step_time_s > 0.0 else 0.0,
        "policy_loss": float(metrics.policy_loss),
        "soft_policy_loss": float(metrics.soft_policy_loss),
        "future_policy_loss": float(metrics.future_policy_loss),
        "wdl_loss": float(metrics.wdl_loss),
        "blended_wdl_loss": float(metrics.blended_wdl_loss),
        "wdl_onehot_loss": float(metrics.wdl_onehot_loss),
        "sf_search_agree_frac": float(metrics.sf_search_agree_frac),
        "sf_search_disagree_sf_low_frac": float(metrics.sf_search_disagree_sf_low_frac),
        "sf_search_disagree_sf_high_frac": float(metrics.sf_search_disagree_sf_high_frac),
        "sf_move_loss": float(metrics.sf_move_loss),
        "sf_move_acc": float(metrics.sf_move_acc),
        "policy_own_acc_top1": float(metrics.policy_own_acc_top1),
        "policy_own_acc_top5": float(metrics.policy_own_acc_top5),
        "policy_own_acc_rows": float(metrics.policy_own_acc_rows),
        "policy_future_acc_top1": float(metrics.policy_future_acc_top1),
        "policy_future_acc_top5": float(metrics.policy_future_acc_top5),
        "policy_future_acc_rows": float(metrics.policy_future_acc_rows),
        "sf_eval_loss": float(metrics.sf_eval_loss),
        "categorical_loss": float(metrics.categorical_loss),
        "volatility_loss": float(metrics.volatility_loss),
        "sf_volatility_loss": float(metrics.sf_volatility_loss),
        "moves_left_loss": float(metrics.moves_left_loss),
        # sf_p0 teacher. `has_sf_p0_frac == 0.0` means the selfplay workers
        # recorded no eligible rows at all this iteration — the teacher is dead
        # regardless of what `m_sf_own` reads. Do not judge either `m_*` column
        # without its `_frac`. CAVEAT: while `rebuild_sf_targets` is on, the
        # cross-ply mask pins this column at 0.0 (a cleared flag is
        # indistinguishable from a never-recorded one) — read
        # `sf_rebuild_masked_p0_frac`, the PRE-mask presence fraction, as the
        # outage detector for the duration of a rebuild experiment.
        "m_sf_own": float(metrics.m_sf_own),
        "m_sf_own_regret": float(metrics.m_sf_own_regret),
        "has_sf_p0_frac": float(metrics.has_sf_p0_frac),
        "has_sf_p0_regret_frac": float(metrics.has_sf_p0_regret_frac),
        # SF-approved-move floor, over the same eligible rows as `m_sf_own_regret`.
        # `sf_policy_floor_binds_frac` is the take-effect observation: a weight
        # that reaches the loss and never binds reads exactly like a dead knob on
        # `m_sf_policy_floor` alone.
        "m_sf_policy_floor": float(metrics.m_sf_policy_floor),
        "sf_policy_floor_binds_frac": float(metrics.sf_policy_floor_binds_frac),
        # Desync alarm over the rows training actually consumed. Unlike the
        # sf_rebuild_* pair below it is computed unconditionally, from the
        # batch's own presence flags, so it is readable on every iteration
        # rather than only while a rebuild experiment runs.
        "sf_labelled_no_multipv_frac": float(metrics.sf_labelled_no_multipv_frac),
        "sf_multipv_checked_frac": float(metrics.sf_multipv_checked_frac),
        # The same alarm on the VALUE half of the label, which carries 0.45 of
        # the trained value target and had no detector at all until 2026-08-03.
        # `sf_eval_pv_orphan_frac` sees the desync pass-through the policy
        # column structurally cannot; the other two are a producer tripwire and
        # the blind spot's own count. See TrainMetrics for what each floor is.
        "sf_wdl_degenerate_frac": float(metrics.sf_wdl_degenerate_frac),
        "sf_wdl_orphaned_frac": float(metrics.sf_wdl_orphaned_frac),
        "sf_eval_pv_orphan_frac": float(metrics.sf_eval_pv_orphan_frac),
        "sf_eval_pv_checked_frac": float(metrics.sf_eval_pv_checked_frac),
        # Terminal-proximal outcome transfer -- see the defaults table above.
        "wdl_terminal_outcome_frac": float(metrics.wdl_terminal_outcome_frac),
        "wdl_terminal_outcome_rows": float(metrics.wdl_terminal_outcome_rows),
        # Rebuild coverage. `sf_rebuild_policy_frac` below `sf_rebuild_wdl_frac`
        # is a Stockfish-DESYNC signal, not a coverage cost: both divide by all
        # batch rows and a healthy labelled row always carries both fields, so
        # the difference is the fully-stripped-label share of the batch — a
        # LOWER bound on contamination (~59% of poisoned rows lose the block).
        # ⚑ All five read 0.0 while `rebuild_sf_targets` is off, which is the
        # default and is in no config, so a zero here is NOT evidence of health;
        # see target_builder.py::SfRebuildCoverage.metric_kwargs for the
        # always-on detectors. `sf_rebuild_masked_frac` counts
        # the cross-ply targets it masked instead of leaving stale, and the
        # per-flag pair decomposes it PRE-mask (the rebuild-mode outage
        # detector — see the has_sf_p0_frac caveat above).
        "sf_rebuild_policy_frac": float(metrics.sf_rebuild_policy_frac),
        "sf_rebuild_wdl_frac": float(metrics.sf_rebuild_wdl_frac),
        "sf_rebuild_masked_frac": float(metrics.sf_rebuild_masked_frac),
        "sf_rebuild_masked_p0_frac": float(metrics.sf_rebuild_masked_p0_frac),
        "sf_rebuild_masked_volatility_frac": float(
            metrics.sf_rebuild_masked_volatility_frac
        ),
        "policy_loss_selfplay": float(metrics.policy_loss_selfplay),
        "policy_loss_curriculum": float(metrics.policy_loss_curriculum),
        "wdl_loss_selfplay": float(metrics.wdl_loss_selfplay),
        "wdl_loss_curriculum": float(metrics.wdl_loss_curriculum),
        "frac_is_selfplay_batch": float(metrics.frac_is_selfplay),
        "frac_tagged_batch": float(metrics.frac_tagged),
        "policy_loss_phase_open": float(metrics.policy_loss_phase_open),
        "policy_loss_phase_mid": float(metrics.policy_loss_phase_mid),
        "policy_loss_phase_end": float(metrics.policy_loss_phase_end),
        "wdl_loss_phase_open": float(metrics.wdl_loss_phase_open),
        "wdl_loss_phase_mid": float(metrics.wdl_loss_phase_mid),
        "wdl_loss_phase_end": float(metrics.wdl_loss_phase_end),
        # Rows each of the three losses above was averaged over, summed across
        # the iteration. Without these a bucket that collected NO rows is
        # indistinguishable from one with a perfect loss, because `masked_mean`
        # clamps its denominator to 1.0 and publishes 0.0.
        "wdl_loss_phase_n_open": float(metrics.wdl_loss_phase_n_open),
        "wdl_loss_phase_n_mid": float(metrics.wdl_loss_phase_n_mid),
        "wdl_loss_phase_n_end": float(metrics.wdl_loss_phase_n_end),
        "policy_loss_phase_n_open": float(metrics.policy_loss_phase_n_open),
        "policy_loss_phase_n_mid": float(metrics.policy_loss_phase_n_mid),
        "policy_loss_phase_n_end": float(metrics.policy_loss_phase_n_end),
        # Grad-norm / clipping, aggregated over every step of the iteration.
        # Previously TensorBoard-only, at a 1-in-10 subsample, in event files
        # that rotate per Ray session — so no ledger yardstick could cite them
        # and audit_realized_config.py could not gate on them (I9).
        "grad_norm_mean": float(metrics.grad_norm_mean),
        "grad_norm_median": float(metrics.grad_norm_median),
        "grad_norm_p95": float(metrics.grad_norm_p95),
        "grad_norm_max": float(metrics.grad_norm_max),
        "grad_clip_rate": float(metrics.grad_clip_rate),
        "grad_adaptive_clip_rate": float(metrics.grad_adaptive_clip_rate),
        "grad_hard_clip_rate": float(metrics.grad_hard_clip_rate),
        "grad_norm_samples": int(metrics.grad_norm_samples),
        # Per-optimizer-group split. The clip acts on the AdamW group alone —
        # the Aurora group's update is the polar factor of the gradient, which
        # is scale-invariant, so a norm clip provably cannot move it — hence
        # `grad_norm_mean`/`_median`/`_p95`/`_max` and the clip rates are all
        # AdamW-group quantities and `grad_norm_adamw` restates that
        # explicitly. `grad_norm_aurora` is the number that previously needed a
        # bespoke offline probe to obtain.
        "grad_norm_aurora": float(metrics.grad_norm_aurora),
        "grad_norm_adamw": float(metrics.grad_norm_adamw),
        "grad_nonfinite_skip_rate": float(metrics.grad_nonfinite_skip_rate),
        # Iteration mean/max of the matrix (trunk) group's LR — the LR the
        # trunk actually trains at. `opt_lr_final` below is the trough of the
        # sqrt_release ramp and is ~9x smaller (I19).
        "opt_lr_mean": float(metrics.opt_lr_mean),
        "opt_lr_max": float(metrics.opt_lr_max),
        # Aurora update/weight ratios. These were TrainMetrics fields that
        # `_train_metrics_dict` never listed, so they reached TensorBoard only
        # -- event files that rotate per Ray session, which is the same reason
        # the grad-norm family above had to be promoted. `aurora_uw_count` is
        # the wiring column: 48 is the production matrix group, 0.0 means no
        # step collected. ⚑ `aurora_uw_effective_ratio_*` is sampled at the
        # sqrt_release sawtooth FLOOR and is ~10x under a typical step by
        # construction (M4-2) -- divide it by `aurora_uw_lr` and multiply by
        # `opt_lr_mean` before reading it as "how big is one step".
        "aurora_uw_count": float(metrics.aurora_uw_count),
        "aurora_uw_lr": float(metrics.aurora_uw_lr),
        "aurora_uw_ratio_median": float(metrics.aurora_uw_ratio_median),
        "aurora_uw_effective_ratio_min": float(metrics.aurora_uw_effective_ratio_min),
        "aurora_uw_effective_ratio_median": float(
            metrics.aurora_uw_effective_ratio_median
        ),
        # Polar residual of the update Aurora applied, at the step count that
        # produced it. `aurora_polar_steps_configured` is the proof-of-effect
        # column for any change to `aurora_polar_steps`: it is read off the
        # optimizer's own param group during the step, not off the yaml.
        # `_sv_ratio_*` -> 1.0 and `_orth_err_*` -> 0.0 as the polar factor
        # converges. Nothing reported polar convergence before this (M4-1 was
        # invisible for the life of the run); `aurora_polar_sv_samples` < 2
        # means a shape class went unsampled, so its 0.0 is "not measured".
        "aurora_polar_steps_configured": float(metrics.aurora_polar_steps_configured),
        "aurora_polar_sv_samples": float(metrics.aurora_polar_sv_samples),
        "aurora_polar_sv_errors": float(metrics.aurora_polar_sv_errors),
        "aurora_polar_sv_ratio_square": float(metrics.aurora_polar_sv_ratio_square),
        "aurora_polar_sv_ratio_rect": float(metrics.aurora_polar_sv_ratio_rect),
        "aurora_polar_orth_err_square": float(metrics.aurora_polar_orth_err_square),
        "aurora_polar_orth_err_rect": float(metrics.aurora_polar_orth_err_rect),
        # Draw-sequence provenance, computed every iteration on the live
        # training path (`Trainer.train_steps`). `batches_drawn` is the total
        # microbatches the iteration pulled from the buffer;
        # `transient_cuda_retry_batches` is how many of those were REPLACEMENT
        # draws after a transient CUDA error, which advances the buffer's RNG
        # and so permanently desynchronises one arm of a paired A/B from the
        # other. Healthy is exactly 0.0.
        #
        # These are here for the same reason the grad-norm and aurora families
        # above are: `_log_metrics` already sends every TrainMetrics field to
        # TensorBoard, and TensorBoard alone is not a sink -- the event files
        # rotate per Ray session, so no ledger yardstick can cite them and
        # audit_realized_config.py cannot gate on them. Until this row carried
        # them, the only live-path record of a retry was a `logging.warning` on
        # a thousand-line console.
        "batches_drawn": float(metrics.batches_drawn),
        "transient_cuda_retry_batches": float(metrics.transient_cuda_retry_batches),
    }


def _mean_std(n: int, total: float, sq_total: float) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    mean = float(total) / float(n)
    var = max(0.0, float(sq_total) / float(n) - mean * mean)
    return mean, var ** 0.5


_TEST_METRIC_KEYS: tuple[str, ...] = (
    "test_loss", "test_policy_loss", "test_soft_policy_loss", "test_future_policy_loss",
    "test_wdl_loss", "test_wdl_onehot_loss",
    "test_sf_move_loss", "test_sf_move_acc", "test_sf_eval_loss",
    "test_policy_own_acc_top1", "test_policy_own_acc_top5",
    "test_policy_own_acc_rows",
    "test_policy_future_acc_top1", "test_policy_future_acc_top5",
    "test_policy_future_acc_rows",
    "test_categorical_loss", "test_volatility_loss", "test_sf_volatility_loss",
    "test_moves_left_loss", "test_wdl_brier", "test_wdl_ece",
    "test_policy_loss_selfplay", "test_policy_loss_curriculum",
    "test_policy_loss_phase_open", "test_policy_loss_phase_mid",
    "test_policy_loss_phase_end",
  # The RULER-side rebuild coverage. `eval_full_pass` pins the SF target
  # rebuild off, so these are 0.0 by construction and a non-zero value means
  # the frozen holdout rebuilt its own targets -- the alarm that
  # `_full_pass_host_batch` forwards its `coverage` sink to keep fail-able.
  # They belong HERE and not only in TensorBoard: TB event files rotate per
  # Ray session (see the grad-norm metrics, promoted to TrainMetrics for
  # exactly this reason), and an alarm whose only reading is in a rotating
  # sink is not an alarm. The train-side twins are in _TRAIN_METRIC_DEFAULTS.
    "test_sf_rebuild_policy_frac", "test_sf_rebuild_wdl_frac",
    "test_sf_rebuild_masked_frac",
    "test_sf_rebuild_masked_p0_frac", "test_sf_rebuild_masked_volatility_frac",
  # The holdout's own SF-label contamination reading. It is NOT redundant with
  # the train-side twin: the holdout is a FROZEN set, so its rate is fixed at
  # the contamination the set was cut with and does not decay as the replay
  # window turns over. A non-zero value here says the ruler itself was cut
  # from poisoned data — which happened (`scratchpad/split/eval_shards`, 14.10 %
  # rejected, docs/experiment_ledger.md 2026-08-01) and was found only by
  # re-gating the pool offline, long after the numbers were published.
    "test_sf_labelled_no_multipv_frac", "test_sf_multipv_checked_frac",
  # VALUE half on the same frozen set, for the same reason: the holdout scores
  # `sf_wdl`-derived quantities, so "was the ruler's own value label detached"
  # is a question about the ruler and it needs its own column here.
    "test_sf_eval_pv_orphan_frac", "test_sf_eval_pv_checked_frac",
)


def _test_and_drift_dict(
    *, tr: TrainingResult, drift: DriftMetrics,
    holdout_frozen: bool, holdout_generation: int,
) -> dict:
    """Holdout-eval metrics + data-drift telemetry. Pre-seed test_iter so Ray
    Tune locks the column on row 1 (else CSV consumers find it missing).

    `test_size` is the number of rows the eval actually SCORED, reported by
    the eval itself (`TrainMetrics.eval_rows`). It has meant three things in
    turn, and only the third is the row count: the holdout BUFFER's size
    (wrong -- reads like an evaluated-row count and is not one), then
    ``test_steps * batch_size`` = 5 x 512 = 2560 draws WITH REPLACEMENT from a
    2000-row set, and now the set itself, because the eval is a deterministic
    full pass (docs/rl_loop_audit.md G14). It is no longer reconstructible
    from config: the last batch of a pass is ragged, so no product of two
    knobs equals it. The buffer size is still emitted under the name that
    means it -- `test_replay`, which is what `audit_realized_config.py`
    already reads. No column is added or removed: Ray's CSV logger fixes the
    header on row 1 and a resume appends without re-heading, so a schema
    change here would misalign every post-restart segment.
    """
    test_dict: dict = {
        "holdout_frozen": int(holdout_frozen),
        "holdout_generation": int(holdout_generation),
        "data_drift_input_l2": float(drift.drift_input_l2),
        "data_drift_wdl_js": float(drift.drift_wdl_js),
        "data_drift_policy_entropy_diff": float(drift.drift_policy_entropy_diff),
        "data_drift_policy_entropy_train": float(drift.drift_policy_entropy_train),
        "data_drift_policy_entropy_holdout": float(drift.drift_policy_entropy_holdout),
        "data_policy_entropy": float(drift.data_policy_entropy),
        "data_unique_positions": float(drift.data_unique_positions),
        "data_wdl_balance": float(drift.data_wdl_balance),
        "test_size": 0,
        "test_iter": -1,
        **{k: float("nan") for k in _TEST_METRIC_KEYS},
    }
    if tr.test_metrics is not None:
        tm = tr.test_metrics
        test_dict.update({
            "test_size": max(0, int(tm.eval_rows)),
            "test_iter": int(tr.test_metrics_source_iter),
            "test_loss": tm.loss,
            "test_policy_loss": tm.policy_loss,
            "test_soft_policy_loss": tm.soft_policy_loss,
            "test_future_policy_loss": tm.future_policy_loss,
            "test_wdl_loss": tm.wdl_loss,
            "test_wdl_onehot_loss": float(tm.wdl_onehot_loss),
            "test_sf_move_loss": tm.sf_move_loss,
            "test_sf_move_acc": tm.sf_move_acc,
            "test_policy_own_acc_top1": float(tm.policy_own_acc_top1),
            "test_policy_own_acc_top5": float(tm.policy_own_acc_top5),
            "test_policy_own_acc_rows": float(tm.policy_own_acc_rows),
            "test_policy_future_acc_top1": float(tm.policy_future_acc_top1),
            "test_policy_future_acc_top5": float(tm.policy_future_acc_top5),
            "test_policy_future_acc_rows": float(tm.policy_future_acc_rows),
            "test_sf_eval_loss": tm.sf_eval_loss,
            "test_categorical_loss": tm.categorical_loss,
            "test_volatility_loss": tm.volatility_loss,
            "test_sf_volatility_loss": tm.sf_volatility_loss,
            "test_moves_left_loss": tm.moves_left_loss,
            "test_wdl_brier": float(tm.wdl_brier),
            "test_wdl_ece": float(tm.wdl_ece),
            "test_policy_loss_selfplay": float(tm.policy_loss_selfplay),
            "test_policy_loss_curriculum": float(tm.policy_loss_curriculum),
            "test_policy_loss_phase_open": float(tm.policy_loss_phase_open),
            "test_policy_loss_phase_mid": float(tm.policy_loss_phase_mid),
            "test_policy_loss_phase_end": float(tm.policy_loss_phase_end),
            # Ruler alarm: 0.0 by construction (the full pass pins the SF
            # target rebuild off), so non-zero means the frozen holdout
            # rebuilt its own targets and `test_loss` changed meaning.
            "test_sf_rebuild_policy_frac": float(tm.sf_rebuild_policy_frac),
            "test_sf_rebuild_wdl_frac": float(tm.sf_rebuild_wdl_frac),
            "test_sf_rebuild_masked_frac": float(tm.sf_rebuild_masked_frac),
            "test_sf_rebuild_masked_p0_frac": float(tm.sf_rebuild_masked_p0_frac),
            "test_sf_rebuild_masked_volatility_frac": float(
                tm.sf_rebuild_masked_volatility_frac
            ),
            # Contamination of the FROZEN holdout set itself. Exactly 0.000000
            # on a sound set; anything else means the ruler was cut from
            # poisoned shards and every `test_*` SF-derived column above is
            # scored against detached labels.
            #
            # ⚑⚑ THE CURRENT HOLDOUT IS NOT SOUND, AND ITS FIRST READING IS
            # KNOWN: 0.101305 (194 no-PV rows of 1915 labelled over 2000),
            # identical in checkpoint_000474/476/478, with
            # `test_sf_multipv_checked_frac` 0.957500. That is ~500x the train
            # row's ~2e-4 post-quarantine residue and it does NOT decay — the
            # set is frozen, so it stays until the holdout is re-cut from
            # post-quarantine shards. Expected on the first live holdout row;
            # it is a real finding about the RULER, not a plumbing fault, and
            # must not be used as a reason to mute the column.
            "test_sf_labelled_no_multipv_frac": float(tm.sf_labelled_no_multipv_frac),
            "test_sf_multipv_checked_frac": float(tm.sf_multipv_checked_frac),
            # Same question on the value half. The full-pass eval path reaches
            # `_prepare_host_arrays` (`_full_pass_host_batch`), which is where
            # the flags are derived, so this reads for real — but ONLY on the
            # array path: the legacy `ReplaySample` list path
            # (`_sample_batch_host`'s else branch) never touches that method
            # and publishes `checked_frac` 0.0, i.e. UNMEASURED. Read the two
            # together; the orphan rate above a zero checked_frac means nothing.
            "test_sf_eval_pv_orphan_frac": float(tm.sf_eval_pv_orphan_frac),
            "test_sf_eval_pv_checked_frac": float(tm.sf_eval_pv_checked_frac),
        })
    return test_dict


def _build_report_dict(
    *,
    tc: TrialConfig,
    trainer,
    pr: PidResult,
    sp: SelfplayResult,
    tr: TrainingResult,
    drift: DriftMetrics,
    eval_dict: dict,
    puzzle_dict: dict,
    probe_dict: dict | None = None,
    # Iteration context
    wdl_regret_used: float,
    sf_nodes_used: int,
    pause_metrics: dict,
    restore: RestoreResult,
    best_loss: float,
    iter_t0: float,
    iteration_idx: int,
    buf_size: int,
    holdout_buf_size: int,
    holdout_frozen: bool,
    holdout_generation: int,
) -> dict:
    """Assemble the per-iteration report dict for Ray Tune."""
    test_dict = _test_and_drift_dict(
        tr=tr, drift=drift,
        holdout_frozen=holdout_frozen, holdout_generation=holdout_generation,
    )
    train_metrics_dict = _train_metrics_dict(tr.metrics)
    diff_priority_mean, diff_priority_std = _mean_std(
        int(sp.diff_focus_records),
        float(sp.diff_focus_priority_sum),
        float(sp.diff_focus_priority_sq_sum),
    )
    replay_priority_mean, replay_priority_std = _mean_std(
        int(sp.replay_priority_n),
        float(sp.replay_priority_sum),
        float(sp.replay_priority_sq_sum),
    )
    gumbel_diag_n = int(sp.gumbel_policy_diag_n)
    replay_wdl_n = int(sp.replay_wdl_0) + int(sp.replay_wdl_1) + int(sp.replay_wdl_2)
    # Outcome categories are data-dependent — a key exists only for outcomes that
    # occurred this iteration. Splicing them as individual `outcome_*` columns gave
    # Ray's CSV logger a per-iteration/segment-varying key set, which (header fixed
    # from row 1, resume appends without re-heading) misaligned every later segment.
    # Emit ONE stable, COMMA-FREE column instead (pipe-separated `cat=count`).
    # Comma-free matters: monitor_pbt.sh parses rows with naive `awk -F','`, so a
    # value with embedded commas (even CSV-quoted) would shift every later field.
    # Nothing consumes the individual outcome columns (verified).
    outcome_stats = "|".join(
        f"{k}={int(v)}" for k, v in sorted(dict(sp.outcome_stats).items())
    )
    pid_diag_dict = _pid_report_dict(pr)

    return {
        "opponent_sf_nodes": int(sf_nodes_used),
        "opponent_sf_nodes_next": int(pr.sf_nodes_next),
        "opponent_wdl_regret_limit": float(wdl_regret_used),
        "opponent_wdl_regret_limit_next": float(pr.wdl_regret_next),
        "iter": int(iteration_idx),
        "global_iter": int(iteration_idx),
        "replay": int(buf_size),
        "test_replay": int(holdout_buf_size),
        "matching_positions": sp.matching_positions,
        "replay_positions_ingested": int(sp.replay_positions_ingested),
        # TRUE replay reuse: trained samples per position that actually entered
        # the buffer. In steady state this IS the mean number of times a
        # position is trained before it leaves the window, so < 1.0 means most
        # data is never trained on at all. Emitted so this can never drift
        # unobserved again (it sat at 0.46 while the config read 2.5).
        "train_views_actual": (
            float(train_metrics_dict.get("train_samples_seen", 0))
            / float(sp.replay_positions_ingested)
            if int(sp.replay_positions_ingested) > 0
            else 0.0
        ),
        "replay_window_before": int(sp.replay_window_before),
        "replay_window_after": int(sp.replay_window_after),
        "replay_window_growth_positions": int(sp.replay_window_growth_positions),
        "replay_window_growth_frac_used": float(sp.replay_window_growth_frac_used),
        "ingest_is_selfplay_tagged": int(sp.ingest_is_selfplay_tagged),
        "ingest_is_selfplay_true": int(sp.ingest_is_selfplay_true),
        "ingest_frac_selfplay": (
            float(sp.ingest_is_selfplay_true) / float(sp.ingest_is_selfplay_tagged)
            if sp.ingest_is_selfplay_tagged > 0
            else 0.0
        ),
        "diff_focus_records": int(sp.diff_focus_records),
        "diff_focus_kept": int(sp.diff_focus_kept),
        "diff_focus_keep_rate": (
            float(sp.diff_focus_kept) / float(sp.diff_focus_records)
            if sp.diff_focus_records > 0
            else 0.0
        ),
        "diff_focus_keep_prob_mean": (
            float(sp.diff_focus_keep_prob_sum) / float(sp.diff_focus_records)
            if sp.diff_focus_records > 0
            else 0.0
        ),
        "diff_focus_keep_limited_frac": (
            float(sp.diff_focus_keep_limited) / float(sp.diff_focus_records)
            if sp.diff_focus_records > 0
            else 0.0
        ),
        "diff_focus_sample_weight_mean": (
            float(sp.diff_focus_sample_weight_sum) / float(sp.diff_focus_records)
            if sp.diff_focus_records > 0
            else 0.0
        ),
        "diff_focus_sample_weight_limited_frac": (
            float(sp.diff_focus_sample_weight_limited) / float(sp.diff_focus_records)
            if sp.diff_focus_records > 0
            else 0.0
        ),
        "diff_focus_priority_mean": float(diff_priority_mean),
        "diff_focus_priority_std": float(diff_priority_std),
        "diff_focus_priority_min": float(sp.diff_focus_priority_min),
        "diff_focus_priority_max": float(sp.diff_focus_priority_max),
        "gumbel_policy_diag_n": gumbel_diag_n,
        "gumbel_policy_top_prob_mean": (
            float(sp.gumbel_policy_top_prob_sum) / float(gumbel_diag_n)
            if gumbel_diag_n > 0 else 0.0
        ),
        "gumbel_policy_action_prob_mean": (
            float(sp.gumbel_policy_action_prob_sum) / float(gumbel_diag_n)
            if gumbel_diag_n > 0 else 0.0
        ),
        "gumbel_policy_entropy_mean": (
            float(sp.gumbel_policy_entropy_sum) / float(gumbel_diag_n)
            if gumbel_diag_n > 0 else 0.0
        ),
        "gumbel_policy_eff_moves_mean": (
            float(sp.gumbel_policy_eff_moves_sum) / float(gumbel_diag_n)
            if gumbel_diag_n > 0 else 0.0
        ),
        "gumbel_policy_candidate_mass_mean": (
            float(sp.gumbel_policy_candidate_mass_sum) / float(gumbel_diag_n)
            if gumbel_diag_n > 0 else 0.0
        ),
        "gumbel_policy_non_candidate_top_prob_mean": (
            float(sp.gumbel_policy_non_candidate_top_prob_sum) / float(gumbel_diag_n)
            if gumbel_diag_n > 0 else 0.0
        ),
        "gumbel_policy_argmax_is_candidate_frac": (
            float(sp.gumbel_policy_argmax_is_candidate_sum) / float(gumbel_diag_n)
            if gumbel_diag_n > 0 else 0.0
        ),
        "gumbel_policy_argmax_is_action_frac": (
            float(sp.gumbel_policy_argmax_is_action_sum) / float(gumbel_diag_n)
            if gumbel_diag_n > 0 else 0.0
        ),
        "gumbel_policy_legal_count_mean": (
            float(sp.gumbel_policy_legal_count_sum) / float(gumbel_diag_n)
            if gumbel_diag_n > 0 else 0.0
        ),
        "gumbel_policy_candidate_count_mean": (
            float(sp.gumbel_policy_candidate_count_sum) / float(gumbel_diag_n)
            if gumbel_diag_n > 0 else 0.0
        ),
        "replay_priority_n": int(sp.replay_priority_n),
        "replay_priority_mean": float(replay_priority_mean),
        "replay_priority_std": float(replay_priority_std),
        "replay_priority_min": float(sp.replay_priority_min),
        "replay_priority_max": float(sp.replay_priority_max),
        "replay_has_policy_frac": (
            float(sp.replay_has_policy_sum) / float(sp.replay_has_policy_n)
            if sp.replay_has_policy_n > 0
            else 0.0
        ),
        "replay_has_sf_wdl_frac": (
            float(sp.replay_has_sf_wdl_sum) / float(sp.replay_has_sf_wdl_n)
            if sp.replay_has_sf_wdl_n > 0
            else 0.0
        ),
        "replay_has_search_wdl_frac": (
            float(sp.replay_has_search_wdl_sum) / float(sp.replay_has_search_wdl_n)
            if sp.replay_has_search_wdl_n > 0
            else 0.0
        ),
        "replay_wdl_win_frac": (
            float(sp.replay_wdl_0) / float(replay_wdl_n) if replay_wdl_n > 0 else 0.0
        ),
        "replay_wdl_draw_frac": (
            float(sp.replay_wdl_1) / float(replay_wdl_n) if replay_wdl_n > 0 else 0.0
        ),
        "replay_wdl_loss_frac": (
            float(sp.replay_wdl_2) / float(replay_wdl_n) if replay_wdl_n > 0 else 0.0
        ),
        "matching_games": int(sp.matching_games),
        # Every game we INGESTED this iteration, current-model or stale.
        # Emitted so "how much selfplay did we pay for" is a metric rather than
        # a sum a reader has to know to compute: matching_games alone looked
        # healthy for days while ~80% of games were being discarded as stale
        # (2026-07-24, workers frozen on an old model_sha).
        #
        # NOT the uploaded total: shards that fail to load are moved to
        # processed/bad/ and shards present in the inbox on resume are
        # quarantined, and neither is counted in either summand. If upload
        # volume itself is ever in question, that needs its own counter --
        # do not read this one as one.
        "total_games_ingested": int(sp.matching_games) + int(sp.distributed_stale_games),
        "avg_game_plies": float(pr.avg_game_plies),
        "adjudication_rate": float(pr.adjudication_rate),
        "tb_adjudication_rate": float(pr.tb_adjudication_rate),
        "draw_rate": float(pr.draw_rate),
        "selfplay_adjudication_rate": float(pr.selfplay_adjudication_rate),
        "selfplay_draw_rate": float(pr.selfplay_draw_rate),
        "curriculum_adjudication_rate": float(pr.curriculum_adjudication_rate),
        "curriculum_draw_rate": float(pr.curriculum_draw_rate),
        "checkmate_rate": float(pr.checkmate_rate),
        "stalemate_rate": float(pr.stalemate_rate),
        "avg_plies_win": float(pr.avg_plies_win),
        "avg_plies_draw": float(pr.avg_plies_draw),
        "avg_plies_loss": float(pr.avg_plies_loss),
        "shared_samples_ingested": int(sp.imported_samples_this_iter),
        "shared_trials_selected": int(sp.shared_summary.get("source_trials_selected", 0)),
        "shared_trials_ingested": int(sp.shared_summary.get("source_trials_ingested", 0)),
        "shared_trials_skipped_repeat": int(sp.shared_summary.get("source_trials_skipped_repeat", 0)),
        "shared_shards_loaded": int(sp.shared_summary.get("source_shards_loaded", 0)),
        "distributed_workers_per_trial": int(tc.distributed_workers_per_trial),
        "distributed_stale_games": int(sp.distributed_stale_games),
        "distributed_stale_positions": int(sp.distributed_stale_positions),
        "backpressure_paused_seconds": float(pause_metrics["paused_seconds"]),
        "backpressure_paused_fraction": float(pause_metrics["paused_fraction"]),
        "backpressure_paused_percent": float(pause_metrics["paused_percent"]),
        "startup_source": str(restore.startup_source),
        "salvage_warmstart_used": (1 if restore.seed_warmstart_used else 0),
        "salvage_warmstart_slot": int(restore.seed_warmstart_slot),
        "salvage_warmstart_slots_total": int(restore.seed_warmstart_slots_total),
        "salvage_origin_used": (1 if restore.salvage_origin_used else 0),
        "salvage_origin_slot": int(restore.salvage_origin_slot),
        "salvage_origin_slots_total": int(restore.salvage_origin_slots_total),
        "train_steps_used": int(tr.steps),
        "train_target_samples": int(tr.target_sample_budget),
        "train_window_target_samples": int(tr.window_target_samples),
        "win": sp.total_w,
        "draw": sp.total_d,
        "loss": sp.total_l,
        "sf_eval_delta6": float(sp.total_sf_d6 / max(1, sp.total_sf_d6_n)) if sp.total_sf_d6_n > 0 else 0.0,
        "sf_eval_delta6_n": sp.total_sf_d6_n,
        "outcome_stats": outcome_stats,
        "sf_nodes": int(sf_nodes_used),
        "sf_nodes_next": int(pr.sf_nodes_next),
        "pid_ema_winrate": float(pr.pid_ema_wr),
        "pid_curriculum_w": int(sp.total_w),
        "pid_curriculum_d": int(sp.total_d),
        "pid_curriculum_l": int(sp.total_l),
        **pid_diag_dict,
        "selfplay_games": int(sp.total_selfplay_games),
        "selfplay_draw_games": int(sp.total_selfplay_draw_games),
        "wdl_regret": float(wdl_regret_used),
        "wdl_regret_next": float(pr.wdl_regret_next),
        "opponent_strength": float(pr.opp_strength),
        "opponent_strength_ema": float(pr.opp_strength_ema),
        # Renamed from `opt_lr`, which read as "the learning rate" but is
        # sampled after the train phase, i.e. at the END of the sqrt_release
        # ramp — the TROUGH, ~9x below the mean the matrix group sees. The
        # honest per-iteration numbers are opt_lr_mean / opt_lr_max, spliced
        # in from train_metrics_dict below (docs/rl_loop_audit.md I19).
        "opt_lr_final": float(trainer.opt.param_groups[0]["lr"]),
        "peak_lr": float(getattr(trainer, "_peak_lr", 0.0)),
        "w_wdl": float(trainer.w_wdl),
        "w_soft": float(trainer.w_soft),
        "w_categorical": float(trainer.w_categorical),
        "w_sf_move": float(trainer.w_sf_move),
        "sf_wdl_frac": float(trainer.sf_wdl_frac),
        "diff_focus_q_weight": tc.diff_focus_q_weight,
        "diff_focus_pol_scale": tc.diff_focus_pol_scale,
        "diff_focus_slope": tc.diff_focus_slope,
        "diff_focus_min": tc.diff_focus_min,
        "feature_dropout_p": tc.feature_dropout_p,
        # Report the effective per-group probability (after sentinel resolution),
        # not the raw -1 sentinel from TrialConfig — the dashboard otherwise
        # shows -1 even when the global feature_dropout_p is active.
        "fdp_king_safety": float(trainer._feature_group_dropout[0][2]),
        "fdp_pins":        float(trainer._feature_group_dropout[1][2]),
        "fdp_pawns":       float(trainer._feature_group_dropout[2][2]),
        "fdp_mobility":    float(trainer._feature_group_dropout[3][2]),
        "fdp_outposts":    float(trainer._feature_group_dropout[4][2]),
        "selfplay_fraction": tc.selfplay_fraction,
        "optimizer_name": tc.optimizer,
        # `mirror_prob` has NO yaml key: `trainer_kwargs_from_config` never
        # reads one, so production runs at the constructor default and a
        # left-right mirror of ~half of every training batch was invisible to
        # `audit_realized_config.py` and to the 319-key config audit. Reported
        # (not plumbed) so the realized value is auditable from the row --
        # adding the yaml key is restart-gated, since the live-yaml validator
        # rejects a key the running code does not define.
        "mirror_prob": float(trainer.mirror_prob),
        # BOTH of these shape only the AUXILIARY `sf_eval` head's row mask.
        # Neither touches the WDL value target (docs/model_heads.md).
        "sf_wdl_conf_power": float(trainer.sf_wdl_conf_power),
        "sf_wdl_draw_scale": float(trainer.sf_wdl_draw_scale),
        "sf_wdl_temperature": float(trainer.sf_wdl_temperature),
        "best_loss": float(best_loss),
        **train_metrics_dict,
  # strict=False: the zero-game invariant guards a state no shipped path can
  # reach, so raising from inside the REPORTING path would take the trial down
  # to protect against something that cannot happen. It logs and degrades here;
  # ``gate_metrics`` still raises everywhere else, which is where the tests
  # pin it.
  # ``gate_passed`` is GONE, not renamed. It was a constant 1 emitted while
  # ``gate_games: 0`` -- a pass indistinguishable from a gate that never ran.
  # ``gate_decision`` is -1/0/1 (not-run / demote / promote) and travels with
  # the games it was computed from; ``gate_metrics`` refuses to emit a promote
  # with zero anchored games at all.
        **gate_metrics(tr.gate_decision, strict=False),
        "ingest_ms": float(sp.ingest_ms),
        "train_ms": float(tr.train_ms),
        "total_iter_ms": float((time.monotonic() - iter_t0) * 1000.0),
        **eval_dict,
        **test_dict,
        **puzzle_dict,
  # `probe_metric_defaults()` on None, never `**(probe_dict or {})`: Ray's CSV
  # logger fixes the header from row 1 and a resume appends without re-heading,
  # so a column set that depends on whether the probes are configured would
  # misalign every later segment of progress.csv. The columns exist from row 1
  # and read nan until a set is named.
        **(probe_metric_defaults() if probe_dict is None else probe_dict),
        "curriculum_winrate_raw": float(pr.curriculum_winrate_raw) if pr.curriculum_winrate_raw is not None else 0.0,
    }
