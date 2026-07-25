"""Per-iteration reporting / persistence helpers for the Ray Tune trainable.

Writers and builders: CSV rows, TensorBoard scalars, best-model tracking,
Ray checkpoint directories, and the per-iteration Ray report dict.
"""
from __future__ import annotations

import csv
import json
import logging
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from chess_anti_engine.tune._utils import (
    SIDECAR_PID_STATE,
    SIDECAR_RNG_STATE,
    SIDECAR_TRIAL_META,
)
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

    for p in ckpts[:-keep_last]:
        shutil.rmtree(p, ignore_errors=True)
    log.info(
        "pruned %d trial checkpoint(s) to keep_last=%d: %s",
        len(ckpts) - keep_last,
        keep_last,
        ", ".join(p.name for p in ckpts[:-keep_last]),
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
    "stale", "train_loss", "best_loss", "win", "draw", "loss", "lr", "startup",
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
    opt_lr: float,
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
                f"{float(opt_lr):.2e}",
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
    Checkpoint,
):
    """Flush replay buffer and save a lightweight checkpoint."""
    buf.flush()
    trainer.save(ckpt_dir / "trainer.pt")
    _write_rng_state_sidecar(ckpt_dir=ckpt_dir, rng=rng)
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
            }, sort_keys=True, indent=2),
        )
    except (OSError, TypeError, ValueError) as exc:
        # trial_meta drives exploit-clone owner tracking; loud failure is the right call.
        log.warning("[trial] failed to write trial_meta.json: %s", exc)
    return Checkpoint.from_directory(str(ckpt_dir))


def _update_best_model(
    *,
    trainer,
    test_metrics,
    train_metrics,
    best_loss: float,
    best_dir: Path,
    best_state_path: Path,
    iteration_idx: int,
    opp_strength_ema: float,
) -> float:
    """Update best model if current loss improved. Returns updated best_loss."""
    cur_loss = (
        float(test_metrics.loss) if test_metrics is not None
        else (float(train_metrics.loss) if train_metrics is not None else float("inf"))
    )
    if cur_loss < best_loss - 1e-12:
        best_loss = cur_loss
        trainer.save(best_dir / "trainer.pt")
        trainer.export_swa(best_dir / "best_model.pt")
        atomic_write_text(
            best_state_path,
            json.dumps({
                "best_loss": float(best_loss),
                "iter": int(iteration_idx),
                "trainer_step": int(getattr(trainer, "step", 0)),
                "source": "test_loss" if test_metrics is not None else "train_loss",
                "opp_strength_ema": float(opp_strength_ema),
            }, indent=2, sort_keys=True),
        )
    return best_loss


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
    "wdl_loss": 0.0, "blended_wdl_loss": 0.0, "sf_move_loss": 0.0, "sf_move_acc": 0.0, "sf_eval_loss": 0.0,
    "sf_search_agree_frac": 0.0,
    "sf_search_disagree_sf_low_frac": 0.0,
    "sf_search_disagree_sf_high_frac": 0.0,
    "categorical_loss": 0.0, "volatility_loss": 0.0, "sf_volatility_loss": 0.0,
    "moves_left_loss": 0.0,
    "policy_loss_selfplay": 0.0, "policy_loss_curriculum": 0.0,
    "wdl_loss_selfplay": 0.0, "wdl_loss_curriculum": 0.0,
    "frac_is_selfplay_batch": 0.0, "frac_tagged_batch": 0.0,
    "policy_loss_open": 0.0, "policy_loss_mid": 0.0, "policy_loss_end": 0.0,
    "wdl_loss_open": 0.0, "wdl_loss_mid": 0.0, "wdl_loss_end": 0.0,
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
        "sf_search_agree_frac": float(metrics.sf_search_agree_frac),
        "sf_search_disagree_sf_low_frac": float(metrics.sf_search_disagree_sf_low_frac),
        "sf_search_disagree_sf_high_frac": float(metrics.sf_search_disagree_sf_high_frac),
        "sf_move_loss": float(metrics.sf_move_loss),
        "sf_move_acc": float(metrics.sf_move_acc),
        "sf_eval_loss": float(metrics.sf_eval_loss),
        "categorical_loss": float(metrics.categorical_loss),
        "volatility_loss": float(metrics.volatility_loss),
        "sf_volatility_loss": float(metrics.sf_volatility_loss),
        "moves_left_loss": float(metrics.moves_left_loss),
        "policy_loss_selfplay": float(metrics.policy_loss_selfplay),
        "policy_loss_curriculum": float(metrics.policy_loss_curriculum),
        "wdl_loss_selfplay": float(metrics.wdl_loss_selfplay),
        "wdl_loss_curriculum": float(metrics.wdl_loss_curriculum),
        "frac_is_selfplay_batch": float(metrics.frac_is_selfplay),
        "frac_tagged_batch": float(metrics.frac_tagged),
        "policy_loss_open": float(metrics.policy_loss_open),
        "policy_loss_mid": float(metrics.policy_loss_mid),
        "policy_loss_end": float(metrics.policy_loss_end),
        "wdl_loss_open": float(metrics.wdl_loss_open),
        "wdl_loss_mid": float(metrics.wdl_loss_mid),
        "wdl_loss_end": float(metrics.wdl_loss_end),
    }


def _mean_std(n: int, total: float, sq_total: float) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    mean = float(total) / float(n)
    var = max(0.0, float(sq_total) / float(n) - mean * mean)
    return mean, var ** 0.5


_TEST_METRIC_KEYS: tuple[str, ...] = (
    "test_loss", "test_policy_loss", "test_soft_policy_loss", "test_future_policy_loss",
    "test_wdl_loss", "test_sf_move_loss", "test_sf_move_acc", "test_sf_eval_loss",
    "test_categorical_loss", "test_volatility_loss", "test_sf_volatility_loss",
    "test_moves_left_loss", "test_wdl_brier", "test_wdl_ece",
    "test_policy_loss_selfplay", "test_policy_loss_curriculum",
    "test_policy_loss_open", "test_policy_loss_mid", "test_policy_loss_end",
)


def _test_and_drift_dict(
    *, tr: TrainingResult, drift: DriftMetrics,
    holdout_buf_size: int, holdout_frozen: bool, holdout_generation: int,
) -> dict:
    """Holdout-eval metrics + data-drift telemetry. Pre-seed test_iter so Ray
    Tune locks the column on row 1 (else CSV consumers find it missing)."""
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
            "test_size": int(holdout_buf_size),
            "test_iter": int(tr.test_metrics_source_iter),
            "test_loss": tm.loss,
            "test_policy_loss": tm.policy_loss,
            "test_soft_policy_loss": tm.soft_policy_loss,
            "test_future_policy_loss": tm.future_policy_loss,
            "test_wdl_loss": tm.wdl_loss,
            "test_sf_move_loss": tm.sf_move_loss,
            "test_sf_move_acc": tm.sf_move_acc,
            "test_sf_eval_loss": tm.sf_eval_loss,
            "test_categorical_loss": tm.categorical_loss,
            "test_volatility_loss": tm.volatility_loss,
            "test_sf_volatility_loss": tm.sf_volatility_loss,
            "test_moves_left_loss": tm.moves_left_loss,
            "test_wdl_brier": float(tm.wdl_brier),
            "test_wdl_ece": float(tm.wdl_ece),
            "test_policy_loss_selfplay": float(tm.policy_loss_selfplay),
            "test_policy_loss_curriculum": float(tm.policy_loss_curriculum),
            "test_policy_loss_open": float(tm.policy_loss_open),
            "test_policy_loss_mid": float(tm.policy_loss_mid),
            "test_policy_loss_end": float(tm.policy_loss_end),
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
        tr=tr, drift=drift, holdout_buf_size=holdout_buf_size,
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
        "opt_lr": float(trainer.opt.param_groups[0]["lr"]),
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
        "sf_wdl_conf_power": float(trainer.sf_wdl_conf_power),
        "sf_wdl_draw_scale": float(trainer.sf_wdl_draw_scale),
        "sf_wdl_temperature": float(trainer.sf_wdl_temperature),
        "best_loss": float(best_loss),
        **train_metrics_dict,
        "gate_passed": (1 if tr.gate_passed else 0),
        "ingest_ms": float(sp.ingest_ms),
        "train_ms": float(tr.train_ms),
        "total_iter_ms": float((time.monotonic() - iter_t0) * 1000.0),
        **eval_dict,
        **test_dict,
        **puzzle_dict,
        "curriculum_winrate_raw": float(pr.curriculum_winrate_raw) if pr.curriculum_winrate_raw is not None else 0.0,
    }
