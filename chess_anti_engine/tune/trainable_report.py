"""Per-iteration reporting / persistence helpers for the Ray Tune trainable.

Writers and builders: CSV rows, TensorBoard scalars, best-model tracking,
Ray checkpoint directories, and the per-iteration Ray report dict.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from chess_anti_engine.utils.atomic import atomic_write_text
from chess_anti_engine.tune.trial_config import (
    DriftMetrics,
    PidResult,
    RestoreResult,
    SelfplayResult,
    TrainingResult,
    TrialConfig,
)


def _prune_trial_checkpoints(*, trial_dir: Path, keep_last: int) -> None:
    """Best-effort deletion of old checkpoint_* dirs inside a Tune trial.

    This complements Ray's `CheckpointConfig(num_to_keep=...)`.
    In particular, it helps when resuming an older experiment whose RunConfig did
    not have checkpoint retention enabled.
    """

    keep_last = int(keep_last)
    if keep_last <= 0:
        return

    ckpts = sorted(
        [p for p in trial_dir.glob("checkpoint_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if len(ckpts) <= keep_last:
        return

    for p in ckpts[:-keep_last]:
        shutil.rmtree(p, ignore_errors=True)


def _write_status_csv_row(
    path: Path,
    *,
    iteration_idx: int,
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
                int(iteration_idx),
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
    Checkpoint,
):
    """Flush replay buffer and save a lightweight checkpoint."""
    buf.flush()
    trainer.save(ckpt_dir / "trainer.pt")
    try:
        atomic_write_text(
            ckpt_dir / "rng_state.json",
            json.dumps(rng.bit_generator.state, sort_keys=True),
        )
    except Exception:
        pass
    try:
        atomic_write_text(
            ckpt_dir / "trial_meta.json",
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
            }, sort_keys=True, indent=2),
        )
    except Exception:
        pass
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
            slot_dir / "pid_state.json",
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
        try:
            shutil.rmtree(slot_dir, ignore_errors=True)
        except Exception:
            pass
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
            try:
                shutil.rmtree(ev_dir)
            except Exception:
                pass

    atomic_write_text(index_path, json.dumps(entries, indent=2))

    # Also emit a salvage-pool-compatible manifest.json so this directory can be
    # consumed directly by `train.sh salvage-restart` without further packaging.
    try:
        import time
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "label": "auto_best_regret",
            "metric": "wdl_regret",
            "top_n": len(entries),
            "entries": [
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
        }
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
    metrics = tr.metrics

    # --- Test / holdout metrics ---
    test_dict: dict = {
        "holdout_frozen": int(1 if holdout_frozen else 0),
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
        "test_loss": float("nan"),
        "test_policy_loss": float("nan"),
        "test_soft_policy_loss": float("nan"),
        "test_future_policy_loss": float("nan"),
        "test_wdl_loss": float("nan"),
        "test_sf_move_loss": float("nan"),
        "test_sf_move_acc": float("nan"),
        "test_sf_eval_loss": float("nan"),
        "test_categorical_loss": float("nan"),
        "test_volatility_loss": float("nan"),
        "test_sf_volatility_loss": float("nan"),
        "test_moves_left_loss": float("nan"),
    }
    if tr.test_metrics is not None:
        tm = tr.test_metrics
        test_dict.update({
            "test_size": int(holdout_buf_size),
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
        })

    return {
        "opponent_sf_nodes": int(sf_nodes_used),
        "opponent_sf_nodes_next": int(pr.sf_nodes_next),
        "opponent_wdl_regret_limit": float(wdl_regret_used),
        "opponent_wdl_regret_limit_next": float(pr.wdl_regret_next),
        "iter": int(iteration_idx),
        "global_iter": int(iteration_idx),
        "replay": int(buf_size),
        "test_replay": int(holdout_buf_size),
        "positions_added": sp.total_positions,
        "replay_positions_ingested": int(sp.replay_positions_ingested),
        "games_generated": int(sp.total_games_generated),
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
        "salvage_warmstart_used": int(1 if restore.seed_warmstart_used else 0),
        "salvage_warmstart_slot": int(restore.seed_warmstart_slot),
        "salvage_warmstart_slots_total": int(restore.seed_warmstart_slots_total),
        "salvage_origin_used": int(1 if restore.salvage_origin_used else 0),
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
        "sf_nodes": int(sf_nodes_used),
        "sf_nodes_next": int(pr.sf_nodes_next),
        "pid_ema_winrate": float(pr.pid_ema_wr),
        "pid_curriculum_w": int(sp.total_w),
        "pid_curriculum_d": int(sp.total_d),
        "pid_curriculum_l": int(sp.total_l),
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
        "w_sf_wdl": float(trainer.w_sf_wdl),
        "diff_focus_q_weight": tc.diff_focus_q_weight,
        "feature_dropout_p": tc.feature_dropout_p,
        "fdp_king_safety": tc.fdp_king_safety,
        "fdp_pins": tc.fdp_pins,
        "fdp_pawns": tc.fdp_pawns,
        "fdp_mobility": tc.fdp_mobility,
        "fdp_outposts": tc.fdp_outposts,
        "selfplay_fraction": tc.selfplay_fraction,
        "optimizer_name": tc.optimizer,
        "sf_wdl_conf_power": float(trainer.sf_wdl_conf_power),
        "sf_wdl_draw_scale": float(trainer.sf_wdl_draw_scale),
        "train_loss": float(metrics.loss) if metrics is not None else 999.0,
        "train_time_s": float(metrics.train_time_s) if metrics is not None else 0.0,
        "optimizer_step_time_s": float(metrics.opt_step_time_s) if metrics is not None else 0.0,
        "trainer_steps_done": int(metrics.train_steps_done) if metrics is not None else 0,
        "train_samples_seen": int(metrics.train_samples_seen) if metrics is not None else 0,
        "trainer_steps_per_s": float(
            metrics.train_steps_done / max(metrics.train_time_s, 1e-9)
        ) if metrics is not None and metrics.train_time_s > 0.0 else 0.0,
        "trainer_samples_per_s": float(
            metrics.train_samples_seen / max(metrics.train_time_s, 1e-9)
        ) if metrics is not None and metrics.train_time_s > 0.0 else 0.0,
        "optimizer_steps_per_s": float(
            metrics.train_steps_done / max(metrics.opt_step_time_s, 1e-9)
        ) if metrics is not None and metrics.opt_step_time_s > 0.0 else 0.0,
        "best_loss": float(best_loss),
        "policy_loss": float(metrics.policy_loss) if metrics is not None else 0.0,
        "soft_policy_loss": float(metrics.soft_policy_loss) if metrics is not None else 0.0,
        "future_policy_loss": float(metrics.future_policy_loss) if metrics is not None else 0.0,
        "wdl_loss": float(metrics.wdl_loss) if metrics is not None else 0.0,
        "sf_move_loss": float(metrics.sf_move_loss) if metrics is not None else 0.0,
        "sf_move_acc": float(metrics.sf_move_acc) if metrics is not None else 0.0,
        "sf_eval_loss": float(metrics.sf_eval_loss) if metrics is not None else 0.0,
        "categorical_loss": float(metrics.categorical_loss) if metrics is not None else 0.0,
        "volatility_loss": float(metrics.volatility_loss) if metrics is not None else 0.0,
        "sf_volatility_loss": float(metrics.sf_volatility_loss) if metrics is not None else 0.0,
        "moves_left_loss": float(metrics.moves_left_loss) if metrics is not None else 0.0,
        "gate_passed": int(1 if tr.gate_passed else 0),
        "ingest_ms": float(sp.ingest_ms),
        "train_ms": float(tr.train_ms),
        "total_iter_ms": float((time.monotonic() - iter_t0) * 1000.0),
        **eval_dict,
        **test_dict,
        **puzzle_dict,
        "curriculum_winrate_raw": float(pr.curriculum_winrate_raw) if pr.curriculum_winrate_raw is not None else 0.0,
    }
