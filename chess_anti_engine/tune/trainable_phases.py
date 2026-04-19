"""Per-iteration phase orchestrators for the Ray Tune trainable.

Each function represents one phase of the main trial loop:
  * _gate_check            — play gate games to measure winrate
  * _run_puzzle_eval_if_due — puzzle-suite evaluation on a schedule
  * _run_eval_games        — fixed-strength eval games
  * _run_training_and_gating — step budget, training, gating, holdout eval
  * _run_pid_and_eval      — PID observe + opponent-strength compute
  * _run_selfplay_phase    — selfplay (dist. or local) + ingest + export
  * _finalize_iteration    — end-of-iter reporting / CSV / prune
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chess_anti_engine.replay.shard import (
    iter_shard_paths,
    load_shard_arrays,
    local_iter_shard_path,
    save_local_shard_arrays,
)
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import TemperatureConfig
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.stockfish import DifficultyPID, StockfishPool, StockfishUCI
from chess_anti_engine.train import Trainer
from chess_anti_engine.tune._utils import concat_array_batches as _concat_array_batches
from chess_anti_engine.tune.distributed_runtime import (
    _ensure_distributed_workers,
    _ensure_inference_broker,
    _ingest_distributed_selfplay,
    _prune_processed_shards,
    _publish_distributed_trial_state,
)
from chess_anti_engine.tune.replay_exchange import _share_top_replay_each_iteration
from chess_anti_engine.utils.atomic import atomic_write_text
from chess_anti_engine.tune.trainable_config_ops import _play_batch_kwargs
from chess_anti_engine.tune.trainable_metrics import (
    _curriculum_winrate_raw_or_none,
    _compute_train_step_budget,
    _games_per_iter_for_iteration,
    _iteration_pause_metrics,
    _opponent_strength,
    _should_retry_iteration_without_games,
)
from chess_anti_engine.tune.trainable_report import (
    _build_report_dict,
    _log_iteration_scalars,
    _prune_trial_checkpoints,
    _update_best_regret_checkpoints,
    _write_status_csv_row,
)
from chess_anti_engine.tune.trial_config import (
    DifficultyState,
    DriftMetrics,
    PidResult,
    RestoreResult,
    SelfplayResult,
    TrainingResult,
    TrialConfig,
)


def _gate_check(
    model: torch.nn.Module,
    *,
    device: str,
    rng: np.random.Generator,
    sf: StockfishUCI | StockfishPool,
    gate_games: int,
    tc: TrialConfig,
    ds: DifficultyState,
) -> tuple[float, int, int, int]:
    """Play gate games to measure winrate. Returns (winrate, W, D, L).

    Uses the current PID regret setting so the acceptance gate measures against
    the same opponent strength the trainer actually trained against; otherwise
    gate can admit regressions by testing against an unrestricted opponent.
    """
    kw = _play_batch_kwargs(tc, ds=ds)
    # Gate: exploit mode — low temperature, no playout cap, minimal search.
    kw["temp"] = TemperatureConfig(temperature=0.3, drop_plies=0, after=0.0, decay_start_move=10, decay_moves=30, endgame=0.1)
    kw["search"] = dataclasses.replace(kw["search"], simulations=tc.gate_mcts_sims, playout_cap_fraction=1.0, fast_simulations=0)
    kw["diff_focus"] = dataclasses.replace(kw["diff_focus"], enabled=False)
    kw["game"] = dataclasses.replace(kw["game"], selfplay_fraction=0.0)
    _gate_samples, gate_stats = play_batch(model, device=device, rng=rng, stockfish=sf, games=gate_games, **kw)
    w, d, l = gate_stats.w, gate_stats.d, gate_stats.l
    total = max(1, w + d + l)
    winrate = (w + 0.5 * d) / total
    return winrate, w, d, l


def _run_puzzle_eval_if_due(
    model: torch.nn.Module,
    puzzle_suite,
    *,
    tc: TrialConfig,
    device: str,
    rng,
    iteration_zero_based: int,
) -> dict:
    """Run puzzle evaluation if due this iteration. Returns metrics dict."""
    _defaults = {"puzzle_accuracy": float("nan"), "puzzle_correct": 0, "puzzle_total": 0}
    if puzzle_suite is None or tc.puzzle_interval <= 0:
        return _defaults
    if iteration_zero_based % tc.puzzle_interval != 0:
        return _defaults
    from chess_anti_engine.eval import run_puzzle_eval
    pr = run_puzzle_eval(model, puzzle_suite, device=device, mcts_simulations=tc.puzzle_simulations, rng=rng)
    return {
        "puzzle_accuracy": pr.accuracy,
        "puzzle_correct": pr.correct,
        "puzzle_total": pr.total,
    }


def _run_eval_games(
    *,
    tc: TrialConfig,
    trainer: Trainer,
    device: str,
    rng: np.random.Generator,
    eval_sf: Any | None,
) -> dict:
    """Run fixed-strength evaluation games (no training data generated)."""
    eval_dict: dict = {
        "eval_win": 0, "eval_draw": 0, "eval_loss": 0, "eval_winrate": 0.0,
    }
    if tc.eval_games <= 0 or eval_sf is None:
        return eval_dict
    kw = _play_batch_kwargs(tc)
    kw["temp"] = TemperatureConfig(temperature=tc.eval_temperature)
    kw["search"] = dataclasses.replace(
        kw["search"],
        simulations=tc.eval_mcts_simulations,
        playout_cap_fraction=1.0,
        fast_simulations=0,
    )
    kw["game"] = dataclasses.replace(
        kw["game"],
        max_plies=tc.eval_max_plies,
        selfplay_fraction=0.0,
    )
    kw["diff_focus"] = dataclasses.replace(kw["diff_focus"], enabled=False)
    kw["opening"] = OpeningConfig()
    _eval_samples, eval_stats = play_batch(
        trainer.model, device=device, rng=rng, stockfish=eval_sf,
        games=tc.eval_games, **kw,
    )
    denom = float(max(1, eval_stats.w + eval_stats.d + eval_stats.l))
    eval_dict.update({
        "eval_win": eval_stats.w,
        "eval_draw": eval_stats.d,
        "eval_loss": eval_stats.l,
        "eval_winrate": (float(eval_stats.w) + 0.5 * float(eval_stats.d)) / denom,
    })
    return eval_dict


def _run_training_and_gating(
    *,
    tc: TrialConfig,
    trainer,
    buf,
    holdout_buf,
    config: dict,
    model_cfg,
    device: str,
    rng: np.random.Generator,
    sf,
    ds: DifficultyState,
    sims: int,
    total_positions: int,
    imported_samples_this_iter: int,
    gate_match_idx: int,
    gate_state_path: Path,
    distributed_server_root: Path,
    distributed_dirs: dict,
    iteration_idx: int,
    iteration_zero_based: int,
    trial_id: str,
    restore: RestoreResult,
) -> TrainingResult:
    """Compute step budget, run training, net gating, and holdout eval."""
    train_t0 = time.monotonic()
    batch_size = tc.batch_size
    accum_steps = max(1, tc.accum_steps)
    skip_train = len(buf) < batch_size
    steps = 0
    target_sample_budget = 0
    window_target_samples = 0
    gate_passed = True
    metrics = None

    if not skip_train:
        train_budget = _compute_train_step_budget(
            positions_added=int(total_positions),
            imported_samples=int(imported_samples_this_iter),
            replay_size=int(len(buf)),
            batch_size=int(batch_size),
            accum_steps=int(accum_steps),
            base_max_steps=int(tc.train_steps),
            train_window_fraction=float(max(0.0, tc.train_window_fraction)),
        )
        steps = int(train_budget["steps"])
        target_sample_budget = int(train_budget["target_sample_budget"])
        window_target_samples = int(train_budget["window_target_samples"])

        if restore.startup_source == "salvage" and bool(restore.salvage_origin_used):
            if int(iteration_zero_based) < tc.salvage_startup_no_share_iters and tc.salvage_startup_max_train_steps > 0:
                steps = min(steps, tc.salvage_startup_max_train_steps)
            elif (
                tc.salvage_startup_post_share_ramp_iters > 0
                and int(iteration_zero_based) < (tc.salvage_startup_no_share_iters + tc.salvage_startup_post_share_ramp_iters)
                and tc.salvage_startup_post_share_max_train_steps > 0
            ):
                steps = min(steps, tc.salvage_startup_post_share_max_train_steps)

        # Save model state for potential rollback (net gating).
        pre_train_state = None
        if tc.gate_games > 0 and (iteration_zero_based % tc.gate_interval == 0):
            pre_train_state = {
                k: v.clone() for k, v in trainer.model.state_dict().items()
            }

        if tc.distributed_pause_selfplay_during_training:
            _publish_distributed_trial_state(
                trainer=trainer, config=config, model_cfg=model_cfg,
                server_root=distributed_server_root, trial_id=trial_id,
                training_iteration=int(iteration_idx),
                trainer_step=int(getattr(trainer, "step", 0)),
                sf_nodes=int(ds.sf_nodes),
                mcts_simulations=int(sims),
                wdl_regret=float(ds.wdl_regret),
                pause_selfplay=True,
                pause_reason="training",
                export_model=False,
            )

        metrics = trainer.train_steps(
            buf,
            batch_size=batch_size,
            steps=steps,
        )

        # Net gating: play gate_games and reject if winrate < threshold.
        if pre_train_state is not None and tc.gate_games > 0:
            gate_wr, gate_w, gate_d, gate_l = _gate_check(
                trainer.model,
                device=device,
                rng=rng,
                sf=sf,
                gate_games=tc.gate_games,
                tc=tc,
                ds=ds,
            )

            gate_match_idx += 1
            try:
                atomic_write_text(
                    gate_state_path,
                    json.dumps(
                        {"matches": int(gate_match_idx)},
                        indent=2,
                        sort_keys=True,
                    ),
                )
            except Exception:
                pass

            try:
                trainer.writer.add_scalar("gate/winrate", float(gate_wr), int(gate_match_idx))
                trainer.writer.add_scalar("gate/win", float(gate_w), int(gate_match_idx))
                trainer.writer.add_scalar("gate/draw", float(gate_d), int(gate_match_idx))
                trainer.writer.add_scalar("gate/loss", float(gate_l), int(gate_match_idx))
                trainer.writer.add_scalar("gate/passed", float(1.0 if gate_wr >= tc.gate_threshold else 0.0), int(gate_match_idx))
            except Exception:
                pass

            if gate_wr < tc.gate_threshold:
                trainer.model.load_state_dict(pre_train_state)
                gate_passed = False

    test_metrics = None
    if len(holdout_buf) >= tc.batch_size:
        test_metrics = trainer.eval_steps(
            holdout_buf,
            batch_size=tc.batch_size,
            steps=tc.test_steps,
        )

    train_ms = (time.monotonic() - train_t0) * 1000.0

    return TrainingResult(
        metrics=metrics,
        test_metrics=test_metrics,
        gate_passed=gate_passed,
        steps=steps,
        target_sample_budget=target_sample_budget,
        window_target_samples=window_target_samples,
        train_ms=train_ms,
        gate_match_idx=gate_match_idx,
    )


def _run_pid_and_eval(
    *,
    tc: TrialConfig,
    config: dict,
    pid: DifficultyPID | None,
    sf: StockfishUCI | StockfishPool | None,
    sp_result: SelfplayResult,
    iteration_zero_based: int,
    opp_strength_ema: float,
    opp_ema_alpha: float,
    ds: DifficultyState,
) -> PidResult:
    """Update PID, compute opponent strength and derived game stats.

    Mutates *pid* (observe + param refresh) and *sf* (set_nodes) in place.
    Applies the full set of live-reloadable PID knobs from *config* each
    iteration so yaml edits reach the running controller.
    """
    wdl_regret_used = ds.wdl_regret
    sf_nodes_used = ds.sf_nodes
    total_w = sp_result.total_w
    total_d = sp_result.total_d
    total_l = sp_result.total_l

    # --- Derived game stats ---
    gen = float(max(1, int(sp_result.total_games_generated)))
    sp = float(max(1, int(sp_result.total_selfplay_games)))
    cur = float(max(1, int(sp_result.total_curriculum_games)))

    curriculum_winrate_raw = _curriculum_winrate_raw_or_none(
        wins=total_w, draws=total_d, losses=total_l,
    )

    # --- PID update ---
    pid_update = None
    pid_ema_wr = float(pid.ema_winrate) if pid is not None else 0.0
    sf_nodes_next = int(sf_nodes_used)

    if pid is not None and (total_w + total_d + total_l) > 0:
        pid.refresh_live_params(config)
        pid_update = pid.observe(wins=total_w, draws=total_d, losses=total_l, force=True)
        pid_ema_wr = float(pid_update.ema_winrate)
        sf_nodes_next = int(pid.nodes)
        if sf is not None:
            sf.set_nodes(int(sf_nodes_next))

    wdl_regret_next = float(pid.wdl_regret) if pid is not None else -1.0

    # --- Opponent strength ---
    opp_strength = _opponent_strength(
        sf_nodes=int(sf_nodes_used),
        ema_winrate=float(pid_ema_wr),
        min_nodes=int(getattr(pid, "min_nodes", 50)) if pid is not None else 50,
        max_nodes=int(getattr(pid, "max_nodes", 50000)) if pid is not None else 50000,
        pid_target_winrate=tc.sf_pid_target_winrate,
        wdl_regret=float(wdl_regret_used),
        wdl_regret_max=tc.sf_pid_wdl_regret_max,
    )
    if opp_strength_ema == 0.0:
        opp_strength_ema = float(opp_strength)
    else:
        opp_strength_ema = (
            opp_ema_alpha * float(opp_strength)
            + (1.0 - opp_ema_alpha) * opp_strength_ema
        )

    return PidResult(
        sf_nodes_next=sf_nodes_next,
        wdl_regret_next=wdl_regret_next,
        pid_ema_wr=pid_ema_wr,
        pid_update=pid_update,
        curriculum_winrate_raw=curriculum_winrate_raw,
        avg_game_plies=float(sp_result.total_game_plies) / gen,
        adjudication_rate=float(sp_result.total_adjudicated_games) / gen,
        draw_rate=float(sp_result.total_draw_games) / gen,
        selfplay_adjudication_rate=float(sp_result.total_selfplay_adjudicated_games) / sp,
        selfplay_draw_rate=float(sp_result.total_selfplay_draw_games) / sp,
        curriculum_adjudication_rate=float(sp_result.total_curriculum_adjudicated_games) / cur,
        curriculum_draw_rate=float(sp_result.total_curriculum_draw_games) / cur,
        checkmate_rate=float(sp_result.total_checkmate_games) / gen,
        stalemate_rate=float(sp_result.total_stalemate_games) / gen,
        avg_plies_win=float(sp_result.total_plies_win) / float(max(1, total_w)) if total_w > 0 else 0.0,
        avg_plies_draw=float(sp_result.total_plies_draw) / float(max(1, total_d)) if total_d > 0 else 0.0,
        avg_plies_loss=float(sp_result.total_plies_loss) / float(max(1, total_l)) if total_l > 0 else 0.0,
        opp_strength=opp_strength,
        opp_strength_ema=opp_strength_ema,
    )


def _run_selfplay_phase(
    *,
    tc: TrialConfig,
    config: dict,
    trainer,
    model_cfg,
    buf,
    holdout_buf,
    holdout_frozen: bool,
    device: str,
    rng,
    sf,
    distributed_dirs: dict,
    distributed_server_root: Path,
    distributed_worker_procs: list,
    distributed_inference_broker_proc,
    prev_published_model_sha: str,
    ds: DifficultyState,
    sims: int,
    iteration_idx: int,
    iteration_zero_based: int,
    trial_id: str,
    trial_dir: Path,
    selfplay_shards_dir: Path,
    replay_shard_dir: Path,
    current_window: int,
    in_salvage_startup_grace: bool,
) -> tuple[SelfplayResult, str, int, subprocess.Popen[bytes] | None]:
    """Run distributed selfplay, ingest, export shards, grow window, cross-trial share.

    Mutates *buf*, *holdout_buf*, *distributed_worker_procs* in place.
    Returns ``(sp_result, prev_published_model_sha, current_window,
    distributed_inference_broker_proc)``.
    """
    total_games = _games_per_iter_for_iteration(tc, iteration_idx)

    # --- Play games (distributed workers upload shards; we ingest) ---
    ingest_t0 = time.monotonic()
    total_sf_d6 = 0.0
    total_sf_d6_n = 0

    distributed_worker_procs[:] = _ensure_distributed_workers(
        config=config,
        trial_dir=trial_dir,
        trial_id=trial_id,
        procs=distributed_worker_procs,
    )
    published_model_sha = _publish_distributed_trial_state(
        trainer=trainer,
        config=config,
        model_cfg=model_cfg,
        server_root=distributed_server_root,
        trial_id=trial_id,
        training_iteration=int(iteration_idx),
        trainer_step=int(getattr(trainer, "step", 0)),
        sf_nodes=int(ds.sf_nodes),
        mcts_simulations=int(sims),
        wdl_regret=float(ds.wdl_regret),
        pause_selfplay=False,
        pause_reason="",
    )
    distributed_inference_broker_proc = _ensure_inference_broker(
        config=config,
        trial_id=trial_id,
        trial_dir=trial_dir,
        publish_dir=distributed_dirs["publish_dir"],
        proc=distributed_inference_broker_proc,
    )
    _shards_before_ingest = set(buf._shard_paths)

    ingest_summary = _ingest_distributed_selfplay(
        buf=buf,
        holdout_buf=holdout_buf,
        holdout_frac=tc.holdout_fraction,
        holdout_frozen=bool(holdout_frozen),
        inbox_dir=distributed_dirs["inbox_dir"],
        processed_dir=distributed_dirs["processed_dir"],
        target_games=int(total_games),
        accepted_model_shas={str(published_model_sha)} | ({str(prev_published_model_sha)} if prev_published_model_sha else set()),
        prev_model_sha=str(prev_published_model_sha) if prev_published_model_sha else None,
        prev_model_max_fraction=tc.distributed_prev_model_max_fraction,
        wait_timeout_s=tc.distributed_wait_timeout_seconds,
        poll_seconds=tc.distributed_worker_poll_seconds,
        rng=rng,
        min_games_fraction=tc.distributed_min_games_fraction,
    )

    buf.flush()
    _new_selfplay_shards = [p for p in buf._shard_paths if p not in _shards_before_ingest]
    total_w = int(ingest_summary["matching_w"])
    total_d = int(ingest_summary["matching_d"])
    total_l = int(ingest_summary["matching_l"])
    total_games_generated = int(ingest_summary["matching_games"])
    total_game_plies = int(ingest_summary["matching_total_game_plies"])
    total_adjudicated_games = int(ingest_summary["matching_adjudicated_games"])
    total_draw_games = int(ingest_summary["matching_total_draw_games"])
    total_selfplay_games = int(ingest_summary["matching_selfplay_games"])
    total_selfplay_adjudicated_games = int(ingest_summary["matching_selfplay_adjudicated_games"])
    total_selfplay_draw_games = int(ingest_summary["matching_selfplay_draw_games"])
    total_curriculum_games = int(ingest_summary["matching_curriculum_games"])
    total_curriculum_adjudicated_games = int(ingest_summary["matching_curriculum_adjudicated_games"])
    total_curriculum_draw_games = int(ingest_summary["matching_curriculum_draw_games"])
    total_positions = int(ingest_summary["matching_positions"])
    total_plies_win = int(ingest_summary.get("matching_plies_win", 0))
    total_plies_draw = int(ingest_summary.get("matching_plies_draw", 0))
    total_plies_loss = int(ingest_summary.get("matching_plies_loss", 0))
    total_checkmate_games = int(ingest_summary.get("matching_checkmate_games", 0))
    total_stalemate_games = int(ingest_summary.get("matching_stalemate_games", 0))
    distributed_stale_positions = int(ingest_summary["stale_positions"])
    distributed_stale_games = int(ingest_summary["stale_games"])
    replay_positions_ingested = int(ingest_summary["positions_replay_added"])
    prev_published_model_sha = str(published_model_sha)

    if iteration_zero_based % 10 == 0:
        _prune_processed_shards(
            processed_dir=distributed_dirs["processed_dir"],
            max_age_seconds=tc.processed_max_age_seconds,
        )

    ingest_ms = (time.monotonic() - ingest_t0) * 1000.0

    # --- Retry if distributed returned no games ---
    if _should_retry_iteration_without_games(total_games_generated=total_games_generated):
        print(
            "[trial] distributed iteration waiting for fresh selfplay: "
            f"trial={trial_id} iter={iteration_idx} replay={len(buf)} "
            f"timeout_s={tc.distributed_wait_timeout_seconds:.1f}"
        )
        time.sleep(max(0.5, tc.distributed_worker_poll_seconds))
        return (
            SelfplayResult(should_retry=True, ingest_ms=ingest_ms),
            prev_published_model_sha,
            current_window,
            distributed_inference_broker_proc,
        )

    # --- Export selfplay shards for sibling trials ---
    _selfplay_export_path = local_iter_shard_path(selfplay_shards_dir, iteration_idx)
    if _new_selfplay_shards:
        _export_batches: list[dict[str, np.ndarray]] = []
        for _sp_path in _new_selfplay_shards:
            try:
                _arrs, _ = load_shard_arrays(_sp_path, lazy=True)
                if int(np.asarray(_arrs["x"]).shape[0]) > 0:
                    _export_batches.append(_arrs)
            except Exception:
                pass
        if _export_batches:
            save_local_shard_arrays(_selfplay_export_path, arrs=_concat_array_batches(_export_batches))
    _keep_selfplay_iters = tc.exploit_replay_max_unseen_iters_per_source + 2
    _all_selfplay_exports = iter_shard_paths(selfplay_shards_dir)
    for _old in _all_selfplay_exports[:-_keep_selfplay_iters]:
        try:
            shutil.rmtree(_old) if _old.is_dir() else _old.unlink(missing_ok=True)
        except Exception:
            pass

    # --- Growing window ---
    if current_window < tc.replay_window_max:
        current_window = min(current_window + tc.replay_window_growth, tc.replay_window_max)
        if buf.capacity < current_window:
            buf.capacity = current_window

    # --- Cross-trial replay sharing ---
    shared_summary: dict = {
        "source_trials_selected": 0,
        "source_trials_ingested": 0,
        "source_trials_skipped_repeat": 0,
        "source_shards_loaded": 0,
        "source_samples_ingested": 0,
    }
    if tc.exploit_replay_share_top_enabled and (not in_salvage_startup_grace):
        shared_summary = _share_top_replay_each_iteration(
            config=config,
            recipient_trial_dir=trial_dir,
            replay_shard_dir=replay_shard_dir,
            buf=buf,
            top_k_trials=tc.exploit_replay_top_k_trials,
            within_best_frac=tc.exploit_replay_top_within_best_frac,
            min_metric=tc.exploit_replay_top_min_metric,
            source_skip_newest=tc.exploit_replay_skip_newest,
            shard_size=tc.shard_size,
            holdout_fraction=tc.holdout_fraction,
            max_unseen_iters_per_source=tc.exploit_replay_max_unseen_iters_per_source,
            max_shards_per_source=tc.exploit_replay_top_shards_per_source,
            share_fraction=tc.exploit_replay_share_fraction,
        )
        if shared_summary["source_samples_ingested"] > 0:
            print(
                "[trial] replay share this iter: "
                f"sources_selected={shared_summary['source_trials_selected']} "
                f"sources_ingested={shared_summary['source_trials_ingested']} "
                f"sources_skipped_repeat={shared_summary['source_trials_skipped_repeat']} "
                f"source_shards_loaded={shared_summary['source_shards_loaded']} "
                f"source_samples_ingested={shared_summary['source_samples_ingested']}"
            )
    elif tc.exploit_replay_share_top_enabled and in_salvage_startup_grace:
        print(
            "[trial] replay share skipped during salvage startup grace: "
            f"iter={iteration_zero_based} grace_iters={tc.salvage_startup_no_share_iters}"
        )

    imported_samples_this_iter = int(shared_summary.get("source_samples_ingested", 0))

    sp = SelfplayResult(
        total_w=total_w, total_d=total_d, total_l=total_l,
        total_games_generated=total_games_generated,
        total_game_plies=total_game_plies,
        total_adjudicated_games=total_adjudicated_games,
        total_draw_games=total_draw_games,
        total_selfplay_games=total_selfplay_games,
        total_selfplay_adjudicated_games=total_selfplay_adjudicated_games,
        total_selfplay_draw_games=total_selfplay_draw_games,
        total_curriculum_games=total_curriculum_games,
        total_curriculum_adjudicated_games=total_curriculum_adjudicated_games,
        total_curriculum_draw_games=total_curriculum_draw_games,
        total_checkmate_games=total_checkmate_games,
        total_stalemate_games=total_stalemate_games,
        total_plies_win=total_plies_win,
        total_plies_draw=total_plies_draw,
        total_plies_loss=total_plies_loss,
        total_positions=total_positions,
        replay_positions_ingested=replay_positions_ingested,
        total_sf_d6=total_sf_d6,
        total_sf_d6_n=total_sf_d6_n,
        distributed_stale_positions=distributed_stale_positions,
        distributed_stale_games=distributed_stale_games,
        shared_summary=shared_summary,
        imported_samples_this_iter=imported_samples_this_iter,
        ingest_ms=ingest_ms,
    )
    return sp, prev_published_model_sha, current_window, distributed_inference_broker_proc


def _finalize_iteration(
    *,
    tc: TrialConfig,
    trainer,
    pid,
    sp: SelfplayResult,
    tr: TrainingResult,
    drift: DriftMetrics,
    pid_result: PidResult,
    eval_dict: dict,
    checkpoint,
    best_loss: float,
    ckpt_dir: Path,
    work_dir: Path,
    trial_dir: Path,
    status_csv_path: Path,
    tune_report_fn,
    puzzle_suite,
    ds: DifficultyState,
    distributed_pause_started_at: float | None,
    distributed_pause_active: bool,
    restore: RestoreResult,
    holdout_frozen: bool,
    holdout_generation: int,
    buf_size: int,
    holdout_buf_size: int,
    iter_t0: float,
    iteration_idx: int,
    iteration_zero_based: int,
    completed_iterations: int,
    device: str,
    rng,
) -> None:
    """End-of-iteration bookkeeping: PID persist, reporting, CSV, prune."""
    wdl_regret_used = ds.wdl_regret
    sf_nodes_used = ds.sf_nodes
    # Persist PID state AFTER observe() so checkpoints carry the
    # post-iteration difficulty.
    if pid is not None:
        try:
            atomic_write_text(
                ckpt_dir / "pid_state.json",
                json.dumps(pid.state_dict(), sort_keys=True, indent=2),
            )
        except Exception:
            pass

    # Save top-3 checkpoints by lowest regret (best-effort). Write to a
    # persistent cross-trial location so Ray's --tune-keep-last-experiments
    # rotation doesn't take them out with the trial dir.
    try:
        cross_trial_dir = getattr(tc, "best_regret_checkpoints_dir", None)
        if cross_trial_dir and str(cross_trial_dir).strip():
            _best_regret_dir = Path(str(cross_trial_dir)).expanduser()
            if not _best_regret_dir.is_absolute():
                # Ray workers run with cwd set to a per-trial tmp dir, so
                # Path.cwd() would silently write to /tmp/ray/... which gets
                # wiped on process restart. Anchor relative paths to the
                # project root (yaml file's grandparent, since yaml lives in
                # <project>/configs/).
                _yaml = getattr(tc, "_yaml_config_path", None)
                if _yaml:
                    _best_regret_dir = Path(_yaml).resolve().parent.parent / _best_regret_dir
                else:
                    _best_regret_dir = Path.cwd() / _best_regret_dir
        else:
            _best_regret_dir = work_dir / "best_regret"
        _update_best_regret_checkpoints(
            trainer=trainer, pid=pid,
            best_regret_dir=_best_regret_dir,
            iteration_idx=iteration_idx,
            opp_strength_ema=pid_result.opp_strength_ema,
            best_loss=best_loss,
        )
    except Exception:
        print(
            f"[best_regret] WARN: outer wrapper caught exception at iter "
            f"{iteration_idx}; best-regret auto-save skipped this iter",
            flush=True,
        )
        traceback.print_exc()

    # Puzzle evaluation (overspecialization canary).
    puzzle_dict = _run_puzzle_eval_if_due(
        trainer.model, puzzle_suite,
        tc=tc, device=device, rng=rng,
        iteration_zero_based=iteration_zero_based,
    )

    pause_metrics = _iteration_pause_metrics(
        iteration_started_at=iter_t0,
        iteration_finished_at=time.monotonic(),
        pause_started_at=distributed_pause_started_at,
        pause_active=distributed_pause_active,
    )
    _log_iteration_scalars(
        writer=trainer.writer,
        pid_result=pid_result,
        wdl_regret_used=wdl_regret_used,
        pause_metrics=pause_metrics,
        restore=restore,
        iteration_step=int(iteration_idx),
    )

    report_dict = _build_report_dict(
        tc=tc, trainer=trainer,
        pr=pid_result, sp=sp, tr=tr, drift=drift,
        eval_dict=eval_dict, puzzle_dict=puzzle_dict,
        wdl_regret_used=wdl_regret_used,
        sf_nodes_used=sf_nodes_used,
        pause_metrics=pause_metrics,
        restore=restore,
        best_loss=best_loss,
        iter_t0=iter_t0,
        iteration_idx=iteration_idx,
        buf_size=buf_size,
        holdout_buf_size=holdout_buf_size,
        holdout_frozen=holdout_frozen,
        holdout_generation=holdout_generation,
    )

    tune_report_fn(report_dict, checkpoint=checkpoint)

    # Write compact status row (best-effort — never crash the trial).
    _write_status_csv_row(
        status_csv_path,
        iteration_idx=iteration_idx,
        opp_strength=pid_result.opp_strength,
        opp_strength_ema=pid_result.opp_strength_ema,
        sf_nodes=sf_nodes_used,
        wdl_regret=wdl_regret_used,
        ingest_ms=sp.ingest_ms,
        train_ms=tr.train_ms,
        total_iter_ms=report_dict["total_iter_ms"],
        steps=tr.steps,
        replay_size=buf_size,
        positions_ingested=sp.replay_positions_ingested,
        stale_games=sp.distributed_stale_games,
        train_loss=float(tr.metrics.loss) if tr.metrics is not None else None,
        best_loss=best_loss,
        total_w=sp.total_w,
        total_d=sp.total_d,
        total_l=sp.total_l,
        opt_lr=float(trainer.opt.param_groups[0]["lr"]),
        startup_source=restore.startup_source,
    )

    # Best-effort: keep disk usage bounded.
    if (completed_iterations + 1) % 5 == 0:
        _prune_trial_checkpoints(
            trial_dir=trial_dir,
            keep_last=tc.tune_num_to_keep,
        )
