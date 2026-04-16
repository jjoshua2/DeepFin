from __future__ import annotations

# Optional dependency module (Ray Tune). Kept import-light so the core package
# works without installing `.[tune]`.

from pathlib import Path
from typing import Any
import csv
import dataclasses
import json
import logging
import shutil
import subprocess
import time

log = logging.getLogger(__name__)

import numpy as np
import torch

from chess_anti_engine.model import (
    ModelConfig,
    build_model,
    load_state_dict_tolerant,
    reinit_volatility_head_parameters_,
    zero_policy_head_parameters_,
)
from chess_anti_engine.replay import ArrayReplayBuffer, DiskReplayBuffer
from chess_anti_engine.replay.shard import (
    copy_or_link_shard,
    iter_shard_paths,
    load_shard_arrays,
    local_iter_shard_path,
    save_local_shard_arrays,
    samples_to_arrays,
)
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import TemperatureConfig
from chess_anti_engine.selfplay.manager import _effective_curriculum_topk
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.stockfish import DifficultyPID, StockfishPool, StockfishUCI, pid_from_config
from chess_anti_engine.train import Trainer, trainer_kwargs_from_config
from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS
from chess_anti_engine.tune.trial_config import DriftMetrics, PidResult, RestoreResult, SelfplayResult, TrainingResult, TrialConfig
from chess_anti_engine.tune.trainable_metrics import (
    _blended_winrate_raw_or_none,
    _compute_drift_metrics,
    _compute_train_step_budget,
    _count_jsonl_rows,
    _dynamic_sf_wdl_weight,
    _games_per_iter_for_iteration,
    _iteration_pause_metrics,
    _mean_entropy,
    _opponent_strength,
    _sample_drift_arrays,
    _should_retry_distributed_iteration_without_games,
    _wdl_hist,
)
from chess_anti_engine.tune.trainable_report import (
    _build_report_dict,
    _log_iteration_scalars,
    _prune_trial_checkpoints,
    _save_trial_checkpoint,
    _update_best_model,
    _update_best_regret_checkpoints,
    _write_status_csv_row,
)
from chess_anti_engine.tune.trainable_config_ops import (
    _TRAINER_WEIGHT_KEYS,
    _play_batch_kwargs,
    _reload_yaml_into_config,
    _resolve_pause_marker_path,
    _resolve_sims,
    _sync_trainer_weights,
    _wait_if_paused,
)
from chess_anti_engine.tune._utils import (
    concat_array_batches as _concat_array_batches,
    resolve_local_override_root as _resolve_local_override_root,
    stable_seed_u32 as _stable_seed_u32,
)
from chess_anti_engine.tune.distributed_runtime import (
    _ensure_distributed_workers,
    _ensure_inference_broker,
    _ingest_distributed_selfplay,
    _prune_processed_shards,
    _publish_distributed_trial_state,
    _quarantine_inbox_shards,
    _set_active_run_prefix,
    _stop_process,
    _stop_worker_processes,
    _trial_server_dirs,
)
from chess_anti_engine.tune.recovery import (
    _load_salvage_manifest_entry,
    _merge_pid_state_from_result_row,
    _select_salvage_seed_slot,
)
from chess_anti_engine.tune.replay_exchange import (
    _latest_trial_result_row,
    _refresh_replay_shards_on_exploit,
    _share_top_replay_each_iteration,
    _trial_replay_shard_dir,
)



def _gate_check(
    model: torch.nn.Module,
    *,
    device: str,
    rng: np.random.Generator,
    sf: StockfishUCI | StockfishPool,
    gate_games: int,
    opponent_random_move_prob: float,
    tc: TrialConfig,
) -> tuple[float, int, int, int]:
    """Play gate games to measure winrate. Returns (winrate, W, D, L)."""
    kw = _play_batch_kwargs(tc)
    # Gate: exploit mode — low temperature, no playout cap, minimal search.
    kw["opponent"] = dataclasses.replace(kw["opponent"], random_move_prob=opponent_random_move_prob)
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
    pid,
    sf,
    current_rand: float,
    skill_level_used: int,
    sims: int,
    wdl_regret_used: float,
    total_positions: int,
    imported_samples_this_iter: int,
    gate_match_idx: int,
    gate_state_path: Path,
    use_distributed_selfplay: bool,
    distributed_server_root: Path | None,
    distributed_dirs: dict | None,
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

        if tc.distributed_pause_selfplay_during_training and use_distributed_selfplay and distributed_dirs is not None:
            assert distributed_server_root is not None
            _publish_distributed_trial_state(
                trainer=trainer, config=config, model_cfg=model_cfg,
                server_root=distributed_server_root, trial_id=trial_id,
                training_iteration=int(iteration_idx),
                trainer_step=int(getattr(trainer, "step", 0)),
                sf_nodes=int(pid.nodes) if pid is not None else tc.sf_nodes,
                random_move_prob=float(current_rand),
                skill_level=int(skill_level_used),
                mcts_simulations=int(sims),
                wdl_regret=float(pid.wdl_regret) if pid is not None else -1.0,
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
                opponent_random_move_prob=current_rand,
                tc=tc,
            )

            gate_match_idx += 1
            try:
                gate_state_path.write_text(
                    json.dumps(
                        {"matches": int(gate_match_idx)},
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
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
    pid: DifficultyPID | None,
    sf: StockfishUCI | StockfishPool | None,
    sp_result: SelfplayResult,
    iteration_zero_based: int,
    opp_strength_ema: float,
    opp_ema_alpha: float,
    current_rand: float,
    sf_nodes_used: int,
    skill_level_used: int,
    wdl_regret_used: float,
) -> PidResult:
    """Update PID, compute opponent strength and derived game stats.

    Mutates *pid* (observe + param refresh) and *sf* (set_nodes) in place.
    """
    total_w = sp_result.total_w
    total_d = sp_result.total_d
    total_l = sp_result.total_l

    # --- Derived game stats ---
    gen = float(max(1, int(sp_result.total_games_generated)))
    sp = float(max(1, int(sp_result.total_selfplay_games)))
    cur = float(max(1, int(sp_result.total_curriculum_games)))

    blended_winrate_raw = _blended_winrate_raw_or_none(
        wins=total_w, draws=total_d, losses=total_l,
    )

    # --- PID update ---
    pid_update = None
    pid_ema_wr = float(pid.ema_winrate) if pid is not None else 0.0
    sf_nodes_next = int(sf_nodes_used)
    random_move_prob_next = float(current_rand)
    skill_level_next = int(skill_level_used)

    if pid is not None and (total_w + total_d + total_l) > 0:
        _step_start = tc.sf_pid_max_rand_step_start
        _step_ramp = tc.sf_pid_max_rand_step_ramp_iters
        if _step_start > 0 and _step_ramp > 0 and iteration_zero_based < _step_ramp:
            pid.max_rand_step = _step_start
        else:
            pid.max_rand_step = tc.sf_pid_max_rand_step
        pid.max_regret_step = tc.sf_pid_max_regret_step
        pid.max_regret_ease_step = tc.sf_pid_max_regret_ease_step
        pid.target = tc.sf_pid_target_winrate
        pid.kp = tc.sf_pid_kp
        pid.ki = tc.sf_pid_ki
        pid.alpha = tc.sf_pid_ema_alpha
        pid_update = pid.observe(wins=total_w, draws=total_d, losses=total_l, force=True)
        pid_ema_wr = float(pid_update.ema_winrate)
        sf_nodes_next = int(pid.nodes)
        random_move_prob_next = float(pid.random_move_prob)
        skill_level_next = int(pid.skill_level)
        if sf is not None:
            sf.set_nodes(int(sf_nodes_next))

    wdl_regret_next = float(pid.wdl_regret) if pid is not None else -1.0

    # --- Opponent strength ---
    _curr_topk = _effective_curriculum_topk(
        random_move_prob=current_rand,
        stage_end=tc.sf_pid_topk_stage_end,
        topk_max=tc.sf_multipv,
        topk_min=tc.sf_pid_topk_min,
    )
    opp_strength = _opponent_strength(
        random_move_prob=float(current_rand),
        sf_nodes=int(sf_nodes_used),
        skill_level=int(skill_level_used),
        ema_winrate=float(pid_ema_wr),
        min_nodes=int(getattr(pid, "min_nodes", 50)) if pid is not None else 50,
        max_nodes=int(getattr(pid, "max_nodes", 50000)) if pid is not None else 50000,
        pid_target_winrate=tc.sf_pid_target_winrate,
        wdl_regret=float(wdl_regret_used),
        wdl_regret_max=tc.sf_pid_wdl_regret_max,
        topk=int(_curr_topk),
        topk_max=tc.sf_multipv,
        topk_min=tc.sf_pid_topk_min,
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
        random_move_prob_next=random_move_prob_next,
        skill_level_next=skill_level_next,
        wdl_regret_next=wdl_regret_next,
        pid_ema_wr=pid_ema_wr,
        pid_update=pid_update,
        blended_winrate_raw=blended_winrate_raw,
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
        curr_topk=int(_curr_topk),
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
    pid,
    use_distributed_selfplay: bool,
    distributed_dirs: dict | None,
    distributed_server_root: Path | None,
    distributed_worker_procs: list,
    distributed_inference_broker_proc,
    prev_published_model_sha: str,
    current_rand: float,
    skill_level_used: int,
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
    """Run selfplay, ingest, export shards, grow window, cross-trial share.

    Mutates *buf*, *holdout_buf*, *distributed_worker_procs* in place.
    Returns ``(sp_result, prev_published_model_sha, current_window,
    distributed_inference_broker_proc)``.
    """
    total_games = _games_per_iter_for_iteration(tc, iteration_idx)

    # --- Play games (distributed or local) ---
    ingest_t0 = time.monotonic()
    _new_selfplay_shards: list = []
    _new_selfplay_batches: list[dict[str, np.ndarray]] = []

    # Accumulators (populated by either branch, bundled into SelfplayResult at end).
    total_w = total_d = total_l = 0
    total_games_generated = 0
    total_game_plies = 0
    total_adjudicated_games = 0
    total_draw_games = 0
    total_selfplay_games = 0
    total_selfplay_adjudicated_games = 0
    total_selfplay_draw_games = 0
    total_curriculum_games = 0
    total_curriculum_adjudicated_games = 0
    total_curriculum_draw_games = 0
    total_checkmate_games = 0
    total_stalemate_games = 0
    total_plies_win = 0
    total_plies_draw = 0
    total_plies_loss = 0
    total_positions = 0
    replay_positions_ingested = 0
    total_sf_d6 = 0.0
    total_sf_d6_n = 0
    distributed_stale_positions = 0
    distributed_stale_games = 0

    if use_distributed_selfplay:
        assert distributed_server_root is not None
        assert distributed_dirs is not None
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
            sf_nodes=int(pid.nodes) if pid is not None else tc.sf_nodes,
            random_move_prob=float(current_rand),
            skill_level=int(skill_level_used),
            mcts_simulations=int(sims),
            wdl_regret=float(pid.wdl_regret) if pid is not None else -1.0,
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
    else:
        kw = _play_batch_kwargs(tc)
        kw["opponent"] = dataclasses.replace(kw["opponent"], random_move_prob=current_rand)
        kw["search"] = dataclasses.replace(kw["search"], simulations=int(sims))

        samples, stats = play_batch(
            trainer.model,
            device=device, rng=rng, stockfish=sf,
            games=tc.selfplay_batch,
            target_games=total_games,
            **kw,
        )

        total_games_generated += int(stats.games)
        total_w += stats.w
        total_d += stats.d
        total_l += stats.l
        total_game_plies += int(getattr(stats, "total_game_plies", 0))
        total_adjudicated_games += int(getattr(stats, "adjudicated_games", getattr(stats, "timeout_games", 0)))
        total_draw_games += int(getattr(stats, "total_draw_games", 0))
        total_selfplay_games += int(getattr(stats, "selfplay_games", 0))
        total_selfplay_adjudicated_games += int(getattr(stats, "selfplay_adjudicated_games", 0))
        total_selfplay_draw_games += int(getattr(stats, "selfplay_draw_games", 0))
        total_curriculum_games += int(getattr(stats, "curriculum_games", 0))
        total_curriculum_adjudicated_games += int(getattr(stats, "curriculum_adjudicated_games", 0))
        total_curriculum_draw_games += int(getattr(stats, "curriculum_draw_games", 0))
        total_checkmate_games += int(getattr(stats, "checkmate_games", 0))
        total_stalemate_games += int(getattr(stats, "stalemate_games", 0))
        total_plies_win += int(getattr(stats, "plies_win", 0))
        total_plies_draw += int(getattr(stats, "plies_draw", 0))
        total_plies_loss += int(getattr(stats, "plies_loss", 0))
        total_positions += stats.positions
        total_sf_d6 += float(getattr(stats, "sf_eval_delta6", 0.0)) * int(getattr(stats, "sf_eval_delta6_n", 0))
        total_sf_d6_n += int(getattr(stats, "sf_eval_delta6_n", 0))
        replay_positions_ingested = int(total_positions)

        holdout_samples: list = []
        train_samples: list = []
        for s in samples:
            if tc.holdout_fraction > 0.0 and (not holdout_frozen) and (rng.random() < tc.holdout_fraction):
                holdout_samples.append(s)
            else:
                train_samples.append(s)
        if holdout_samples:
            holdout_buf.add_many_arrays(samples_to_arrays(holdout_samples))
        if train_samples:
            train_arrs = samples_to_arrays(train_samples)
            buf.add_many_arrays(train_arrs)
            _new_selfplay_batches.append(train_arrs)
        del samples

    ingest_ms = (time.monotonic() - ingest_t0) * 1000.0

    # --- Retry if distributed returned no games ---
    if _should_retry_distributed_iteration_without_games(
        use_distributed_selfplay=use_distributed_selfplay,
        total_games_generated=total_games_generated,
    ):
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
    if use_distributed_selfplay:
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
    else:
        if _new_selfplay_batches:
            save_local_shard_arrays(_selfplay_export_path, arrs=_concat_array_batches(_new_selfplay_batches))
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
    current_rand: float,
    wdl_regret_used: float,
    sf_nodes_used: int,
    skill_level_used: int,
    use_distributed_selfplay: bool,
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
    # Persist PID state AFTER observe() so checkpoints carry the
    # post-iteration difficulty that produced random_move_prob_next.
    if pid is not None:
        try:
            (ckpt_dir / "pid_state.json").write_text(
                json.dumps(pid.state_dict(), sort_keys=True, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # Save top-3 checkpoints by lowest regret (best-effort).
    try:
        _update_best_regret_checkpoints(
            trainer=trainer, pid=pid,
            best_regret_dir=work_dir / "best_regret",
            iteration_idx=iteration_idx,
            opp_strength_ema=pid_result.opp_strength_ema,
            best_loss=best_loss,
        )
    except Exception:
        pass

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
        current_rand=current_rand,
        wdl_regret_used=wdl_regret_used,
        pause_metrics=pause_metrics,
        restore=restore,
        iteration_step=int(iteration_idx),
    )

    report_dict = _build_report_dict(
        tc=tc, trainer=trainer,
        pr=pid_result, sp=sp, tr=tr, drift=drift,
        eval_dict=eval_dict, puzzle_dict=puzzle_dict,
        current_rand=current_rand,
        wdl_regret_used=wdl_regret_used,
        sf_nodes_used=sf_nodes_used,
        skill_level_used=skill_level_used,
        use_distributed_selfplay=use_distributed_selfplay,
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
        skill_level=skill_level_used,
        sf_nodes=sf_nodes_used,
        current_rand=current_rand,
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


def _restore_checkpoint_or_salvage(
    *,
    config: dict,
    trainer,
    device: str,
    trial_id: str,
    trial_dir: Path,
    base_seed: int,
    active_seed: int,
    rng: np.random.Generator,
    ckpt,
    Checkpoint,
) -> tuple[RestoreResult, np.random.Generator]:
    """Restore from Ray checkpoint, salvage seed pool, or start fresh.

    Mutates *trainer* (model/optimizer load), *config* (donor config overlay),
    and may reseed *rng* on exploit clone.  Returns ``(restore_result, rng)``.
    """
    salvage_restore_donor_config = bool(config.get("salvage_restore_donor_config", False))
    salvage_restore_pid_state = bool(config.get("salvage_restore_pid_state", False))
    salvage_restore_full_trainer_state = bool(config.get("salvage_restore_full_trainer_state", False))

    rr = RestoreResult(active_seed=active_seed)

    restored_rng_state = None
    restored_trial_meta = None
    restored_owner_optimizer = ""

    if ckpt is not None:
        ckpt_dir = Path(ckpt.to_directory())
        maybe = ckpt_dir / "trainer.pt"
        pid_path = ckpt_dir / "pid_state.json"
        if pid_path.exists():
            try:
                rr.restored_pid_state = json.loads(pid_path.read_text(encoding="utf-8"))
            except Exception:
                rr.restored_pid_state = None
        rng_path = ckpt_dir / "rng_state.json"
        if rng_path.exists():
            try:
                restored_rng_state = json.loads(rng_path.read_text(encoding="utf-8"))
            except Exception:
                restored_rng_state = None
        trial_meta_path = ckpt_dir / "trial_meta.json"
        if trial_meta_path.exists():
            try:
                restored_trial_meta = json.loads(trial_meta_path.read_text(encoding="utf-8"))
            except Exception:
                restored_trial_meta = None
        if isinstance(restored_trial_meta, dict):
            restored_owner_optimizer = str(restored_trial_meta.get("optimizer", "") or "")
        current_optimizer = str(config.get("optimizer", "nadamw")).lower()
        if maybe.exists():
            model_only_restore = False
            if isinstance(restored_trial_meta, dict):
                owner_trial_id = str(restored_trial_meta.get("owner_trial_id", ""))
                if owner_trial_id and owner_trial_id != trial_id:
                    if restored_owner_optimizer:
                        model_only_restore = restored_owner_optimizer.lower() != current_optimizer
                    elif bool(config.get("search_optimizer", False)):
                        model_only_restore = True
            if model_only_restore:
                ckpt_data = torch.load(str(maybe), map_location=device)
                load_state_dict_tolerant(
                    trainer.model, ckpt_data["model"],
                    label="checkpoint_model_only",
                )
                del ckpt_data
                trainer._init_swa()
                rr.startup_source = "checkpoint_model_only"
            else:
                trainer.load(maybe)
                rr.startup_source = "checkpoint"
            if "lr" in config:
                trainer.set_peak_lr(float(config["lr"]), rescale_current=False)
    elif isinstance(config.get("salvage_seed_pool_dir"), str) and str(config.get("salvage_seed_pool_dir", "")).strip():
        seed_pool_dir = Path(str(config.get("salvage_seed_pool_dir"))).expanduser()
        if not seed_pool_dir.is_dir():
            raise RuntimeError(f"salvage_seed_pool_dir not found: {seed_pool_dir}")

        picked_dir, picked_slot, num_slots = _select_salvage_seed_slot(
            seed_pool_dir=seed_pool_dir,
            trial_dir=trial_dir,
            trial_id=trial_id,
        )
        if picked_dir is None:
            raise RuntimeError(
                f"salvage requested but no seed slot could be selected for "
                f"trial_id={trial_id} from {seed_pool_dir}"
            )

        maybe = Path(picked_dir) / "trainer.pt"
        if not maybe.exists():
            raise RuntimeError(f"salvage seed missing trainer.pt: {maybe}")

        rr.startup_source = "salvage"
        rr.seed_warmstart_used = True
        rr.seed_warmstart_slot = int(picked_slot)
        rr.seed_warmstart_slots_total = int(num_slots)
        rr.seed_warmstart_dir = Path(picked_dir)
        rr.seed_warmstart_replay_dir = rr.seed_warmstart_dir / "replay_shards"
        rr.salvage_origin_used = True
        rr.salvage_origin_slot = int(rr.seed_warmstart_slot)
        rr.salvage_origin_slots_total = int(rr.seed_warmstart_slots_total)
        rr.salvage_origin_dir = str(rr.seed_warmstart_dir.resolve())
        seed_entry = _load_salvage_manifest_entry(
            seed_pool_dir=seed_pool_dir,
            slot=rr.seed_warmstart_slot,
        )
        seed_warmstart_manifest_row: dict | None = None
        if isinstance(seed_entry, dict):
            result_row = seed_entry.get("result_row")
            if isinstance(result_row, dict):
                seed_warmstart_manifest_row = result_row
                if salvage_restore_donor_config:
                    donor_cfg = result_row.get("config")
                    if isinstance(donor_cfg, dict):
                        for k in ("lr", "cosmos_gamma", *_TRAINER_WEIGHT_KEYS):
                            if k in donor_cfg:
                                config[k] = donor_cfg[k]
                        if "lr" in config:
                            trainer.set_peak_lr(float(config["lr"]), rescale_current=False)
                        if "cosmos_gamma" in config and hasattr(trainer.opt, "gamma"):
                            trainer.opt.gamma = float(config["cosmos_gamma"])
                        for wk in _TRAINER_WEIGHT_KEYS:
                            if wk in config:
                                setattr(trainer, wk, float(config[wk]))

        if salvage_restore_full_trainer_state:
            trainer.load(maybe)
        else:
            salvage_ckpt = torch.load(str(maybe), map_location=device)
            load_state_dict_tolerant(
                trainer.model, salvage_ckpt["model"],
                label="salvage",
            )
            del salvage_ckpt
        if bool(config.get("salvage_reinit_volatility_heads", False)):
            reinit = reinit_volatility_head_parameters_(trainer.model)
            if reinit:
                print(f"[trial] Reinitialized salvage volatility heads: {', '.join(reinit)}")
        print(
            f"[trial] salvage warmstart loaded slot={rr.seed_warmstart_slot} "
            f"of {rr.seed_warmstart_slots_total} from {rr.seed_warmstart_dir}"
        )
        if salvage_restore_pid_state:
            pid_path = rr.seed_warmstart_dir / "pid_state.json"
            if pid_path.exists():
                try:
                    rr.restored_pid_state = json.loads(pid_path.read_text(encoding="utf-8"))
                except Exception:
                    rr.restored_pid_state = None
            rr.restored_pid_state, pid_manifest_overrides = _merge_pid_state_from_result_row(
                pid_state=rr.restored_pid_state,
                result_row=seed_warmstart_manifest_row,
            )
            if pid_manifest_overrides:
                print(
                    "[trial] salvage PID overrides from manifest row: "
                    + ", ".join(pid_manifest_overrides)
                )

    # Post-restore metadata fixups.
    restored_owner_trial_id = ""
    if isinstance(restored_trial_meta, dict):
        restored_owner_trial_id = str(restored_trial_meta.get("owner_trial_id", ""))
        rr.restored_owner_trial_dir = str(restored_trial_meta.get("owner_trial_dir", ""))
        restored_owner_optimizer = str(restored_trial_meta.get("optimizer", restored_owner_optimizer))
        rr.salvage_origin_used = bool(restored_trial_meta.get("salvage_origin_used", rr.salvage_origin_used))
        rr.salvage_origin_slot = int(restored_trial_meta.get("salvage_origin_slot", rr.salvage_origin_slot))
        rr.salvage_origin_slots_total = int(
            restored_trial_meta.get("salvage_origin_slots_total", rr.salvage_origin_slots_total)
        )
        rr.salvage_origin_dir = str(restored_trial_meta.get("salvage_origin_dir", rr.salvage_origin_dir or ""))
        rr.global_iter = int(restored_trial_meta.get("global_iter", 0))
    rr.cross_trial_restore = bool(
        ckpt is not None and restored_owner_trial_id and restored_owner_trial_id != trial_id
    )
    if rr.cross_trial_restore and rr.startup_source == "checkpoint":
        rr.startup_source = "exploit_restore"
    elif rr.cross_trial_restore and rr.startup_source == "checkpoint_model_only":
        rr.startup_source = "exploit_restore_model_only"

    if restored_rng_state is not None and not rr.cross_trial_restore:
        try:
            rng.bit_generator.state = restored_rng_state
        except Exception:
            pass

    if ckpt is not None and rr.cross_trial_restore:
        # PB2 exploit clone: fork RNG stream so recipients do not replay donor opening sequences.
        restore_rows = _count_jsonl_rows(trial_dir / "result.json")
        fork_seed = _stable_seed_u32(
            base_seed,
            trial_id,
            "exploit",
            restore_rows,
            int(getattr(trainer, "step", 0)),
        )
        rr.active_seed = int(fork_seed)
        rng = np.random.default_rng(rr.active_seed)
        torch.manual_seed(rr.active_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rr.active_seed)
        print(
            f"[trial] PB2 exploit restore detected: owner={restored_owner_trial_id} "
            f"recipient={trial_id} fork_seed={rr.active_seed} "
            f"owner_optimizer={restored_owner_optimizer or 'unknown'} "
            f"recipient_optimizer={str(config.get('optimizer', 'nadamw')).lower()} "
            f"restore_mode={rr.startup_source}"
        )
        # Inherit donor's EMA so the recipient isn't ranked on stale
        # pre-exploit history.
        if rr.restored_owner_trial_dir:
            _donor_row = _latest_trial_result_row(Path(rr.restored_owner_trial_dir))
            if _donor_row is not None:
                _donor_ema = _donor_row.get(
                    "opponent_strength_ema",
                    _donor_row.get("opponent_strength", 0.0),
                )
                rr.opp_strength_ema = float(_donor_ema)
                print(
                    f"[trial] exploit: inherited donor opp_strength_ema={rr.opp_strength_ema:.1f}"
                )

    return rr, rng


def _maybe_load_bootstrap(
    *,
    tc: TrialConfig,
    trainer,
    device: str,
    ckpt,
    restore: RestoreResult,
) -> None:
    """Load pre-trained bootstrap model weights on fresh start (no checkpoint/warmstart).

    IMPORTANT: Only loads MODEL WEIGHTS — skips optimizer/scheduler/step.
    The bootstrap was trained for ~13k steps with its own LR schedule; carrying that
    state causes: (1) step=13323 skips warmup entirely, (2) scheduler resumes
    mid-cosine-cycle with near-zero LR then spikes on restart, (3) optimizer momentum
    buffers from bootstrap data cause wrong gradient directions on selfplay data,
    (4) PB2's lr perturbation has no effect because scheduler's base_lr is locked.
    """
    if not (tc.bootstrap_checkpoint and ckpt is None and (not restore.seed_warmstart_used)):
        return
    bp = Path(tc.bootstrap_checkpoint)
    if not bp.exists():
        print(f"[trial] WARNING: bootstrap checkpoint not found: {bp}")
        return
    print(f"[trial] Loading pre-trained bootstrap model weights: {bp}")
    ckpt_data = torch.load(str(bp), map_location=device)
    model_sd = ckpt_data.get("model") or ckpt_data.get("model_state_dict") or ckpt_data
    load_state_dict_tolerant(trainer.model, model_sd, label="bootstrap")
    if tc.bootstrap_zero_policy_heads:
        zeroed = zero_policy_head_parameters_(trainer.model)
        if zeroed:
            print(f"[trial] Zeroed bootstrap policy heads: {', '.join(zeroed)}")
    if tc.bootstrap_reinit_volatility_heads:
        reinit = reinit_volatility_head_parameters_(trainer.model)
        if reinit:
            print(f"[trial] Reinitialized bootstrap volatility heads: {', '.join(reinit)}")
    # Re-sync SWA with the newly loaded weights (AveragedModel deep-copies at init).
    trainer._init_swa()
    del ckpt_data


def _init_replay_buffers(
    *,
    tc: TrialConfig,
    config: dict,
    restore: RestoreResult,
    trial_dir: Path,
    work_dir: Path,
    rng,
    ckpt,
) -> tuple:
    """Set up replay + holdout buffers, seeding shards from warmstart/shared/exploit.

    Returns ``(buf, holdout_buf, current_window, replay_shard_dir,
    selfplay_shards_dir)``.
    """
    current_window = tc.replay_window_start
    replay_shard_dir = _trial_replay_shard_dir(config=config, trial_dir=trial_dir)
    selfplay_shards_dir = work_dir / "selfplay_shards"
    selfplay_shards_dir.mkdir(parents=True, exist_ok=True)

    # Optional warmstart replay from salvage seed slot (fresh trials only).
    if (
        restore.seed_warmstart_used
        and (not restore.cross_trial_restore)
        and restore.seed_warmstart_replay_dir is not None
        and restore.seed_warmstart_replay_dir.is_dir()
        and (not iter_shard_paths(replay_shard_dir))
    ):
        replay_shard_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for sp in iter_shard_paths(restore.seed_warmstart_replay_dir):
            copy_or_link_shard(sp, replay_shard_dir / sp.name)
            copied += 1
        if copied:
            print(
                f"[trial] Seeded {copied} replay shards from salvage slot "
                f"{restore.seed_warmstart_slot} ({restore.seed_warmstart_replay_dir})"
            )

    shared_shards_loaded = 0

    # Seed replay buffer with shared iter-0 data (played once from bootstrap net).
    # Only copy if this is a fresh trial (no existing shards in replay_shard_dir).
    if tc.shared_shards_dir and not iter_shard_paths(replay_shard_dir) and (not restore.cross_trial_restore):
        src = Path(tc.shared_shards_dir)
        if src.is_dir():
            replay_shard_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for sp in iter_shard_paths(src):
                copy_or_link_shard(sp, replay_shard_dir / sp.name)
                copied += 1
            if copied:
                shared_shards_loaded = int(copied)
                print(f"[trial] Seeded {copied} shared iter-0 shards from {src}")

    if restore.cross_trial_restore and tc.exploit_replay_refresh_enabled:
        donor_trial_dir = Path(restore.restored_owner_trial_dir).expanduser() if restore.restored_owner_trial_dir else None
        refresh_summary = _refresh_replay_shards_on_exploit(
            config=config,
            replay_shard_dir=replay_shard_dir,
            recipient_trial_dir=trial_dir,
            donor_trial_dir=donor_trial_dir,
            keep_recent_fraction=tc.exploit_replay_local_keep_recent_fraction,
            keep_older_fraction=tc.exploit_replay_local_keep_older_fraction,
            donor_shards=tc.exploit_replay_donor_shards,
            donor_skip_newest=tc.exploit_replay_skip_newest,
            shard_size=tc.shard_size,
            holdout_fraction=tc.holdout_fraction,
        )
        print(
            "[trial] replay refresh after exploit: "
            f"local_before={refresh_summary['local_before']} "
            f"deleted={refresh_summary['local_deleted']} "
            f"local_recent_deleted={refresh_summary['local_recent_deleted']} "
            f"local_older_deleted={refresh_summary['local_older_deleted']} "
            f"after_keep={refresh_summary['local_after_keep']} "
            f"donor_available={refresh_summary['donor_available']} "
            f"donor_selected={refresh_summary['donor_selected']} "
            f"donor_copied={refresh_summary['donor_copied']} "
            f"final={refresh_summary['local_final']}"
        )
        # Set current_window based on shards actually kept on disk after refresh,
        # so the DiskReplayBuffer capacity is correct from construction and
        # _enforce_window doesn't aggressively evict on first add_many.
        kept_after_refresh = iter_shard_paths(replay_shard_dir)
        if kept_after_refresh:
            current_window = max(int(current_window), len(kept_after_refresh) * tc.shard_size)
            print(
                f"[trial] exploit restore: pre-set window={current_window} "
                f"for {len(kept_after_refresh)} kept shards"
            )

    buf = DiskReplayBuffer(
        current_window,
        shard_dir=replay_shard_dir,
        rng=rng,
        shuffle_cap=tc.shuffle_buffer_size,
        shard_size=tc.shard_size,
        refresh_interval=tc.shuffle_refresh_interval,
        refresh_shards=tc.shuffle_refresh_shards,
        draw_cap_frac=tc.shuffle_draw_cap_frac,
        wl_max_ratio=tc.shuffle_wl_max_ratio,
    )

    # Preserve intentionally seeded replay (resume, salvage warmstart, shared-shard
    # bootstrap), but keep plain fresh starts at replay_window_start so easy early
    # games evict promptly instead of inheriting stale local shards.
    seeded_replay_start = bool(ckpt is not None or restore.seed_warmstart_used or shared_shards_loaded > 0)
    if seeded_replay_start:
        current_window = max(int(current_window), int(len(buf)))
    buf.capacity = int(current_window)
    print(
        f"[trial] buffer init: startup_source={restore.startup_source} "
        f"seeded={seeded_replay_start} cross_trial={restore.cross_trial_restore} "
        f"len(buf)={len(buf)} capacity={buf.capacity} "
        f"tracked_shards={len(buf._shard_paths)} total_pos={buf._total_positions}"
    )
    holdout_buf = ArrayReplayBuffer(tc.holdout_capacity, rng=rng)

    return buf, holdout_buf, current_window, replay_shard_dir, selfplay_shards_dir


def train_trial(config: dict):
    """Ray Tune trainable.

    Reports metrics per outer-loop iteration. Supports checkpoint restore.
    """

    from ray.tune import report as _tune_report, get_checkpoint as _tune_get_checkpoint
    from ray.tune import Checkpoint, get_context as _tune_get_context

    # Re-read YAML and overlay all keys EXCEPT those PB2 is actively searching.
    # This lets --resume pick up config changes without clobbering tuned hyperparams.
    _reload_yaml_into_config(config, config.get("_yaml_config_path"))
    if "device" not in config:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    tc = TrialConfig.from_dict(config)
    _yaml_path = tc._yaml_config_path

    _ctx = _tune_get_context()
    base_seed = tc.seed
    trial_id = str(_ctx.get_trial_id() or "trial")
    trial_seed = _stable_seed_u32(base_seed, trial_id)
    active_seed = int(trial_seed)
    rng = np.random.default_rng(active_seed)
    torch.manual_seed(active_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(active_seed)

    device = tc.device

    model_cfg = ModelConfig(
        kind=tc.model,
        embed_dim=tc.embed_dim,
        num_layers=tc.num_layers,
        num_heads=tc.num_heads,
        ffn_mult=tc.ffn_mult,
        use_smolgen=tc.use_smolgen,
        use_nla=tc.use_nla,
        use_gradient_checkpointing=tc.gradient_checkpointing,
    )
    model = build_model(model_cfg)

    # Use Ray-provided trial directory for ALL per-trial state (checkpoints,
    # replay shards, gate state, best model, TensorBoard logs).
    # IMPORTANT: Do NOT use config["work_dir"] here — it points to the shared
    # runs/pbt2_small/ directory. Using it caused all 10 trials to write
    # checkpoints to the same directory, making PB2 unable to clone checkpoints
    # ("no checkpoint for trial X. Skip exploit.").
    trial_dir = Path(_ctx.get_trial_dir())
    work_dir = trial_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    # Compact status CSV — reset on each process start so checkpoint-restore rows don't accumulate.
    _STATUS_CSV_PATH = trial_dir / "status.csv"
    _STATUS_COLS = [
        "iter", "global_iter", "opp", "opp_ema", "skill", "sf_nodes", "rmp", "regret",
        "ingest_s", "train_s", "iter_s", "steps", "replay", "pos_added",
        "stale", "train_loss", "best_loss", "win", "draw", "loss", "lr", "startup",
    ]
    with _STATUS_CSV_PATH.open("w", newline="") as _f:
        csv.writer(_f).writerow(_STATUS_COLS)

    # Gate match counter (so we can log gate scalars against match #, not iteration).
    gate_state_path = work_dir / "gate_state.json"
    gate_match_idx = 0
    if gate_state_path.exists():
        try:
            d = json.loads(gate_state_path.read_text(encoding="utf-8"))
            gate_match_idx = int(d.get("matches", 0))
        except Exception:
            gate_match_idx = 0

    # Best-model tracking (per trial)
    best_state_path = work_dir / "best.json"
    best_dir = work_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    opp_strength_ema = 0.0
    _OPP_EMA_ALPHA = 0.3  # smoothing factor — higher = more responsive
    if best_state_path.exists():
        try:
            d = json.loads(best_state_path.read_text(encoding="utf-8"))
            best_loss = float(d.get("best_loss", d.get("loss", best_loss)))
            opp_strength_ema = float(d.get("opp_strength_ema", 0.0))
        except Exception:
            pass

    trainer_ctor = trainer_kwargs_from_config(
        config | {"device": device}, log_dir=work_dir / "tb",
    )
    trainer = Trainer(model, **trainer_ctor)

    ckpt = _tune_get_checkpoint()
    restore, rng = _restore_checkpoint_or_salvage(
        config=config, trainer=trainer, device=device,
        trial_id=trial_id, trial_dir=trial_dir,
        base_seed=base_seed, active_seed=active_seed,
        rng=rng, ckpt=ckpt, Checkpoint=Checkpoint,
    )
    restored_pid_state = restore.restored_pid_state
    global_iter = restore.global_iter
    opp_strength_ema = restore.opp_strength_ema

    # Rebuild tc — _restore_checkpoint_or_salvage may overlay donor config.
    tc = TrialConfig.from_dict(config)

    buf, holdout_buf, current_window, replay_shard_dir, selfplay_shards_dir = _init_replay_buffers(
        tc=tc, config=config, restore=restore,
        trial_dir=trial_dir, work_dir=work_dir, rng=rng, ckpt=ckpt,
    )

    _maybe_load_bootstrap(tc=tc, trainer=trainer, device=device, ckpt=ckpt, restore=restore)

    holdout_frozen = False
    holdout_generation = 0

    use_distributed_selfplay = (
        tc.distributed_workers_per_trial > 0
        and bool((tc.distributed_server_root or "").strip())
        and bool((tc.distributed_server_url or "").strip())
    )

    need_local_sf = (not use_distributed_selfplay) or (tc.gate_games > 0)
    sf = None
    if need_local_sf:
        if tc.sf_workers > 1:
            sf = StockfishPool(
                path=tc.stockfish_path,
                nodes=tc.sf_nodes,
                num_workers=tc.sf_workers,
                multipv=tc.sf_multipv,
                hash_mb=tc.sf_hash_mb,
            )
        else:
            sf = StockfishUCI(
                tc.stockfish_path,
                nodes=tc.sf_nodes,
                multipv=tc.sf_multipv,
                hash_mb=tc.sf_hash_mb,
            )

    distributed_server_root = (
        _resolve_local_override_root(
            raw_root=str(tc.distributed_server_root),
            tune_work_dir=tc.work_dir or str(work_dir),
            suffix="server",
        )
        if use_distributed_selfplay
        else None
    )
    if distributed_server_root is not None:
        _set_active_run_prefix(server_root=distributed_server_root, trial_id=trial_id)
    distributed_dirs = (
        _trial_server_dirs(server_root=distributed_server_root, trial_id=trial_id)
        if distributed_server_root is not None
        else None
    )
    distributed_worker_procs: list[subprocess.Popen[bytes]] = []
    distributed_inference_broker_proc: subprocess.Popen[bytes] | None = None

    if use_distributed_selfplay and ckpt is not None and distributed_dirs is not None:
        quarantined = _quarantine_inbox_shards(
            inbox_dir=distributed_dirs["inbox_dir"],
            processed_dir=distributed_dirs["processed_dir"],
            reason=f"{restore.startup_source}_resume",
        )
        if int(quarantined["moved_shards"]) > 0:
            print(
                "[trial] quarantined preexisting inbox shards on resume: "
                f"trial={trial_id} moved={quarantined['moved_shards']} "
                f"dst={quarantined['quarantine_root']}"
            )

    eval_sf = None
    if tc.eval_games > 0:
        # For fixed-strength evaluation, use a dedicated engine instance with its own node limit.
        eval_sf = StockfishUCI(
            tc.stockfish_path,
            nodes=tc.eval_sf_nodes,
            multipv=1,
            hash_mb=tc.sf_hash_mb,
        )

    pid = None
    if tc.sf_pid_enabled:
        pid = pid_from_config(config)
        if restored_pid_state is not None:
            try:
                pid.load_state_dict(restored_pid_state)
            except Exception:
                pass

    # Optional puzzle evaluation suite.
    puzzle_suite = None
    if tc.puzzle_epd and tc.puzzle_interval > 0:
        from chess_anti_engine.eval import load_epd
        try:
            puzzle_suite = load_epd(tc.puzzle_epd)
        except FileNotFoundError:
            puzzle_suite = None

    pause_marker_path = _resolve_pause_marker_path(tc=tc, trial_dir=trial_dir)

    if use_distributed_selfplay:
        assert distributed_server_root is not None
        assert distributed_dirs is not None
        current_rand_init = (
            float(pid.random_move_prob)
            if pid is not None
            else tc.sf_pid_random_move_prob_start
        )
        current_nodes_init = int(pid.nodes) if pid is not None else tc.sf_nodes
        current_skill_init = int(pid.skill_level) if pid is not None else 0
        sims_init = _resolve_sims(tc, trainer, max_sims=tc.mcts_simulations)
        prev_published_model_sha = _publish_distributed_trial_state(
            trainer=trainer,
            config=config,
            model_cfg=model_cfg,
            server_root=distributed_server_root,
            trial_id=trial_id,
            training_iteration=0,
            trainer_step=int(getattr(trainer, "step", 0)),
            sf_nodes=current_nodes_init,
            random_move_prob=current_rand_init,
            skill_level=current_skill_init,
            mcts_simulations=int(sims_init),
            wdl_regret=float(pid.wdl_regret) if pid is not None else -1.0,
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
        distributed_worker_procs = _ensure_distributed_workers(
            config=config,
            trial_dir=trial_dir,
            trial_id=trial_id,
            procs=distributed_worker_procs,
        )
    distributed_pause_active = False
    distributed_pause_started_at: float | None = None
    prev_published_model_sha: str = ""

    ckpt_dir = work_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        iterations = tc.iterations
        completed_iterations = 0
        while completed_iterations < iterations:
            iteration_zero_based = int(global_iter)
            iteration_idx = iteration_zero_based + 1
            iter_t0 = time.monotonic()

            # Live-reload YAML config each iteration so changes apply without restart.
            _reload_yaml_into_config(config, _yaml_path)
            tc = TrialConfig.from_dict(config)

            in_salvage_startup_grace = (
                restore.startup_source == "salvage"
                and bool(restore.salvage_origin_used)
                and int(iteration_zero_based) < tc.salvage_startup_no_share_iters
            )
            _wait_if_paused(
                pause_marker_path=pause_marker_path,
                poll_seconds=tc.pause_poll_seconds,
                trial_id=trial_id,
                iteration=iteration_idx,
            )

            # Difficulty knobs used for this iteration's selfplay (kept fixed across
            # selfplay chunks). PID is updated once per iteration AFTER training so
            # changes align to net updates rather than chunk noise.
            current_rand = float(pid.random_move_prob) if pid is not None else tc.sf_pid_random_move_prob_start
            wdl_regret_used = float(pid.wdl_regret) if pid is not None else -1.0
            sf_nodes_used = (
                int(getattr(sf, "nodes", 0) or 0)
                if sf is not None
                else (int(pid.nodes) if pid is not None else tc.sf_nodes)
            )
            skill_level_used = int(getattr(pid, "skill_level", 0) or 0) if pid is not None else 0

            base_sims = tc.mcts_simulations
            sims = _resolve_sims(tc, trainer, max_sims=base_sims)

            sp, prev_published_model_sha, current_window, distributed_inference_broker_proc = _run_selfplay_phase(
                tc=tc, config=config, trainer=trainer, model_cfg=model_cfg,
                buf=buf, holdout_buf=holdout_buf,
                holdout_frozen=holdout_frozen,
                device=device, rng=rng, sf=sf, pid=pid,
                use_distributed_selfplay=use_distributed_selfplay,
                distributed_dirs=distributed_dirs,
                distributed_server_root=distributed_server_root,
                distributed_worker_procs=distributed_worker_procs,
                distributed_inference_broker_proc=distributed_inference_broker_proc,
                prev_published_model_sha=prev_published_model_sha,
                current_rand=current_rand,
                skill_level_used=skill_level_used,
                sims=sims,
                iteration_idx=iteration_idx,
                iteration_zero_based=iteration_zero_based,
                trial_id=trial_id,
                trial_dir=trial_dir,
                selfplay_shards_dir=selfplay_shards_dir,
                replay_shard_dir=replay_shard_dir,
                current_window=current_window,
                in_salvage_startup_grace=in_salvage_startup_grace,
            )
            distributed_pause_active = False
            distributed_pause_started_at = None
            if sp.should_retry:
                continue

            if (not holdout_frozen) and tc.freeze_holdout_at > 0 and len(holdout_buf) >= tc.freeze_holdout_at:
                holdout_frozen = True

            drift = _compute_drift_metrics(
                buf=buf, holdout_buf=holdout_buf,
                drift_sample_size=tc.drift_sample_size,
            )

            # Optional holdout reset based on input drift threshold.
            if (
                tc.reset_holdout_on_drift
                and (tc.drift_threshold > 0.0)
                and (drift.drift_input_l2 > tc.drift_threshold)
            ):
                holdout_buf.clear()
                holdout_frozen = False
                holdout_generation += 1

            _sync_trainer_weights(trainer, config, tc, wdl_regret_used, current_rand)

            tr = _run_training_and_gating(
                tc=tc, trainer=trainer, buf=buf, holdout_buf=holdout_buf,
                config=config, model_cfg=model_cfg,
                device=device, rng=rng, pid=pid, sf=sf,
                current_rand=current_rand,
                skill_level_used=skill_level_used,
                sims=sims,
                wdl_regret_used=wdl_regret_used,
                total_positions=sp.total_positions,
                imported_samples_this_iter=sp.imported_samples_this_iter,
                gate_match_idx=gate_match_idx,
                gate_state_path=gate_state_path,
                use_distributed_selfplay=use_distributed_selfplay,
                distributed_server_root=distributed_server_root,
                distributed_dirs=distributed_dirs,
                iteration_idx=iteration_idx,
                iteration_zero_based=iteration_zero_based,
                trial_id=trial_id,
                restore=restore,
            )
            gate_match_idx = tr.gate_match_idx

            eval_dict = _run_eval_games(
                tc=tc, trainer=trainer, device=device, rng=rng,
                eval_sf=eval_sf,
            )

            # Flush replay + save checkpoint (model+optimizer+step + PID state).
            checkpoint = _save_trial_checkpoint(
                trainer=trainer,
                buf=buf,
                ckpt_dir=ckpt_dir,
                rng=rng,
                trial_id=trial_id,
                trial_dir=trial_dir,
                config=config,
                base_seed=base_seed,
                restore=restore,
                iteration_idx=iteration_idx,
                Checkpoint=Checkpoint,
            )

            # Best-model tracking: prefer holdout loss when available, skip if no training yet.
            best_loss = _update_best_model(
                trainer=trainer, test_metrics=tr.test_metrics, train_metrics=tr.metrics,
                best_loss=best_loss, best_dir=best_dir, best_state_path=best_state_path,
                iteration_idx=iteration_idx, opp_strength_ema=opp_strength_ema,
            )

            pid_result = _run_pid_and_eval(
                tc=tc, pid=pid, sf=sf,
                sp_result=sp,
                iteration_zero_based=iteration_zero_based,
                opp_strength_ema=opp_strength_ema,
                opp_ema_alpha=_OPP_EMA_ALPHA,
                current_rand=current_rand,
                sf_nodes_used=sf_nodes_used,
                skill_level_used=skill_level_used,
                wdl_regret_used=wdl_regret_used,
            )
            opp_strength_ema = pid_result.opp_strength_ema

            _finalize_iteration(
                tc=tc, trainer=trainer, pid=pid,
                sp=sp, tr=tr, drift=drift,
                pid_result=pid_result,
                eval_dict=eval_dict,
                checkpoint=checkpoint,
                best_loss=best_loss,
                ckpt_dir=ckpt_dir,
                work_dir=work_dir,
                trial_dir=trial_dir,
                status_csv_path=_STATUS_CSV_PATH,
                tune_report_fn=_tune_report,
                puzzle_suite=puzzle_suite,
                current_rand=current_rand,
                wdl_regret_used=wdl_regret_used,
                sf_nodes_used=sf_nodes_used,
                skill_level_used=skill_level_used,
                use_distributed_selfplay=use_distributed_selfplay,
                distributed_pause_started_at=distributed_pause_started_at,
                distributed_pause_active=distributed_pause_active,
                restore=restore,
                holdout_frozen=holdout_frozen,
                holdout_generation=holdout_generation,
                buf_size=len(buf),
                holdout_buf_size=len(holdout_buf),
                iter_t0=iter_t0,
                iteration_idx=iteration_idx,
                iteration_zero_based=iteration_zero_based,
                completed_iterations=completed_iterations,
                device=device,
                rng=rng,
            )

            global_iter = int(iteration_idx)
            completed_iterations += 1
    finally:
        _stop_worker_processes(distributed_worker_procs)
        _stop_process(distributed_inference_broker_proc)
        if sf is not None:
            sf.close()
        if eval_sf is not None:
            eval_sf.close()
