from __future__ import annotations

import csv
import json
import subprocess
import time

# Optional dependency module (Ray Tune). Kept import-light so the core package
# works without installing `.[tune]`.
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.stockfish import StockfishPool, StockfishUCI, pid_from_config
from chess_anti_engine.train import Trainer, trainer_kwargs_from_config
from chess_anti_engine.tune._utils import (
    resolve_local_override_root as _resolve_local_override_root,
)
from chess_anti_engine.tune._utils import (
    stable_seed_u32 as _stable_seed_u32,
)
from chess_anti_engine.tune.distributed_runtime import (
    _ensure_distributed_workers,
    _ensure_inference_broker,
    _publish_distributed_trial_state,
    _quarantine_inbox_shards,
    _set_active_run_prefix,
    _stop_process,
    _stop_worker_processes,
    _trial_server_dirs,
)
from chess_anti_engine.tune.trainable_config_ops import (
    _reload_yaml_into_config,
    _resolve_pause_marker_path,
    _resolve_sims,
    _sync_trainer_weights,
    _wait_if_paused,
)
from chess_anti_engine.tune.trainable_init import (
    _init_replay_buffers,
    _maybe_load_bootstrap,
    _restore_checkpoint_or_salvage,
)
from chess_anti_engine.tune.trainable_metrics import _compute_drift_metrics
from chess_anti_engine.tune.trainable_phases import (
    _finalize_iteration,
    _run_eval_games,
    _run_pid_and_eval,
    _run_selfplay_phase,
    _run_training_and_gating,
)
from chess_anti_engine.tune.trainable_report import (
    _save_trial_checkpoint,
    _update_best_model,
)
from chess_anti_engine.tune.trial_config import DifficultyState, TrialConfig


def train_trial(config: dict):
    """Ray Tune trainable.

    Reports metrics per outer-loop iteration. Supports checkpoint restore.
    """

    from ray.tune import Checkpoint
    from ray.tune import get_checkpoint as _tune_get_checkpoint
    from ray.tune import get_context as _tune_get_context
    from ray.tune import report as _tune_report

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
        "iter", "global_iter", "opp", "opp_ema", "sf_nodes", "regret",
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
    trainer = Trainer(model, model_config=model_cfg, **trainer_ctor)

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

    if not (
        tc.distributed_workers_per_trial > 0
        and bool((tc.distributed_server_root or "").strip())
        and bool((tc.distributed_server_url or "").strip())
    ):
        raise RuntimeError(
            "Trainable requires distributed selfplay to be configured: "
            "distributed_workers_per_trial>0 plus server_root/server_url. "
            "run.py auto-enables this for --mode train; if you see this from "
            "a direct trainable invocation, call run.py --mode train (or tune) "
            "or populate the distributed_* keys."
        )

  # Local SF is only needed for gate-check games; distributed workers run
  # their own SF subprocesses.
    sf = None
    if tc.gate_games > 0:
        if tc.sf_workers > 1:
            sf = StockfishPool(
                path=tc.stockfish_path,
                nodes=tc.sf_nodes,
                num_workers=tc.sf_workers,
                multipv=tc.sf_multipv,
                hash_mb=tc.sf_hash_mb,
                syzygy_path=tc.syzygy_path,
            )
        else:
            sf = StockfishUCI(
                tc.stockfish_path,
                nodes=tc.sf_nodes,
                multipv=tc.sf_multipv,
                hash_mb=tc.sf_hash_mb,
                syzygy_path=tc.syzygy_path,
            )

    distributed_server_root = _resolve_local_override_root(
        raw_root=str(tc.distributed_server_root),
        tune_work_dir=tc.work_dir or str(work_dir),
        suffix="server",
    )
    _set_active_run_prefix(server_root=distributed_server_root, trial_id=trial_id)
    distributed_dirs = _trial_server_dirs(server_root=distributed_server_root, trial_id=trial_id)
    distributed_worker_procs: list[subprocess.Popen[bytes]] = []
    distributed_inference_broker_proc: subprocess.Popen[bytes] | None = None

    if ckpt is not None:
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
            syzygy_path=tc.syzygy_path,
        )

    pid = None
    if tc.sf_pid_enabled:
        pid = pid_from_config(config)
        if restored_pid_state is not None:
            try:
                pid.load_state_dict(restored_pid_state)
            except Exception:
                pass
  # Keep local sf in lock-step with restored PID nodes so gate games
  # on resume play at the same difficulty as distributed workers.
        if sf is not None:
            try:
                sf.set_nodes(int(pid.nodes))
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

    current_nodes_init = int(pid.nodes) if pid is not None else tc.sf_nodes
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
            ds = DifficultyState.from_pid(pid, sf, tc)

            base_sims = tc.mcts_simulations
            sims = _resolve_sims(tc, trainer, max_sims=base_sims)

            sp, prev_published_model_sha, current_window, distributed_inference_broker_proc = _run_selfplay_phase(
                tc=tc, config=config, trainer=trainer, model_cfg=model_cfg,
                buf=buf, holdout_buf=holdout_buf,
                holdout_frozen=holdout_frozen,
                rng=rng,
                distributed_dirs=distributed_dirs,
                distributed_server_root=distributed_server_root,
                distributed_worker_procs=distributed_worker_procs,
                distributed_inference_broker_proc=distributed_inference_broker_proc,
                prev_published_model_sha=prev_published_model_sha,
                ds=ds,
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

            _sync_trainer_weights(trainer, config, tc, ds)

            tr = _run_training_and_gating(
                tc=tc, trainer=trainer, buf=buf, holdout_buf=holdout_buf,
                config=config, model_cfg=model_cfg,
                device=device, rng=rng, sf=sf,
                ds=ds,
                sims=sims,
                total_positions=sp.total_positions,
                imported_samples_this_iter=sp.imported_samples_this_iter,
                gate_match_idx=gate_match_idx,
                gate_state_path=gate_state_path,
                distributed_server_root=distributed_server_root,
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
                tc=tc, config=config, pid=pid, sf=sf,
                sp_result=sp,
                opp_strength_ema=opp_strength_ema,
                opp_ema_alpha=_OPP_EMA_ALPHA,
                ds=ds,
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
                ds=ds,
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
