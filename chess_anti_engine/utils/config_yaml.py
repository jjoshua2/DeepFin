from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict.

    We keep this tiny so the rest of the codebase doesn't need to import YAML.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load config files. Install pyyaml.") from e

    data = yaml.safe_load(p.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict, got {type(data).__name__}")
    return data


def flatten_run_config_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map a nested YAML config into run.py argparse defaults.

    Supports both:
    - flat keys matching argparse destinations (e.g. sf_nodes, sf_policy_temp)
    - nested sections: stockfish/selfplay/train/model/tune
    """

    out: dict[str, Any] = {}

    # Pass through any flat keys.
    for k, v in cfg.items():
        if not isinstance(k, str):
            continue
        if k in {
            # core
            "mode",
            "seed",
            "device",
            "iterations",
            "work_dir",
            "replay_capacity",
            "bootstrap_dir",
            "bootstrap_checkpoint",
            "bootstrap_zero_policy_heads",
            "worker_wheel_path",
            "bootstrap_max_positions",
            "bootstrap_train_steps",
            "shared_shards_dir",
            "pause_file",
            "pause_poll_seconds",
            "salvage_seed_pool_dir",
            "salvage_source_run_id",
            "salvage_top_n",
            "salvage_out_dir",
            "salvage_metric",
            "salvage_copy_replay",
            "salvage_restore_donor_config",
            "salvage_restore_full_trainer_state",
            "salvage_startup_no_share_iters",
            "salvage_startup_max_train_steps",
            "salvage_startup_post_share_ramp_iters",
            "salvage_startup_post_share_max_train_steps",
            "replay_window_start",
            "replay_window_max",
            "replay_window_growth",
            # model
            "model",
            "embed_dim",
            "num_layers",
            "num_heads",
            "ffn_mult",
            "no_smolgen",
            "use_nla",
            "gradient_checkpointing",
            # stockfish
            "stockfish_path",
            "sf_nodes",
            "sf_workers",
            "sf_multipv",
            "sf_hash_mb",
            "sf_pid_enabled",
            "sf_pid_target_winrate",
            "sf_pid_ema_alpha",
            "sf_pid_deadzone",
            "sf_pid_rate_limit",
            "sf_pid_min_games_between_adjust",
            "sf_pid_kp",
            "sf_pid_ki",
            "sf_pid_kd",
            "sf_pid_integral_clamp",
            "sf_pid_min_nodes",
            "sf_pid_max_nodes",
            "sf_pid_initial_skill_level",
            "sf_pid_skill_min",
            "sf_pid_skill_max",
            "sf_pid_skill_promote_nodes",
            "sf_pid_skill_demote_nodes",
            "sf_pid_skill_nodes_on_promote",
            "sf_pid_skill_nodes_on_demote",
            # selfplay
            "games_per_iter",
            "games_per_iter_start",
            "games_per_iter_ramp_iters",
            "selfplay_batch",
            "temperature",
            "temperature_drop_plies",
            "temperature_after",
            "temperature_decay_start_move",
            "temperature_decay_moves",
            "temperature_endgame",
            "max_plies",
            "mcts",
            "mcts_simulations",
            "mcts_start_simulations",
            "mcts_ramp_steps",
            "mcts_ramp_exponent",
            "playout_cap_fraction",
            "fast_simulations",
            "opening_book_path",
            "opening_book_max_plies",
            "opening_book_max_games",
            "opening_book_prob",
            "random_start_plies",
            "sf_policy_temp",
            "sf_policy_label_smooth",
            "syzygy_path",
            "syzygy_policy",
            "timeout_adjudication_threshold",
            # selfplay diff focus
            "diff_focus_enabled",
            "diff_focus_q_weight",
            "diff_focus_pol_scale",
            "diff_focus_slope",
            "diff_focus_min",
            # categorical value head
            "categorical_bins",
            "hlgauss_sigma",
            # train
            "optimizer",
            "cosmos_rank",
            "cosmos_gamma",
            "train_steps",
            "train_window_fraction",
            "batch_size",
            "lr",
            "no_amp",
            "feature_dropout_p",
            "w_volatility",
            "search_w_volatility",
            "accum_steps",
            "warmup_steps",
            "warmup_lr_start",
            "lr_eta_min",
            "lr_T0",
            "lr_T_mult",
            "grad_clip",
            "zclip_z_thresh",
            "zclip_alpha",
            "zclip_max_norm",
            "use_compile",
            "swa_start",
            "swa_freq",
            # loss weights
            "w_policy",
            "w_soft",
            "w_future",
            "w_wdl",
            "w_sf_move",
            "w_sf_eval",
            "w_categorical",
            "w_sf_volatility",
            "w_moves_left",
            # puzzle eval
            "puzzle_epd",
            "puzzle_interval",
            "puzzle_simulations",
            # tune/eval
            "num_samples",
            "max_concurrent_trials",
            "cpus_per_trial",
            "distributed_workers_per_trial",
            "distributed_worker_sf_workers",
            "distributed_worker_poll_seconds",
            "distributed_worker_device",
            "distributed_worker_use_compile",
            "distributed_worker_auto_tune",
            "distributed_worker_target_batch_seconds",
            "distributed_worker_min_games_per_batch",
            "distributed_worker_max_games_per_batch",
            "distributed_worker_shared_cache_dir",
            "distributed_min_workers_per_trial",
            "distributed_max_worker_delta_per_rebalance",
            "distributed_server_port",
            "distributed_server_host",
            "distributed_server_public_url",
            "distributed_wait_timeout_seconds",
            "pbt_synch",
            "gpbt_pairwise_lr",
            "gpbt_pairwise_momentum",
            "gpbt_quantile_fraction",
            "gpbt_resample_probability",
            "tune_metric",
            "tune_mode",
            "tune_num_to_keep",
            "tune_keep_last_experiments",
            "eval_games",
            "eval_sf_nodes",
            "eval_mcts_simulations",
            "holdout_fraction",
            "holdout_capacity",
            "test_steps",
            "freeze_holdout_at",
            "reset_holdout_on_drift",
            "drift_threshold",
            "drift_sample_size",
            "search_feature_dropout_p",
            "w_volatility",
            "search_w_volatility",
            "search_optimizer",
            "search_optimizer_choices",
            "asha_optimizer_only",
            "asha_optimizer_repeats",
        }:
            out[k] = v

    # Nested sections.
    stockfish = cfg.get("stockfish")
    if isinstance(stockfish, dict):
        if "path" in stockfish:
            out["stockfish_path"] = stockfish.get("path")
        if "nodes" in stockfish:
            out["sf_nodes"] = stockfish.get("nodes")
        if "workers" in stockfish:
            out["sf_workers"] = stockfish.get("workers")
        if "multipv" in stockfish:
            out["sf_multipv"] = stockfish.get("multipv")
        if "hash_mb" in stockfish:
            out["sf_hash_mb"] = stockfish.get("hash_mb")

        # PID parameters
        if "pid_enabled" in stockfish:
            out["sf_pid_enabled"] = stockfish.get("pid_enabled")
        for k in [
            "pid_target_winrate",
            "pid_ema_alpha",
            "pid_deadzone",
            "pid_rate_limit",
            "pid_min_games_between_adjust",
            "pid_kp",
            "pid_ki",
            "pid_kd",
            "pid_integral_clamp",
            "pid_min_nodes",
            "pid_max_nodes",
            "pid_initial_skill_level",
            "pid_skill_min",
            "pid_skill_max",
            "pid_skill_promote_nodes",
            "pid_skill_demote_nodes",
            "pid_skill_nodes_on_promote",
            "pid_skill_nodes_on_demote",
            "pid_random_move_prob_start",
            "pid_random_move_prob_min",
            "pid_random_move_prob_max",
            "pid_random_move_stage_end",
            "pid_max_rand_step",
        ]:
            if k in stockfish:
                out[f"sf_{k}"] = stockfish.get(k)

    selfplay = cfg.get("selfplay")
    if isinstance(selfplay, dict):
        for k in [
            "games_per_iter",
            "games_per_iter_start",
            "games_per_iter_ramp_iters",
            "temperature",
            "temperature_drop_plies",
            "temperature_after",
            "temperature_decay_start_move",
            "temperature_decay_moves",
            "temperature_endgame",
            "max_plies",
            "mcts",
            "mcts_simulations",
            "mcts_start_simulations",
            "mcts_ramp_steps",
            "mcts_ramp_exponent",
            "playout_cap_fraction",
            "fast_simulations",
            "opening_book_path",
            "opening_book_max_plies",
            "opening_book_max_games",
            "opening_book_prob",
            "random_start_plies",
            "sf_policy_temp",
            "sf_policy_label_smooth",
            "syzygy_path",
            "syzygy_policy",
            "timeout_adjudication_threshold",
            "diff_focus_enabled",
            "diff_focus_q_weight",
            "diff_focus_pol_scale",
            "diff_focus_slope",
            "diff_focus_min",
            "categorical_bins",
            "hlgauss_sigma",
            "fpu_reduction",
            "fpu_at_root",
            "selfplay_batch",
            "selfplay_fraction",
        ]:
            if k in selfplay:
                out[k] = selfplay.get(k)

    train = cfg.get("train")
    if isinstance(train, dict):
        if "optimizer" in train:
            out["optimizer"] = train.get("optimizer")
        if "lr" in train:
            out["lr"] = train.get("lr")
        if "batch_size" in train:
            out["batch_size"] = train.get("batch_size")
        if "train_steps" in train:
            out["train_steps"] = train.get("train_steps")
        if "train_window_fraction" in train:
            out["train_window_fraction"] = train.get("train_window_fraction")
        if "no_amp" in train:
            out["no_amp"] = train.get("no_amp")
        if "feature_dropout_p" in train:
            out["feature_dropout_p"] = train.get("feature_dropout_p")
        if "w_volatility" in train:
            out["w_volatility"] = train.get("w_volatility")
        for k in ["cosmos_rank", "cosmos_gamma", "accum_steps", "warmup_steps", "lr_eta_min", "lr_T0", "lr_T_mult", "grad_clip", "zclip_z_thresh", "zclip_alpha", "zclip_max_norm", "use_compile", "swa_start", "swa_freq", "w_policy", "w_soft", "w_future", "w_wdl", "w_sf_move", "w_sf_eval", "w_categorical", "w_sf_volatility", "w_moves_left", "w_sf_wdl", "sf_wdl_floor", "sf_wdl_floor_at"]:
            if k in train:
                out[k] = train.get(k)
        if "device" in train and "device" not in out:
            # allow train.device to set global device if the top-level key isn't present
            out["device"] = train.get("device")

    model = cfg.get("model")
    if isinstance(model, dict):
        if "kind" in model:
            out["model"] = model.get("kind")
        for k in ["embed_dim", "num_layers", "num_heads", "ffn_mult", "use_nla", "gradient_checkpointing"]:
            if k in model:
                out[k] = model.get(k)
        if "use_smolgen" in model:
            out["no_smolgen"] = not bool(model.get("use_smolgen"))

    tune = cfg.get("tune")
    if isinstance(tune, dict):
        for k in [
            "num_samples",
            "max_concurrent_trials",
            "cpus_per_trial",
            "distributed_workers_per_trial",
            "distributed_worker_sf_workers",
            "distributed_worker_poll_seconds",
            "distributed_worker_device",
            "distributed_worker_use_compile",
            "distributed_worker_auto_tune",
            "distributed_worker_target_batch_seconds",
            "distributed_worker_min_games_per_batch",
            "distributed_worker_max_games_per_batch",
            "distributed_min_workers_per_trial",
            "distributed_max_worker_delta_per_rebalance",
            "distributed_server_port",
            "distributed_server_host",
            "distributed_server_public_url",
            "distributed_wait_timeout_seconds",
            "tune_metric",
            "tune_mode",
            "tune_num_to_keep",
            "tune_keep_last_experiments",
            "eval_games",
            "eval_sf_nodes",
            "eval_mcts_simulations",
            "holdout_fraction",
            "holdout_capacity",
            "test_steps",
            "freeze_holdout_at",
            "reset_holdout_on_drift",
            "drift_threshold",
            "drift_sample_size",
            "search_feature_dropout_p",
            "search_w_volatility",
            "search_diff_focus",
            "search_loss_weights",
            "search_categorical_bins",
            "search_smolgen",
            "search_nla",
            "search_optimizer",
            "search_optimizer_choices",
            "asha_optimizer_only",
            "asha_optimizer_repeats",
            "tune_scheduler",
            "pbt_synch",
            "gpbt_pairwise_lr",
            "gpbt_pairwise_momentum",
            "gpbt_quantile_fraction",
            "gpbt_resample_probability",
            "gpus_per_trial",
            "pb2_perturbation_interval",
            "min_replay_size",
            "gate_games",
            "gate_threshold",
            "gate_interval",
            "gate_mcts_sims",
            "shuffle_buffer_size",
            "shard_size",
            "exploit_replay_refresh_enabled",
            "exploit_replay_keep_fraction",
            "exploit_replay_donor_shards",
            "exploit_replay_skip_newest",
            "exploit_replay_share_top_enabled",
            "exploit_replay_top_k_trials",
            "exploit_replay_top_within_best_frac",
            "exploit_replay_top_min_metric",
            "exploit_replay_max_unseen_iters_per_source",
            "exploit_replay_top_shards_per_source",
            "exploit_replay_local_keep_recent_fraction",
            "exploit_replay_local_keep_older_fraction",
            "pause_file",
            "pause_poll_seconds",
            "salvage_seed_pool_dir",
            "salvage_restore_donor_config",
            "salvage_restore_full_trainer_state",
            "salvage_startup_no_share_iters",
            "salvage_startup_max_train_steps",
            "salvage_startup_post_share_ramp_iters",
            "salvage_startup_post_share_max_train_steps",
        ]:
            if k in tune:
                out[k] = tune.get(k)
        # Pass through pb2_bounds_* keys (dynamic, any number of them).
        for k, v in tune.items():
            if k.startswith("pb2_bounds_"):
                out[k] = v

    # Keep defaults dict clean.
    return {k: v for k, v in out.items() if v is not None}
