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


# ---------------------------------------------------------------------------
# Declarative section → argparse key mappings.
#
# Each section maps (nested_key → flat_key).  When nested_key == flat_key the
# entry can be listed in the simpler _PASSTHROUGH tuple instead.
# ---------------------------------------------------------------------------

# Keys that pass through 1:1 from a nested section (or flat root) to argparse.
_CORE_KEYS = (
    "mode", "seed", "device", "iterations", "work_dir",
    "replay_capacity", "replay_window_start", "replay_window_max", "replay_window_growth",
    "bootstrap_dir", "bootstrap_checkpoint", "bootstrap_zero_policy_heads",
    "bootstrap_reinit_volatility_heads", "worker_wheel_path",
    "bootstrap_max_positions", "bootstrap_train_steps", "shared_shards_dir",
    "pause_file", "pause_poll_seconds",
    "best_regret_checkpoints_dir",
    "salvage_seed_pool_dir", "salvage_source_run_id", "salvage_top_n",
    "salvage_out_dir", "salvage_metric", "salvage_copy_replay",
    "salvage_reinit_volatility_heads", "salvage_restore_pid_state",
    "salvage_restore_donor_config", "salvage_restore_full_trainer_state",
    "salvage_startup_no_share_iters", "salvage_startup_max_train_steps",
    "salvage_startup_post_share_ramp_iters", "salvage_startup_post_share_max_train_steps",
    "puzzle_epd", "puzzle_interval", "puzzle_simulations",
)

# stockfish section: 1:1 passthrough (YAML key == flat config key).
# The section is just visual grouping — keys use the same names as code.
_STOCKFISH_KEYS = (
    "stockfish_path", "sf_nodes", "sf_workers", "sf_multipv", "sf_hash_mb",
    "sf_pid_enabled",
    "sf_pid_target_winrate", "sf_pid_ema_alpha", "sf_pid_deadzone", "sf_pid_rate_limit",
    "sf_pid_min_games_between_adjust",
    "sf_pid_kp", "sf_pid_ki", "sf_pid_kd", "sf_pid_integral_clamp",
    "sf_pid_min_nodes", "sf_pid_max_nodes",
    "sf_pid_initial_skill_level", "sf_pid_skill_min", "sf_pid_skill_max",
    "sf_pid_skill_promote_nodes", "sf_pid_skill_demote_nodes",
    "sf_pid_skill_nodes_on_promote", "sf_pid_skill_nodes_on_demote",
    "sf_pid_random_move_prob_start", "sf_pid_random_move_prob_min", "sf_pid_random_move_prob_max",
    "sf_pid_random_move_stage_end", "sf_pid_topk_stage_end", "sf_pid_topk_min",
    "sf_pid_suboptimal_wdl_regret_max", "sf_pid_suboptimal_wdl_regret_min",
    "sf_pid_max_rand_step", "sf_pid_max_rand_step_start", "sf_pid_max_rand_step_ramp_iters",
    "sf_pid_wdl_regret_start", "sf_pid_wdl_regret_min", "sf_pid_wdl_regret_max",
    "sf_pid_wdl_regret_stage_end", "sf_pid_max_regret_step", "sf_pid_max_regret_ease_step",
)
# Backwards compat: old short YAML names still work inside stockfish: section.
_STOCKFISH_LEGACY: dict[str, str] = {
    "path": "stockfish_path",
    "nodes": "sf_nodes",
    "workers": "sf_workers",
    "multipv": "sf_multipv",
    "hash_mb": "sf_hash_mb",
    "pid_enabled": "sf_pid_enabled",
    **{f"pid_{k}": f"sf_pid_{k}" for k in (
        "target_winrate", "ema_alpha", "deadzone", "rate_limit",
        "min_games_between_adjust",
        "kp", "ki", "kd", "integral_clamp",
        "min_nodes", "max_nodes",
        "initial_skill_level", "skill_min", "skill_max",
        "skill_promote_nodes", "skill_demote_nodes",
        "skill_nodes_on_promote", "skill_nodes_on_demote",
        "random_move_prob_start", "random_move_prob_min", "random_move_prob_max",
        "random_move_stage_end", "topk_stage_end", "topk_min",
        "suboptimal_wdl_regret_max", "suboptimal_wdl_regret_min",
        "max_rand_step", "max_rand_step_start", "max_rand_step_ramp_iters",
        "wdl_regret_start", "wdl_regret_min", "wdl_regret_max",
        "wdl_regret_stage_end", "max_regret_step", "max_regret_ease_step",
    )},
}

# selfplay section: all 1:1 passthrough.
_SELFPLAY_KEYS = (
    "games_per_iter", "games_per_iter_start", "games_per_iter_ramp_iters",
    "selfplay_batch", "selfplay_fraction",
    "temperature", "temperature_drop_plies", "temperature_after",
    "temperature_decay_start_move", "temperature_decay_moves", "temperature_endgame",
    "max_plies",
    "mcts", "mcts_simulations", "mcts_start_simulations", "mcts_ramp_steps", "mcts_ramp_exponent",
    "playout_cap_fraction", "fast_simulations",
    "fpu_reduction", "fpu_at_root",
    "opening_book_path", "opening_book_max_plies", "opening_book_max_games", "opening_book_prob",
    "opening_book_path_2", "opening_book_max_plies_2", "opening_book_max_games_2", "opening_book_mix_prob_2",
    "random_start_plies",
    "sf_policy_temp", "sf_policy_label_smooth", "soft_policy_temp",
    "syzygy_path", "syzygy_policy",
    "timeout_adjudication_threshold",
    "diff_focus_enabled", "diff_focus_q_weight", "diff_focus_pol_scale",
    "diff_focus_slope", "diff_focus_min",
    "categorical_bins", "hlgauss_sigma",
)

# model section: mostly 1:1 except kind→model and use_smolgen→no_smolgen (inverted).
_MODEL_PASSTHROUGH = (
    "embed_dim", "num_layers", "num_heads", "ffn_mult", "use_nla", "gradient_checkpointing",
)

# train section: all 1:1 passthrough.
_TRAIN_KEYS = (
    "optimizer", "cosmos_rank", "cosmos_gamma",
    "lr", "batch_size", "train_steps", "train_window_fraction",
    "no_amp", "feature_dropout_p",
    "fdp_king_safety", "fdp_pins", "fdp_pawns", "fdp_mobility", "fdp_outposts",
    "w_volatility",
    "accum_steps", "warmup_steps", "warmup_lr_start", "lr_eta_min", "lr_T0", "lr_T_mult",
    "grad_clip", "zclip_z_thresh", "zclip_alpha", "zclip_max_norm",
    "use_compile", "swa_start", "swa_freq",
    "w_policy", "w_soft", "w_future", "w_wdl", "w_sf_move", "w_sf_eval",
    "w_categorical", "w_sf_volatility", "w_moves_left",
    "w_sf_wdl", "sf_wdl_conf_power", "sf_wdl_draw_scale", "sf_wdl_floor", "sf_wdl_floor_at", "sf_wdl_floor_at_regret",
)

# tune section: all 1:1 passthrough.
_TUNE_KEYS = (
    "num_samples", "max_concurrent_trials", "cpus_per_trial", "gpus_per_trial",
    "distributed_workers_per_trial", "distributed_worker_sf_workers",
    "distributed_worker_poll_seconds", "distributed_worker_device",
    "distributed_worker_use_compile", "distributed_worker_compile_mode",
    "distributed_worker_aot_dir", "distributed_worker_threaded",
    "distributed_worker_selfplay_threads", "distributed_worker_auto_tune",
    "distributed_worker_target_batch_seconds",
    "distributed_worker_min_games_per_batch", "distributed_worker_max_games_per_batch",
    "distributed_worker_upload_target_positions", "distributed_worker_upload_flush_seconds",
    "distributed_worker_shared_cache_dir",
    "distributed_worker_username", "distributed_worker_password",
    "distributed_min_workers_per_trial", "distributed_max_worker_delta_per_rebalance",
    "distributed_server_port", "distributed_server_host", "distributed_server_public_url",
    "distributed_server_root_override", "tune_replay_root_override",
    "distributed_upload_compact_shard_size", "distributed_upload_compact_max_age_seconds",
    "distributed_inference_broker_enabled", "distributed_inference_shared_broker",
    "distributed_inference_batch_wait_ms",
    "distributed_inference_use_compile",
    "distributed_inference_max_batch_per_slot",
    "distributed_pause_selfplay_during_training", "distributed_wait_timeout_seconds",
    "distributed_min_games_fraction",
    "tune_metric", "tune_mode", "tune_num_to_keep", "tune_keep_last_experiments",
    "tune_scheduler",
    "eval_games", "eval_sf_nodes", "eval_mcts_simulations",
    "holdout_fraction", "holdout_capacity", "test_steps",
    "freeze_holdout_at", "reset_holdout_on_drift", "drift_threshold", "drift_sample_size",
    "search_feature_dropout_p", "search_w_volatility",
    "search_diff_focus", "search_loss_weights", "search_categorical_bins",
    "search_smolgen", "search_nla",
    "search_optimizer", "search_optimizer_choices",
    "asha_optimizer_only", "asha_optimizer_repeats",
    "pbt_synch",
    "gpbt_pairwise_lr", "gpbt_pairwise_momentum",
    "gpbt_inertia_weight", "gpbt_winner_weight",
    "gpbt_quantile_fraction", "gpbt_resample_probability",
    "pb2_perturbation_interval", "min_replay_size",
    "gate_games", "gate_threshold", "gate_interval", "gate_mcts_sims",
    "shuffle_buffer_size", "shuffle_refresh_interval", "shuffle_refresh_shards",
    "shuffle_draw_cap_frac", "shuffle_wl_max_ratio",
    "shard_size",
    "exploit_replay_refresh_enabled", "exploit_replay_keep_fraction",
    "exploit_replay_donor_shards", "exploit_replay_skip_newest",
    "exploit_replay_share_top_enabled", "exploit_replay_top_k_trials",
    "exploit_replay_top_within_best_frac", "exploit_replay_top_min_metric",
    "exploit_replay_max_unseen_iters_per_source", "exploit_replay_top_shards_per_source",
    "exploit_replay_local_keep_recent_fraction", "exploit_replay_local_keep_older_fraction",
    "exploit_replay_share_fraction",
    "pause_file", "pause_poll_seconds",
    "best_regret_checkpoints_dir",
    "salvage_seed_pool_dir", "salvage_reinit_volatility_heads",
    "salvage_restore_pid_state", "salvage_restore_donor_config",
    "salvage_restore_full_trainer_state",
    "salvage_startup_no_share_iters", "salvage_startup_max_train_steps",
    "salvage_startup_post_share_ramp_iters", "salvage_startup_post_share_max_train_steps",
)


def _build_flat_allowlist() -> frozenset[str]:
    """Derive the set of keys accepted at the YAML root level."""
    keys: set[str] = set(_CORE_KEYS)
    keys.update(_STOCKFISH_KEYS)
    # selfplay, train, tune: flat name == nested name
    keys.update(_SELFPLAY_KEYS)
    # model has special handling but these flat names are accepted
    keys.update(_MODEL_PASSTHROUGH)
    keys.add("model")  # kind → model
    keys.add("no_smolgen")  # use_smolgen → no_smolgen (inverted)
    keys.update(_TRAIN_KEYS)
    keys.update(_TUNE_KEYS)
    return frozenset(keys)


_FLAT_ALLOWLIST = _build_flat_allowlist()


def flatten_run_config_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map a nested YAML config into run.py argparse defaults.

    Supports both:
    - flat keys matching argparse destinations (e.g. sf_nodes, sf_policy_temp)
    - nested sections: stockfish/selfplay/train/model/tune
    """

    out: dict[str, Any] = {}

    # --- Pass 1: flat keys at the YAML root level ---
    for k, v in cfg.items():
        if isinstance(k, str) and k in _FLAT_ALLOWLIST:
            out[k] = v

    # --- Pass 2: nested sections (can override flat keys) ---

    stockfish = cfg.get("stockfish")
    if isinstance(stockfish, dict):
        for k in _STOCKFISH_KEYS:
            if k in stockfish:
                out[k] = stockfish[k]
        # Backwards compat: accept old short names (e.g. "path" -> "stockfish_path").
        # New-style key wins if both are present.
        for old_k, new_k in _STOCKFISH_LEGACY.items():
            if old_k in stockfish and new_k not in stockfish:
                out[new_k] = stockfish[old_k]

    selfplay = cfg.get("selfplay")
    if isinstance(selfplay, dict):
        for k in _SELFPLAY_KEYS:
            if k in selfplay:
                out[k] = selfplay[k]

    train = cfg.get("train")
    if isinstance(train, dict):
        for k in _TRAIN_KEYS:
            if k in train:
                out[k] = train[k]
        # allow train.device to set global device if the top-level key isn't present
        if "device" in train and "device" not in out:
            out["device"] = train["device"]

    model = cfg.get("model")
    if isinstance(model, dict):
        if "kind" in model:
            out["model"] = model["kind"]
        for k in _MODEL_PASSTHROUGH:
            if k in model:
                out[k] = model[k]
        if "use_smolgen" in model:
            out["no_smolgen"] = not bool(model["use_smolgen"])

    tune = cfg.get("tune")
    if isinstance(tune, dict):
        for k in _TUNE_KEYS:
            if k in tune:
                out[k] = tune[k]
        # Pass through pb2_bounds_* keys (dynamic, any number of them).
        for k, v in tune.items():
            if k.startswith("pb2_bounds_"):
                out[k] = v

    # Keep defaults dict clean.
    return {k: v for k, v in out.items() if v is not None}
