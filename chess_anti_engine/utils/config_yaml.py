from __future__ import annotations

from pathlib import Path
from typing import Any

from chess_anti_engine.config_keys import TRAINER_WEIGHT_KEYS


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict.

    We keep this tiny so the rest of the codebase doesn't need to import YAML.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    try:
        import yaml
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
    "sf_pid_target_winrate", "sf_pid_ema_alpha",
    "sf_pid_min_games_between_adjust",
    "sf_pid_min_nodes", "sf_pid_max_nodes",
    "sf_pid_wdl_regret_start", "sf_pid_wdl_regret_min", "sf_pid_wdl_regret_max",
    "sf_pid_wdl_regret_stage_end",
    "sf_pid_regret_window",
    "sf_pid_regret_max_step",
    "sf_pid_regret_max_step_frac",
    "sf_pid_regret_ease_step_frac",
    "sf_pid_regret_safety_floor",
    "sf_pid_regret_emergency_ease_step",
    "sf_pid_regret_recency_half_life",
    "sf_pid_regret_deadband_sigma",
    "sf_pid_regret_degen_step_frac",
    "sf_pid_nodes_window",
    "sf_pid_nodes_max_step",
    "sf_pid_nodes_max_step_frac",
    "sf_pid_nodes_safety_floor",
    "sf_pid_nodes_emergency_ease_step",
    "sf_pid_nodes_recency_half_life",
    "sf_pid_nodes_deadband_sigma",
    "sf_pid_nodes_degen_step_frac",
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
        "target_winrate", "ema_alpha",
        "min_games_between_adjust",
        "min_nodes", "max_nodes",
        "wdl_regret_start", "wdl_regret_min", "wdl_regret_max",
        "wdl_regret_stage_end",
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
    "sf_wdl_use_cp_logistic", "sf_wdl_cp_slope", "sf_wdl_cp_draw_width",
    "syzygy_path", "syzygy_rescore_policy", "syzygy_adjudicate",
    "syzygy_adjudicate_fraction", "syzygy_in_search",
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
    "use_compile", "compile_mode", "log_level", "swa_start", "swa_freq",
    *TRAINER_WEIGHT_KEYS,
    "sf_wdl_frac_floor", "sf_wdl_floor_at_regret",
)

# tune section: all 1:1 passthrough.
_TUNE_KEYS = (
    "num_samples", "max_concurrent_trials", "cpus_per_trial", "gpus_per_trial",
    "distributed_workers_per_trial", "distributed_worker_sf_workers",
    "distributed_worker_poll_seconds", "distributed_worker_device",
    "distributed_worker_use_compile", "distributed_worker_compile_mode",
    "distributed_worker_inference_fp8",
    "distributed_worker_aot_dir", "distributed_worker_threaded",
    "distributed_worker_selfplay_threads", "distributed_worker_auto_tune",
    "distributed_worker_threaded_dispatcher",
    "distributed_worker_dispatcher_batch_wait_ms",
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
    "distributed_prefetch_shards",
    "distributed_async_test_eval", "distributed_async_test_eval_timeout_s",
    "distributed_min_games_fraction",
    "distributed_prev_model_max_fraction",
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


def _check_unknown(section: str, section_cfg: dict, allowed: set[str]) -> None:
    """Raise ValueError listing YAML keys that are not in the allowlist.

    Fail-loud beats silently dropping keys; stale configs from before a
    simplification would otherwise run with surprising defaults.
    """
    unknown = sorted(k for k in section_cfg if k not in allowed)
    if unknown:
        raise ValueError(
            f"Unknown keys in yaml '{section}:' section: {unknown}. "
            "These are typically leftovers from a prior config schema; "
            "delete them or update to the current names."
        )


_SECTION_NAMES = frozenset({"stockfish", "selfplay", "train", "model", "tune"})


def _flatten_root_keys(cfg: dict[str, Any], out: dict[str, Any]) -> None:
    """Copy root-level keys (those not nested under sections) into ``out``.

    Raises if any root-level key is neither a known section nor in the flat
    allowlist — typically a leftover from a prior config schema.
    """
    root_unknown: list[str] = []
    for k, v in cfg.items():
        if k in _SECTION_NAMES:
            continue
        if k in _FLAT_ALLOWLIST:
            out[k] = v
        else:
            root_unknown.append(k)
    if root_unknown:
        raise ValueError(
            f"Unknown yaml root-level keys: {sorted(root_unknown)}. "
            "Nest them under stockfish/selfplay/train/model/tune, rename to "
            "a current flat key, or delete if retired."
        )


def _copy_section_keys(out: dict[str, Any], section: dict[str, Any], keys: tuple[str, ...]) -> None:
    """Copy each key in ``keys`` from ``section`` into ``out`` if present."""
    for k in keys:
        if k in section:
            out[k] = section[k]


def _apply_stockfish_section(out: dict[str, Any], section: dict[str, Any]) -> None:
    _check_unknown("stockfish", section, set(_STOCKFISH_KEYS) | set(_STOCKFISH_LEGACY.keys()))
    _copy_section_keys(out, section, _STOCKFISH_KEYS)
  # Legacy alias map (e.g. "path" -> "stockfish_path"); new-style wins if both
  # are present.
    for old_k, new_k in _STOCKFISH_LEGACY.items():
        if old_k in section and new_k not in section:
            out[new_k] = section[old_k]


def _apply_train_section(out: dict[str, Any], section: dict[str, Any]) -> None:
    _check_unknown("train", section, set(_TRAIN_KEYS) | {"device"})
    _copy_section_keys(out, section, _TRAIN_KEYS)
  # train.device sets the global device only if not set at the root level.
    if "device" in section and "device" not in out:
        out["device"] = section["device"]


def _apply_model_section(out: dict[str, Any], section: dict[str, Any]) -> None:
    _check_unknown("model", section, set(_MODEL_PASSTHROUGH) | {"kind", "use_smolgen"})
    if "kind" in section:
        out["model"] = section["kind"]
    _copy_section_keys(out, section, _MODEL_PASSTHROUGH)
    if "use_smolgen" in section:
        out["no_smolgen"] = not bool(section["use_smolgen"])


def _apply_tune_section(out: dict[str, Any], section: dict[str, Any]) -> None:
  # pb2_bounds_* are dynamic — any number of them allowed.
    unknown = sorted(k for k in section if k not in _TUNE_KEYS and not k.startswith("pb2_bounds_"))
    if unknown:
        raise ValueError(
            f"Unknown keys in yaml 'tune:' section: {unknown}. "
            "These are typically leftovers from a prior config schema; "
            "delete them or update to the current names."
        )
    _copy_section_keys(out, section, _TUNE_KEYS)
    for k, v in section.items():
        if k.startswith("pb2_bounds_"):
            out[k] = v


def flatten_run_config_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map a nested YAML config into run.py argparse defaults.

    Supports both:
    - flat keys matching argparse destinations (e.g. sf_nodes, sf_policy_temp)
    - nested sections: stockfish/selfplay/train/model/tune

    Raises ValueError if a recognized section contains unknown keys.
    """
    out: dict[str, Any] = {}
    _flatten_root_keys(cfg, out)

  # Nested sections override matching flat keys.
    stockfish = cfg.get("stockfish")
    if isinstance(stockfish, dict):
        _apply_stockfish_section(out, stockfish)

    selfplay = cfg.get("selfplay")
    if isinstance(selfplay, dict):
        _check_unknown("selfplay", selfplay, set(_SELFPLAY_KEYS))
        _copy_section_keys(out, selfplay, _SELFPLAY_KEYS)

    train = cfg.get("train")
    if isinstance(train, dict):
        _apply_train_section(out, train)

    model = cfg.get("model")
    if isinstance(model, dict):
        _apply_model_section(out, model)

    tune = cfg.get("tune")
    if isinstance(tune, dict):
        _apply_tune_section(out, tune)

    return {k: v for k, v in out.items() if v is not None}
