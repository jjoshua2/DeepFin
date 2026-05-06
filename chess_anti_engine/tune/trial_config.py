from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS

if TYPE_CHECKING:
    from chess_anti_engine.train.trainer import TrainMetrics

StartupSource = Literal[
    "fresh",
    "checkpoint",
    "checkpoint_model_only",
    "salvage",
    "exploit_restore",
    "exploit_restore_model_only",
]


@dataclass
class TrialConfig:
    """Typed, validated view of the flat Ray Tune config dict.

    Constructed once per iteration from the raw config dict.
    Read-only within each iteration -- PB2/YAML mutations happen
    to the underlying dict, then TrialConfig is rebuilt.
    """

  # --- Global / control ---
    seed: int = 0
    device: str = ""  # resolved at runtime
    iterations: int = 10
    work_dir: str = ""
    pause_file: str | None = None
    pause_poll_seconds: int = 60
    _yaml_config_path: str | None = None

  # --- Model architecture (startup-only) ---
    model: str = "transformer"
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_mult: float = 2.0
    use_smolgen: bool = True
    use_nla: bool = False
    use_qk_rmsnorm: bool = False
    gradient_checkpointing: bool = False

  # --- Training ---
    lr: float = 0.0003
    optimizer: str = "nadamw"
    cosmos_gamma: float = 0.0
    batch_size: int = 128
    accum_steps: int = 1
    train_steps: int = 25
    train_window_fraction: float = 0.0
    test_steps: int = 10
    search_optimizer: bool = False
    feature_dropout_p: float = 0.0
    fdp_king_safety: float = -1.0
    fdp_pins: float = -1.0
    fdp_pawns: float = -1.0
    fdp_mobility: float = -1.0
    fdp_outposts: float = -1.0

  # --- Selfplay ---
    games_per_iter: int = 1
    games_per_iter_start: int = 0  # 0 means use games_per_iter
    games_per_iter_ramp_iters: int = 0
    selfplay_batch: int = 10
    selfplay_fraction: float = 0.0
    max_plies: int = 240
    mcts: str = "puct"
    mcts_simulations: int = 50
    mcts_start_simulations: int = 50
    mcts_ramp_steps: int = 10_000
    mcts_ramp_exponent: float = 2.0
    progressive_mcts: bool = True
    playout_cap_fraction: float = 0.25
    fast_simulations: int = 8
    fpu_reduction: float = 1.2
    fpu_at_root: float = 1.0

  # --- Temperature ---
    temperature: float = 1.0
    temperature_drop_plies: int = 0
    temperature_after: float = 0.0
    temperature_decay_start_move: int = 20
    temperature_decay_moves: int = 60
    temperature_endgame: float = 0.6

  # --- Opening books ---
    opening_book_path: str | None = None
    opening_book_max_plies: int = 4
    opening_book_max_games: int = 200_000
    opening_book_prob: float = 1.0
    opening_book_path_2: str | None = None
    opening_book_max_plies_2: int = 16
    opening_book_max_games_2: int = 200_000
    opening_book_mix_prob_2: float = 0.0
    random_start_plies: int = 0

  # --- SF policy / game ---
    sf_policy_temp: float = 0.25
    sf_policy_label_smooth: float = 0.05
    sf_wdl_use_cp_logistic: bool = False
    sf_wdl_cp_slope: float = 0.010
    sf_wdl_cp_draw_width: float = 60.0
    soft_policy_temp: float = 2.0
    timeout_adjudication_threshold: float = 0.90
    volatility_source: str = "raw"
    syzygy_path: str | None = None
    syzygy_rescore_policy: bool = False
    categorical_bins: int = DEFAULT_CATEGORICAL_BINS
    hlgauss_sigma: float = 0.04

  # --- Diff focus ---
    diff_focus_enabled: bool = True
    diff_focus_q_weight: float = 6.0
    diff_focus_pol_scale: float = 3.5
    diff_focus_slope: float = 3.0
    diff_focus_min: float = 0.025

  # --- Stockfish ---
    stockfish_path: str = ""
    sf_nodes: int = 500
    sf_workers: int = 1
    sf_multipv: int = 1
    sf_hash_mb: int = 16
    sf_pid_enabled: bool = True

  # --- PID controller ---
    sf_pid_target_winrate: float = 0.60
    sf_pid_ema_alpha: float = 0.03
    sf_pid_wdl_regret_max: float = 1.0

  # --- Loss weights ---
    sf_wdl_frac: float = 0.5
    sf_wdl_frac_floor: float = 0.10
    search_wdl_frac: float = 0.0
    sf_wdl_floor_at_regret: float = 0.0
    sf_wdl_temperature: float = 1.0
    sf_search_dampen_sf_low: float = 0.0
    sf_search_dampen_sf_high: float = 0.0

  # --- Replay buffer ---
    replay_window_start: int = 100_000
    replay_window_max: int = 1_000_000
    replay_window_growth: int = 10_000
    shuffle_buffer_size: int = 20_000
    shard_size: int = 1000
    shuffle_refresh_interval: int = 5
    shuffle_refresh_shards: int = 3
    shuffle_draw_cap_frac: float = 0.90
    shuffle_wl_max_ratio: float = 1.5
    shared_shards_dir: str | None = None

  # --- Holdout / evaluation ---
    holdout_fraction: float = 0.02
    holdout_capacity: int = 50_000
    freeze_holdout_at: int = 0
    reset_holdout_on_drift: bool = False
    drift_threshold: float = 0.0
    drift_sample_size: int = 256
    eval_games: int = 0
    eval_sf_nodes: int = 0  # 0 means fallback to sf_nodes
    eval_mcts_simulations: int = 0  # 0 means fallback to mcts_simulations
    eval_temperature: float = 0.25
    eval_max_plies: int = 0  # 0 means fallback to max_plies

  # --- Gate ---
    gate_games: int = 0
    gate_interval: int = 1
    gate_threshold: float = 0.50
    gate_mcts_sims: int = 1

  # --- Puzzle ---
    puzzle_epd: str | None = None
    puzzle_interval: int = 1
    puzzle_simulations: int = 200

  # --- Distributed ---
    distributed_workers_per_trial: int = 1
    distributed_server_root: str | None = None
    distributed_server_url: str | None = None
    distributed_wait_timeout_seconds: float = 900.0
    distributed_worker_poll_seconds: float = 1.0
    distributed_min_games_fraction: float = 0.5
    distributed_prev_model_max_fraction: float = 0.33
    distributed_pause_selfplay_during_training: bool = False
    processed_max_age_seconds: float = 43200.0
    # Background shard prefetcher: zarr decompress moves to a daemon thread
    # during train phase. Default off until measured in production.
    distributed_prefetch_shards: bool = False
    # Run holdout test eval in a daemon thread on a snapshot of the
    # post-train model — overlaps eval (~30-50s) with the next iter's
    # selfplay phase. Trade-off: test_metrics in row N reports loss for
    # iter N-1's model; row gains a `test_iter` field to disambiguate.
    distributed_async_test_eval: bool = False
    distributed_async_test_eval_timeout_s: float = 120.0

  # --- Exploit replay sharing ---
    exploit_replay_refresh_enabled: bool = True
    exploit_replay_local_keep_recent_fraction: float = 0.20
    exploit_replay_local_keep_older_fraction: float = 0.65
    exploit_replay_donor_shards: int = -1
    exploit_replay_skip_newest: int = 0
    exploit_replay_share_top_enabled: bool = False
    exploit_replay_top_k_trials: int = 5
    exploit_replay_top_within_best_frac: float = 0.10
    exploit_replay_top_min_metric: float = -1e9
    exploit_replay_top_shards_per_source: int = 0
    exploit_replay_max_unseen_iters_per_source: int = 2
    exploit_replay_share_fraction: float = 1.0

  # --- Persistent best-regret snapshots (cross-trial, cross-experiment) ---
  # None means write in-trial at work_dir / "best_regret" (legacy). A path makes the
  # top-N snapshots persist even after Ray rotates the trial dir.
    best_regret_checkpoints_dir: str | None = "data/best_regret_checkpoints"

  # --- Salvage / bootstrap ---
    salvage_seed_pool_dir: str | None = None
    salvage_reinit_volatility_heads: bool = False
    salvage_restore_donor_config: bool = False
    salvage_restore_pid_state: bool = False
    salvage_restore_full_trainer_state: bool = False
    salvage_startup_no_share_iters: int = 0
    salvage_startup_max_train_steps: int = 0
    salvage_startup_post_share_ramp_iters: int = 0
    salvage_startup_post_share_max_train_steps: int = 0
    bootstrap_checkpoint: str | None = None
    bootstrap_zero_policy_heads: bool = False
    bootstrap_reinit_volatility_heads: bool = False

  # --- Tune bookkeeping ---
    tune_num_to_keep: int = 2

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> TrialConfig:
        """Parse a flat config dict into a typed TrialConfig.

        This is the SINGLE source of truth for default values.
        Keys not present in config get the dataclass default.
        """
        def _get(key: str, default: Any) -> Any:
            v = config.get(key)
            return default if v is None else v

        return cls(
  # --- Global ---
            seed=int(config.get("seed", 0)),
            device=str(config.get("device", "cuda")),
            iterations=int(config.get("iterations", 10)),
            work_dir=str(config.get("work_dir", "")),
            pause_file=_get("pause_file", None),
            pause_poll_seconds=int(config.get("pause_poll_seconds", 60)),
            _yaml_config_path=_get("_yaml_config_path", None),

  # --- Model ---
            model=str(config.get("model", "transformer")),
            embed_dim=int(config.get("embed_dim", 256)),
            num_layers=int(config.get("num_layers", 6)),
            num_heads=int(config.get("num_heads", 8)),
            ffn_mult=float(config.get("ffn_mult", 2)),
            use_smolgen=bool(config.get("use_smolgen", True)),
            use_nla=bool(config.get("use_nla", False)),
            use_qk_rmsnorm=bool(config.get("use_qk_rmsnorm", False)),
            gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),

  # --- Training ---
            lr=float(config["lr"]) if "lr" in config else 0.0003,
            optimizer=str(config.get("optimizer", "nadamw")),
            cosmos_gamma=float(config["cosmos_gamma"]) if "cosmos_gamma" in config else 0.0,
            batch_size=int(config.get("batch_size", 128)),
            accum_steps=int(config.get("accum_steps", 1)),
            train_steps=int(config.get("train_steps", 25)),
            train_window_fraction=float(config.get("train_window_fraction", 0.0)),
            test_steps=int(config.get("test_steps", 10)),
            search_optimizer=bool(config.get("search_optimizer", False)),
            feature_dropout_p=float(config.get("feature_dropout_p", 0.0)),
            fdp_king_safety=float(config.get("fdp_king_safety", -1)),
            fdp_pins=float(config.get("fdp_pins", -1)),
            fdp_pawns=float(config.get("fdp_pawns", -1)),
            fdp_mobility=float(config.get("fdp_mobility", -1)),
            fdp_outposts=float(config.get("fdp_outposts", -1)),

  # --- Selfplay ---
            games_per_iter=int(config.get("games_per_iter", 1)),
            games_per_iter_start=int(config.get("games_per_iter_start", config.get("games_per_iter", 1))),
            games_per_iter_ramp_iters=int(config.get("games_per_iter_ramp_iters", 0)),
            selfplay_batch=int(config.get("selfplay_batch", 10)),
            selfplay_fraction=float(config.get("selfplay_fraction", 0.0)),
            max_plies=int(config.get("max_plies", 240)),
            mcts=str(config.get("mcts", "puct")),
            mcts_simulations=int(config.get("mcts_simulations", 50)),
            mcts_start_simulations=int(config.get("mcts_start_simulations", 50)),
            mcts_ramp_steps=int(config.get("mcts_ramp_steps", 10_000)),
            mcts_ramp_exponent=float(config.get("mcts_ramp_exponent", 2.0)),
            progressive_mcts=bool(config.get("progressive_mcts", True)),
            playout_cap_fraction=float(config.get("playout_cap_fraction", 0.25)),
            fast_simulations=int(config.get("fast_simulations", 8)),
            fpu_reduction=float(config.get("fpu_reduction", 1.2)),
            fpu_at_root=float(config.get("fpu_at_root", 1.0)),

  # --- Temperature ---
            temperature=float(config.get("temperature", 1.0)),
            temperature_drop_plies=int(config.get("temperature_drop_plies", 0)),
            temperature_after=float(config.get("temperature_after", 0.0)),
            temperature_decay_start_move=int(config.get("temperature_decay_start_move", 20)),
            temperature_decay_moves=int(config.get("temperature_decay_moves", 60)),
            temperature_endgame=float(config.get("temperature_endgame", 0.6)),

  # --- Opening books ---
            opening_book_path=_get("opening_book_path", None),
            opening_book_max_plies=int(config.get("opening_book_max_plies", 4)),
            opening_book_max_games=int(config.get("opening_book_max_games", 200_000)),
            opening_book_prob=float(config.get("opening_book_prob", 1.0)),
            opening_book_path_2=_get("opening_book_path_2", None),
            opening_book_max_plies_2=int(config.get("opening_book_max_plies_2", 16)),
            opening_book_max_games_2=int(config.get("opening_book_max_games_2", 200_000)),
            opening_book_mix_prob_2=float(config.get("opening_book_mix_prob_2", 0.0)),
            random_start_plies=int(config.get("random_start_plies", 0)),

  # --- SF policy / game ---
            sf_policy_temp=float(config.get("sf_policy_temp", 0.25)),
            sf_policy_label_smooth=float(config.get("sf_policy_label_smooth", 0.05)),
            sf_wdl_use_cp_logistic=bool(config.get("sf_wdl_use_cp_logistic", False)),
            sf_wdl_cp_slope=float(config.get("sf_wdl_cp_slope", 0.010)),
            sf_wdl_cp_draw_width=float(config.get("sf_wdl_cp_draw_width", 60.0)),
            soft_policy_temp=float(config.get("soft_policy_temp", 2.0)),
            timeout_adjudication_threshold=float(config.get("timeout_adjudication_threshold", 0.90)),
            volatility_source=str(config.get("volatility_source", "raw")),
            syzygy_path=_get("syzygy_path", None),
            syzygy_rescore_policy=bool(config.get("syzygy_rescore_policy", False)),
            categorical_bins=int(config.get("categorical_bins", DEFAULT_CATEGORICAL_BINS)),
            hlgauss_sigma=float(config.get("hlgauss_sigma", 0.04)),

  # --- Diff focus ---
            diff_focus_enabled=bool(config.get("diff_focus_enabled", True)),
            diff_focus_q_weight=float(config.get("diff_focus_q_weight", 6.0)),
            diff_focus_pol_scale=float(config.get("diff_focus_pol_scale", 3.5)),
            diff_focus_slope=float(config.get("diff_focus_slope", 3.0)),
            diff_focus_min=float(config.get("diff_focus_min", 0.025)),

  # --- Stockfish ---
            stockfish_path=str(config.get("stockfish_path", "")),
            sf_nodes=int(config.get("sf_nodes", 500)),
            sf_workers=int(config.get("sf_workers", 1)),
            sf_multipv=int(config.get("sf_multipv", 1)),
            sf_hash_mb=int(config.get("sf_hash_mb", 16)),
            sf_pid_enabled=bool(config.get("sf_pid_enabled", True)),

  # --- PID ---
            sf_pid_target_winrate=float(config.get("sf_pid_target_winrate", 0.60)),
            sf_pid_ema_alpha=float(config.get("sf_pid_ema_alpha", 0.03)),
            sf_pid_wdl_regret_max=float(config.get("sf_pid_wdl_regret_max", 1.0)),

  # --- Loss weights ---
            sf_wdl_frac=float(config.get("sf_wdl_frac", 0.5)),
            sf_wdl_frac_floor=float(config.get("sf_wdl_frac_floor", 0.10)),
            sf_wdl_temperature=float(config.get("sf_wdl_temperature", 1.0)),
            sf_search_dampen_sf_low=float(config.get("sf_search_dampen_sf_low", 0.0)),
            sf_search_dampen_sf_high=float(config.get("sf_search_dampen_sf_high", 0.0)),
            search_wdl_frac=float(config.get("search_wdl_frac", 0.0)),
            sf_wdl_floor_at_regret=float(config.get("sf_wdl_floor_at_regret", 0.0)),

  # --- Replay buffer ---
            replay_window_start=int(config.get("replay_window_start", 100_000)),
            replay_window_max=int(config.get("replay_window_max", config.get("replay_capacity", 1_000_000))),
            replay_window_growth=int(config.get("replay_window_growth", 10_000)),
            shuffle_buffer_size=int(config.get("shuffle_buffer_size", 20_000)),
            shard_size=int(config.get("shard_size", 1000)),
            shuffle_refresh_interval=int(config.get("shuffle_refresh_interval", 5)),
            shuffle_refresh_shards=int(config.get("shuffle_refresh_shards", 3)),
            shuffle_draw_cap_frac=float(config.get("shuffle_draw_cap_frac", 0.90)),
            shuffle_wl_max_ratio=float(config.get("shuffle_wl_max_ratio", 1.5)),
            shared_shards_dir=_get("shared_shards_dir", None),

  # --- Holdout / evaluation ---
            holdout_fraction=float(config.get("holdout_fraction", 0.02)),
            holdout_capacity=int(config.get("holdout_capacity", 50_000)),
            freeze_holdout_at=int(config.get("freeze_holdout_at", 0)),
            reset_holdout_on_drift=bool(config.get("reset_holdout_on_drift", False)),
            drift_threshold=float(config.get("drift_threshold", 0.0)),
            drift_sample_size=int(config.get("drift_sample_size", 256)),
            eval_games=int(config.get("eval_games", 0)),
            eval_sf_nodes=int(_get("eval_sf_nodes", None) or config.get("sf_nodes", 500)),
            eval_mcts_simulations=int(_get("eval_mcts_simulations", None) or config.get("mcts_simulations", 50)),
            eval_temperature=float(config.get("eval_temperature", 0.25)),
            eval_max_plies=int(_get("eval_max_plies", None) or config.get("max_plies", 240)),

  # --- Gate ---
            gate_games=int(config.get("gate_games", 0)),
            gate_interval=int(config.get("gate_interval", 1)),
            gate_threshold=float(config.get("gate_threshold", 0.50)),
            gate_mcts_sims=int(config.get("gate_mcts_sims", 1)),

  # --- Puzzle ---
            puzzle_epd=_get("puzzle_epd", None),
            puzzle_interval=int(config.get("puzzle_interval", 1)),
            puzzle_simulations=int(config.get("puzzle_simulations", 200)),

  # --- Distributed ---
            distributed_workers_per_trial=max(1, int(config.get("distributed_workers_per_trial", 1))),
            distributed_server_root=_get("distributed_server_root", None),
            distributed_server_url=_get("distributed_server_url", None),
            distributed_wait_timeout_seconds=float(config.get("distributed_wait_timeout_seconds", 900.0)),
            distributed_worker_poll_seconds=float(config.get("distributed_worker_poll_seconds", 1.0)),
            distributed_min_games_fraction=float(config.get("distributed_min_games_fraction", 0.5)),
            distributed_prev_model_max_fraction=float(config.get("distributed_prev_model_max_fraction", 0.33)),
            distributed_pause_selfplay_during_training=bool(config.get("distributed_pause_selfplay_during_training", False)),
            processed_max_age_seconds=float(config.get("processed_max_age_seconds", 43200.0)),
            distributed_prefetch_shards=bool(config.get("distributed_prefetch_shards", False)),
            distributed_async_test_eval=bool(config.get("distributed_async_test_eval", False)),
            distributed_async_test_eval_timeout_s=float(config.get("distributed_async_test_eval_timeout_s", 120.0)),

  # --- Exploit replay sharing ---
            exploit_replay_refresh_enabled=bool(config.get("exploit_replay_refresh_enabled", True)),
            exploit_replay_local_keep_recent_fraction=float(config.get("exploit_replay_local_keep_recent_fraction", 0.20)),
            exploit_replay_local_keep_older_fraction=float(config.get("exploit_replay_local_keep_older_fraction", 0.65)),
            exploit_replay_donor_shards=int(config.get("exploit_replay_donor_shards", -1)),
            exploit_replay_skip_newest=int(config.get("exploit_replay_skip_newest", 0)),
            exploit_replay_share_top_enabled=bool(config.get("exploit_replay_share_top_enabled", False)),
            exploit_replay_top_k_trials=int(config.get("exploit_replay_top_k_trials", 5)),
            exploit_replay_top_within_best_frac=float(config.get("exploit_replay_top_within_best_frac", 0.10)),
            exploit_replay_top_min_metric=float(config.get("exploit_replay_top_min_metric", -1e9)),
            exploit_replay_top_shards_per_source=int(config.get("exploit_replay_top_shards_per_source", 0)),
            exploit_replay_max_unseen_iters_per_source=int(config.get("exploit_replay_max_unseen_iters_per_source", 2)),
            exploit_replay_share_fraction=float(config.get("exploit_replay_share_fraction", 1.0)),

  # --- Persistent best-regret snapshots ---
            best_regret_checkpoints_dir=_get("best_regret_checkpoints_dir", "data/best_regret_checkpoints"),

  # --- Salvage / bootstrap ---
            salvage_seed_pool_dir=_get("salvage_seed_pool_dir", None),
            salvage_reinit_volatility_heads=bool(config.get("salvage_reinit_volatility_heads", False)),
            salvage_restore_donor_config=bool(config.get("salvage_restore_donor_config", False)),
            salvage_restore_pid_state=bool(config.get("salvage_restore_pid_state", False)),
            salvage_restore_full_trainer_state=bool(config.get("salvage_restore_full_trainer_state", False)),
            salvage_startup_no_share_iters=max(0, int(config.get("salvage_startup_no_share_iters", 0))),
            salvage_startup_max_train_steps=int(config.get("salvage_startup_max_train_steps", 0)),
            salvage_startup_post_share_ramp_iters=int(config.get("salvage_startup_post_share_ramp_iters", 0)),
            salvage_startup_post_share_max_train_steps=int(config.get("salvage_startup_post_share_max_train_steps", 0)),
            bootstrap_checkpoint=_get("bootstrap_checkpoint", None),
            bootstrap_zero_policy_heads=bool(config.get("bootstrap_zero_policy_heads", False)),
            bootstrap_reinit_volatility_heads=bool(config.get("bootstrap_reinit_volatility_heads", False)),

  # --- Tune ---
            tune_num_to_keep=int(config.get("tune_num_to_keep", 2)),
        )


# ---------------------------------------------------------------------------
# Result dataclasses for phase functions
# ---------------------------------------------------------------------------

@dataclass
class SelfplayResult:
    """Accumulated stats from one iteration of selfplay + ingest."""

  # Win / draw / loss
    total_w: int = 0
    total_d: int = 0
    total_l: int = 0

  # Game counts
    total_games_generated: int = 0
    total_game_plies: int = 0
    total_adjudicated_games: int = 0
    total_tb_adjudicated_games: int = 0
    total_draw_games: int = 0

  # Selfplay-only subset
    total_selfplay_games: int = 0
    total_selfplay_adjudicated_games: int = 0
    total_selfplay_draw_games: int = 0

  # Curriculum subset
    total_curriculum_games: int = 0
    total_curriculum_adjudicated_games: int = 0
    total_curriculum_draw_games: int = 0

  # Endgame stats
    total_checkmate_games: int = 0
    total_stalemate_games: int = 0
    total_plies_win: int = 0
    total_plies_draw: int = 0
    total_plies_loss: int = 0

  # Positions / ingest
    total_positions: int = 0
    replay_positions_ingested: int = 0
  # Per-sample is_selfplay tag accounting (this iter's ingested training rows).
    ingest_is_selfplay_tagged: int = 0
    ingest_is_selfplay_true: int = 0

  # SF evaluation deltas
    total_sf_d6: float = 0.0
    total_sf_d6_n: int = 0

  # Distributed stale data
    distributed_stale_positions: int = 0
    distributed_stale_games: int = 0

  # Cross-trial sharing
    shared_summary: dict = field(default_factory=dict)
    imported_samples_this_iter: int = 0

  # Timing
    ingest_ms: float = 0.0

  # Control flow
    should_retry: bool = False


@dataclass
class DriftMetrics:
    """Drift and diversity metrics between training and holdout buffers."""

    drift_input_l2: float = 0.0
    drift_wdl_js: float = 0.0
    drift_policy_entropy_diff: float = 0.0
    drift_policy_entropy_train: float = 0.0
    drift_policy_entropy_holdout: float = 0.0

  # Always-on data diversity (from training buffer only)
    data_policy_entropy: float = 0.0
    data_unique_positions: float = 0.0
    data_wdl_balance: float = 0.0


@dataclass
class TrainingResult:
    """Output of the training + gating phase."""

    metrics: TrainMetrics | None = None
    test_metrics: TrainMetrics | None = None
    gate_passed: bool = True
    steps: int = 0
    target_sample_budget: int = 0
    window_target_samples: int = 0
    train_ms: float = 0.0
    gate_match_idx: int = 0
    # Iter whose model produced ``test_metrics``. Equals ``training_iteration``
    # for sync eval; lags by 1 for ``distributed_async_test_eval``.
    test_metrics_source_iter: int = -1


@dataclass(frozen=True)
class DifficultyState:
    """Opponent difficulty at the start of an iteration (pre-observe).

    Snapshot of PID/SF state used to drive selfplay, training weights,
    reporting, and PID observation for this iteration.
    """

    wdl_regret: float
    sf_nodes: int

    @classmethod
    def from_pid(cls, pid: Any, sf: Any, tc: TrialConfig) -> DifficultyState:
        """Build the per-iteration difficulty snapshot.

        PID is the source of truth when present; sf is a fallback for gate-only
        configurations where no PID exists. After PID restore, the caller is
        expected to sync ``sf.set_nodes(pid.nodes)`` so sf.nodes == pid.nodes;
        this method still prefers pid.nodes to make divergence impossible.
        """
        if pid is not None:
            return cls(
                wdl_regret=float(pid.wdl_regret),
                sf_nodes=int(pid.nodes),
            )
        return cls(
            wdl_regret=-1.0,
            sf_nodes=int(getattr(sf, "nodes", 0) or 0) if sf is not None else tc.sf_nodes,
        )


@dataclass
class PidResult:
    """Output of the PID update + eval games + opponent strength phase."""

  # PID outputs (next-iteration values)
    sf_nodes_next: int = 0
    wdl_regret_next: float = -1.0
    pid_ema_wr: float = 0.0
    pid_update: object | None = None

  # Derived game stats
    curriculum_winrate_raw: float | None = None
    avg_game_plies: float = 0.0
    adjudication_rate: float = 0.0
    tb_adjudication_rate: float = 0.0
    draw_rate: float = 0.0
    selfplay_adjudication_rate: float = 0.0
    selfplay_draw_rate: float = 0.0
    curriculum_adjudication_rate: float = 0.0
    curriculum_draw_rate: float = 0.0
    checkmate_rate: float = 0.0
    stalemate_rate: float = 0.0
    avg_plies_win: float = 0.0
    avg_plies_draw: float = 0.0
    avg_plies_loss: float = 0.0

  # Opponent strength
    opp_strength: float = 0.0
    opp_strength_ema: float = 0.0


@dataclass
class RestoreResult:
    """Output of checkpoint / salvage / fresh-start restore logic."""

    startup_source: StartupSource = "fresh"
    restored_pid_state: dict | None = None
    global_iter: int = 0
    opp_strength_ema: float = 0.0
    active_seed: int = 0
    seed_warmstart_used: bool = False
    seed_warmstart_slot: int = -1
    seed_warmstart_slots_total: int = 0
    seed_warmstart_dir: Any = None
    seed_warmstart_replay_dir: Any = None
    salvage_origin_used: bool = False
    salvage_origin_slot: int = -1
    salvage_origin_slots_total: int = 0
    salvage_origin_dir: str = ""
    cross_trial_restore: bool = False
    restored_owner_trial_dir: str = ""
    restored_window: int = 0
