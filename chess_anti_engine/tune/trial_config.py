from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from chess_anti_engine.mcts.gumbel import (
    DEFAULT_VOLATILITY_ANCHOR,
    POLICY_TEMP_MAX,
    POLICY_TEMP_MIN,
    SELFPLAY_GUMBEL_C_SCALE,
)
from chess_anti_engine.train.target_builder import SfTargetParams
from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS
from chess_anti_engine.tune.promotion_gate import GateDecision
from chess_anti_engine.utils.architecture import (
    DEFAULT_PHASE_PIECE_THRESHOLDS,
    normalize_embed_dim_by_layer,
    normalize_ffn_mult_by_layer,
    normalize_phase_piece_thresholds,
)

if TYPE_CHECKING:
    from chess_anti_engine.train.trainer import TrainMetrics

# Single home of the SF target-construction defaults (see
# trainer.resolve_sf_target_params, which derives from the same dataclass).
_SF_TARGET_DEFAULTS = SfTargetParams()

StartupSource = Literal[
    "fresh",
    "checkpoint",
    "checkpoint_model_only",
    "salvage",
    "exploit_restore",
    "exploit_restore_model_only",
]


def _optional_unit_fraction(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    frac = float(value)
    if not math.isfinite(frac) or frac < 0.0 or frac > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return frac


def _nonnegative_float(value: Any, *, name: str) -> float:
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return val


def _policy_temperature(value: Any, *, name: str) -> float:
    """Policy prior temperature: inside ``[POLICY_TEMP_MIN, POLICY_TEMP_MAX]``.

    ``mcts.gumbel.apply_policy_temp`` treats ``policy_temp <= 0`` as a no-op, so
    a mis-typed 0 would be accepted and then ignored --- the exact silent-ignore
    failure the plumbing tests exist to prevent. Reject it at load instead.

    ⚑ ``> 0 and finite`` is not enough, which is why this is a band and not a
    sign check. ``1e300`` is finite and positive and divides every logit to zero: a
    uniform prior, i.e. search with the policy head switched off, accepted by
    the validator and published to every worker as a working temperature. The
    band is imported from ``mcts.gumbel`` rather than restated here so the
    loader and the hot-path predicate ``policy_temp_active`` cannot drift into
    disagreeing about which values are real temperatures --- a guard has to
    share the criterion's instrument.
    """
    val = float(value)
    if not math.isfinite(val) or not (POLICY_TEMP_MIN <= val <= POLICY_TEMP_MAX):
        raise ValueError(
            f"{name} must be finite and in [{POLICY_TEMP_MIN}, {POLICY_TEMP_MAX}], "
            f"got {value!r}"
        )
    return val


def _at_least_one_float(value: Any, *, name: str) -> float:
    val = float(value)
    if not math.isfinite(val) or val < 1.0:
        raise ValueError(f"{name} must be finite and >= 1.0, got {value!r}")
    return val


def _validate_sf_refute_net_blend(value: Any) -> float:
    """Fail loud on sf_refute_opp_policy_net_blend > 0.

    The blend math + interface are implemented, but the net-visit provider at an
    opponent turn is a deliberately-unwired seam (running a net MCTS search on a
    non-net-color slot is architecturally invasive). Rather than silently no-op a
    requested blend, reject any positive value so the config is honest about what
    is wired. See selfplay/stockfish_turn.py::_sf_refute_net_visit_provider.
    """
    blend = float(value)
    if not math.isfinite(blend) or blend < 0.0 or blend > 1.0:
        raise ValueError(
            f"sf_refute_opp_policy_net_blend must be in [0, 1], got {value!r}",
        )
    if blend > 0.0:
        raise ValueError(
            "sf_refute_opp_policy_net_blend > 0 is not yet wired: the net-visit "
            "provider at an opponent turn is an unimplemented seam "
            "(_sf_refute_net_visit_provider). Keep it at 0.0 until that lands.",
        )
    return blend


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
    embed_dim_by_layer: tuple[int, ...] | None = None
    ffn_mult: float = 2.0
    ffn_mult_by_layer: tuple[float, ...] | None = None
    use_smolgen: bool = True
    use_nla: bool = False
    use_qk_rmsnorm: bool = False
    gradient_checkpointing: bool = False
    input_pos_encoding: str = "none"
    qkv_projection: str = "fused"
    use_deepnorm: bool = False
    policy_encoding: str = "lc0_1858"
    input_history_encoding: str = "legacy"
    input_extra_features: str = "v2_threats"
    use_dynamic_relations: bool = False
    dynamic_relation_count: int = 5
    policy_dynamic_relations: bool = False
    input_global_embedding: str = "none"
    input_global_embedding_channels: int = 0
    input_square_embedding: str = "none"
    smolgen_mode: str = "shared"
    smolgen_pooling: str = "flatten"
    smolgen_hidden_channels: int = 32
    smolgen_hidden_sz: int = 256
    smolgen_gen_sz: int = 256
    smolgen_bias_scale: str = "none"
    smolgen_bias_norm: str = "none"
    arc_attention_bias: str = "none"
    smolgen_relation_basis: bool = False
    smolgen_relation_norm: str = "none"
    smolgen_relation_coeff_norm: str = "none"
    smolgen_relation_scale: str = "none"
    phase_output_adapter: bool = False
    phase_output_adapter_dim: int = 64
    phase_smolgen: bool = False
    phase_piece_thresholds: tuple[int, int] = DEFAULT_PHASE_PIECE_THRESHOLDS

  # --- Training ---
    lr: float = 0.0003
    optimizer: str = "nadamw"
    batch_size: int = 128
    accum_steps: int = 1
    train_steps: int = 25
    train_window_fraction: float = 0.0
    train_views_per_ingested_position: float = 0.0
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
    slot_oversubscribe: float = 1.0
    max_plies: int = 240
    mcts: str = "puct"
    mcts_simulations: int = 50
    mcts_start_simulations: int = 50
    mcts_ramp_steps: int = 10_000
    mcts_ramp_exponent: float = 2.0
    progressive_mcts: bool = True
    playout_cap_fraction: float = 0.25
    full_ply_pair_fraction: float = 0.0
    fast_simulations: int = 8
    fpu_reduction: float = 1.2
    fpu_at_root: float = 1.0
    gumbel_topk: int = 16
  # lc0 PolicyTemperature on the SELFPLAY search prior; 1.0 = no-op.
  # See SearchConfig.gumbel_policy_temp for the cost note.
    gumbel_policy_temp: float = 1.0
  # C17 batching knobs. Carried here so the gate/eval matches built by
  # _play_batch_kwargs search the SAME way distributed selfplay does; without
  # them those matches silently ran the 56%-duplicate vloss_weight=0 arm while
  # the live yaml asked for 1.
    gumbel_target_batch: int = 0
    gumbel_vloss_weight: int = 0
    volatility_q_scale: float = 0.0
    volatility_fpu: float = 0.0
    volatility_anchor: float = DEFAULT_VOLATILITY_ANCHOR
    gumbel_c_scale: float = SELFPLAY_GUMBEL_C_SCALE
    gumbel_scale: float = 1.0
    gumbel_scale_after: float = 0.0
    gumbel_scale_decay_start_move: int = 0
    gumbel_scale_decay_moves: int = 0
    curriculum_gumbel_scale: float = 0.0
    curriculum_gumbel_scale_after: float = 0.0
    curriculum_gumbel_scale_decay_start_move: int = 0
    curriculum_gumbel_scale_decay_moves: int = 0

  # --- Temperature ---
    temperature: float = 1.0
    temperature_drop_plies: int = 0
    temperature_after: float = 0.0
    temperature_decay_start_move: int = 20
    temperature_decay_moves: int = 60
    temperature_endgame: float = 0.6
    selfplay_temperature: float | None = None
    selfplay_temperature_decay_start_move: int | None = None
    selfplay_temperature_decay_moves: int | None = None
    selfplay_temperature_endgame: float | None = None

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
  # Blind-spot FEN seeding (see selfplay/opening.py OpeningConfig)
    opening_fen_list_path: str | None = None
    opening_fen_prob: float = 0.0
    opening_fen_net_side_to_move: bool = True
    opening_fen_selfplay_only: bool = False
    opening_fen_dole_per_iter: int = 0
    # Upper bound on seeded games per iteration as a fraction of games_per_iter
    # (0 = uncapped, the pre-2026-07-24 behavior). The dole hands the whole seed
    # list out every iteration, so without this the seeded share of selfplay is
    # just pool_size/capacity and silently reached 100%.
    opening_fen_dole_max_fraction: float = 0.0
    opening_fen_sf_refute_frac: float = 0.0
    opening_fen_sf_refute_plies: int = 5
  # SF-refute opp-row recording (STAGED, all default off — see OpeningConfig).
    sf_refute_full_node_moves: bool = False
    sf_refute_record_opp_rows: bool = False
    sf_refute_opp_policy_net_blend: float = 0.0

  # --- SF policy / game ---
    sf_policy_temp: float = 0.25
    sf_policy_score_mode: str = "wdl"
    sf_policy_cp_temp: float = 16.2
    sf_policy_label_smooth: float = 0.05
    sf_wdl_use_cp_logistic: bool = False
    sf_wdl_cp_slope: float = 0.010
    sf_wdl_cp_draw_width: float = 60.0
    soft_policy_temp: float = 2.0
    timeout_adjudication_threshold: float = 0.90
    volatility_source: str = "raw"
    syzygy_path: str | None = None
    stockfish_syzygy_path: str | None = None
    syzygy_rescore_policy: bool = False
    syzygy_adjudicate: bool = False
    syzygy_adjudicate_fraction: float = 1.0
    syzygy_in_search: bool = False
    categorical_bins: int = DEFAULT_CATEGORICAL_BINS
    hlgauss_sigma: float = 0.04
    categorical_blend_frac: float = 0.0
    categorical_search_blend_frac: float = 0.0
    record_lc0_root_input: bool = False
    history_rep_fix: bool = False
    record_dense_sf_policy: bool = True
    record_sf_p0_policy: bool = False
    record_sf_p0_regret: bool = False
    record_fast_ply_value: bool = False
    blindspot_harvest_out_path: str = ""
  # Distributed workers only: persist in-flight games at a selfplay-session
  # teardown and resume them in the next session, instead of abandoning ~256
  # partial games (and their ~698k-node SF labels) every restart. Default off.
    selfplay_resume_inflight_games: bool = False

  # --- Diff focus ---
    diff_focus_enabled: bool = True
    diff_focus_q_weight: float = 6.0
    diff_focus_pol_scale: float = 3.5
    diff_focus_slope: float = 3.0
    diff_focus_min: float = 0.025

  # --- Stockfish ---
    stockfish_path: str = ""
    sf_nodes: int = 500
    sf_move_nodes: int = 0
    sf_fast_ply_node_scale: float = 0.25
    sf_label_nodes_cap: int = 0
    sf_label_nodes_floor: int = 0
    sf_label_escalate_q_gap: float = 0.0
    sf_label_escalate_nodes: int = 3_000_000
    sf_label_escalate_max_per_game: int = 2
    sf_workers: int = 1
    sf_multipv: int = 1
    sf_hash_mb: int = 16
    sf_nice: int = 0
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
    replay_window_growth_frac: float | None = None
    shuffle_buffer_size: int = 20_000
    shard_size: int = 1000
    # Recompute the 29 v2 threat planes for stored 146-plane chunks instead
    # of zero-padding (only acts when input_extra_features is v2_threats).
    replay_upgrade_v1_planes: bool = False
    shuffle_refresh_interval: int = 5
    shuffle_refresh_shards: int = 3
    # Exponent on the shuffle-refresh shard draw's recency weight
    # (weight ∝ rank**α, oldest rank 1). 1.0 = the linear draw production has
    # always run and the ONLY value that reproduces it bit-for-bit; 0.0 =
    # uniform over the window. Construction-time (see
    # trainable_config_ops.construction_only_config_keys).
    replay_shard_recency_exponent: float = 1.0
    shuffle_draw_cap_frac: float = 0.90
    shuffle_wl_max_ratio: float = 1.5
    # Surprise-priority shaping at shuffle append (live-reloadable). Weight
    # added per unit of stored |SF value label - search value| on full rows;
    # demotion priority for fast value-only rows whose 32-sim search read
    # agreed (within a full point) with the game outcome. 0.0 / 1.0 = off.
    replay_sf_gap_priority_weight: float = 0.0
    replay_sf_gap_priority_signed: bool = False
    replay_fast_low_surprise_priority: float = 1.0
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
  # gate_games / gate_threshold / gate_mcts_sims are DEAD KEYS kept only so a
  # live yaml carrying them still validates. The 1-sim vs-Stockfish gate they
  # configured is gone; ``gate_games`` at anything but 0 now raises at startup
  # rather than silently doing nothing (chess_anti_engine/tune/promotion_gate).
    gate_games: int = 0
    gate_interval: int = 1
    gate_threshold: float = 0.50
    gate_mcts_sims: int = 1
  # The anchored promotion gate's knobs (gate_mode, gate_window_iters,
  # gate_min_iters, gate_min_games_per_side, gate_demote_delta_elo, gate_alpha,
  # gate_max_hold_iters) are DELIBERATELY ABSENT from this dataclass. They are
  # read once, at construction, by ``promotion_gate.gate_config_from_dict``
  # straight from the config dict. Parsing them here too would give the gate
  # two sources of truth, and the copy nothing reads is the one that rots: an
  # earlier revision of this class carried the pre-review floor of 40 and line
  # of -50.0 for exactly as long as it took a reviewer to grep ``tc.gate_``
  # and find nothing.

  # --- Puzzle ---
    puzzle_epd: str | None = None
    puzzle_interval: int = 1
    puzzle_simulations: int = 200

  # --- Era-forgetting probes (chess_anti_engine/eval/era_probe.py) ---
  # Two FROZEN row sets scored with the current weights every
  # `era_probe_interval` iterations: one cut from an OLD era, one re-cut from
  # the newest shards. Their divergence is the treadmill's fingerprint. Both
  # default to "" = off, so the live yaml is untouched until an operator names
  # a set built by scripts/build_era_probe_set.py.
  #
  # The two paths and the row cap are CONSTRUCTION-ONLY (see
  # trainable_config_ops.construction_only_config_keys): the sets are loaded
  # once at trial startup, so a live edit would read back correct from the
  # trial's config while the probe kept scoring the launch-time set — the
  # exact class of lie that classification exists to refuse. Changing which
  # rows a column is measured over is a RULER change and must invalidate the
  # column's history, which is a restart, not a reload.
    era_probe_path: str = ""
    era_probe_inwindow_path: str = ""
    era_probe_rows: int = 2048
  # Live-reloadable: read fresh off `tc` on the iteration they act on, so an
  # operator can throttle or silence the probe on a contended box without a
  # restart. `era_probe_interval <= 0` disables scoring (the sets stay loaded
  # and the columns read nan).
    era_probe_interval: int = 1
    era_probe_batch_size: int = 512

  # --- Distributed ---
    distributed_workers_per_trial: int = 1
    distributed_server_root: str | None = None
    distributed_server_url: str | None = None
    distributed_wait_timeout_seconds: float = 900.0
    distributed_worker_poll_seconds: float = 1.0
    distributed_min_games_fraction: float = 0.5
    distributed_prev_model_max_fraction: float = 0.33
    distributed_stale_pause_target_games: int = -1
    distributed_pause_selfplay_during_training: bool = False
    processed_max_age_seconds: float = 43200.0
    # Background shard prefetcher: zarr decompress moves to a daemon thread
    # during train phase. Default off until measured in production.
    distributed_prefetch_shards: bool = False
    # Byte budget for the prefetch queue (audit A19). The producer is
    # shard-arrival rate and the consumer is once-per-iteration, so without
    # this the queue is unbounded: a 2000-row shard decodes to ~102 MB, i.e.
    # ~10.2 GB at 100 queued. 768 MB is ~7.5 shards -- generous for steady
    # state, where the queue holds the 1-3 shards that land during one train
    # phase. Over budget the prefetcher stops decoding and leaves shards on
    # disk for the iter-time ingest path; nothing is dropped.
    distributed_prefetch_max_queued_mb: int = 768
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

        num_layers = int(config.get("num_layers", 6))
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
            num_layers=num_layers,
            num_heads=int(config.get("num_heads", 8)),
            embed_dim_by_layer=normalize_embed_dim_by_layer(
                config.get("embed_dim_by_layer"),
                num_layers=num_layers,
            ),
            ffn_mult=float(config.get("ffn_mult", 2)),
            ffn_mult_by_layer=normalize_ffn_mult_by_layer(
                config.get("ffn_mult_by_layer"),
                num_layers=num_layers,
            ),
            use_smolgen=bool(config.get("use_smolgen", True)),
            use_nla=bool(config.get("use_nla", False)),
            use_qk_rmsnorm=bool(config.get("use_qk_rmsnorm", False)),
            gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
            input_pos_encoding=str(config.get("input_pos_encoding", "none")),
            qkv_projection=str(config.get("qkv_projection", "fused")),
            use_deepnorm=bool(config.get("use_deepnorm", False)),
            policy_encoding=str(config.get("policy_encoding", "lc0_1858")),
            input_history_encoding=str(config.get("input_history_encoding", "legacy")),
            input_extra_features=str(config.get("input_extra_features", "v2_threats")),
            use_dynamic_relations=bool(config.get("use_dynamic_relations", False)),
            dynamic_relation_count=int(config.get("dynamic_relation_count", 5)),
            policy_dynamic_relations=bool(config.get("policy_dynamic_relations", False)),
            input_global_embedding=str(config.get("input_global_embedding", "none")),
            input_global_embedding_channels=int(config.get("input_global_embedding_channels", 0)),
            input_square_embedding=str(config.get("input_square_embedding", "none")),
            smolgen_mode=str(config.get("smolgen_mode", "shared")),
            smolgen_pooling=str(config.get("smolgen_pooling", "flatten")),
            smolgen_hidden_channels=int(config.get("smolgen_hidden_channels", 32)),
            smolgen_hidden_sz=int(config.get("smolgen_hidden_sz", 256)),
            smolgen_gen_sz=int(config.get("smolgen_gen_sz", 256)),
            smolgen_bias_scale=str(config.get("smolgen_bias_scale", "none")),
            smolgen_bias_norm=str(config.get("smolgen_bias_norm", "none")),
            arc_attention_bias=str(config.get("arc_attention_bias", "none")),
            smolgen_relation_basis=bool(config.get("smolgen_relation_basis", False)),
            smolgen_relation_norm=str(config.get("smolgen_relation_norm", "none")),
            smolgen_relation_coeff_norm=str(config.get("smolgen_relation_coeff_norm", "none")),
            smolgen_relation_scale=str(config.get("smolgen_relation_scale", "none")),
            phase_output_adapter=bool(config.get("phase_output_adapter", False)),
            phase_output_adapter_dim=int(config.get("phase_output_adapter_dim", 64)),
            phase_smolgen=bool(config.get("phase_smolgen", False)),
            phase_piece_thresholds=normalize_phase_piece_thresholds(
                config.get("phase_piece_thresholds")
            ),

  # --- Training ---
            lr=float(config["lr"]) if "lr" in config else 0.0003,
            optimizer=str(config.get("optimizer", "nadamw")),
            batch_size=int(config.get("batch_size", 128)),
            accum_steps=int(config.get("accum_steps", 1)),
            train_steps=int(config.get("train_steps", 25)),
            train_window_fraction=float(config.get("train_window_fraction", 0.0)),
            train_views_per_ingested_position=float(config.get("train_views_per_ingested_position", 0.0)),
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
            slot_oversubscribe=_at_least_one_float(
                config.get("slot_oversubscribe", 1.0), name="slot_oversubscribe",
            ),
            max_plies=int(config.get("max_plies", 240)),
            mcts=str(config.get("mcts", "puct")),
            mcts_simulations=int(config.get("mcts_simulations", 50)),
            mcts_start_simulations=int(config.get("mcts_start_simulations", 50)),
            mcts_ramp_steps=int(config.get("mcts_ramp_steps", 10_000)),
            mcts_ramp_exponent=float(config.get("mcts_ramp_exponent", 2.0)),
            progressive_mcts=bool(config.get("progressive_mcts", True)),
            playout_cap_fraction=float(config.get("playout_cap_fraction", 0.25)),
            full_ply_pair_fraction=float(config.get("full_ply_pair_fraction", 0.0)),
            fast_simulations=int(config.get("fast_simulations", 8)),
            fpu_reduction=float(config.get("fpu_reduction", 1.2)),
            fpu_at_root=float(config.get("fpu_at_root", 1.0)),
            gumbel_topk=max(1, int(config.get("gumbel_topk", 16))),
            gumbel_policy_temp=_policy_temperature(
                config.get("gumbel_policy_temp", 1.0), name="gumbel_policy_temp",
            ),
            gumbel_target_batch=max(0, int(config.get("gumbel_target_batch", 0))),
            gumbel_vloss_weight=max(0, int(config.get("gumbel_vloss_weight", 0))),
            volatility_q_scale=float(config.get("volatility_q_scale", 0.0)),
            volatility_fpu=float(config.get("volatility_fpu", 0.0)),
            volatility_anchor=float(config.get("volatility_anchor", DEFAULT_VOLATILITY_ANCHOR)),
            gumbel_c_scale=_nonnegative_float(
                config.get("gumbel_c_scale", SELFPLAY_GUMBEL_C_SCALE), name="gumbel_c_scale",
            ),
            gumbel_scale=_nonnegative_float(config.get("gumbel_scale", 1.0), name="gumbel_scale"),
            gumbel_scale_after=_nonnegative_float(
                config.get("gumbel_scale_after", 0.0),
                name="gumbel_scale_after",
            ),
            gumbel_scale_decay_start_move=max(0, int(config.get("gumbel_scale_decay_start_move", 0))),
            gumbel_scale_decay_moves=max(0, int(config.get("gumbel_scale_decay_moves", 0))),
            curriculum_gumbel_scale=_nonnegative_float(
                config.get("curriculum_gumbel_scale", 0.0),
                name="curriculum_gumbel_scale",
            ),
            curriculum_gumbel_scale_after=_nonnegative_float(
                config.get("curriculum_gumbel_scale_after", 0.0),
                name="curriculum_gumbel_scale_after",
            ),
            curriculum_gumbel_scale_decay_start_move=max(
                0,
                int(config.get("curriculum_gumbel_scale_decay_start_move", 0)),
            ),
            curriculum_gumbel_scale_decay_moves=max(
                0,
                int(config.get("curriculum_gumbel_scale_decay_moves", 0)),
            ),

  # --- Temperature ---
            temperature=float(config.get("temperature", 1.0)),
            temperature_drop_plies=int(config.get("temperature_drop_plies", 0)),
            temperature_after=float(config.get("temperature_after", 0.0)),
            temperature_decay_start_move=int(config.get("temperature_decay_start_move", 20)),
            temperature_decay_moves=int(config.get("temperature_decay_moves", 60)),
            temperature_endgame=float(config.get("temperature_endgame", 0.6)),
            selfplay_temperature=(
                None
                if config.get("selfplay_temperature") is None
                else float(_get("selfplay_temperature", 0.0))
            ),
            selfplay_temperature_decay_start_move=(
                None
                if config.get("selfplay_temperature_decay_start_move") is None
                else int(_get("selfplay_temperature_decay_start_move", 0))
            ),
            selfplay_temperature_decay_moves=(
                None
                if config.get("selfplay_temperature_decay_moves") is None
                else int(_get("selfplay_temperature_decay_moves", 0))
            ),
            selfplay_temperature_endgame=(
                None
                if config.get("selfplay_temperature_endgame") is None
                else float(_get("selfplay_temperature_endgame", 0.0))
            ),

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
            opening_fen_list_path=_get("opening_fen_list_path", None),
            opening_fen_prob=float(config.get("opening_fen_prob", 0.0)),
            opening_fen_net_side_to_move=bool(
                config.get("opening_fen_net_side_to_move", True)
            ),
            opening_fen_selfplay_only=bool(config.get("opening_fen_selfplay_only", False)),
            opening_fen_dole_per_iter=int(config.get("opening_fen_dole_per_iter", 0)),
            opening_fen_dole_max_fraction=float(
                config.get("opening_fen_dole_max_fraction", 0.0)
            ),
            opening_fen_sf_refute_frac=float(config.get("opening_fen_sf_refute_frac", 0.0)),
            opening_fen_sf_refute_plies=int(config.get("opening_fen_sf_refute_plies", 5)),
            sf_refute_full_node_moves=bool(config.get("sf_refute_full_node_moves", False)),
            sf_refute_record_opp_rows=bool(config.get("sf_refute_record_opp_rows", False)),
            sf_refute_opp_policy_net_blend=_validate_sf_refute_net_blend(
                config.get("sf_refute_opp_policy_net_blend", 0.0),
            ),

  # --- SF policy / game ---
  # Defaults derive from SfTargetParams, the single home of these seven (the
  # trainer-side rebuild resolves the same yaml keys through it).
            sf_policy_temp=float(
                config.get("sf_policy_temp", _SF_TARGET_DEFAULTS.sf_policy_temp)
            ),
            sf_policy_score_mode=str(
                config.get(
                    "sf_policy_score_mode", _SF_TARGET_DEFAULTS.sf_policy_score_mode
                )
            ),
            sf_policy_cp_temp=float(
                config.get("sf_policy_cp_temp", _SF_TARGET_DEFAULTS.sf_policy_cp_temp)
            ),
            sf_policy_label_smooth=float(
                config.get(
                    "sf_policy_label_smooth", _SF_TARGET_DEFAULTS.sf_policy_label_smooth
                )
            ),
            sf_wdl_use_cp_logistic=bool(
                config.get(
                    "sf_wdl_use_cp_logistic", _SF_TARGET_DEFAULTS.sf_wdl_use_cp_logistic
                )
            ),
            sf_wdl_cp_slope=float(
                config.get("sf_wdl_cp_slope", _SF_TARGET_DEFAULTS.sf_wdl_cp_slope)
            ),
            sf_wdl_cp_draw_width=float(
                config.get(
                    "sf_wdl_cp_draw_width", _SF_TARGET_DEFAULTS.sf_wdl_cp_draw_width
                )
            ),
            soft_policy_temp=float(config.get("soft_policy_temp", 2.0)),
            timeout_adjudication_threshold=float(config.get("timeout_adjudication_threshold", 0.90)),
            volatility_source=str(config.get("volatility_source", "raw")),
            syzygy_path=_get("syzygy_path", None),
            stockfish_syzygy_path=_get("stockfish_syzygy_path", None),
            syzygy_rescore_policy=bool(config.get("syzygy_rescore_policy", False)),
            syzygy_adjudicate=bool(config.get("syzygy_adjudicate", False)),
            syzygy_adjudicate_fraction=float(config.get("syzygy_adjudicate_fraction", 1.0)),
            syzygy_in_search=bool(config.get("syzygy_in_search", False)),
            categorical_bins=int(config.get("categorical_bins", DEFAULT_CATEGORICAL_BINS)),
            hlgauss_sigma=float(config.get("hlgauss_sigma", 0.04)),
            categorical_blend_frac=float(config.get("categorical_blend_frac", 0.0)),
            categorical_search_blend_frac=float(
                config.get("categorical_search_blend_frac", 0.0)
            ),
            record_lc0_root_input=bool(config.get("record_lc0_root_input", False)),
            history_rep_fix=bool(config.get("history_rep_fix", False)),
            record_dense_sf_policy=bool(config.get("record_dense_sf_policy", True)),
            record_sf_p0_policy=bool(config.get("record_sf_p0_policy", False)),
            record_sf_p0_regret=bool(config.get("record_sf_p0_regret", False)),
            record_fast_ply_value=bool(config.get("record_fast_ply_value", False)),
            blindspot_harvest_out_path=str(config.get("blindspot_harvest_out_path", "")),
            selfplay_resume_inflight_games=bool(
                config.get("selfplay_resume_inflight_games", False)
            ),

  # --- Diff focus ---
            diff_focus_enabled=bool(config.get("diff_focus_enabled", True)),
            diff_focus_q_weight=float(config.get("diff_focus_q_weight", 6.0)),
            diff_focus_pol_scale=float(config.get("diff_focus_pol_scale", 3.5)),
            diff_focus_slope=float(config.get("diff_focus_slope", 3.0)),
            diff_focus_min=float(config.get("diff_focus_min", 0.025)),

  # --- Stockfish ---
            stockfish_path=str(config.get("stockfish_path", "")),
            sf_nodes=int(config.get("sf_nodes", 500)),
            sf_move_nodes=int(config.get("sf_move_nodes", 0)),
            sf_fast_ply_node_scale=float(config.get("sf_fast_ply_node_scale", 0.25)),
            sf_label_nodes_cap=int(config.get("sf_label_nodes_cap", 0)),
            sf_label_nodes_floor=int(config.get("sf_label_nodes_floor", 0)),
            sf_label_escalate_q_gap=float(config.get("sf_label_escalate_q_gap", 0.0)),
            sf_label_escalate_nodes=int(config.get("sf_label_escalate_nodes", 3_000_000)),
            sf_label_escalate_max_per_game=int(
                config.get("sf_label_escalate_max_per_game", 2)
            ),
            sf_workers=int(config.get("sf_workers", 1)),
            sf_multipv=int(config.get("sf_multipv", 1)),
            sf_hash_mb=int(config.get("sf_hash_mb", 16)),
            sf_nice=int(config.get("sf_nice", 0)),
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
            replay_window_max=int(config.get("replay_window_max", 1_000_000)),
            replay_window_growth=int(config.get("replay_window_growth", 10_000)),
            replay_window_growth_frac=_optional_unit_fraction(
                config.get("replay_window_growth_frac"),
                name="replay_window_growth_frac",
            ),
            shuffle_buffer_size=int(config.get("shuffle_buffer_size", 20_000)),
            shard_size=int(config.get("shard_size", 1000)),
            replay_upgrade_v1_planes=bool(config.get("replay_upgrade_v1_planes", False)),
            shuffle_refresh_interval=int(config.get("shuffle_refresh_interval", 5)),
            shuffle_refresh_shards=int(config.get("shuffle_refresh_shards", 3)),
            replay_shard_recency_exponent=float(
                config.get("replay_shard_recency_exponent", 1.0)),
            shuffle_draw_cap_frac=float(config.get("shuffle_draw_cap_frac", 0.90)),
            shuffle_wl_max_ratio=float(config.get("shuffle_wl_max_ratio", 1.5)),
            replay_sf_gap_priority_weight=float(
                config.get("replay_sf_gap_priority_weight", 0.0)),
            # str().lower() coercion so a stringy "off"/"no"/"false" (which
            # bool() would read as True) disables signed mode, while a native
            # YAML bool still works.
            replay_sf_gap_priority_signed=str(
                config.get("replay_sf_gap_priority_signed", False),
            ).strip().lower() in ("1", "true", "yes", "on"),
            replay_fast_low_surprise_priority=float(
                config.get("replay_fast_low_surprise_priority", 1.0)),
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
            era_probe_path=str(config.get("era_probe_path", "") or ""),
            era_probe_inwindow_path=str(
                config.get("era_probe_inwindow_path", "") or ""),
            era_probe_rows=int(config.get("era_probe_rows", 2048)),
            era_probe_interval=int(config.get("era_probe_interval", 1)),
            era_probe_batch_size=int(config.get("era_probe_batch_size", 512)),

  # --- Distributed ---
            distributed_workers_per_trial=max(1, int(config.get("distributed_workers_per_trial", 1))),
            distributed_server_root=_get("distributed_server_root", None),
            distributed_server_url=_get("distributed_server_url", None),
            distributed_wait_timeout_seconds=float(config.get("distributed_wait_timeout_seconds", 900.0)),
            distributed_worker_poll_seconds=float(config.get("distributed_worker_poll_seconds", 1.0)),
            distributed_min_games_fraction=float(config.get("distributed_min_games_fraction", 0.5)),
            distributed_prev_model_max_fraction=float(config.get("distributed_prev_model_max_fraction", 0.33)),
            distributed_stale_pause_target_games=int(config.get("distributed_stale_pause_target_games", -1)),
            distributed_pause_selfplay_during_training=bool(config.get("distributed_pause_selfplay_during_training", False)),
            processed_max_age_seconds=float(config.get("processed_max_age_seconds", 43200.0)),
            distributed_prefetch_shards=bool(config.get("distributed_prefetch_shards", False)),
            distributed_prefetch_max_queued_mb=int(config.get("distributed_prefetch_max_queued_mb", 768)),
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

  # Game counts. "matching" = games whose shard model_sha is accepted for THIS
  # iteration; stale-model games are ingested too but counted separately in
  # distributed_stale_games. matching + stale = everything uploaded.
    matching_games: int = 0
    total_game_plies: int = 0
    total_adjudicated_games: int = 0
    total_tb_adjudicated_games: int = 0
    total_draw_games: int = 0

  # Anchored promotion-gate split of the vs-SF (curriculum) outcomes above, by
  # which published model played them: "cur" = the model published this
  # iteration, "prev" = last iteration's. Both faced the same handicapped
  # Stockfish, so their difference is a free A/B of one training iteration.
    gate_cur_w: int = 0
    gate_cur_d: int = 0
    gate_cur_l: int = 0
    gate_prev_w: int = 0
    gate_prev_d: int = 0
    gate_prev_l: int = 0
  # Games-weighted mean ``wdl_regret`` each arm was actually played at, from
  # the shards' own ``ShardMeta.opponent_wdl_regret_limit``. NaN when the
  # shards predate that field. These are NOT decoration: model and difficulty
  # ship in one manifest, so the prev arm is one PID step behind and the
  # anchored delta carries a controller term whose size is
  # ``(cur - prev) * dWR/dregret``. See promotion_gate's "THE PID LAG DOES NOT
  # CANCEL".
    gate_cur_wdl_regret: float = float("nan")
    gate_prev_wdl_regret: float = float("nan")
  # False on an iteration whose publish crossed a hold boundary: the anchored
  # labels are then inverted (or span many iterations) and the sample must not
  # enter the window. See GateHoldController.sample_is_valid.
    gate_sample_valid: bool = True

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

  # Positions / ingest. matching_positions is CURRENT-MODEL only;
  # replay_positions_ingested is every position that entered the buffer and is
  # the correct denominator for any per-position budget.
    matching_positions: int = 0
    replay_positions_ingested: int = 0
    replay_window_before: int = 0
    replay_window_after: int = 0
    replay_window_growth_positions: int = 0
    replay_window_growth_frac_used: float = 0.0
  # Per-sample is_selfplay tag accounting (this iter's ingested training rows).
    ingest_is_selfplay_tagged: int = 0
    ingest_is_selfplay_true: int = 0
    diff_focus_records: int = 0
    diff_focus_kept: int = 0
    diff_focus_keep_prob_sum: float = 0.0
    diff_focus_keep_limited: int = 0
    diff_focus_sample_weight_sum: float = 0.0
    diff_focus_sample_weight_limited: int = 0
    diff_focus_priority_sum: float = 0.0
    diff_focus_priority_sq_sum: float = 0.0
    diff_focus_priority_min: float = 0.0
    diff_focus_priority_max: float = 0.0
    gumbel_policy_diag_n: int = 0
    gumbel_policy_top_prob_sum: float = 0.0
    gumbel_policy_action_prob_sum: float = 0.0
    gumbel_policy_entropy_sum: float = 0.0
    gumbel_policy_eff_moves_sum: float = 0.0
    gumbel_policy_candidate_mass_sum: float = 0.0
    gumbel_policy_non_candidate_top_prob_sum: float = 0.0
    gumbel_policy_argmax_is_candidate_sum: int = 0
    gumbel_policy_argmax_is_action_sum: int = 0
    gumbel_policy_legal_count_sum: int = 0
    gumbel_policy_candidate_count_sum: int = 0
    replay_priority_n: int = 0
    replay_priority_sum: float = 0.0
    replay_priority_sq_sum: float = 0.0
    replay_priority_min: float = 0.0
    replay_priority_max: float = 0.0
    replay_has_policy_n: int = 0
    replay_has_policy_sum: int = 0
    replay_has_sf_wdl_n: int = 0
    replay_has_sf_wdl_sum: int = 0
    replay_has_search_wdl_n: int = 0
    replay_has_search_wdl_sum: int = 0
    replay_wdl_0: int = 0
    replay_wdl_1: int = 0
    replay_wdl_2: int = 0
    outcome_stats: dict[str, int] = field(default_factory=dict)

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
  # NOT a bool. The reported metric must distinguish "the gate did not run"
  # from "the gate passed"; the old bool defaulted to True and emitted
  # ``gate_passed: 1`` for 200+ iterations with zero games played.
    gate_decision: GateDecision = field(default_factory=GateDecision)
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
  # d(winrate)/d(wdl_regret) from the PID's most recent inverse fit. Carried
  # here because the promotion gate needs it to convert the difficulty gap
  # between its two arms into Elo, and ``ds`` is already threaded to the gating
  # phase -- a second loose local in ``train_trial`` is what this dataclass
  # exists to avoid. NaN whenever the PID had no usable fit (deadband, airbag,
  # or fewer than 3 history points), which is honest: no fit means no
  # prediction, not a prediction of zero.
    regret_fit_slope: float = float("nan")

    @classmethod
    def from_pid(cls, pid: Any, sf: Any, tc: TrialConfig) -> DifficultyState:
        """Build the per-iteration difficulty snapshot.

        PID is the source of truth when present; sf is a fallback for gate-only
        configurations where no PID exists. After PID restore, the caller is
        expected to sync ``sf.set_nodes(pid.nodes)`` so sf.nodes == pid.nodes;
        this method still prefers pid.nodes to make divergence impossible.
        """
        if pid is not None:
            slope = getattr(pid, "last_regret_fit_slope", None)
            return cls(
                wdl_regret=float(pid.wdl_regret),
                sf_nodes=int(pid.nodes),
                regret_fit_slope=(
                    float(slope) if slope is not None else float("nan")
                ),
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
  # None means "this restore carries no opponent-strength EMA", which is NOT
  # the same as 0.0 and is why the field is optional. The loop treats 0.0 as
  # "no EMA yet" and re-seeds from the instantaneous opponent_strength of the
  # first post-restart iteration -- the single row the post-restart winrate
  # truncation makes least trustworthy -- so a restore that unconditionally
  # assigned its default wiped the restored value every restart (audit T1,
  # 6/6 live process segments). Only a restore that actually found one (the
  # checkpoint's trial_meta.json, or a PB2 donor's result row) sets it.
    opp_strength_ema: float | None = None
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
  # Where the holdout sidecar was restored from (a Ray checkpoint dir or a
  # salvage seed slot), plus the two scalars that ride in its trial_meta.json.
  # None means "nothing to restore from" -- a fresh start, which is the only
  # case where an empty holdout is the truth rather than a loss.
    holdout_state_dir: Any = None
    holdout_frozen: bool = False
    holdout_generation: int = 0
  # Identity of the MEASUREMENT the restored generation was earned under, so a
  # ruler change across the restart bumps the generation instead of hiding
  # inside it. "" on any checkpoint written before this was recorded.
    holdout_ruler: str = ""
  # The flat yaml key set the process that wrote the restored checkpoint was
  # running, banked in its trial_meta.json. Empty for a fresh start and for any
  # checkpoint written before it was banked; the startup check treats empty as
  # "no baseline" and stays silent rather than reporting every key as deleted.
    restored_yaml_keys: frozenset[str] = frozenset()
  # The promotion gate's `gate_state.json` as found in a salvage seed slot, for
  # a trial whose own durable dir has none. None on every other path -- an
  # ordinary `--resume` reads the trial's own file and never consults this
  # (audit T10: a salvage restart used to start with an empty gate window while
  # a resume kept it, with nothing saying the two differed).
    restored_gate_state: dict | None = None
