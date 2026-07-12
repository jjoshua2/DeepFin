from __future__ import annotations

import pytest

from chess_anti_engine.tune.trainable_phases import (
    _compute_replay_window_update,
    _selfplay_diagnostic_fields_from_ingest,
)
from chess_anti_engine.tune.trial_config import TrialConfig


def test_from_empty_dict() -> None:
    """All defaults should work with an empty config dict."""
    tc = TrialConfig.from_dict({})
    assert tc.model == "transformer"
    assert tc.categorical_blend_frac == 0.0  # off by default
    assert tc.embed_dim == 256
    assert tc.input_extra_features == "v2_threats"
    assert tc.lr == 0.0003
    assert tc.optimizer == "nadamw"
    assert tc.sf_nodes == 500
    assert tc.sf_move_nodes == 0
    assert tc.slot_oversubscribe == 1.0
    assert tc.mcts_simulations == 50
    assert tc.gumbel_topk == 16
    assert tc.gumbel_c_scale == 0.1
    assert tc.gumbel_scale == 1.0
    assert tc.gumbel_scale_after == 0.0
    assert tc.gumbel_scale_decay_start_move == 0
    assert tc.gumbel_scale_decay_moves == 0
    assert tc.curriculum_gumbel_scale == 0.0
    assert tc.curriculum_gumbel_scale_after == 0.0
    assert tc.curriculum_gumbel_scale_decay_start_move == 0
    assert tc.curriculum_gumbel_scale_decay_moves == 0
    assert tc.sf_pid_enabled is True
    assert tc.sf_pid_wdl_regret_max == 1.0
    assert tc.diff_focus_q_weight == 6.0
    assert tc.exploit_replay_skip_newest == 0
    assert tc.exploit_replay_max_unseen_iters_per_source == 2
    assert tc.stockfish_path == ""
    assert tc.pause_file is None
    assert tc.opening_book_path is None
    assert tc.syzygy_adjudicate is False
    assert tc.syzygy_adjudicate_fraction == 1.0
    assert tc.syzygy_in_search is False
    assert tc.stockfish_syzygy_path is None


def test_from_dict_overrides() -> None:
    """Config values should override defaults."""
    tc = TrialConfig.from_dict({
        "lr": 0.001,
        "sf_nodes": 5000,
        "sf_move_nodes": 10000,
        "mcts_simulations": 100,
        "gumbel_topk": 24,
        "gumbel_c_scale": 0.2,
        "gumbel_scale": 0.5,
        "gumbel_scale_after": 0.1,
        "gumbel_scale_decay_start_move": 15,
        "gumbel_scale_decay_moves": 15,
        "curriculum_gumbel_scale": 0.25,
        "curriculum_gumbel_scale_after": 0.05,
        "curriculum_gumbel_scale_decay_start_move": 10,
        "curriculum_gumbel_scale_decay_moves": 20,
        "stockfish_path": "/usr/bin/stockfish",
        "stockfish_syzygy_path": "/ssd/tb",
        "sf_pid_ema_alpha": 0.50,
        "batch_size": 256,
        "syzygy_path": "/tb",
        "syzygy_adjudicate": True,
        "syzygy_adjudicate_fraction": 0.25,
        "syzygy_in_search": True,
        "categorical_blend_frac": 0.5,
        "slot_oversubscribe": 2.0,
    })
    assert tc.lr == 0.001
    assert tc.sf_nodes == 5000
    assert tc.sf_move_nodes == 10000
    assert tc.slot_oversubscribe == 2.0
    assert tc.mcts_simulations == 100
    assert tc.gumbel_topk == 24
    assert tc.gumbel_c_scale == 0.2
    assert tc.gumbel_scale == 0.5
    assert tc.gumbel_scale_after == 0.1
    assert tc.gumbel_scale_decay_start_move == 15
    assert tc.gumbel_scale_decay_moves == 15
    assert tc.curriculum_gumbel_scale == 0.25
    assert tc.curriculum_gumbel_scale_after == 0.05
    assert tc.curriculum_gumbel_scale_decay_start_move == 10
    assert tc.curriculum_gumbel_scale_decay_moves == 20
    assert tc.stockfish_path == "/usr/bin/stockfish"
    assert tc.stockfish_syzygy_path == "/ssd/tb"
    assert tc.sf_pid_ema_alpha == 0.50
    assert tc.batch_size == 256
    assert tc.syzygy_path == "/tb"
    assert tc.syzygy_adjudicate is True
    assert tc.syzygy_adjudicate_fraction == 0.25
    assert tc.syzygy_in_search is True
    assert tc.categorical_blend_frac == 0.5
    # Unset values still get defaults
    assert tc.model == "transformer"
    assert tc.sf_pid_enabled is True


def test_model_architecture_keys_are_parsed() -> None:
    tc = TrialConfig.from_dict({
        "num_layers": 3,
        "embed_dim_by_layer": [256, 384, 320],
        "ffn_mult_by_layer": [1.0, 1.25, 1.5],
        "policy_encoding": "lc0_1858",
        "input_history_encoding": "lc0_root_legacy_meta",
        "input_pos_encoding": "arc_adapter",
        "qkv_projection": "split",
        "use_deepnorm": True,
        "input_global_embedding": "bt4_board_adapter",
        "input_global_embedding_channels": 24,
        "input_square_embedding": "add",
        "smolgen_mode": "per_layer",
        "smolgen_pooling": "mean",
        "smolgen_hidden_channels": 16,
        "smolgen_hidden_sz": 128,
        "smolgen_gen_sz": 192,
        "smolgen_bias_scale": "layer_head",
        "smolgen_bias_norm": "center_rms",
        "arc_attention_bias": "basic",
        "smolgen_relation_basis": True,
        "smolgen_relation_norm": "branch_center_rms",
        "smolgen_relation_coeff_norm": "rms",
        "smolgen_relation_scale": "layer",
        "phase_output_adapter": True,
        "phase_output_adapter_dim": 48,
        "phase_smolgen": True,
        "phase_piece_thresholds": [12, 21],
    })
    assert tc.num_layers == 3
    assert tc.embed_dim_by_layer == (256, 384, 320)
    assert tc.ffn_mult_by_layer == (1.0, 1.25, 1.5)
    assert tc.policy_encoding == "lc0_1858"
    assert tc.input_history_encoding == "lc0_root_legacy_meta"
    assert tc.input_pos_encoding == "arc_adapter"
    assert tc.qkv_projection == "split"
    assert tc.use_deepnorm is True
    assert tc.input_global_embedding == "bt4_board_adapter"
    assert tc.input_global_embedding_channels == 24
    assert tc.input_square_embedding == "add"
    assert tc.smolgen_mode == "per_layer"
    assert tc.smolgen_pooling == "mean"
    assert tc.smolgen_hidden_channels == 16
    assert tc.smolgen_hidden_sz == 128
    assert tc.smolgen_gen_sz == 192
    assert tc.smolgen_bias_scale == "layer_head"
    assert tc.smolgen_bias_norm == "center_rms"
    assert tc.arc_attention_bias == "basic"
    assert tc.smolgen_relation_basis is True
    assert tc.smolgen_relation_norm == "branch_center_rms"
    assert tc.smolgen_relation_coeff_norm == "rms"
    assert tc.smolgen_relation_scale == "layer"
    assert tc.phase_output_adapter is True
    assert tc.phase_output_adapter_dim == 48
    assert tc.phase_smolgen is True
    assert tc.phase_piece_thresholds == (12, 21)


def test_model_architecture_rejects_mismatched_ffn_schedule() -> None:
    with pytest.raises(ValueError, match="length"):
        TrialConfig.from_dict({"num_layers": 3, "ffn_mult_by_layer": [1.0, 1.25]})


def test_model_architecture_rejects_mismatched_embed_schedule() -> None:
    with pytest.raises(ValueError, match="length"):
        TrialConfig.from_dict({"num_layers": 3, "embed_dim_by_layer": [256, 384]})


def test_fallback_keys() -> None:
    """Keys with fallbacks (eval_sf_nodes -> sf_nodes) should chain correctly."""
    # When neither is set, get default sf_nodes
    tc = TrialConfig.from_dict({})
    assert tc.eval_sf_nodes == 500

    # When sf_nodes is set but eval_sf_nodes is not, inherit sf_nodes
    tc = TrialConfig.from_dict({"sf_nodes": 5000})
    assert tc.eval_sf_nodes == 5000

    # When eval_sf_nodes is explicitly set, use it
    tc = TrialConfig.from_dict({"sf_nodes": 5000, "eval_sf_nodes": 1000})
    assert tc.eval_sf_nodes == 1000

    # Explicit None for eval overrides should fall back to parent key
    tc = TrialConfig.from_dict({"sf_nodes": 3000, "eval_sf_nodes": None})
    assert tc.eval_sf_nodes == 3000

    # Same for eval_mcts_simulations
    tc = TrialConfig.from_dict({"mcts_simulations": 100})
    assert tc.eval_mcts_simulations == 100

    tc = TrialConfig.from_dict({"mcts_simulations": 100, "eval_mcts_simulations": None})
    assert tc.eval_mcts_simulations == 100

    # Same for eval_max_plies
    tc = TrialConfig.from_dict({"max_plies": 300})
    assert tc.eval_max_plies == 300

    tc = TrialConfig.from_dict({"max_plies": 300, "eval_max_plies": None})
    assert tc.eval_max_plies == 300

    tc = TrialConfig.from_dict({"replay_window_growth_frac": 0.75})
    assert tc.replay_window_growth_frac == 0.75


@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf"), float("nan")])
def test_replay_window_growth_frac_rejects_out_of_range_values(value: float) -> None:
    with pytest.raises(ValueError, match="replay_window_growth_frac"):
        TrialConfig.from_dict({"replay_window_growth_frac": value})


@pytest.mark.parametrize("value", [-0.01, float("inf"), float("nan")])
def test_gumbel_c_scale_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="gumbel_c_scale"):
        TrialConfig.from_dict({"gumbel_c_scale": value})


def test_replay_window_growth_frac_controls_runtime_growth() -> None:
    after, growth, frac = _compute_replay_window_update(
        current_window=1_000,
        replay_window_max=2_000,
        fixed_growth=500,
        growth_frac=0.10,
        positions_ingested=333,
    )

    assert after == 1_034
    assert growth == 34
    assert frac == 0.10


def test_replay_window_fixed_growth_remains_fallback() -> None:
    after, growth, frac = _compute_replay_window_update(
        current_window=1_000,
        replay_window_max=1_100,
        fixed_growth=500,
        growth_frac=None,
        positions_ingested=333,
    )

    assert after == 1_100
    assert growth == 100
    assert frac == 0.0


def test_replay_window_live_max_shrink_caps_window() -> None:
    after, growth, frac = _compute_replay_window_update(
        current_window=2_000,
        replay_window_max=1_500,
        fixed_growth=500,
        growth_frac=0.25,
        positions_ingested=333,
    )

    assert after == 1_500
    assert growth == 0
    assert frac == 0.25


def test_selfplay_diagnostic_fields_are_copied_from_ingest_summary() -> None:
    fields = _selfplay_diagnostic_fields_from_ingest({
        "matching_diff_focus_records": 5,
        "matching_diff_focus_kept": 3,
        "matching_diff_focus_keep_prob_sum": 2.5,
        "matching_diff_focus_keep_limited": 1,
        "matching_diff_focus_sample_weight_sum": 4.5,
        "matching_diff_focus_sample_weight_limited": 2,
        "matching_diff_focus_priority_sum": 10.0,
        "matching_diff_focus_priority_sq_sum": 25.0,
        "matching_diff_focus_priority_min": 0.5,
        "matching_diff_focus_priority_max": 3.5,
        "matching_gumbel_policy_diag_n": 7,
        "matching_gumbel_policy_top_prob_sum": 1.25,
        "matching_gumbel_policy_action_prob_sum": 1.0,
        "matching_gumbel_policy_entropy_sum": 8.0,
        "matching_gumbel_policy_eff_moves_sum": 12.0,
        "matching_gumbel_policy_candidate_mass_sum": 6.5,
        "matching_gumbel_policy_non_candidate_top_prob_sum": 0.75,
        "matching_gumbel_policy_argmax_is_candidate_sum": 6,
        "matching_gumbel_policy_argmax_is_action_sum": 4,
        "matching_gumbel_policy_legal_count_sum": 210,
        "matching_gumbel_policy_candidate_count_sum": 70,
        "matching_outcome_stats": {"book_uho": 4},
        "replay_priority_n": 11,
        "replay_priority_sum": 13.0,
        "replay_priority_sq_sum": 17.0,
        "replay_priority_min": 0.25,
        "replay_priority_max": 2.0,
        "replay_has_policy_n": 11,
        "replay_has_policy_sum": 10,
        "replay_has_sf_wdl_n": 9,
        "replay_has_sf_wdl_sum": 8,
        "replay_has_search_wdl_n": 7,
        "replay_has_search_wdl_sum": 6,
        "replay_wdl_0": 3,
        "replay_wdl_1": 5,
        "replay_wdl_2": 2,
    })

    assert fields["diff_focus_records"] == 5
    assert fields["diff_focus_priority_max"] == 3.5
    assert fields["gumbel_policy_diag_n"] == 7
    assert fields["gumbel_policy_legal_count_sum"] == 210
    assert fields["outcome_stats"] == {"book_uho": 4}
    assert fields["replay_priority_n"] == 11
    assert fields["replay_has_sf_wdl_sum"] == 8
    assert fields["replay_wdl_1"] == 5


@pytest.mark.parametrize("value", [-0.01, float("inf"), float("nan")])
def test_gumbel_scale_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="gumbel_scale"):
        TrialConfig.from_dict({"gumbel_scale": value})


@pytest.mark.parametrize("value", [-0.01, float("inf"), float("nan")])
def test_gumbel_scale_after_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="gumbel_scale_after"):
        TrialConfig.from_dict({"gumbel_scale_after": value})


@pytest.mark.parametrize("value", [-0.01, float("inf"), float("nan")])
def test_curriculum_gumbel_scale_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="curriculum_gumbel_scale"):
        TrialConfig.from_dict({"curriculum_gumbel_scale": value})


@pytest.mark.parametrize("value", [-0.01, float("inf"), float("nan")])
def test_curriculum_gumbel_scale_after_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="curriculum_gumbel_scale_after"):
        TrialConfig.from_dict({"curriculum_gumbel_scale_after": value})


@pytest.mark.parametrize("value", [0.99, float("inf"), float("nan")])
def test_slot_oversubscribe_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="slot_oversubscribe"):
        TrialConfig.from_dict({"slot_oversubscribe": value})


def test_games_per_iter_start_fallback() -> None:
    """games_per_iter_start falls back to games_per_iter."""
    tc = TrialConfig.from_dict({"games_per_iter": 200})
    assert tc.games_per_iter_start == 200

    tc = TrialConfig.from_dict({"games_per_iter": 200, "games_per_iter_start": 32})
    assert tc.games_per_iter_start == 32


def test_from_real_config() -> None:
    """Load from a real YAML config to verify no crashes."""
    from pathlib import Path

    from chess_anti_engine.utils.config_yaml import (
        flatten_run_config_defaults,
        load_yaml_file,
    )

    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    if not config_path.exists():
        return  # skip if config not available

    raw = load_yaml_file(config_path)
    flat = flatten_run_config_defaults(raw)
    tc = TrialConfig.from_dict(flat)

    assert tc.model == "transformer"
    assert tc.sf_pid_enabled is True
    assert tc.embed_dim == 768  # default.yaml uses BT3 scale


def test_retired_replay_capacity_key_is_rejected() -> None:
    """replay_window_max is the sole replay-window ceiling knob."""
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    with pytest.raises(ValueError, match="replay_capacity"):
        flatten_run_config_defaults({"replay_capacity": 500_000})
