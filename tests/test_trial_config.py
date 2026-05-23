from __future__ import annotations

import pytest

from chess_anti_engine.tune.trial_config import TrialConfig


def test_from_empty_dict() -> None:
    """All defaults should work with an empty config dict."""
    tc = TrialConfig.from_dict({})
    assert tc.model == "transformer"
    assert tc.embed_dim == 256
    assert tc.lr == 0.0003
    assert tc.optimizer == "nadamw"
    assert tc.sf_nodes == 500
    assert tc.sf_move_nodes == 0
    assert tc.mcts_simulations == 50
    assert tc.gumbel_topk == 16
    assert tc.gumbel_scale == 1.0
    assert tc.gumbel_scale_after == 0.0
    assert tc.gumbel_scale_decay_start_move == 0
    assert tc.gumbel_scale_decay_moves == 0
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
        "gumbel_scale": 0.5,
        "gumbel_scale_after": 0.1,
        "gumbel_scale_decay_start_move": 15,
        "gumbel_scale_decay_moves": 15,
        "stockfish_path": "/usr/bin/stockfish",
        "stockfish_syzygy_path": "/ssd/tb",
        "sf_pid_ema_alpha": 0.50,
        "batch_size": 256,
        "syzygy_path": "/tb",
        "syzygy_adjudicate": True,
        "syzygy_adjudicate_fraction": 0.25,
        "syzygy_in_search": True,
    })
    assert tc.lr == 0.001
    assert tc.sf_nodes == 5000
    assert tc.sf_move_nodes == 10000
    assert tc.mcts_simulations == 100
    assert tc.gumbel_topk == 24
    assert tc.gumbel_scale == 0.5
    assert tc.gumbel_scale_after == 0.1
    assert tc.gumbel_scale_decay_start_move == 15
    assert tc.gumbel_scale_decay_moves == 15
    assert tc.stockfish_path == "/usr/bin/stockfish"
    assert tc.stockfish_syzygy_path == "/ssd/tb"
    assert tc.sf_pid_ema_alpha == 0.50
    assert tc.batch_size == 256
    assert tc.syzygy_path == "/tb"
    assert tc.syzygy_adjudicate is True
    assert tc.syzygy_adjudicate_fraction == 0.25
    assert tc.syzygy_in_search is True
    # Unset values still get defaults
    assert tc.model == "transformer"
    assert tc.sf_pid_enabled is True


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

    # Same for replay_window_max -> replay_capacity
    tc = TrialConfig.from_dict({"replay_capacity": 500_000})
    assert tc.replay_window_max == 500_000

    tc = TrialConfig.from_dict({"replay_window_growth_frac": 0.75})
    assert tc.replay_window_growth_frac == 0.75


@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf"), float("nan")])
def test_replay_window_growth_frac_rejects_out_of_range_values(value: float) -> None:
    with pytest.raises(ValueError, match="replay_window_growth_frac"):
        TrialConfig.from_dict({"replay_window_growth_frac": value})


@pytest.mark.parametrize("value", [-0.01, float("inf"), float("nan")])
def test_gumbel_scale_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="gumbel_scale"):
        TrialConfig.from_dict({"gumbel_scale": value})


@pytest.mark.parametrize("value", [-0.01, float("inf"), float("nan")])
def test_gumbel_scale_after_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="gumbel_scale_after"):
        TrialConfig.from_dict({"gumbel_scale_after": value})


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
