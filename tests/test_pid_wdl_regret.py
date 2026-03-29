from __future__ import annotations

from chess_anti_engine.stockfish.pid import DifficultyPID


def test_pid_negative_regret_stage_end_keeps_regret_adaptive_but_ungated() -> None:
    pid = DifficultyPID(
        initial_nodes=100,
        min_nodes=50,
        max_nodes=1000,
        min_games_between_adjust=1,
        target_winrate=0.52,
        ema_alpha=1.0,
        deadzone=0.0,
        rate_limit=0.10,
        initial_skill_level=0,
        skill_min=0,
        skill_max=20,
        initial_random_move_prob=0.0,
        random_move_stage_end=0.0,
        initial_wdl_regret=0.50,
        wdl_regret_min=0.01,
        wdl_regret_max=1.0,
        wdl_regret_stage_end=-1.0,
        max_regret_step=0.01,
    )

    upd = pid.observe(wins=10, draws=0, losses=0)

    assert upd.wdl_regret_changed is True
    assert upd.wdl_regret_after < upd.wdl_regret_before
    assert upd.nodes_after > upd.nodes_before
