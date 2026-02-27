from __future__ import annotations

from chess_anti_engine.stockfish.pid import DifficultyPID


def test_pid_respects_min_games_between_adjust_and_rate_limit():
    pid = DifficultyPID(
        initial_nodes=1000,
        target_winrate=0.52,
        ema_alpha=1.0,  # make EMA equal to the latest batch winrate
        deadzone=0.0,
        rate_limit=0.10,
        min_games_between_adjust=20,
        kp=100.0,  # saturate controller so rate limiter dominates
        ki=0.0,
        kd=0.0,
        min_nodes=1000,
        max_nodes=1000000,
        # Disable skill ladder for this test (testing node-only behavior)
        skill_promote_nodes=10_000_000,
        skill_demote_nodes=0,
    )

    u1 = pid.observe(wins=10, draws=0, losses=0)
    assert u1.adjusted is False
    assert u1.nodes_after == 1000

    # Now reach the minimum adjustment period
    u2 = pid.observe(wins=10, draws=0, losses=0)
    assert u2.adjusted is True
    assert u2.nodes_before == 1000
    assert u2.nodes_after == 1100  # +10% cap

    # If the controller wants to decrease below min_nodes, it must clamp.
    u3 = pid.observe(wins=0, draws=0, losses=20)
    assert u3.adjusted is True
    assert u3.nodes_before == 1100
    assert u3.nodes_after == 1000  # clamped to min_nodes


def test_pid_deadzone_prevents_adjustment():
    pid = DifficultyPID(
        initial_nodes=2000,
        target_winrate=0.52,
        ema_alpha=1.0,
        deadzone=0.05,
        rate_limit=0.10,
        min_games_between_adjust=20,
        kp=100.0,
        ki=0.0,
        kd=0.0,
        # Disable skill ladder for this test
        skill_promote_nodes=10_000_000,
        skill_demote_nodes=0,
    )

    # winrate = 11/20 = 0.55 -> within [0.47, 0.57]
    u = pid.observe(wins=11, draws=0, losses=9)
    assert u.adjusted is False
    assert u.nodes_after == 2000
