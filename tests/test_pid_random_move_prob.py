from chess_anti_engine.stockfish.pid import DifficultyPID


def test_pid_random_move_prob_random_first_gates_nodes() -> None:
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
        initial_random_move_prob=0.95,
        random_move_prob_min=0.0,
        random_move_prob_max=1.0,
        random_move_stage_end=0.50,
    )

    # Network is too strong (wins all games) -> PID should make opponent stronger:
    # decrease random_move_prob. Nodes should stay fixed until random_move_prob <= stage_end.
    upd = pid.observe(wins=10, draws=0, losses=0)
    assert upd.random_move_prob_after < upd.random_move_prob_before
    assert upd.nodes_after == upd.nodes_before == 100

    # Force stage complete and check nodes update starts applying.
    pid.random_move_prob = 0.40
    upd2 = pid.observe(wins=10, draws=0, losses=0)
    assert upd2.nodes_after >= upd2.nodes_before


def test_pid_random_move_prob_can_increase_when_network_loses() -> None:
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
        initial_random_move_prob=0.50,
        random_move_prob_min=0.0,
        random_move_prob_max=1.0,
        random_move_stage_end=0.50,
    )

    upd = pid.observe(wins=0, draws=0, losses=10)
    assert upd.random_move_prob_after > upd.random_move_prob_before


def test_pid_random_stage_has_hysteresis_before_refreezing_nodes() -> None:
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
        initial_random_move_prob=0.50,
        random_move_prob_min=0.0,
        random_move_prob_max=1.0,
        random_move_stage_end=0.50,
        random_move_stage_reenter=0.70,
    )

    # Stage starts complete at boundary, so nodes can move immediately.
    first = pid.observe(wins=10, draws=0, losses=0)
    assert first.nodes_after > first.nodes_before

    # A small regression above stage_end should not immediately re-freeze nodes.
    pid.random_move_prob = 0.60
    second = pid.observe(wins=0, draws=0, losses=10)
    assert second.random_move_prob_after > second.random_move_prob_before
    assert second.nodes_after < second.nodes_before


def test_pid_random_stage_can_reenter_on_large_regression() -> None:
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
        initial_random_move_prob=0.50,
        random_move_prob_min=0.0,
        random_move_prob_max=1.0,
        random_move_stage_end=0.50,
        random_move_stage_reenter=0.70,
    )

    # Push to a strong regression that should re-enter random-first gating.
    pid.random_move_prob = 0.75
    upd = pid.observe(wins=0, draws=0, losses=10)
    assert upd.random_move_prob_after > upd.random_move_prob_before
    assert upd.nodes_after == upd.nodes_before
