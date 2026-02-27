from chess_anti_engine.selfplay.temperature import temperature_for_ply


def test_temperature_for_ply_no_drop():
    assert temperature_for_ply(ply=1, temperature=1.25, drop_plies=0, after=0.0) == 1.25
    assert temperature_for_ply(ply=50, temperature=1.25, drop_plies=-1, after=0.0) == 1.25


def test_temperature_for_ply_drop_boundary_step():
    assert temperature_for_ply(ply=1, temperature=1.25, drop_plies=4, after=0.0) == 1.25
    assert temperature_for_ply(ply=3, temperature=1.25, drop_plies=4, after=0.0) == 1.25
    assert temperature_for_ply(ply=4, temperature=1.25, drop_plies=4, after=0.0) == 0.0
    assert temperature_for_ply(ply=10, temperature=1.25, drop_plies=4, after=0.0) == 0.0


def test_temperature_for_ply_linear_decay_precedence():
    # Linear decay takes precedence over step schedule when enabled.
    t = temperature_for_ply(
        ply=20,
        temperature=1.0,
        drop_plies=5,
        after=0.0,
        decay_start_move=20,
        decay_moves=60,
        endgame=0.6,
    )
    assert abs(t - 1.0) < 1e-9


def test_temperature_for_ply_linear_decay_midpoint():
    # Halfway through the decay: 1.0 -> 0.6 over 60 moves
    t = temperature_for_ply(
        ply=50,  # start=20, so 30/60 = 0.5
        temperature=1.0,
        drop_plies=0,
        after=0.0,
        decay_start_move=20,
        decay_moves=60,
        endgame=0.6,
    )
    assert abs(t - 0.8) < 1e-9


def test_temperature_for_ply_linear_decay_clamp_endgame():
    t = temperature_for_ply(
        ply=100,
        temperature=1.0,
        drop_plies=0,
        after=0.0,
        decay_start_move=20,
        decay_moves=60,
        endgame=0.6,
    )
    assert abs(t - 0.6) < 1e-9


def test_temperature_for_ply_nonzero_after_step_only():
    assert temperature_for_ply(ply=5, temperature=1.25, drop_plies=2, after=0.5) == 0.5
