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


def test_temperature_bucket_rounding_stays_tight_for_default_decay():
    """Manager buckets by round(temp, 2); adjacent bucket splits should stay small."""
    params = dict(
        temperature=1.0,
        drop_plies=0,
        after=0.0,
        decay_start_move=20,
        decay_moves=60,
        endgame=0.6,
    )
    vals = [(ply, temperature_for_ply(ply=ply, **params)) for ply in range(1, 121)]

    # Equal temperatures must always land in the same bucket.
    for i in range(1, len(vals)):
        prev_ply, prev_t = vals[i - 1]
        ply, t = vals[i]
        if prev_t == t:
            assert round(prev_t, 2) == round(t, 2), (prev_ply, ply, prev_t, t)

    # For the current default linear decay, when adjacent plies do fall into
    # different 2-decimal buckets, the actual temperature drift should still be tiny.
    max_rel = 0.0
    for i in range(1, len(vals)):
        prev_ply, prev_t = vals[i - 1]
        ply, t = vals[i]
        if round(prev_t, 2) != round(t, 2):
            abs_diff = abs(prev_t - t)
            rel_diff = abs_diff / max(prev_t, t)
            max_rel = max(max_rel, rel_diff)
            assert abs_diff <= 0.01, (prev_ply, ply, prev_t, t, abs_diff)
            assert rel_diff <= 0.02, (prev_ply, ply, prev_t, t, rel_diff)

    assert max_rel > 0.0
