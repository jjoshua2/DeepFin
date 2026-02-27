from chess_anti_engine.selfplay.budget import progressive_mcts_simulations


def test_progressive_mcts_simulations_monotone_and_clamped():
    start = 50
    max_sims = 800
    ramp = 10_000

    assert progressive_mcts_simulations(0, start=start, max_sims=max_sims, ramp_steps=ramp) == start
    assert progressive_mcts_simulations(ramp, start=start, max_sims=max_sims, ramp_steps=ramp) == max_sims
    assert progressive_mcts_simulations(ramp * 10, start=start, max_sims=max_sims, ramp_steps=ramp) == max_sims

    # monotone non-decreasing over a small sample
    vals = [progressive_mcts_simulations(s, start=start, max_sims=max_sims, ramp_steps=ramp) for s in [0, 1, 10, 100, 1000, 5000, 9999, 10_000]]
    assert vals == sorted(vals)


def test_progressive_mcts_simulations_handles_bad_params():
    # ramp_steps<=0 -> max
    assert progressive_mcts_simulations(0, start=50, max_sims=800, ramp_steps=0) == 800

    # exponent<=0 -> linear-ish, but still clamped
    v = progressive_mcts_simulations(5000, start=50, max_sims=800, ramp_steps=10_000, exponent=0.0)
    assert 50 <= v <= 800
