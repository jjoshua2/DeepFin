from chess_anti_engine.worker import tune_games_per_batch


def test_tune_games_per_batch_no_elapsed():
    assert tune_games_per_batch(current=8, elapsed_s=0.0, target_s=30.0, min_games=1, max_games=64) == 8


def test_tune_games_per_batch_increase_for_fast_client():
    # If we finished in 10s but target is 30s, we should scale up (clamped by max_change_factor=1.5)
    assert tune_games_per_batch(current=8, elapsed_s=10.0, target_s=30.0, min_games=1, max_games=64) == 12


def test_tune_games_per_batch_decrease_for_slow_client():
    # If we finished in 60s but target is 30s, scale down by 0.5 (clamped within 1/1.5 .. 1.5)
    assert tune_games_per_batch(current=12, elapsed_s=60.0, target_s=30.0, min_games=1, max_games=64) == 8


def test_tune_games_per_batch_respects_bounds():
    assert tune_games_per_batch(current=1, elapsed_s=1.0, target_s=1000.0, min_games=1, max_games=2) == 2
    assert tune_games_per_batch(current=2, elapsed_s=1000.0, target_s=1.0, min_games=1, max_games=2) == 1
