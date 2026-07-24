"""Replay-reuse denominator: the views budget must divide by positions
ACTUALLY INGESTED, not by matching (current-model) positions.

Until 2026-07-24 it used matching_positions while 4.5-6.5x more positions
entered the buffer, so the live run sat at 0.46 true views while the config
read 2.5 -- over half of all ingested data was never trained on once.
"""

from __future__ import annotations

import pytest

from chess_anti_engine.tune.trainable_metrics import _compute_train_step_budget
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

# Live 2026-07-24 numbers.
MATCHING = 4_577
INGESTED = 25_693
BATCH = 512


def _budget(positions: int, views: float) -> dict[str, int]:
    return _compute_train_step_budget(
        positions_added=positions,
        imported_samples=0,
        replay_size=1_500_000,
        batch_size=BATCH,
        accum_steps=1,
        base_max_steps=800,
        train_window_fraction=0.04,
        train_views_per_ingested_position=views,
    )


def test_matching_denominator_reproduces_the_bug() -> None:
    """The old denominator yields ~0.46 true views at a configured 2.5."""
    steps = _budget(MATCHING, 2.5)["steps"]
    true_views = steps * BATCH / INGESTED
    assert true_views < 0.6, f"expected the historical ~0.46, got {true_views:.2f}"


def test_ingested_denominator_delivers_the_configured_ratio() -> None:
    """With the fix, configured 2.5 actually means ~2.5 views."""
    steps = _budget(INGESTED, 2.5)["steps"]
    true_views = steps * BATCH / INGESTED
    assert 2.4 <= true_views <= 2.6, f"got {true_views:.2f}"


@pytest.mark.parametrize("views", [1.0, 2.5, 4.0])
def test_true_views_tracks_config_across_the_range(views: float) -> None:
    steps = _budget(INGESTED, views)["steps"]
    assert abs(steps * BATCH / INGESTED - views) < 0.1


def test_reuse_is_invariant_to_the_stale_ratio() -> None:
    """The whole point of views mode. Under the old denominator the drifting
    stale ratio (4.5-6.5x) moved true reuse; under the new one it cannot."""
    seen = set()
    for ingested in (25_693, 26_854, 27_509, 29_912):
        steps = _budget(ingested, 2.5)["steps"]
        seen.add(round(steps * BATCH / ingested, 1))
    assert seen == {2.5}, f"true views drifted across ingest volumes: {seen}"


def test_old_key_is_a_hard_error_not_a_silent_alias() -> None:
    cfg = {"train": {"train_views_per_position": 14.0}}
    with pytest.raises(ValueError, match="RENAMED"):
        flatten_run_config_defaults(cfg)


def test_old_key_error_tells_you_the_conversion() -> None:
    cfg = {"train": {"train_views_per_position": 14.0}}
    with pytest.raises(ValueError, match="DIVIDE IT BY") as exc:
        flatten_run_config_defaults(cfg)
    msg = str(exc.value)
    assert "5.6" in msg, "the error must state the conversion factor"
    assert "train_views_per_ingested_position" in msg


def test_new_key_is_accepted() -> None:
    cfg = {"train": {"train_views_per_ingested_position": 2.5}}
    out = flatten_run_config_defaults(cfg)
    assert out["train_views_per_ingested_position"] == 2.5


def test_views_mode_off_falls_back_to_window_fraction() -> None:
    """0 disables views mode; the window-fraction path must still work."""
    steps = _budget(INGESTED, 0.0)["steps"]
    assert steps == _compute_train_step_budget(
        positions_added=INGESTED, imported_samples=0, replay_size=1_500_000,
        batch_size=BATCH, accum_steps=1, base_max_steps=800,
        train_window_fraction=0.04, train_views_per_ingested_position=0.0,
    )["steps"]
    assert steps > 0
