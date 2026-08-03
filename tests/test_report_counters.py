"""Report-dict pins for the per-iteration selfplay counters.

These counters are the only window onto "how much selfplay did we pay for",
and a wrong one is invisible: matching_games alone read healthy for days while
~80% of games were discarded as stale (2026-07-24, workers frozen on an old
model_sha). So the arithmetic gets a test rather than trust.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from chess_anti_engine.tune.trainable_report import _build_report_dict
from chess_anti_engine.tune.trial_config import (
    DriftMetrics,
    PidResult,
    RestoreResult,
    SelfplayResult,
    TrainingResult,
    TrialConfig,
)


def _fake_trainer(*, mirror_prob: float = 0.5) -> Any:
    """Minimum trainer surface _build_report_dict reads (loss weights + LR)."""
    return SimpleNamespace(
        opt=SimpleNamespace(param_groups=[{"lr": 3e-4}]),
        w_wdl=1.0,
        w_soft=1.0,
        w_sf_move=1.0,
        w_categorical=1.0,
        sf_wdl_frac=0.5,
        sf_wdl_temperature=1.0,
        sf_wdl_draw_scale=1.0,
        sf_wdl_conf_power=1.0,
        mirror_prob=mirror_prob,
        # (name, groups, effective_p) per group; only [i][2] is read.
        _feature_group_dropout=[(f"g{i}", (), 0.0) for i in range(8)],
    )


def _report(sp: SelfplayResult, *, mirror_prob: float = 0.5) -> dict:
    return _build_report_dict(
        tc=TrialConfig(),
        trainer=_fake_trainer(mirror_prob=mirror_prob),
        pr=PidResult(),
        sp=sp,
        tr=TrainingResult(),
        drift=DriftMetrics(),
        eval_dict={},
        puzzle_dict={},
        wdl_regret_used=0.07,
        sf_nodes_used=5_000,
        pause_metrics={
            "paused_seconds": 0.0,
            "paused_fraction": 0.0,
            "paused_percent": 0.0,
        },
        restore=RestoreResult(),
        best_loss=1.0,
        iter_t0=0.0,
        iteration_idx=1,
        buf_size=10,
        holdout_buf_size=1,
        holdout_frozen=False,
        holdout_generation=0,
    )


def test_total_games_ingested_sums_matching_and_stale() -> None:
    """The frozen-fleet iteration: 445 current-model games, 1661 stale."""
    row = _report(SelfplayResult(matching_games=445, distributed_stale_games=1661))
    assert row["matching_games"] == 445
    assert row["distributed_stale_games"] == 1661
    assert row["total_games_ingested"] == 2106


def test_total_games_ingested_equals_matching_when_nothing_is_stale() -> None:
    """A healthy fleet: the new metric must not double-count."""
    row = _report(SelfplayResult(matching_games=445, distributed_stale_games=0))
    assert row["total_games_ingested"] == 445


def test_matching_positions_is_reported_under_its_new_name() -> None:
    row = _report(SelfplayResult(matching_positions=9_810))
    assert row["matching_positions"] == 9_810
    assert "positions_added" not in row


def test_matching_games_is_reported_under_its_new_name() -> None:
    row = _report(SelfplayResult(matching_games=445))
    assert row["matching_games"] == 445
    assert "games_generated" not in row


def test_mirror_prob_reaches_the_result_row() -> None:
    """`mirror_prob` has no yaml key, so the ROW is the only place it is legible.

    `_build_report_dict` is the last point where the dict can be edited: its
    return value is handed to `tune_report_fn` (== `ray.tune.report`, wired at
    trainable.py:977) at trainable_phases.py:1184, with only an `.update()` of
    the replay-priority stats in between. So a column missing here is a column
    missing from the Ray result row.

    Without this assertion the reporting line can be deleted with every test
    still green -- which is the failure this whole item exists to prevent: a
    value that is accepted and then silently not observed.
    """
    row = _report(SelfplayResult())
    assert "mirror_prob" in row
    assert row["mirror_prob"] == 0.5


def test_mirror_prob_reports_the_trainers_realized_value() -> None:
    """Not a constant that happens to match the constructor default."""
    assert _report(SelfplayResult(), mirror_prob=0.25)["mirror_prob"] == 0.25
    assert _report(SelfplayResult(), mirror_prob=0.0)["mirror_prob"] == 0.0
