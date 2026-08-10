"""The diff-focus regime guard, pinned against the run it was built from.

The bands are not opinions: every row below is a real per-iteration value read
out of the live trial's tfevents (trial ``379f6``, 2026-08-09/10). The guard is
useless unless it stays silent through the pre-bundle steady state AND fires on
the first few iterations after the search-authority bundle, so both halves are
asserted directly rather than through a synthetic proxy.
"""

from __future__ import annotations

from typing import Any

import pytest

from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.tune.diff_focus_guard import (
    ALARM_DETAIL_KEY,
    ALARM_KEY,
    REGIME_BANDS,
    WARMUP_ITERATIONS,
    evaluate_diff_focus_regime,
)

# Measured pre-bundle steady state (iters 560-735; keep_rate sd 0.0063 over 176
# iterations, so these are representative, not cherry-picked).
PRE_BUNDLE = {
    "diff_focus_keep_rate": 0.8029,
    "diff_focus_keep_limited_frac": 0.3737,
    "diff_focus_priority_mean": 0.8937,
    "replay_priority_mean": 1.0979,
    "grad_hard_clip_rate": 0.0,
    "diff_focus_records": 11_872,   # measured median over trial 379f6
    "replay_priority_n": 11_300,
    "grad_norm_samples": 88,
}

# The ten iterations immediately after the bundle restart at iter 736, in order.
# Read off tfevents; these are the values that ran unnoticed for ~140 iterations.
POST_BUNDLE_KEEP_RATE = [
    0.8300, 0.8597, 0.8976, 0.9216, 0.9301, 0.9339, 0.9360, 0.9284, 0.9360, 0.9354,
]
POST_BUNDLE_KEEP_LIMITED = [
    0.3272, 0.2765, 0.2237, 0.1813, 0.1644, 0.1570, 0.1554, 0.1651, 0.1509, 0.1463,
]
POST_BUNDLE_REPLAY_PRIORITY = [
    1.7304, 2.7436, 3.1302, 3.3495, 3.5149, 3.7664, 3.7268, 3.6223, 3.6674, 3.9357,
]

STEADY: dict[str, Any] = {
    "enabled": True, "trial_iterations_completed": 50, "report_iteration": 700,
}


def _row(**over) -> dict:
    row = dict(PRE_BUNDLE)
    row.update(over)
    return row


def test_pre_bundle_steady_state_does_not_alarm() -> None:
    """The regime the run was GAINING Elo in must read clean."""
    out = evaluate_diff_focus_regime(_row(), **STEADY)
    assert out[ALARM_KEY] == 0, out[ALARM_DETAIL_KEY]
    assert out[ALARM_DETAIL_KEY] == ""


@pytest.mark.parametrize(
    ("offset", "keep", "limited", "priority"),
    list(zip(
        range(len(POST_BUNDLE_KEEP_RATE)),
        POST_BUNDLE_KEEP_RATE,
        POST_BUNDLE_KEEP_LIMITED,
        POST_BUNDLE_REPLAY_PRIORITY,
        strict=True,
    )),
)
def test_post_bundle_iterations_alarm_within_four_iterations(
    offset: int, keep: float, limited: float, priority: float,
) -> None:
    """THE NEGATIVE CONTROL, run against the real incident.

    Iterations 736-745 are the actual telemetry from the change that cost ~76
    Elo. The guard must be silent on none of the settled ones and must have
    fired by the fourth (iter 739) -- ~140 iterations earlier than a human did.
    """
    out = evaluate_diff_focus_regime(
        _row(
            diff_focus_keep_rate=keep,
            diff_focus_keep_limited_frac=limited,
            replay_priority_mean=priority,
            # diff_focus_priority_mean tracks replay_priority_mean closely
            # (0.894 vs 1.098 pre-bundle); scale it the same way rather than
            # letting a stale pre-bundle value mask a breach.
            diff_focus_priority_mean=priority * 0.894 / 1.0979,
        ),
        enabled=True,
        # The bundle arrived on a RESTART, so the trial counter is `offset`.
        # Passing the real value makes a future WARMUP_ITERATIONS increase turn
        # this test red instead of silently delaying detection.
        trial_iterations_completed=offset,
        report_iteration=736 + offset,
    )
    if offset >= 3:
        assert out[ALARM_KEY] == 1, f"iter {736 + offset} slipped through"
    # The first three are the ramp; only require that by iter 739 it is caught.


# Iteration 5 of this same trial, verbatim: the HEALTHIEST-era extreme, from a
# full-length iteration (12165 records) while the run was gaining Elo. It is the
# high-water mark of a 700-iteration downward drift in keep_rate, so the 560-735
# calibration window is that drift's BOTTOM, not its centre. Pinned so that
# tightening a band back onto the calibration window fails the suite.
EARLY_HEALTHY = {
    "diff_focus_keep_rate": 0.8732,
    "diff_focus_keep_limited_frac": 0.2630,
    "diff_focus_priority_mean": 1.1174,
    "replay_priority_mean": 1.2605,
    "grad_hard_clip_rate": 0.0,
    "diff_focus_records": 12_165,
    "replay_priority_n": 11_300,
    "grad_norm_samples": 88,
}


def test_early_healthy_iteration_does_not_alarm() -> None:
    """A cold-buffer iteration from the run's best era must read clean.

    The post-revert restart begins exactly such an era, so this is the likely
    next operational state, not a hypothetical.
    """
    out = evaluate_diff_focus_regime(dict(EARLY_HEALTHY), **STEADY)
    assert out[ALARM_KEY] == 0, out[ALARM_DETAIL_KEY]


def test_alarm_names_the_metric_and_uses_no_commas() -> None:
    out = evaluate_diff_focus_regime(
        _row(diff_focus_keep_rate=0.9360, diff_focus_keep_limited_frac=0.1463),
        **STEADY,
    )
    assert out[ALARM_KEY] == 1
    detail = str(out[ALARM_DETAIL_KEY])
    assert "diff_focus_keep_rate" in detail
    assert "diff_focus_keep_limited_frac" in detail
    # progress.csv is parsed with naive `awk -F','` by scripts/monitor_pbt.sh.
    assert "," not in detail


def test_every_band_can_actually_fire_on_each_side_it_declares() -> None:
    """A band nobody can breach is a gate that cannot fail.

    For every entry, push that one metric past each bound it declares and
    require the alarm. Catches a typo'd band, a key renamed out from under the
    table, and a one-sided band declared on the wrong side.
    """
    for key, (low, high, _denom_key, _denom_min) in REGIME_BANDS.items():
        assert key in PRE_BUNDLE, f"{key} has no representative baseline"
        for bound, direction in ((low, -1.0), (high, 1.0)):
            if bound is None:
                continue
            out = evaluate_diff_focus_regime(
                _row(**{key: bound + direction * (abs(bound) * 0.5 + 0.5)}), **STEADY,
            )
            assert out[ALARM_KEY] == 1, f"{key} band {direction:+.0f} side cannot fire"
            assert key in str(out[ALARM_DETAIL_KEY])


def test_empty_denominator_is_skipped_not_alarmed() -> None:
    """A paused or short iteration reports 0.0 everywhere; that is not a breach."""
    out = evaluate_diff_focus_regime(
        _row(
            diff_focus_keep_rate=0.0,
            diff_focus_keep_limited_frac=0.0,
            diff_focus_priority_mean=0.0,
            replay_priority_mean=0.0,
            diff_focus_records=0,
            replay_priority_n=0,
            grad_norm_samples=0,
        ),
        **STEADY,
    )
    assert out[ALARM_KEY] == 0, out[ALARM_DETAIL_KEY]


def test_disabled_curriculum_is_not_alarmed() -> None:
    out = evaluate_diff_focus_regime(
        _row(diff_focus_keep_rate=1.0, diff_focus_keep_limited_frac=0.0),
        enabled=False, trial_iterations_completed=50, report_iteration=700,
    )
    assert out[ALARM_KEY] == 0
    assert out[ALARM_DETAIL_KEY] == "disabled"


def test_warmup_uses_the_trials_own_counter_not_the_restored_global_one() -> None:
    """A salvage warm start restores global_iter ~672; the warmup must not
    read that as 'long past warmup' on iteration 0, nor as 'still warming'
    forever."""
    breach = _row(diff_focus_keep_rate=0.9360)
    warming = evaluate_diff_focus_regime(
        breach, enabled=True,
        trial_iterations_completed=WARMUP_ITERATIONS - 1, report_iteration=672,
    )
    assert warming[ALARM_KEY] == 0
    assert warming[ALARM_DETAIL_KEY] == "warmup"

    armed = evaluate_diff_focus_regime(
        breach, enabled=True,
        trial_iterations_completed=WARMUP_ITERATIONS, report_iteration=675,
    )
    assert armed[ALARM_KEY] == 1


def test_guard_reads_keys_the_report_actually_emits() -> None:
    """Every guarded key and every denominator must be a real report column.

    This is the wiring half: a band on a key nobody emits is silent forever,
    which is the failure this whole module exists to stop.
    """
    import ast
    import inspect
    import textwrap

    from chess_anti_engine.tune import trainable_report

    # Parse the EMITTER and collect the string keys of its dict literals. A
    # substring search over the source text passes when the old name survives
    # only in a rename COMMENT -- verified: that mutation left the suite green,
    # and this file is in a repo whose comment density makes it likely.
    emitted: set[str] = set()
    for fn in (trainable_report._build_report_dict, trainable_report._train_metrics_dict):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                emitted.update(
                    k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                )
    assert len(emitted) > 100, f"AST walk found only {len(emitted)} keys; parse broke"
    for key, (_low, _high, denom_key, _min) in REGIME_BANDS.items():
        assert key in emitted, f"{key} is not emitted by trainable_report"
        assert denom_key in emitted, f"{denom_key} is not emitted"


def test_guard_is_wired_into_the_reported_row() -> None:
    """The call must sit between the priority-mass merge and the report.

    Order is load-bearing: called earlier, the guard reads a dict that has no
    priority-mass columns; called after `tune_report_fn`, its own columns never
    reach progress.csv. A guard that runs and is not logged is the exact defect
    it was written to catch.
    """
    import inspect

    from chess_anti_engine.tune import trainable_phases

    src = inspect.getsource(trainable_phases)
    merge = src.index("report_dict.update(replay_priority_stats")
    call = src.index("evaluate_diff_focus_regime(")
    report = src.index("tune_report_fn(report_dict, checkpoint=checkpoint)")
    assert merge < call < report


def test_priority_mass_raw_means_are_free_of_the_config_weights() -> None:
    """`*_raw_mean` must not move when `diff_focus_pol_scale` does.

    That independence is the entire reason the columns were added: the
    `*_share` columns apply TODAY's weight to rows whose stored priority
    carries yesterday's, so they cannot be read across a recalibration.
    """
    import numpy as np

    n = 64
    arrs = {
        "has_policy": np.ones(n, dtype=bool),
        "has_priority_policy_kl": np.ones(n, dtype=bool),
        "has_priority_q_delta": np.ones(n, dtype=bool),
        "priority_policy_kl": np.full(n, 0.5, dtype=np.float32),
        "priority_q_delta": np.full(n, -0.25, dtype=np.float32),
    }
    pri = np.ones(n, dtype=np.float32)

    seen = []
    for pol_scale in (3.5, 0.45):
        buf = DiskReplayBuffer.__new__(DiskReplayBuffer)
        buf.diff_focus_pol_scale = pol_scale
        buf.diff_focus_q_weight = 6.0
        buf.sf_gap_priority_weight = 0.0
        buf.fast_low_surprise_priority = 1.0
        buf._pmass = DiskReplayBuffer._pmass_zero()
        buf._accumulate_priority_mass(arrs, pri)
        seen.append(DiskReplayBuffer.pop_priority_mass_stats(buf))

    assert seen[0]["replay_pmass_kl_raw_mean"] == pytest.approx(0.5)
    assert seen[1]["replay_pmass_kl_raw_mean"] == pytest.approx(0.5)
    assert seen[0]["replay_pmass_qd_raw_mean"] == pytest.approx(0.25)
    assert seen[1]["replay_pmass_qd_raw_mean"] == pytest.approx(0.25)
    assert seen[0]["replay_pmass_kl_raw_rows"] == n
    # ...while the share DOES move, which is exactly the trap being documented.
    assert seen[0]["replay_pmass_kl_share"] != pytest.approx(
        seen[1]["replay_pmass_kl_share"],
    )
