from __future__ import annotations

from types import SimpleNamespace

from chess_anti_engine.tune.trainable_report import (
    _pid_report_dict,
    _pid_step_diag_dict,
)


def _full_diag() -> SimpleNamespace:
    return SimpleNamespace(
        reason="fit", changed=True, value_before=0.05, value_after=0.04,
        applied_delta=-0.01, raw_delta=-0.02, cap=0.1, observation_se=0.03,
        raw_deadband=0.01, ema_deadband=0.01, history_len=5,
        tighten_gain_applied=1.0, crash_ease_applied=False,
        predicted_value=0.5, fit_slope=2.0,
    )


def test_pid_step_diag_dict_key_set_is_stable() -> None:
    """A lever that didn't run (diag=None) must emit the SAME keys as one that did,
    so Ray's first-row-fixes-the-header CSV logger never sees a varying schema."""
    assert set(_pid_step_diag_dict("pid_regret", None)) == set(
        _pid_step_diag_dict("pid_regret", _full_diag())
    )


def test_pid_report_dict_key_set_is_stable() -> None:
    """The PID report block has an identical key set whether or not an update ran
    and regardless of which levers were active."""
    pr_none = SimpleNamespace(pid_update=None)
    pr_regret_only = SimpleNamespace(pid_update=SimpleNamespace(
        raw_winrate=0.6, observation_se=0.03, regret_frozen=False, nodes_active=False,
        regret_diag=_full_diag(), nodes_diag=None,
    ))
    pr_both = SimpleNamespace(pid_update=SimpleNamespace(
        raw_winrate=0.6, observation_se=0.03, regret_frozen=True, nodes_active=True,
        regret_diag=_full_diag(), nodes_diag=_full_diag(),
    ))
    keys_none = set(_pid_report_dict(pr_none))  # pyright: ignore[reportArgumentType]
    assert keys_none == set(_pid_report_dict(pr_regret_only))  # pyright: ignore[reportArgumentType]
    assert keys_none == set(_pid_report_dict(pr_both))  # pyright: ignore[reportArgumentType]
    assert keys_none, "must emit the full key set, not an empty dict"
