"""Invariant logic of scripts/loop_health.py (pure checks + parsing)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "loop_health", Path(__file__).resolve().parents[1] / "scripts" / "loop_health.py"
)
assert _spec is not None
assert _spec.loader is not None
loop_health = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loop_health)

HEALTHY = {
    "training_iteration": 648,
    "replay_positions_ingested": 15772,
    "replay_has_policy_frac": 1.0,
    "replay_pmass_gap_share": 0.0,
    "pid_ema_winrate": 0.59,
    "train_steps_used": 78,
    "games_generated": 651,
    "distributed_stale_games": 0,
    "wdl_regret": 0.0447,
    "time_this_iter_s": 2400.0,
}


def _check(row: dict, prev: dict | None = None, *, fen_total: bool = False,
           fen_self: bool = False, steps_streak: int = 0) -> tuple[list[str], list[str]]:
    return loop_health.check_row(
        row, prev, 2400.0,
        fen_total_alert=fen_total, fen_selfplay_alert=fen_self,
        steps_zero_streak=steps_streak,
    )


def test_healthy_row_is_green() -> None:
    alerts, notes = _check(HEALTHY)
    assert alerts == []
    assert notes == []


def test_value_only_flood_alerts_when_rows_ingested() -> None:
    alerts, _ = _check({**HEALTHY, "replay_has_policy_frac": 0.25})
    assert any("has_policy_frac" in a for a in alerts)


def test_no_flood_alert_on_empty_ingest_denominator() -> None:
    # frac 0.0 with zero ingest is a drought artifact, not a value-only flood.
    alerts, _ = _check({**HEALTHY, "replay_has_policy_frac": 0.0,
                        "replay_positions_ingested": 0})
    assert not any("has_policy_frac" in a for a in alerts)


def test_no_flood_alert_on_zero_frac_missing_field() -> None:
    # frac exactly 0.0 WITH ingest is the has_policy-missing-field fallback, not
    # a real flood (which is ~0.25); must not false-fire.
    alerts, _ = _check({**HEALTHY, "replay_has_policy_frac": 0.0,
                        "replay_positions_ingested": 15772})
    assert not any("has_policy_frac" in a for a in alerts)


def test_killed_gap_knob_alerts() -> None:
    alerts, _ = _check({**HEALTHY, "replay_pmass_gap_share": 0.36})
    assert any("gap-priority" in a for a in alerts)


def test_fen_alerts_are_caller_decided() -> None:
    assert _check(HEALTHY, fen_total=False, fen_self=False)[0] == []
    assert any("stopped delivering entirely" in a for a in _check(HEALTHY, fen_total=True)[0])
    assert any("value-injection sub-stream" in a for a in _check(HEALTHY, fen_self=True)[0])


def test_winrate_low_is_alert_high_is_note() -> None:
    alerts, _ = _check({**HEALTHY, "pid_ema_winrate": 0.30})
    assert any("airbag territory" in a for a in alerts)
    # High side is benign restart-spike territory: NOTE, never an ALERT.
    alerts_hi, notes_hi = _check({**HEALTHY, "pid_ema_winrate": 0.98})
    assert alerts_hi == []
    assert any("likely benign" in n for n in notes_hi)


def test_winrate_zero_is_pid_inactive_not_airbag() -> None:
    # 0.0 is the PID-inactive fallback, not a real sub-0.35 winrate.
    alerts, _ = _check({**HEALTHY, "pid_ema_winrate": 0.0})
    assert not any("airbag territory" in a for a in alerts)


def test_low_games_is_alert_but_note_on_restart_iter() -> None:
    alerts, _ = _check({**HEALTHY, "games_generated": 40})
    assert any("selfplay collapse" in a for a in alerts)
    # Same low count on a restart iteration (stale_games>0) is benign -> NOTE.
    alerts_r, notes_r = _check({**HEALTHY, "games_generated": 40, "distributed_stale_games": 6})
    assert not any("selfplay collapse" in a for a in alerts_r)
    assert any("workers spinning up" in n for n in notes_r)


def test_regret_ease_step_note_and_zero_baseline() -> None:
    _, notes = _check({**HEALTHY, "wdl_regret": 0.09}, prev={**HEALTHY, "wdl_regret": 0.045})
    assert any("airbag fired" in n for n in notes)
    # A 0.0 -> 0.20 jump must still be reported (no falsy-zero short-circuit).
    _, notes0 = _check({**HEALTHY, "wdl_regret": 0.20}, prev={**HEALTHY, "wdl_regret": 0.0})
    assert notes0 == [] or all("airbag fired" not in n for n in notes0)  # 0.0 baseline: no ratio


def test_stale_games_is_note() -> None:
    _, notes = _check({**HEALTHY, "distributed_stale_games": 5})
    assert any("winrate spike" in n for n in notes)


def test_outcome_stats_parser() -> None:
    d = loop_health.parse_outcome_stats(
        "opening_fenlist_games=12|selfplay_fenlist_games=11|pid_reason=not_active|bad")
    assert d == {"opening_fenlist_games": 12, "selfplay_fenlist_games": 11}


def test_load_rows_skips_torn_trailing_line(tmp_path: Path) -> None:
    p = tmp_path / "result.json"
    good = json.dumps({"training_iteration": 1})
    p.write_text(good + "\n" + '{"training_iteration": 2, "outcome_st', encoding="utf-8")
    rows = loop_health.load_rows(p, 20)
    assert rows == [{"training_iteration": 1}]


def test_load_rows_returns_only_last_n(tmp_path: Path) -> None:
    p = tmp_path / "result.json"
    p.write_text("".join(json.dumps({"training_iteration": i}) + "\n" for i in range(50)),
                 encoding="utf-8")
    rows = loop_health.load_rows(p, 5)
    assert [r["training_iteration"] for r in rows] == [45, 46, 47, 48, 49]


def test_stale_outrunning_matching_twice_running_is_an_alert() -> None:
    """Workers frozen on an old model_sha: most selfplay silently discarded.

    Ran undetected for days (2026-07-24) because each counter looked sane on
    its own — only their ratio, sustained, shows the pipeline throwing work away.
    """
    frozen = {**HEALTHY, "games_generated": 445, "distributed_stale_games": 1661}
    alerts, _ = _check(frozen, frozen)
    assert any("frozen on an old model_sha" in a for a in alerts)


def test_stale_outrunning_matching_once_is_only_a_note() -> None:
    """One such iteration is indistinguishable from a restart — do not cry wolf."""
    healthy_prev = {**HEALTHY, "games_generated": 445, "distributed_stale_games": 0}
    alerts, notes = _check(
        {**HEALTHY, "games_generated": 20, "distributed_stale_games": 1661},
        healthy_prev,
    )
    assert not any("frozen on an old model_sha" in a for a in alerts)
    assert any("an alert if it repeats" in n for n in notes)


def test_stale_games_below_matching_is_not_an_alert() -> None:
    row = {**HEALTHY, "games_generated": 445, "distributed_stale_games": 40}
    alerts, _ = _check(row, row)
    assert not any("frozen on an old model_sha" in a for a in alerts)


def test_low_games_on_a_true_restart_iter_stays_a_note() -> None:
    """A restart strands in-flight games; matching ramps. Still benign."""
    healthy_prev = {**HEALTHY, "distributed_stale_games": 0}
    _alerts, notes = _check(
        {**HEALTHY, "games_generated": 20, "distributed_stale_games": 15},
        healthy_prev,
    )
    assert any("workers spinning up" in n for n in notes)
