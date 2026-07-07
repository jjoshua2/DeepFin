"""Invariant logic of scripts/loop_health.py (pure checks, no I/O)."""
from __future__ import annotations

import importlib.util
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
    "replay_has_policy_frac": 1.0,
    "replay_pmass_gap_share": 0.0,
    "replay_pmass_fast_share": 0.0,
    "pid_ema_winrate": 0.59,
    "train_steps_used": 78,
    "games_generated": 651,
    "distributed_stale_games": 0,
    "wdl_regret": 0.0447,
    "time_this_iter_s": 2400.0,
}


def _check(row: dict, prev: dict | None = None, fen_streak: int = 0,
           steps_streak: int = 0) -> tuple[list[str], list[str]]:
    return loop_health.check_row(row, prev, 2400.0, fen_streak, steps_streak)


def test_healthy_row_is_green() -> None:
    alerts, notes = _check(HEALTHY)
    assert alerts == []
    assert notes == []


def test_value_only_flood_alerts() -> None:
    alerts, _ = _check({**HEALTHY, "replay_has_policy_frac": 0.25})
    assert any("has_policy_frac" in a for a in alerts)


def test_killed_priority_knob_alerts() -> None:
    alerts, _ = _check({**HEALTHY, "replay_pmass_gap_share": 0.36})
    assert any("gap-priority" in a for a in alerts)


def test_fen_flatline_needs_streak() -> None:
    assert _check(HEALTHY, fen_streak=2)[0] == []
    alerts, _ = _check(HEALTHY, fen_streak=3)
    assert any("FEN seeding" in a for a in alerts)


def test_winrate_airbag_band() -> None:
    alerts, _ = _check({**HEALTHY, "pid_ema_winrate": 0.30})
    assert any("airbag territory" in a for a in alerts)
    assert _check({**HEALTHY, "pid_ema_winrate": 0.74})[0] == []


def test_regret_ease_step_is_note_not_alert() -> None:
    alerts, notes = _check({**HEALTHY, "wdl_regret": 0.09}, prev={**HEALTHY, "wdl_regret": 0.045})
    assert alerts == []
    assert any("airbag fired" in n for n in notes)


def test_stale_games_is_note() -> None:
    _, notes = _check({**HEALTHY, "distributed_stale_games": 5})
    assert any("winrate spike" in n for n in notes)


def test_outcome_stats_parser() -> None:
    d = loop_health.parse_outcome_stats(
        "opening_fenlist_games=12|selfplay_fenlist_games=11|pid_reason=not_active|bad")
    assert d == {"opening_fenlist_games": 12, "selfplay_fenlist_games": 11}
