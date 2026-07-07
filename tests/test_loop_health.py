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
    rows = loop_health.load_rows(p)
    assert rows == [{"training_iteration": 1}]
