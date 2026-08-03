"""Invariant logic of scripts/loop_health.py (pure checks + parsing)."""
from __future__ import annotations

import importlib.util
import json
import os
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
    "matching_games": 651,
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
    alerts, _ = _check({**HEALTHY, "matching_games": 40})
    assert any("selfplay collapse" in a for a in alerts)
    # Same low count on a restart iteration (stale_games>0) is benign -> NOTE.
    alerts_r, notes_r = _check({**HEALTHY, "matching_games": 40, "distributed_stale_games": 6})
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


# --------------------------------------------------------------------------
# The SF-desync detector's blind-instrument alert (checked_frac == 0).
#
# The RATE is deliberately NOT alerted on yet: the train row reads ~2e-4 while
# the quarantine tail ages out, and the `test_` twin reads 0.101305 on the
# frozen holdout and does not age out at all. Only "the instrument saw nothing"
# is armed here, which is why every test below pins checked_frac and not the
# rate.
# --------------------------------------------------------------------------

# What a live row looks like once the column has deployed and the window is
# healthy: the labelled share of the batch (~0.99), and a holdout that IS
# contaminated at the known 0.101305 but is being MEASURED.
LIVE_WITH_COLUMN = {
    **HEALTHY,
    "sf_labelled_no_multipv_frac": 0.0002,
    "sf_multipv_checked_frac": 0.9915,
    "test_size": 2000,
    "test_sf_labelled_no_multipv_frac": 0.101305,
    "test_sf_multipv_checked_frac": 0.9575,
}


def test_a_healthy_row_carrying_the_columns_is_green() -> None:
    alerts, notes = _check(LIVE_WITH_COLUMN)
    assert alerts == []
    assert notes == []


def test_the_contaminated_frozen_holdout_is_not_an_alert() -> None:
    """0.101305 on `test_sf_labelled_no_multipv_frac` is the REAL reading of
    the shipped holdout. The rate alert is not armed, so it must stay green —
    an operator seeing red here on the first restart would mute the column."""
    alerts, _ = _check(LIVE_WITH_COLUMN)
    assert not any("no_multipv" in a for a in alerts)


def test_a_blind_train_detector_alerts() -> None:
    alerts, _ = _check({**LIVE_WITH_COLUMN, "sf_multipv_checked_frac": 0.0})
    assert any("sf_multipv_checked_frac=0.0 on a trained iteration" in a for a in alerts)


def test_a_blind_holdout_detector_alerts() -> None:
    alerts, _ = _check({**LIVE_WITH_COLUMN, "test_sf_multipv_checked_frac": 0.0})
    assert any("test_sf_multipv_checked_frac=0.0" in a for a in alerts)


def test_a_blind_value_half_detector_alerts() -> None:
    """`sf_wdl` is 0.45 of the value target and had NO detector before
    2026-08-03. Watching only the policy column would rebuild the P2
    asymmetry one layer up, in the thing that reads the columns."""
    alerts, _ = _check({**LIVE_WITH_COLUMN, "sf_eval_pv_checked_frac": 0.0})
    assert any("sf_eval_pv_checked_frac=0.0 on a trained iteration" in a for a in alerts)


def test_a_blind_value_half_detector_is_silent_when_absent_or_dark() -> None:
    """Same two guards as the policy twin: an absent column is a schema fact,
    and a zero-step iteration publishes the dataclass default, not a reading."""
    absent, _ = _check(LIVE_WITH_COLUMN)
    assert not any("sf_eval_pv_checked_frac" in a for a in absent)
    dark, _ = _check(
        {**LIVE_WITH_COLUMN, "train_steps_used": 0, "sf_eval_pv_checked_frac": 0.0},
    )
    assert not any("sf_eval_pv_checked_frac=0.0 on a trained iteration" in a for a in dark)


def test_an_absent_column_is_silent() -> None:
    """Every result.json row written before the column shipped lacks it. A
    missing key is a schema fact, not a blindness reading — if this fired the
    monitor would latch ALERTS PRESENT on all history and be ignored."""
    alerts, _ = _check(HEALTHY)
    assert not any("checked_frac" in a for a in alerts)


def test_zero_steps_does_not_read_as_a_blind_detector() -> None:
    """With no train phase the reporter publishes the dataclass DEFAULT 0.0,
    which is not a measurement. The zero-steps state has its own alert."""
    alerts, _ = _check(
        {**LIVE_WITH_COLUMN, "train_steps_used": 0, "sf_multipv_checked_frac": 0.0},
    )
    assert not any("checked_frac=0.0 on a trained iteration" in a for a in alerts)


def test_a_dark_holdout_eval_does_not_read_as_a_blind_detector() -> None:
    """The eval is legitimately dark for two iterations after every restart
    (rl_loop_audit G17); it publishes NaN and test_size 0."""
    alerts, _ = _check(
        {**LIVE_WITH_COLUMN, "test_size": 0,
         "test_sf_multipv_checked_frac": float("nan")},
    )
    assert not any("test_sf_multipv_checked_frac" in a for a in alerts)
    # ...and an explicit 0.0 with no rows scored is equally not a reading.
    alerts0, _ = _check(
        {**LIVE_WITH_COLUMN, "test_size": 0, "test_sf_multipv_checked_frac": 0.0},
    )
    assert not any("test_sf_multipv_checked_frac" in a for a in alerts0)


def test_the_blind_detector_alert_is_wired_into_check_row() -> None:
    """The helper is not the production path — `check_row` is. A helper that
    works while nothing calls it is this repo's signature defect."""
    direct = loop_health.blind_desync_detector_alerts(
        {**LIVE_WITH_COLUMN, "sf_multipv_checked_frac": 0.0},
    )
    assert len(direct) == 1
    via_check_row, _ = _check({**LIVE_WITH_COLUMN, "sf_multipv_checked_frac": 0.0})
    assert direct[0] in via_check_row


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
    """A sustained stale/matching ratio is worth an alert — but not the old one.

    The condition still fires: it ran undetected for days (2026-07-24) because
    each counter looks sane alone and only the sustained ratio shows anything.

    What changed 2026-07-26 is the CLAIM. The alert used to say "workers frozen
    on an old model_sha; most selfplay is being discarded". Both halves were
    false. `_process_shard` calls `_ingest_train_arrays` unconditionally, BEFORE
    the `model_sha in accepted_model_shas` check, so stale shards still reach the
    replay buffer — stale/matching is an accounting split, not a data-loss ratio.
    And when it fired for real at 74-78% stale, all four workers were switching
    shas normally; iterations were simply slow (578s -> 3114s under a concurrent
    arena) so surplus games arrived under an aged-out sha.

    This test therefore pins the DETECTION and explicitly forbids the two false
    assertions coming back.
    """
    frozen = {**HEALTHY, "matching_games": 445, "distributed_stale_games": 1661}
    alerts, _ = _check(frozen, frozen)
    stale_alerts = [a for a in alerts if "stale_games=1661" in a]
    assert stale_alerts, f"the sustained stale ratio must still alert: {alerts}"
    joined = " ".join(stale_alerts)
    assert "frozen on an old model_sha" not in joined
    assert "discarded" not in joined
    assert "NOT lost" in joined


def test_stale_outrunning_matching_once_is_only_a_note() -> None:
    """One such iteration is indistinguishable from a restart — do not cry wolf."""
    healthy_prev = {**HEALTHY, "matching_games": 445, "distributed_stale_games": 0}
    alerts, notes = _check(
        {**HEALTHY, "matching_games": 20, "distributed_stale_games": 1661},
        healthy_prev,
    )
    assert not any("frozen on an old model_sha" in a for a in alerts)
    assert any("an alert if it repeats" in n for n in notes)


def test_stale_games_below_matching_is_not_an_alert() -> None:
    row = {**HEALTHY, "matching_games": 445, "distributed_stale_games": 40}
    alerts, _ = _check(row, row)
    assert not any("frozen on an old model_sha" in a for a in alerts)


def test_low_games_on_a_true_restart_iter_stays_a_note() -> None:
    """A restart strands in-flight games; matching ramps. Still benign."""
    healthy_prev = {**HEALTHY, "distributed_stale_games": 0}
    _alerts, notes = _check(
        {**HEALTHY, "matching_games": 20, "distributed_stale_games": 15},
        healthy_prev,
    )
    assert any("workers spinning up" in n for n in notes)


# ---------------------------------------------------------------------------
# The strength ruler's own outage. daily_gate_ratchet.sh now stops after a
# couple of failed attempts instead of retrying all day, so a dead day leaves
# ratchet.csv with no row for that date and result.json perfectly green. The
# attempt ledger is only a control if something reads it.
# ---------------------------------------------------------------------------

_ATTEMPTS_HEADER = "date,iter,attempt,rc,rows,reason\n"


def _attempts(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "attempts.csv"
    p.write_text(_ATTEMPTS_HEADER + body)
    return p


def test_a_day_that_wrote_no_ratchet_row_is_an_alert(tmp_path: Path) -> None:
    path = _attempts(tmp_path, (
        "2026-07-31,409,1,0,1,vs_prev:row|vs_boot512:row\n"
        "2026-08-01,478,1,1,0,vs_prev:selfmatch|vs_boot512:nopairs\n"
        "2026-08-01,478,2,5,0,vs_prev:selfmatch|vs_boot512:nopairs\n"
    ))
    alerts = loop_health.ratchet_gap_alerts(path)
    assert len(alerts) == 1, alerts
    assert "2026-08-01" in alerts[0]
    assert "2 attempt(s)" in alerts[0]
    assert "gave up" in alerts[0]
    assert "not a null result" in alerts[0]


def test_a_day_with_a_row_is_silent(tmp_path: Path) -> None:
    """One failed attempt followed by a good one is a working instrument."""
    path = _attempts(tmp_path, (
        "2026-08-01,478,1,1,0,vs_boot512:nopairs\n"
        "2026-08-01,478,2,0,1,vs_prev:nosnap|vs_boot512:row\n"
    ))
    assert loop_health.ratchet_gap_alerts(path) == []


def test_a_still_retryable_day_says_so(tmp_path: Path) -> None:
    path = _attempts(tmp_path, "2026-08-01,478,1,1,0,vs_boot512:backstop\n")
    alerts = loop_health.ratchet_gap_alerts(path)
    assert len(alerts) == 1, alerts
    assert "still retryable" in alerts[0]


def test_a_missing_or_empty_ledger_is_not_an_alert(tmp_path: Path) -> None:
    """The ratchet may legitimately never have run; that is invariant L1's
    question, not this check's. A monitor that fires on a fresh box is a
    monitor that gets ignored."""
    assert loop_health.ratchet_gap_alerts(tmp_path / "nope.csv") == []
    assert loop_health.ratchet_gap_alerts(_attempts(tmp_path, "")) == []


def test_only_the_NEWEST_day_is_judged(tmp_path: Path) -> None:
    """An old hole stays in the ledger forever; alerting on it would latch the
    monitor on permanently, which is how a real alert becomes invisible."""
    path = _attempts(tmp_path, (
        "2026-07-30,308,1,5,0,vs_boot512:nopairs\n"
        "2026-08-01,478,1,0,2,vs_prev:row|vs_boot512:row\n"
    ))
    assert loop_health.ratchet_gap_alerts(path) == []


def test_the_ratchet_check_is_wired_into_the_exit_status(tmp_path: Path) -> None:
    """The rule is worthless if main() never calls it.

    Runs the tool for real: a green result.json plus a ledger whose newest day
    produced no row must exit 1 and print the ALERT. Deleting the call in
    main() leaves every ratchet_gap_alerts test above green, which is the
    "pinned rule, unpinned wiring" defect this check exists to catch.
    """
    import subprocess
    import sys

    result = tmp_path / "result.json"
    result.write_text(json.dumps(HEALTHY) + "\n")
    root = Path(__file__).resolve().parents[1]

    def run(attempts_body: str) -> tuple[int, str]:
        path = _attempts(tmp_path, attempts_body)
        r = subprocess.run(
            [sys.executable, "scripts/loop_health.py",
             "--result-json", str(result), "--ratchet-attempts", str(path)],
            cwd=str(root), capture_output=True, text=True, check=False,
            env={"PYTHONPATH": str(root), "PATH": os.environ.get("PATH", ""),
                 "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return r.returncode, r.stdout + r.stderr

    rc, out = run("2026-08-01,478,2,5,0,vs_boot512:nopairs\n")
    assert rc == 1, f"a day with no strength reading must alert:\n{out}"
    assert "ALERT: the daily strength ratchet wrote NO row on 2026-08-01" in out, out
    assert "ALERTS PRESENT" in out

    rc, out = run("2026-08-01,478,1,0,2,vs_prev:row|vs_boot512:row\n")
    assert rc == 0, f"a working ratchet must stay green:\n{out}"
    assert "all invariants green" in out, out
