from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import audit_realized_config as ar

CONFIG = {
    "selfplay_fraction": 0.5,
    "train_views_per_ingested_position": 2.5,
    "opening_fen_dole_max_fraction": 0.25,
}


def _row(
    *,
    it: int = 100,
    games: int = 400,
    selfplay: int = 200,
    stale: int = 0,
    views: float = 2.5,
    seeded: int = 20,
    positions_added: int = 5000,
    ingested: int = 5000,
    timestamp: float = 1_000_000.0,
    reported_s: float = 1200.0,
    isr: int = 5,
) -> dict:
    return {
        "training_iteration": it,
        "games_generated": games,
        "selfplay_games": selfplay,
        "distributed_stale_games": stale,
        "train_views_actual": views,
        "positions_added": positions_added,
        "replay_positions_ingested": ingested,
        "timestamp": timestamp,
        "time_this_iter_s": reported_s,
        "iterations_since_restore": isr,
        "outcome_stats": f"selfplay_fenlist_games={seeded}",
        "config": dict(CONFIG),
    }


def _stats(rows: list[dict]) -> list[dict]:
    return [ar.parse_outcome_stats(str(r.get("outcome_stats") or "")) for r in rows]


def test_matching_knobs_produce_no_findings(capsys) -> None:
    # selfplay 200/400 = 0.5, views 2.5, seeded 20/200 = 0.1 <= 0.25 cap.
    rows = [_row(), _row(it=101, timestamp=1_001_200.0)]
    findings = ar.audit_knobs(rows, _stats(rows))
    assert findings == []
    assert "DIVERGENT" not in capsys.readouterr().out


def test_realized_selfplay_share_divergence_is_reported() -> None:
    """The live 2026-07-24 case: configured 0.50, realized ~0.86."""
    rows = [_row(selfplay=345), _row(it=101, selfplay=345, timestamp=1_001_200.0)]
    findings = ar.audit_knobs(rows, _stats(rows))
    assert any("selfplay_fraction" in f for f in findings)


def test_a_cap_knob_passes_when_under_and_fails_when_over() -> None:
    under = [_row(seeded=20), _row(it=101, seeded=20, timestamp=1_001_200.0)]
    assert not any(
        "opening_fen_dole_max_fraction" in f for f in ar.audit_knobs(under, _stats(under))
    )
    # Seeded == all selfplay games: the cap is inert (realized 1.0 vs 0.25).
    over = [_row(seeded=200), _row(it=101, seeded=200, timestamp=1_001_200.0)]
    assert any(
        "opening_fen_dole_max_fraction" in f for f in ar.audit_knobs(over, _stats(over))
    )


def test_a_knob_nothing_ever_realizes_is_flagged_not_silently_skipped() -> None:
    """An absent metric must not read as agreement — that is how views hid."""
    rows = [_row(), _row(it=101, timestamp=1_001_200.0)]
    for r in rows:
        del r["train_views_actual"]
    findings = ar.audit_knobs(rows, _stats(rows))
    assert any("NOTHING realized it" in f for f in findings)


def test_frozen_fleet_needs_two_iters_to_alert() -> None:
    once = [_row(stale=0), _row(it=101, stale=1600, timestamp=1_001_200.0)]
    assert not any("frozen on an old model_sha" in f for f in ar.audit_counters(once))
    twice = [_row(stale=1600), _row(it=101, stale=1600, timestamp=1_001_200.0)]
    assert any("frozen on an old model_sha" in f for f in ar.audit_counters(twice))


def test_positions_added_undercount_is_reported_with_its_factor() -> None:
    """This ratio IS the views-denominator error factor — surface it numerically."""
    rows = [_row(positions_added=1000, ingested=6220) for _ in range(2)]
    findings = ar.audit_counters(rows)
    assert any("6.22x" in f for f in findings)


def test_wall_clock_gap_from_an_unrequested_restart_is_reported() -> None:
    """iterations_since_restore failing to advance is the restart tell."""
    rows = [
        _row(timestamp=1_000_000.0, reported_s=1124.0, isr=1),
        _row(it=101, timestamp=1_006_687.0, reported_s=1208.0, isr=1),
    ]
    findings = ar.audit_wall_clock(rows)
    assert any("unrequested restart" in f for f in findings)
    assert any("5479s" in f for f in findings)


def test_wall_clock_clean_when_reported_matches_the_timestamp_delta() -> None:
    rows = [
        _row(timestamp=1_000_000.0, reported_s=1200.0, isr=1),
        _row(it=101, timestamp=1_001_200.0, reported_s=1200.0, isr=2),
    ]
    assert ar.audit_wall_clock(rows) == []


def test_yaml_on_disk_disagreeing_with_effective_config_is_reported(tmp_path: Path) -> None:
    """A rejected live reload leaves every knob at its pre-edit value, silently."""
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("selfplay:\n  selfplay_fraction: 0.9\n", encoding="utf-8")
    rows = [_row(), _row(it=101, timestamp=1_001_200.0)]
    findings = ar.audit_live_yaml(rows, yaml_path)
    assert any("live reload was likely REJECTED" in f for f in findings)


def test_yaml_matching_effective_config_is_clean(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("selfplay:\n  selfplay_fraction: 0.5\n", encoding="utf-8")
    rows = [_row(), _row(it=101, timestamp=1_001_200.0)]
    assert ar.audit_live_yaml(rows, yaml_path) == []


def test_main_exits_nonzero_on_divergence(tmp_path: Path, monkeypatch, capsys) -> None:
    result = tmp_path / "result.json"
    rows = [_row(selfplay=345, stale=1600), _row(it=101, selfplay=345, stale=1600,
                                                 timestamp=1_001_200.0)]
    result.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["audit", "--result-json", str(result), "--last", "2"])
    with pytest.raises(SystemExit) as exc:
        ar.main()
    assert exc.value.code == 1
    assert "AUDIT FOUND" in capsys.readouterr().out


def test_main_is_green_and_exits_zero_when_everything_agrees(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    result = tmp_path / "result.json"
    rows = [
        _row(timestamp=1_000_000.0, isr=1),
        _row(it=101, timestamp=1_001_200.0, isr=2),
    ]
    result.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["audit", "--result-json", str(result), "--last", "2"])
    ar.main()
    assert "AUDIT OK" in capsys.readouterr().out


def test_window_smaller_than_two_iters_is_rejected(monkeypatch) -> None:
    """Wall-clock and frozen-fleet checks both need a previous row."""
    monkeypatch.setattr("sys.argv", ["audit", "--result-json", "x", "--last", "1"])
    with pytest.raises(SystemExit) as exc:
        ar.main()
    assert exc.value.code != 0
