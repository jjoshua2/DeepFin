from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from scripts import audit_realized_config as ar

CONFIG = {
    "selfplay_fraction": 0.5,
    "train_views_per_ingested_position": 2.5,
    "opening_fen_dole_max_fraction": 0.25,
    # The dole cap is a fraction of games_per_iter, not of selfplay.
    "games_per_iter": 400,
}


def _row(
    *,
    it: int = 100,
    games: int = 400,
    selfplay: int = 200,
    stale: int = 0,
    views: float = 2.5,
    seeded: int = 20,
    seeded_backed: int = 0,
    matching_positions: int = 5000,
    ingested: int = 5000,
    timestamp: float = 1_000_000.0,
    reported_s: float = 1200.0,
    isr: int = 5,
) -> dict:
    return {
        "training_iteration": it,
        "matching_games": games,
        "selfplay_games": selfplay,
        "distributed_stale_games": stale,
        "train_views_actual": views,
        "matching_positions": matching_positions,
        "replay_positions_ingested": ingested,
        "timestamp": timestamp,
        "time_this_iter_s": reported_s,
        "iterations_since_restore": isr,
        # outcome_stats is PIPE-delimited, not space-delimited.
        "outcome_stats": (
            f"selfplay_fenlist_games={seeded}"
            f"|selfplay_fenlist_backed_games={seeded_backed}"
        ),
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


def test_selfplay_mix_is_context_and_never_a_finding(capsys) -> None:
    """selfplay_fraction is a per-slot roll, not a completed-game share.

    The completed share is EXPECTED to run above the knob whenever curriculum
    games outlast selfplay games, so reporting the gap as a divergence trains
    readers to ignore the tool. The live 0.86-vs-0.50 reading was reported as a
    knob failure on this tool's first run; it is not one.
    """
    rows = [_row(selfplay=345), _row(it=101, selfplay=345, timestamp=1_001_200.0)]
    assert not any("selfplay_fraction" in f for f in ar.audit_knobs(rows, _stats(rows)))
    ar.report_selfplay_mix(rows, _stats(rows))
    out = capsys.readouterr().out
    assert "0.8625" in out or "0.862" in out
    assert "DIVERGENT" not in out


def test_a_cap_knob_passes_when_under_and_fails_when_over() -> None:
    under = [_row(seeded=20), _row(it=101, seeded=20, timestamp=1_001_200.0)]
    assert not any(
        "opening_fen_dole_max_fraction" in f for f in ar.audit_knobs(under, _stats(under))
    )
    # 150/400 games_per_iter = 0.375 > the 0.25 cap.
    over = [_row(seeded=150), _row(it=101, seeded=150, timestamp=1_001_200.0)]
    assert any(
        "opening_fen_dole_max_fraction" in f for f in ar.audit_knobs(over, _stats(over))
    )


def test_the_dole_cap_denominator_is_games_per_iter_not_selfplay() -> None:
    """110 seeded of 400 total with 200 selfplay MEETS the cap exactly.

    Dividing by selfplay_games instead would read 0.55 and call a compliant cap
    a 2x breach -- inflated by 1/selfplay_fraction. This pins the base in both
    directions, so swapping the denominator fails here.
    """
    ok_rows = [_row(seeded=100), _row(it=101, seeded=100, timestamp=1_001_200.0)]
    assert not any(
        "opening_fen_dole_max_fraction" in f
        for f in ar.audit_knobs(ok_rows, _stats(ok_rows))
    )
    # Same seeded count, half the selfplay: still compliant, because the base
    # is games_per_iter. A selfplay denominator would flag this one.
    thin = [
        _row(seeded=100, selfplay=100),
        _row(it=101, seeded=100, selfplay=100, timestamp=1_001_200.0),
    ]
    assert not any(
        "opening_fen_dole_max_fraction" in f for f in ar.audit_knobs(thin, _stats(thin))
    )


def test_the_dole_cap_counts_the_backed_seed_channels_too() -> None:
    """Backed seeds are the GOAL channel; omitting them hides a real breach."""
    rows = [
        _row(seeded=20, seeded_backed=140),
        _row(it=101, seeded=20, seeded_backed=140, timestamp=1_001_200.0),
    ]
    assert any(
        "opening_fen_dole_max_fraction" in f for f in ar.audit_knobs(rows, _stats(rows))
    )


def test_a_cap_configured_to_zero_is_uncapped_not_a_breach(capsys) -> None:
    """0 disables the cap in distributed_runtime; reporting it as a breach is noise."""
    rows = [_row(seeded=300), _row(it=101, seeded=300, timestamp=1_001_200.0)]
    for r in rows:
        r["config"] = dict(CONFIG, opening_fen_dole_max_fraction=0.0)
    findings = ar.audit_knobs(rows, _stats(rows))
    assert not any("opening_fen_dole_max_fraction" in f for f in findings)
    assert "UNCAPPED" in capsys.readouterr().out


def test_a_window_spanning_a_knob_change_judges_only_the_current_value(capsys) -> None:
    """A median across a deploy is a number no configuration ever asked for.

    This produced the wrong "the dole cap is entirely inert" verdict on the
    tool's first real run: five pre-deploy iterations with no cap at all
    dominated the median over three capped ones.
    """
    old = _row(seeded=300)
    old["config"] = dict(CONFIG)
    del old["config"]["opening_fen_dole_max_fraction"]
    new = _row(it=101, seeded=20, timestamp=1_001_200.0)
    findings = ar.audit_knobs([old, new], _stats([old, new]))
    assert not any("opening_fen_dole_max_fraction" in f for f in findings)
    assert "changed inside this window" in capsys.readouterr().out


def test_a_malformed_config_value_is_a_finding_not_a_traceback() -> None:
    rows = [_row(), _row(it=101, timestamp=1_001_200.0)]
    for r in rows:
        r["config"] = dict(CONFIG, train_views_per_ingested_position=None)
    findings = ar.audit_knobs(rows, _stats(rows))
    assert any("not a number" in f for f in findings)


def test_a_knob_missing_from_the_config_is_a_finding_not_a_silent_skip() -> None:
    """"Nothing can check this knob" is the state every one of these bugs hid in."""
    rows = [_row(), _row(it=101, timestamp=1_001_200.0)]
    for r in rows:
        cfg = dict(CONFIG)
        del cfg["train_views_per_ingested_position"]
        r["config"] = cfg
    findings = ar.audit_knobs(rows, _stats(rows))
    assert any("cannot audit" in f for f in findings)


def test_views_reading_high_is_not_a_finding_but_low_is() -> None:
    """The budget clamps views UP from below (fresh-samples floor, drought mode)."""
    high = [_row(views=4.0), _row(it=101, views=4.0, timestamp=1_001_200.0)]
    assert not any(
        "train_views_per_ingested_position" in f for f in ar.audit_knobs(high, _stats(high))
    )
    low = [_row(views=0.46), _row(it=101, views=0.46, timestamp=1_001_200.0)]
    assert any(
        "train_views_per_ingested_position" in f for f in ar.audit_knobs(low, _stats(low))
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


def test_matching_positions_undercount_is_reported_with_its_factor() -> None:
    """This ratio IS the views-denominator error factor — surface it numerically."""
    rows = [_row(matching_positions=1000, ingested=6220) for _ in range(2)]
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
    yaml_path.write_text(
        "selfplay:\n  opening_fen_dole_max_fraction: 0.9\n", encoding="utf-8"
    )
    rows = [_row(), _row(it=101, timestamp=1_001_200.0)]
    findings = ar.audit_live_yaml(rows, yaml_path)
    assert any("live reload was likely REJECTED" in f for f in findings)


def test_yaml_matching_effective_config_is_clean(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "selfplay:\n  opening_fen_dole_max_fraction: 0.25\n", encoding="utf-8"
    )
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


# ---------------------------------------------------------------------------
# J5: params.json is the LAUNCH config, and nothing in it says so.
# ---------------------------------------------------------------------------

_RESTART_KEYS = frozenset({"embed_dim", "opening_book_path"})


def _classify(params, yaml_cfg, realized, **kw):
    return ar.classify_config_provenance(
        params, yaml_cfg, realized, restart_keys=_RESTART_KEYS, **kw
    )


def test_stale_params_json_is_reported_by_name_and_is_not_a_finding() -> None:
    """The launch value differing from the running one is normal -- and invisible.

    It has to be listed rather than merely counted: the whole failure mode is
    someone reading one specific key out of that file.
    """
    report, findings = _classify(
        {"opening_fen_dole_per_iter": 1, "w_wdl": 4.0},
        {"opening_fen_dole_per_iter": 0, "w_wdl": 4.0},
        {"opening_fen_dole_per_iter": 0, "w_wdl": 4.0},
    )
    assert findings == []
    assert any("STALE-IN-PARAMS-JSON  1 live-reloadable" in line for line in report)
    assert any("opening_fen_dole_per_iter: params.json=1 running=0" in line
               for line in report)
    assert not any("w_wdl" in line for line in report)


def test_a_rejected_live_reload_is_a_finding() -> None:
    """yaml != running on a live-reloadable key means the whole reload bounced."""
    report, findings = _classify({}, {"w_wdl": 6.0}, {"w_wdl": 4.0})
    assert any("RELOAD-NOT-APPLIED w_wdl" in line for line in report)
    assert any("rejects the WHOLE yaml reload" in f for f in findings)


def test_a_restart_required_key_the_yaml_moved_is_a_finding() -> None:
    """Here params.json IS the authority, and the yaml is the misleading source."""
    _, findings = _classify({"embed_dim": 512}, {"embed_dim": 768}, {"embed_dim": 512})
    assert any("restart required" in f and "embed_dim" in f for f in findings)


def test_a_yaml_edited_after_the_newest_row_is_unresolved_not_a_finding() -> None:
    """The realized row is always minutes old; a newer yaml has not been read yet."""
    report, findings = _classify(
        {}, {"w_wdl": 6.0}, {"w_wdl": 4.0}, yaml_is_newer_than_row=True,
    )
    assert findings == []
    assert any("UNRESOLVED" in line and "next iteration" in line for line in report)


def test_a_rotating_key_is_expected_not_a_finding() -> None:
    """opening_fen_list_path is rewritten between iterations by the retire loop."""
    key = "opening_fen_list_path"
    assert key in ar._PROVENANCE_ROTATING_KEYS
    report, findings = _classify({}, {key: "retire_61.txt"}, {key: "retire_60.txt"})
    assert findings == []
    assert any("EXPECTED" in line and key in line for line in report)


def test_pb2_searched_keys_are_not_compared_against_the_yaml() -> None:
    """A searched key is SUPPOSED to diverge from the yaml -- that is what PB2 does."""
    _, findings = _classify({}, {"lr": 0.0003}, {"lr": 0.0001}, searched_keys={"lr"})
    assert findings == []


def test_provenance_needs_all_three_sources_before_it_says_anything(
    tmp_path: Path, capsys
) -> None:
    """No yaml -> SKIP, not a silent green. Absence of a check is not a pass."""
    rows = [_row(), _row(it=101, timestamp=1_001_200.0)]
    assert ar.audit_config_provenance(rows, tmp_path / "missing.yaml", None) == []
    assert "SKIP" in capsys.readouterr().out


def test_provenance_section_names_params_json_as_the_launch_config(
    tmp_path: Path, capsys
) -> None:
    """End to end through the real restart-key set, on a synthetic trial.

    The row timestamp is deliberately AHEAD of the yaml's mtime. With a 1970
    timestamp every difference is downgraded to UNRESOLVED and the section can
    no longer report anything, so this test would pass without exercising the
    comparison at all.
    """
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "selfplay:\n  opening_fen_dole_max_fraction: 0.25\n", encoding="utf-8"
    )
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"opening_fen_dole_max_fraction": 0.9}), encoding="utf-8")
    later = time.time() + 3600.0
    rows = [_row(timestamp=later - 1200.0), _row(it=101, timestamp=later)]
    findings = ar.audit_config_provenance(rows, yaml_path, params)
    out = capsys.readouterr().out
    assert findings == []
    assert "UNRESOLVED" not in out, "the yaml must look older than the row here"
    assert "LAUNCH config" in out
    assert "opening_fen_dole_max_fraction: params.json=0.9 running=0.25" in out

    # Same trial, yaml moved: now it must speak up rather than stay green.
    yaml_path.write_text(
        "selfplay:\n  opening_fen_dole_max_fraction: 0.9\n", encoding="utf-8"
    )
    findings = ar.audit_config_provenance(rows, yaml_path, params)
    assert any("opening_fen_dole_max_fraction" in f for f in findings)
