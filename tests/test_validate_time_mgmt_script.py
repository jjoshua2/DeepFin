"""Pure-logic tests for scripts/validate_time_mgmt.py.

The game-playing needs a GPU + checkpoint, so these cover only the testable
core: knob parsing/flags, the match argv, and PGN aggregation (flag attribution,
scoring, the pass/fail gate)."""
from __future__ import annotations

import pytest

from tests.script_loading import load_script_module


def _mod():
    return load_script_module("validate_time_mgmt.py", "validate_time_mgmt_test_module")


def _pgn(white: str, black: str, result: str, termination: str = "rules") -> str:
    return (
        f'[White "{white}"]\n[Black "{black}"]\n'
        f'[Result "{result}"]\n[Termination "{termination}"]\n\n{result}\n\n'
    )


def test_parse_knobs_roundtrip_and_rejects_unknown() -> None:
    m = _mod()
    assert m.parse_knobs("abort_factor=0.7,time_budget_scale=1.5") == {
        "abort_factor": 0.7, "time_budget_scale": 1.5,
    }
    assert m.parse_knobs("") == {}
    with pytest.raises(ValueError, match="unknown knob"):
        m.parse_knobs("aborptfactor=0.7")  # typo
    with pytest.raises(ValueError, match="expected name=value"):
        m.parse_knobs("abort_factor")


def test_knob_flags_and_engine_cmd_are_deterministic() -> None:
    m = _mod()
    knobs = {"time_budget_scale": 1.5, "abort_factor": 0.6}
    # Sorted order regardless of dict insertion order.
    assert m.knob_flags(knobs) == ["--abort-factor", "0.6", "--time-budget-scale", "1.5"]
    cmd = m.build_engine_cmd("/ckpts/t.pt", "cuda", knobs)
    assert "--checkpoint /ckpts/t.pt" in cmd
    assert "--abort-factor 0.6" in cmd
    assert "--time-budget-scale 1.5" in cmd
    # Empty knobs -> bare command, no stray flags.
    assert m.build_engine_cmd("/ckpts/t.pt", "cpu", {}).endswith("--device cpu")


def test_chunk_sims_is_a_sweepable_knob() -> None:
    m = _mod()
    # chunk_sims couples NPS and stop-granularity; the sweep must reach it.
    assert m.parse_knobs("chunk_sims=2048") == {"chunk_sims": 2048.0}
    knobs = {"chunk_sims": 2048.0, "optimum_fraction": 0.6}
    # Int-valued knob renders without a trailing .0 (argparse --chunk-sims is int).
    assert m.knob_flags(knobs) == [
        "--chunk-sims", "2048", "--optimum-fraction", "0.6",
    ]


def test_opponent_mode_gates_flag_safety_on_our_side_only() -> None:
    m = _mod()
    # Baseline is a fixed external opponent ("opp"); candidate is "cand".
    # Opponent flags (0-1 + time => White=opp lost on time): not OUR bug.
    opp_flag = _pgn("opp", "cand", "0-1", termination="time")
    r = m.aggregate_pgn(
        opp_flag, label_baseline="opp", label_candidate="cand", opponent_mode=True,
    )
    assert r.baseline_flags == 1
    assert r.candidate_flags == 0
    assert r.flag_safe is True  # opponent's flag doesn't fail our gate
    assert m.validation_passed([r]) is True

    # But OUR flag (1-0 + time => Black=cand lost on time) still fails the gate.
    our_flag = _pgn("opp", "cand", "1-0", termination="time")
    r2 = m.aggregate_pgn(
        our_flag, label_baseline="opp", label_candidate="cand", opponent_mode=True,
    )
    assert r2.candidate_flags == 1
    assert r2.flag_safe is False
    assert m.validation_passed([r2]) is False


def test_build_match_argv_wires_clock_and_labels() -> None:
    m = _mod()
    argv = m.build_match_argv(
        "match_vs_uci.py",
        engine_baseline="ENG_A", engine_candidate="ENG_B",
        label_baseline="base", label_candidate="cand",
        clock_base_ms=10000, clock_inc_ms=100,
        games=20, max_plies=300, pgn_out="/tmp/x.pgn",
        openings="/tmp/o.epd",
    )
    # Baseline must be engine A so PGN White/Black carry the labels we attribute on.
    assert argv[argv.index("--engine-a") + 1] == "ENG_A"
    assert argv[argv.index("--label-a") + 1] == "base"
    assert argv[argv.index("--clock-base-ms") + 1] == "10000"
    assert argv[argv.index("--openings") + 1] == "/tmp/o.epd"


def test_aggregate_pgn_scores_from_candidate_perspective() -> None:
    m = _mod()
    # Two paired games: cand white & wins, then cand black & draws -> 1.5/2.
    pgn = _pgn("base", "cand", "0-1") + _pgn("cand", "base", "1/2-1/2")
    r = m.aggregate_pgn(pgn, label_baseline="base", label_candidate="cand")
    assert r.games == 2
    assert r.candidate_points == pytest.approx(1.5)
    assert r.score == pytest.approx(0.75)
    assert r.cand_wins == 1
    assert r.draws == 1
    assert r.cand_losses == 0
    assert r.flag_safe is True


def test_aggregate_pgn_attributes_time_flags_to_the_right_side() -> None:
    m = _mod()
    # result 1-0 + Termination time => Black lost on time. Black is the candidate.
    cand_flag = _pgn("base", "cand", "1-0", termination="time")
    r1 = m.aggregate_pgn(cand_flag, label_baseline="base", label_candidate="cand")
    assert r1.candidate_flags == 1
    assert r1.baseline_flags == 0
    assert r1.flag_safe is False

    # result 0-1 + time => White lost on time. White is the baseline here.
    base_flag = _pgn("base", "cand", "0-1", termination="time")
    r2 = m.aggregate_pgn(base_flag, label_baseline="base", label_candidate="cand")
    assert r2.baseline_flags == 1
    assert r2.candidate_flags == 0
    assert r2.flag_safe is False


def test_validation_gate_and_elo() -> None:
    m = _mod()
    safe = m.aggregate_pgn(
        _pgn("base", "cand", "0-1") + _pgn("cand", "base", "1-0"),
        label_baseline="base", label_candidate="cand",
    )  # cand wins both -> score 1.0 (Elo saturates to None), no flags
    assert safe.flag_safe
    assert safe.elo is None
    assert m.validation_passed([safe]) is True

    flagged = m.aggregate_pgn(
        _pgn("base", "cand", "1-0", termination="time"),
        label_baseline="base", label_candidate="cand",
    )
    assert m.validation_passed([safe, flagged]) is False
    summary = m.format_summary([safe, flagged], baseline_label="base")
    assert "FAIL" in summary
    assert "FLAGGED" in summary


def test_aggregate_pgn_ignores_unfinished_games() -> None:
    m = _mod()
    r = m.aggregate_pgn(
        _pgn("base", "cand", "0-1") + _pgn("base", "cand", "*"),
        label_baseline="base", label_candidate="cand",
    )
    assert r.games == 1  # the '*' (unfinished) game is not counted
