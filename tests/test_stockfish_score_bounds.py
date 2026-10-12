"""Node-limited UCI score lines must not become exact teacher point targets.

Stockfish tags an aspiration fail with ``lowerbound`` / ``upperbound`` and
prints the window edge as ``score cp`` / ``score mate``. ``go nodes`` can end
on that line. The production value and policy labels are the cp logistic of
whatever ``StockfishUCI.search`` retained, so a bound has to stay out of that
snapshot.

These tests drive the real ``search`` readline loop with a generated engine
transcript and then the real label consumers. They do not claim a particular
engine binary emitted the transcript.
"""
from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest
import yaml

from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.selfplay.stockfish_turn import (
    _collect_sf_pv_candidates,
    _sf_result_wdl_for_record,
    flip_wdl_pov,
)
from chess_anti_engine.stockfish.uci import StockfishResult, StockfishUCI
from chess_anti_engine.stockfish.wdl import cp_to_wdl

_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_NODES = 5000


def _production_logistic() -> tuple[bool, float, float]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "pbt2_small.yaml"
    selfplay = yaml.safe_load(config_path.read_text(encoding="utf-8"))["selfplay"]
    return (
        bool(selfplay["sf_wdl_use_cp_logistic"]),
        float(selfplay["sf_wdl_cp_slope"]),
        float(selfplay["sf_wdl_cp_draw_width"]),
    )


def _search_transcript(
    tmp_path: Path,
    info_lines: list[str],
    *,
    bestmove: str,
    nodes: int = _NODES,
    multipv: int = 1,
) -> tuple[StockfishResult, list[str]]:
    log_path = tmp_path / "commands.log"
    engine_py = tmp_path / "engine.py"
    printed = "\n".join(
        f"        print({line!r}, flush=True)" for line in info_lines
    )
    engine_py.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"log = Path({str(log_path)!r})\n"
        "for line in sys.stdin:\n"
        "    cmd = line.strip()\n"
        "    with log.open('a', encoding='utf-8') as stream:\n"
        "        stream.write(cmd + '\\n')\n"
        "    if cmd == 'uci':\n"
        "        print('uciok', flush=True)\n"
        "    elif cmd == 'isready':\n"
        "        print('readyok', flush=True)\n"
        "    elif cmd.startswith('go '):\n"
        f"{printed}\n"
        f"        print('bestmove {bestmove}', flush=True)\n",
        encoding="utf-8",
    )
    engine_sh = tmp_path / "engine.sh"
    engine_sh.write_text(
        f"#!/usr/bin/env bash\nexec {sys.executable} {engine_py}\n",
        encoding="utf-8",
    )
    engine_sh.chmod(engine_sh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    engine = StockfishUCI(
        str(engine_sh), nodes=nodes, multipv=multipv, read_timeout_s=5.0,
    )
    try:
        result = engine.search(_START_FEN, nodes=nodes)
    finally:
        engine.close()
    commands = log_path.read_text(encoding="utf-8").splitlines()
    assert f"go nodes {nodes}" in commands
    assert not any(cmd.startswith("go depth ") for cmd in commands)
    return result, commands


def _value_and_policy(
    result: StockfishResult,
) -> tuple[object, list[int], list[float]]:
    use_cp, slope, width = _production_logistic()
    assert use_cp is True
    value = _sf_result_wdl_for_record(
        result,
        sf_wdl_use_cp_logistic=use_cp,
        sf_wdl_cp_slope=slope,
        sf_wdl_cp_draw_width=width,
    )
    legal = {
        uci_to_policy_index("e2e4", True),
        uci_to_policy_index("d2d4", True),
    }
    assert -1 not in legal
    indices, scores = _collect_sf_pv_candidates(
        result,
        _turn=True,
        legal_set=legal,
        sf_wdl_use_cp_logistic=use_cp,
        sf_wdl_cp_slope=slope,
        sf_wdl_cp_draw_width=width,
        sf_policy_score_mode="wdl",
    )
    return value, indices, scores


def _expected_value(cp: int | None, mate: int | None):
    _use_cp, slope, width = _production_logistic()
    stm = cp_to_wdl(cp, mate, slope=slope, draw_width_cp=width)
    return flip_wdl_pov(stm)


def _expected_policy_score(cp: int | None, mate: int | None) -> float:
    _use_cp, slope, width = _production_logistic()
    wdl = cp_to_wdl(cp, mate, slope=slope, draw_width_cp=width)
    return float(wdl[0] + 0.5 * wdl[1])


def test_exact_score_is_the_value_and_policy_point_target(tmp_path: Path) -> None:
    result, _commands = _search_transcript(
        tmp_path,
        [
            "info depth 6 seldepth 8 multipv 1 score cp 40 "
            "wdl 500 400 100 nodes 800 nps 1000 hashfull 0 tbhits 0 time 1 pv e2e4",
        ],
        bestmove="e2e4",
    )
    value, indices, scores = _value_and_policy(result)
    assert result.cp == 40
    assert result.mate is None
    assert value == pytest.approx(_expected_value(40, None))
    assert indices == [uci_to_policy_index("e2e4", True)]
    assert scores == pytest.approx([_expected_policy_score(40, None)])


def test_bound_only_node_limited_line_is_not_a_point_target(tmp_path: Path) -> None:
    """The only score in the stream is an upper bound.

    Accepting it makes the logistic value and the policy score the window
    edge. With no earlier exact line there is no point target to keep.
    """
    result, _commands = _search_transcript(
        tmp_path,
        [
            "info depth 7 seldepth 10 multipv 1 score cp 54 upperbound "
            "wdl 600 300 100 nodes 5000 nps 1000 hashfull 0 tbhits 0 time 2 pv e2e4",
        ],
        bestmove="e2e4",
    )
    value, indices, scores = _value_and_policy(result)
    assert result.cp is None
    assert result.mate is None
    assert result.wdl is None
    assert result.pvs == []
    assert value is None
    assert indices == []
    assert scores == []
    assert result.nodes == 5000
    assert result.depth == 7


def test_exact_then_bound_keeps_the_exact_point_target(tmp_path: Path) -> None:
    result, _commands = _search_transcript(
        tmp_path,
        [
            "info depth 6 seldepth 8 multipv 1 score cp 40 "
            "wdl 500 400 100 nodes 800 nps 1000 hashfull 0 tbhits 0 time 1 pv e2e4",
            "info depth 7 seldepth 11 multipv 1 score cp 400 lowerbound "
            "wdl 900 80 20 nodes 5000 nps 1000 hashfull 0 tbhits 0 time 2 pv e2e4",
        ],
        bestmove="e2e4",
    )
    value, indices, scores = _value_and_policy(result)
    assert result.cp == 40
    assert result.mate is None
    assert result.pvs[0].cp == 40
    assert result.pvs[0].move_uci == "e2e4"
    assert value == pytest.approx(_expected_value(40, None))
    assert value != pytest.approx(_expected_value(400, None))
    assert indices == [uci_to_policy_index("e2e4", True)]
    assert scores == pytest.approx([_expected_policy_score(40, None)])


def test_exact_mate_then_mate_bound_keeps_the_exact_mate(tmp_path: Path) -> None:
    result, _commands = _search_transcript(
        tmp_path,
        [
            "info depth 4 seldepth 6 multipv 1 score mate 3 "
            "wdl 1000 0 0 nodes 400 nps 1000 hashfull 0 tbhits 0 time 1 pv e2e4",
            "info depth 5 seldepth 8 multipv 1 score mate 1 lowerbound "
            "wdl 1000 0 0 nodes 5000 nps 1000 hashfull 0 tbhits 0 time 2 pv e2e4",
        ],
        bestmove="e2e4",
    )
    value, _indices, scores = _value_and_policy(result)
    # Mate 1 and mate 3 both saturate the production logistic, so the stored
    # mate — not the WDL — is what shows the bound was dropped.
    assert result.mate == 3
    assert result.cp is None
    assert result.pvs[0].mate == 3
    assert result.pvs[0].mate != 1
    assert value == pytest.approx(_expected_value(None, 3))
    assert scores == pytest.approx([_expected_policy_score(None, 3)])


def test_multipv_bound_does_not_replace_its_rank_or_the_other_rank(tmp_path: Path) -> None:
    result, _commands = _search_transcript(
        tmp_path,
        [
            "info depth 5 seldepth 8 multipv 1 score cp 40 "
            "wdl 500 400 100 nodes 900 nps 1000 hashfull 0 tbhits 0 time 1 pv e2e4",
            "info depth 5 seldepth 8 multipv 2 score cp 10 "
            "wdl 300 500 200 nodes 900 nps 1000 hashfull 0 tbhits 0 time 1 pv d2d4",
            "info depth 6 currmove d2d4 currmovenumber 2",
            "info depth 6 seldepth 10 multipv 1 score cp 200 lowerbound "
            "wdl 900 80 20 nodes 5000 nps 1000 hashfull 0 tbhits 0 time 2 pv e2e4",
            "info depth 6 seldepth 10 multipv 2 score cp 12 "
            "wdl 320 500 180 nodes 5000 nps 1000 hashfull 0 tbhits 0 time 2 pv d2d4",
        ],
        bestmove="e2e4",
        multipv=2,
    )
    value, indices, scores = _value_and_policy(result)
    by_move = {
        pv.move_uci: pv.cp for pv in result.pvs
    }
    assert by_move == {"e2e4": 40, "d2d4": 12}
    assert result.cp == 40
    assert value == pytest.approx(_expected_value(40, None))
    assert value != pytest.approx(_expected_value(200, None))
    e2 = uci_to_policy_index("e2e4", True)
    d2 = uci_to_policy_index("d2d4", True)
    assert indices == [e2, d2]
    assert scores == pytest.approx([
        _expected_policy_score(40, None),
        _expected_policy_score(12, None),
    ])


def test_later_exact_multipv_rank_replaces_only_that_rank(tmp_path: Path) -> None:
    result, _commands = _search_transcript(
        tmp_path,
        [
            "info depth 5 seldepth 8 multipv 1 score cp 20 "
            "wdl 400 500 100 nodes 700 nps 1000 hashfull 0 tbhits 0 time 1 pv e2e4",
            "info depth 5 seldepth 8 multipv 2 score cp 10 "
            "wdl 300 500 200 nodes 700 nps 1000 hashfull 0 tbhits 0 time 1 pv d2d4",
            "info depth 6 seldepth 9 multipv 1 score cp 40 "
            "wdl 500 400 100 nodes 4000 nps 1000 hashfull 0 tbhits 0 time 2 pv e2e4",
        ],
        bestmove="e2e4",
        multipv=2,
    )
    _value, indices, scores = _value_and_policy(result)
    by_move = {pv.move_uci: pv.cp for pv in result.pvs}
    assert by_move == {"e2e4": 40, "d2d4": 10}
    assert result.cp == 40
    e2 = uci_to_policy_index("e2e4", True)
    d2 = uci_to_policy_index("d2d4", True)
    assert indices == [e2, d2]
    assert scores == pytest.approx([
        _expected_policy_score(40, None),
        _expected_policy_score(10, None),
    ])


def test_partial_exact_cp_after_wdl_uses_the_newer_cp(tmp_path: Path) -> None:
    """An exact cp with no WDL on the later line is still the point target.

    Production logistic reads ``res.cp`` / ``pv.cp``. This control is the
    exact-score half of a split line, not a bound.
    """
    result, _commands = _search_transcript(
        tmp_path,
        [
            "info depth 8 seldepth 10 multipv 1 score cp 20 "
            "wdl 400 500 100 nodes 1000 nps 1000 hashfull 0 tbhits 0 time 1 pv e2e4",
            "info depth 9 seldepth 12 multipv 1 score cp 80 "
            "nodes 4000 nps 1000 hashfull 0 tbhits 0 time 2 pv e2e4",
        ],
        bestmove="e2e4",
    )
    value, _indices, scores = _value_and_policy(result)
    assert result.cp == 80
    assert value == pytest.approx(_expected_value(80, None))
    assert value != pytest.approx(_expected_value(20, None))
    assert scores == pytest.approx([_expected_policy_score(80, None)])
