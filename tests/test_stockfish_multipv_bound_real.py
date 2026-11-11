"""MultiPV parsing on REAL Stockfish transcripts that end on a bound line.

Fixture: ``tests/fixtures/stockfish_multipv_bound_cases.json`` (provenance in
the file). Every expected value below was read off those transcripts by hand
from the ``info`` lines, not produced by running the parser.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.selfplay.stockfish_turn import (
    _choose_curriculum_opponent_move,
    _collect_sf_label_meta,
    _collect_sf_pv_candidates,
    _collect_sparse_pv_rows,
    _sf_eval_pv_orphaned,
)
from chess_anti_engine.stockfish.uci import StockfishResult, _SearchInfoAccumulator

_FIXTURE = Path(__file__).parent / "fixtures" / "stockfish_multipv_bound_cases.json"
CASES = json.loads(_FIXTURE.read_text())["cases"]


def _parse(name: str) -> StockfishResult:
    case = CASES[name]
    acc = _SearchInfoAccumulator()
    for text in case["lines"]:
        acc.consume(text.split())
    return acc.result(case["bestmove"])


def _rows(res: StockfishResult) -> list[tuple[str, int | None, int | None]]:
    return [(pv.move_uci, pv.cp, pv.mate) for pv in res.pvs]


def _legal(name: str) -> tuple[bool, np.ndarray]:
    board = chess.Board(CASES[name]["fen"])
    legal = np.array(
        [uci_to_policy_index(m.uci(), board.turn) for m in board.legal_moves], dtype=np.int32,
    )
    return board.turn, legal


def _stored_record(name: str, res: StockfishResult):
    """The two blocks a shard stores, built by the production stampers."""
    turn, legal = _legal(name)

    class Rec:
        sf_multipv_raw = _collect_sparse_pv_rows(res, turn=turn, legal_set=set(legal.tolist()))
        sf_label_meta = _collect_sf_label_meta(res)

    return Rec


def _final_bounds(name: str) -> dict[str, tuple[str, int]]:
    """move -> (bound kind, bound cp) for moves whose LAST line is a bound."""
    last: dict[str, list[str]] = {}
    for text in CASES[name]["lines"]:
        tok = text.split()
        last[tok[tok.index("pv") + 1]] = tok
    out = {}
    for move, tok in last.items():
        for kind in ("lowerbound", "upperbound"):
            if kind in tok:
                out[move] = (kind, int(tok[tok.index("score") + 2]))
    return out


def test_upperbound_contradicting_a_kept_exact_score_is_clamped() -> None:
    # c5e4 was exact at -391 (depth 9); the final line says it is now at most
    # -465. The emitted -391 would put it ABOVE f8g7 (-406) on stale evidence.
    res = _parse("upperbound_rank3_clamped")
    assert _rows(res) == [
        ("b4c3", -307, None), ("g5g4", -417, None), ("c5e4", -465, None),
        ("f8g7", -406, None), ("d6e6", -448, None), ("d6h6", -481, None),
    ]
    # The clamped row carries the bound line's own WDL ("wdl 0 0 1000").
    assert res.pvs[2].wdl is not None
    np.testing.assert_allclose(res.pvs[2].wdl, [0.0, 0.0, 1.0])
    assert (res.cp, res.mate) == (-307, None)


def test_lowerbound_contradicting_a_kept_exact_score_is_clamped() -> None:
    # d3f5 failed high at rank 2 ("cp 484 lowerbound") over its exact 471.
    # Before the parser fix this transcript emitted d3f3 twice and lost d3f5.
    res = _parse("lowerbound_rank2_clamped")
    assert _rows(res) == [
        ("d3e4", 569, None), ("d3f5", 484, None), ("d3f3", 441, None),
        ("d3c2", 453, None), ("d3g6", 365, None), ("d3b5", 329, None),
    ]


@pytest.mark.parametrize("name", sorted(CASES))
def test_emitted_scores_respect_the_final_bounds_except_the_scalar_row(name: str) -> None:
    res = _parse(name)
    emitted = {pv.move_uci: pv for pv in res.pvs}
    assert len(emitted) == len(res.pvs)  # one row per move
    for move, (kind, edge) in _final_bounds(name).items():
        pv = emitted.get(move)
        if pv is None or pv.cp is None:
            continue
        if res.pvs[0].move_uci == move and (pv.cp, pv.mate) == (res.cp, res.mate):
            continue  # the scalar's own row; see the next test
        assert (pv.cp <= edge) if kind == "upperbound" else (pv.cp >= edge), (move, kind, edge)


def test_scalar_row_is_not_clamped_so_row0_keeps_matching_the_value_scalar() -> None:
    # c6c7 was exact at 360 at rank 1 (the value scalar); the search then
    # stopped on "multipv 1 score cp 430 lowerbound pv c6c7". Clamping row 0
    # to 430 would split it from res.cp and fire the eval/PV orphan detector
    # (a desync fingerprint) on ~3-5% of labelled rows in the banks.
    res = _parse("scalar_row_lowerbound_not_clamped")
    assert _final_bounds("scalar_row_lowerbound_not_clamped")["c6c7"] == ("lowerbound", 430)
    assert (res.pvs[0].move_uci, res.pvs[0].cp, res.cp) == ("c6c7", 360, 360)
    assert not _sf_eval_pv_orphaned(_stored_record("scalar_row_lowerbound_not_clamped", res))


@pytest.mark.parametrize(
    ("name", "scalar_move"),
    [("benign_orphan_rank1_fail_high", "d6b4"), ("benign_orphan_rank1_fail_low", "e7e6")],
)
def test_benign_orphans_keep_bestmove_at_row0(name: str, scalar_move: str) -> None:
    """The two orphan-detector hits real Stockfish produced (2 of 1000).

    In both, rank 1 ends on a bound for SF's ``bestmove`` while the scalar is
    the previous exact rank-1 line of another move. Reordering rows to put the
    scalar's move first is NOT a fix: ``_choose_curriculum_opponent_move``
    plays ``cand_indices[0]`` as "SF's best" on full-strength (SF-refute)
    plies, so it would play the stale move instead of ``bestmove``; and in the
    fail-high case no emitted row carries the scalar at all (d6b4's newer
    exact line is -583, the scalar -416), so reordering cannot clear the flag.
    """
    case = CASES[name]
    res = _parse(name)
    assert res.pvs[0].move_uci == case["bestmove"] != scalar_move
    assert _sf_eval_pv_orphaned(_stored_record(name, res))  # documented known rate
    turn, legal = _legal(name)
    ids, scores = _collect_sf_pv_candidates(res, _turn=turn, legal_set=set(legal.tolist()))
    played = _choose_curriculum_opponent_move(
        rng=np.random.default_rng(0), legal_indices=legal, cand_indices=ids,
        cand_scores=scores, regret_limit=math.inf,
    )
    assert played == uci_to_policy_index(case["bestmove"], turn)
    if name == "benign_orphan_rank1_fail_high":
        assert all(pv.cp != res.cp for pv in res.pvs)
        assert {pv.move_uci: pv.cp for pv in res.pvs}[scalar_move] == -583
