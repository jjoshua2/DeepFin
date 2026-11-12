"""Move identity must survive synthetic MultiPV reranking and bound updates.

These transcripts are fixtures, not captured output from a running engine.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pytest

from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.selfplay.stockfish_turn import (
    _build_sf_policy_target,
    _collect_sf_pv_candidates,
    _sf_eval_pv_orphaned,
)
from chess_anti_engine.stockfish.uci import StockfishUCI, _SearchInfoAccumulator


A, B, C = "e2e4", "d2d4", "g1f3"


def line(rank: int, move: str, cp: int, *, depth: int = 5, bound: str = "") -> str:
    return f"info depth {depth} nodes 100 multipv {rank} score cp {cp} {bound} pv {move}"


def accumulate(lines: list[str]) -> _SearchInfoAccumulator:
    acc = _SearchInfoAccumulator()
    for text in lines:
        acc.consume(text.split())
    return acc


def pairs(acc: _SearchInfoAccumulator) -> list[tuple[str, int | None]]:
    return [(pv.move_uci, pv.cp) for pv in acc.result(B).pvs]


class TestMultiPVMoveIdentity(unittest.TestCase):
    def test_complete_reranked_bound_reuses_its_own_exact_score(self):
        # B's exact 10 is consistent with "at most 200" and kept; it
        # contradicts "at least 200" and is clamped to the bound.
        for bound, b_cp in (("lowerbound", 200), ("upperbound", 10)):
            with self.subTest(bound=bound):
                acc = accumulate([line(1, A, 40), line(2, B, 10),
                                  line(1, B, 200, depth=6, bound=bound),
                                  line(2, A, 35, depth=6)])
                assert pairs(acc) == [(B, b_cp), (A, 35)]
                assert acc.result(B).cp == 40

    def test_unknown_bound_move_evicts_previous_rank_occupant(self):
        acc = accumulate([line(1, A, 40), line(2, B, 10),
                          line(1, C, 200, bound="lowerbound")])
        assert pairs(acc) == [(B, 10)]

    def test_partial_exact_rerank_is_unique_and_uses_latest_score(self):
        acc = accumulate([line(1, A, 40), line(2, B, 10), line(2, A, 35)])
        assert pairs(acc) == [(A, 35)]

    def test_three_way_rerank_preserves_distinct_candidates(self):
        acc = accumulate([line(1, A, 40), line(2, B, 10), line(3, C, -10),
                          line(1, B, 200, bound="lowerbound"),
                          line(2, C, 25), line(3, A, 30)])
        assert pairs(acc) == [(B, 200), (C, 25), (A, 30)]

    def test_exact_snapshot_rerank_remains_rank_ordered(self):
        acc = accumulate([line(1, A, 40), line(2, B, 10),
                          line(1, B, 45), line(2, A, 35)])
        assert pairs(acc) == [(B, 45), (A, 35)]

    def test_mixed_depths_are_not_discarded(self):
        acc = accumulate([line(1, A, 40, depth=6), line(2, B, 10, depth=5)])
        assert pairs(acc) == [(A, 40), (B, 10)]

    def test_same_move_bound_preserves_exact_tuple(self):
        acc = accumulate(["info multipv 1 score cp 40 wdl 400 300 300 pv e2e4",
                          "info multipv 1 score cp 900 lowerbound wdl 900 90 10 pv e2e4"])
        res = acc.result(A)
        assert res.pvs[0].cp == 40
        assert res.pvs[0].wdl is not None
        assert res.wdl is not None
        np.testing.assert_allclose(res.pvs[0].wdl, [0.4, 0.3, 0.3])
        np.testing.assert_allclose(res.wdl, [0.4, 0.3, 0.3])

    def test_rerank_preserves_move_specific_mate_and_wdl(self):
        acc = accumulate([line(1, A, 40),
                          "info multipv 2 score mate 3 wdl 1000 0 0 pv d2d4",
                          line(1, B, 200, bound="lowerbound"), line(2, A, 35)])
        pv = acc.result(B).pvs[0]
        assert (pv.move_uci, pv.cp, pv.mate) == (B, None, 3)
        assert pv.wdl is not None
        np.testing.assert_array_equal(pv.wdl, [1, 0, 0])

    def test_no_pv_bound_preserves_unknown_identity_behavior(self):
        acc = accumulate([line(1, A, 40), "info multipv 1 score cp 80 lowerbound"])
        assert pairs(acc) == [(A, 40)]

    def test_bound_only_result_has_no_exact_candidates(self):
        acc = accumulate([line(1, A, 80, bound="lowerbound")])
        assert pairs(acc) == []
        assert acc.result(A).cp is None

    def test_diagnostic_string_does_not_change_identity(self):
        acc = accumulate([line(1, A, 40), "info string multipv 1 score cp 80 pv d2d4"])
        assert pairs(acc) == [(A, 40)]

    def test_new_exact_tuple_does_not_inherit_old_wdl(self):
        acc = accumulate(["info multipv 1 score cp 40 wdl 400 300 300 pv e2e4",
                          line(1, A, 50)])
        assert acc.result(A).pvs[0].wdl is None
        assert acc.result(A).pvs[0].cp == 50

    def test_nodes_depth_and_bestmove_still_update(self):
        acc = accumulate([line(1, A, 40),
                          "info depth 7 nodes 999 multipv 1 score cp 80 lowerbound pv d2d4"])
        res = acc.result(B)
        assert (res.nodes, res.depth, res.bestmove_uci, res.cp) == (999, 7, B, 40)

    def test_each_search_has_its_own_exact_history(self):
        accumulate([line(1, A, 40)])
        acc = accumulate([line(1, A, 80, bound="lowerbound")])
        assert pairs(acc) == []

    def test_repeated_result_is_stable(self):
        acc = accumulate([line(1, A, 40), line(2, A, 30)])
        first = pairs(acc)
        assert first == pairs(acc)
        assert first == [(A, 30)]

    def test_three_move_policy_has_one_weight_per_move(self):
        acc = accumulate([line(1, A, 40), line(2, B, 10), line(3, C, -10),
                          line(1, B, 200, bound="lowerbound"),
                          line(2, C, 25), line(3, A, 30)])
        legal = np.array([uci_to_policy_index(m, True) for m in (A, B, C)], dtype=np.int32)
        ids, scores = _collect_sf_pv_candidates(
            acc.result(B), _turn=True, legal_set=set(legal.tolist()), sf_policy_score_mode="cp")
        assert len(ids) == len(set(ids))
        assert dict(zip(ids, scores, strict=True)) == dict(zip(legal.tolist(), [30, 200, 25], strict=True))
        result = _build_sf_policy_target(ids, scores, legal_indices=legal,
                                        sf_policy_temp=100.0, sf_policy_label_smooth=0.1)
        expected = np.exp(np.array([30, 200, 25], dtype=np.float64) / 100.0)
        expected /= expected.sum()
        np.testing.assert_allclose(result[legal], expected, rtol=1e-6, atol=1e-7)
        assert float(result.sum()) == pytest.approx(1.0, rel=0, abs=5e-7)

    def test_two_move_policy_retains_both_moves(self):
        acc = accumulate([line(1, A, 40), line(2, B, 10),
                          line(1, B, 200, bound="lowerbound"), line(2, A, 35)])
        legal = np.array([uci_to_policy_index(m, True) for m in (A, B)], dtype=np.int32)
        ids, scores = _collect_sf_pv_candidates(acc.result(B), _turn=True,
                                               legal_set=set(legal.tolist()), sf_policy_score_mode="cp")
        target = _build_sf_policy_target(ids, scores, legal_indices=legal,
                                        sf_policy_temp=100.0, sf_policy_label_smooth=0.0)
        assert float(target[legal[1]]) > 0.0
        expected_a = 1.0 / (1.0 + np.exp((200 - 35) / 100.0))
        assert float(target[legal[0]]) == pytest.approx(float(expected_a), rel=0, abs=5e-7)

    def test_preserved_root_scalar_can_trigger_existing_pv_health_flag(self):
        # This is an explicit compatibility caveat, not evidence of desync.
        acc = accumulate([line(1, A, 40), line(2, B, 10),
                          line(1, B, 200, bound="lowerbound"), line(2, A, 35)])
        res = acc.result(B)
        assert (res.cp, res.pvs[0].cp) == (40, 200)
        record = SimpleNamespace(
            sf_multipv_raw=np.array([[0, res.pvs[0].cp, 0, -1, -1]], dtype=np.int16),
            sf_label_meta=np.array([100, 5, res.cp, 0, -1, -1], dtype=np.int32),
        )
        assert _sf_eval_pv_orphaned(record)


@unittest.skipUnless(os.name == "posix", "the UCI wrapper uses a PTY")
class TestMultiPVSyntheticProtocol(unittest.TestCase):
    def test_owned_synthetic_uci_process_reorders_and_resets_search_history(self):
        fixture = '''import sys
searches = 0
for raw in sys.stdin:
    cmd = raw.strip()
    if cmd == "uci":
        print("id name synthetic-multipv-fixture", flush=True)
        print("uciok", flush=True)
    elif cmd == "isready":
        print("readyok", flush=True)
    elif cmd.startswith("go "):
        searches += 1
        if searches == 1:
            lines = ["info depth 5 multipv 1 score cp 40 pv e2e4",
                     "info depth 5 multipv 2 score cp 10 pv d2d4",
                     "info depth 6 multipv 1 score cp 200 lowerbound pv d2d4",
                     "info depth 6 multipv 2 score cp 35 pv e2e4"]
        else:
            lines = ["info depth 1 multipv 1 score cp 200 lowerbound pv d2d4"]
        for text in lines:
            print(text, flush=True)
        print("bestmove d2d4", flush=True)
    elif cmd == "quit":
        break
'''
        with tempfile.TemporaryDirectory(prefix="synthetic-multipv-") as directory:
            path = Path(directory) / "uci-fixture"
            path.write_text(f"#!{sys.executable}\n" + fixture)
            path.chmod(0o700)
            engine = StockfishUCI(str(path), nodes=100, multipv=2, nice=0, read_timeout_s=3)
            try:
                fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                first = engine.search(fen)
                assert [(p.move_uci, p.cp) for p in first.pvs] == [(B, 200), (A, 35)]
                second = engine.search(fen)
                assert second.pvs == []
            finally:
                engine.close()
            assert engine.proc.poll() is not None


if __name__ == "__main__":
    unittest.main()
