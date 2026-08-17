"""Adaptive SF-label escalation on net-vs-label disagreement (flag-gated).

Covers stockfish_turn's sf_label_escalate_* mechanism: the default-off no-op
(no extra engine calls, no record marker), the fire path (deep cold-TT
re-query replaces the recorded label; original preserved for the harvester),
the per-game cap, POV correctness, the blocking finalize flush, and the
finalize/harvester consumers of ``sf_wdl_original``.
"""
from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.selfplay.blindspot_harvest import _harvest_sf_wdl
from chess_anti_engine.selfplay.config import GameConfig, OpponentConfig
from chess_anti_engine.selfplay.finalize import _count_label_escalations
from chess_anti_engine.selfplay.state import _NetRecord
from chess_anti_engine.selfplay.stockfish_turn import (
    flip_wdl_pov,
    flush_async_sf_labels_for_records,
    poll_async_sf_labels,
    submit_async_curriculum_move_queries,
    submit_async_sf_labels_from_curriculum_moves,
    submit_async_sf_label_queries,
)
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishResult


class _FakeCBoard:
    turn = True
    occ_white = 0
    occ_black = 0
    castling = 0

    def __init__(self) -> None:
        self._legal_indices = np.array([0], dtype=np.int64)

    def fen(self) -> str:
        return chess.STARTING_FEN

    def legal_move_indices(self) -> np.ndarray:
        return self._legal_indices

    def is_game_over(self) -> bool:
        return False


class _QueuedFakePool(StockfishPool):
    """Records submit calls (incl. ``fresh``) and pops queued results."""

    def __init__(self, results: list[StockfishResult]) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.calls: list[dict] = []
        self._results = list(results)

    def submit(
        self, fen: str, *, nodes=None, syzygy_path=None, fresh: bool = False,
        searchmoves=None,
    ):
        # Recorded, so an escalation that silently narrowed its root would show
        # up here rather than being absorbed by the fake.
        self.calls.append(
            {
                "fen": fen, "nodes": nodes, "syzygy_path": syzygy_path,
                "fresh": fresh,
                "searchmoves": None if searchmoves is None else list(searchmoves),
            },
        )
        fut: Future = Future()
        fut.set_result(self._results.pop(0))
        return fut


def _res(wdl: tuple[float, float, float]) -> StockfishResult:
    """A label result whose PV1 WDL is ``wdl`` (side-to-move = opponent POV)."""
    return StockfishResult(
        bestmove_uci="a2a3", wdl=np.array(wdl, dtype=np.float32), pvs=[],
    )


def _record(search_wdl: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> _NetRecord:
    return _NetRecord(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_probs=np.zeros((POLICY_SIZE,), dtype=np.float32),
        net_wdl_est=np.array(search_wdl, dtype=np.float32),
        search_wdl_est=np.array(search_wdl, dtype=np.float32),
        pov_color=chess.WHITE,
        ply_index=0,
        has_policy=True,
        priority=1.0,
        sample_weight=1.0,
        keep_prob=1.0,
        legal_mask=np.zeros((POLICY_SIZE,), dtype=np.uint8),
    )


def _state(
    *,
    results: list[StockfishResult],
    game: GameConfig,
    record: _NetRecord | None = None,
) -> Any:
    return SimpleNamespace(
        batch_size=1,
        pending_sf_labels=[],
        pending_sf_moves={},
        stockfish=_QueuedFakePool(results),
        samples_per_game=[[record if record is not None else _record()]],
        cboards=[_FakeCBoard()],
        base_nodes=100,
        last_net_full=[True],
        tb_adj_roll_arr=[True],
        game=game,
        rng=np.random.default_rng(0),
        opponent=OpponentConfig(),
        move_idx_history=[[]],
        mcts_tree=None,
        root_ids=[-1],
        done_arr=np.zeros((1,), dtype=np.int8),
        finalized_arr=np.zeros((1,), dtype=np.int8),
        selfplay_arr=np.zeros((1,), dtype=np.int8),
        sf_label_escalations=[0],
    )


# Opponent-POV label WDLs (flipped to record POV on attach). The record's
# search says the NET is winning (q=+1), so the "disagreeing" original label
# (opponent winning -> record q=-0.85) gaps by 1.85; the escalated deep result
# sides with the net (record q=+0.85).
_ORIG_WDL = (0.9, 0.05, 0.05)
_ESC_WDL = (0.05, 0.05, 0.9)


def test_escalation_fires_and_records_escalated_label() -> None:
    game = GameConfig(sf_label_escalate_q_gap=0.5)
    state = _state(results=[_res(_ORIG_WDL), _res(_ESC_WDL)], game=game)
    rec = state.samples_per_game[0][0]

    assert submit_async_sf_label_queries(state, [0]) == 1
    # Poll 1: original label completed, gap trips -> escalated re-query
    # replaces the pending future (non-blocking; nothing attached yet).
    assert poll_async_sf_labels(state) == (0, 0)
    assert len(state.pending_sf_labels) == 1
    assert len(state.stockfish.calls) == 2
    esc_call = state.stockfish.calls[1]
    assert esc_call["nodes"] == 3_000_000  # sf_label_escalate_nodes default
    assert esc_call["fresh"] is True  # cold-TT re-search (ucinewgame)
    assert esc_call["fen"] == chess.STARTING_FEN  # the P1 query position

    # Poll 2: escalated result attaches; the recorded label is the DEEP one,
    # the original is preserved for the harvester.
    assert poll_async_sf_labels(state) == (1, 0)
    assert state.pending_sf_labels == []
    np.testing.assert_allclose(
        rec.sf_wdl, flip_wdl_pov(np.array(_ESC_WDL, dtype=np.float32)),
    )
    np.testing.assert_allclose(
        rec.sf_wdl_original, flip_wdl_pov(np.array(_ORIG_WDL, dtype=np.float32)),
    )
    assert state.sf_label_escalations == [1]


def test_flag_off_is_noop_with_no_extra_engine_calls() -> None:
    """Default q_gap 0.0: same disagreement, but no re-query, no marker."""
    state = _state(results=[_res(_ORIG_WDL), _res(_ESC_WDL)], game=GameConfig())
    rec = state.samples_per_game[0][0]

    submit_async_sf_label_queries(state, [0])
    assert poll_async_sf_labels(state) == (1, 0)

    assert len(state.stockfish.calls) == 1  # exactly the plain label query
    assert state.stockfish.calls[0]["fresh"] is False
    np.testing.assert_allclose(
        rec.sf_wdl, flip_wdl_pov(np.array(_ORIG_WDL, dtype=np.float32)),
    )
    assert rec.sf_wdl_original is None
    assert state.sf_label_escalations == [0]


def test_no_escalation_below_threshold() -> None:
    """Agreeing evals never trip the gate even with the flag on."""
    game = GameConfig(sf_label_escalate_q_gap=0.8)
    rec = _record(search_wdl=(0.6, 0.2, 0.2))  # net q = +0.4
    # Opponent-POV (0.2, 0.2, 0.6) flips to record q = +0.4 -> gap 0.
    state = _state(results=[_res((0.2, 0.2, 0.6))], game=game, record=rec)

    submit_async_sf_label_queries(state, [0])
    assert poll_async_sf_labels(state) == (1, 0)

    assert len(state.stockfish.calls) == 1
    assert rec.sf_wdl_original is None
    assert state.sf_label_escalations == [0]


def test_per_game_cap_respected() -> None:
    game = GameConfig(sf_label_escalate_q_gap=0.5, sf_label_escalate_max_per_game=1)
    state = _state(
        results=[_res(_ORIG_WDL), _res(_ESC_WDL), _res(_ORIG_WDL)], game=game,
    )
    rec1 = state.samples_per_game[0][0]

    submit_async_sf_label_queries(state, [0])
    poll_async_sf_labels(state)  # escalate rec1 (budget -> 1/1)
    assert poll_async_sf_labels(state) == (1, 0)  # attach escalated rec1
    assert rec1.sf_wdl_original is not None

    # Second full ply in the SAME game: gap trips again but the budget is
    # exhausted -> the original label attaches directly, no third+1 SF call.
    rec2 = _record()
    state.samples_per_game[0].append(rec2)
    submit_async_sf_label_queries(state, [0])
    assert poll_async_sf_labels(state) == (1, 0)

    assert len(state.stockfish.calls) == 3  # label1, escalation1, label2
    assert rec2.sf_wdl_original is None
    assert rec2.sf_wdl is not None
    np.testing.assert_allclose(
        np.asarray(rec2.sf_wdl), flip_wdl_pov(np.array(_ORIG_WDL, dtype=np.float32)),
    )
    assert state.sf_label_escalations == [1]


def test_finalize_flush_blocks_through_escalation() -> None:
    """The finalize flush escalates too (record is about to be emitted), by
    blocking on the deep result instead of re-queuing."""
    game = GameConfig(sf_label_escalate_q_gap=0.5)
    state = _state(results=[_res(_ORIG_WDL), _res(_ESC_WDL)], game=game)
    rec = state.samples_per_game[0][0]

    submit_async_sf_label_queries(state, [0])
    attached, failed = flush_async_sf_labels_for_records(state, [rec])

    assert (attached, failed) == (1, 0)
    assert state.pending_sf_labels == []
    assert len(state.stockfish.calls) == 2
    assert state.stockfish.calls[1]["fresh"] is True
    np.testing.assert_allclose(
        rec.sf_wdl, flip_wdl_pov(np.array(_ESC_WDL, dtype=np.float32)),
    )
    assert rec.sf_wdl_original is not None


def test_escalated_label_uses_same_pov_flip_as_plain_path() -> None:
    """POV correctness: when the deep re-search returns the SAME eval as the
    original, the recorded label is bit-identical to the flag-off path."""
    game = GameConfig(sf_label_escalate_q_gap=0.5)
    state_on = _state(results=[_res(_ORIG_WDL), _res(_ORIG_WDL)], game=game)
    state_off = _state(results=[_res(_ORIG_WDL)], game=GameConfig())

    for st in (state_on, state_off):
        submit_async_sf_label_queries(st, [0])
        flush_async_sf_labels_for_records(st, [st.samples_per_game[0][0]])

    rec_on = state_on.samples_per_game[0][0]
    rec_off = state_off.samples_per_game[0][0]
    np.testing.assert_array_equal(rec_on.sf_wdl, rec_off.sf_wdl)
    # Escalated but unmoved: marker present, identical label.
    np.testing.assert_array_equal(rec_on.sf_wdl_original, rec_on.sf_wdl)


def test_curriculum_label_reuse_path_carries_escalation_context() -> None:
    """Labels reused from full-strength curriculum move futures escalate via
    the same pending pipeline (context captured at reuse time)."""
    game = GameConfig(sf_label_escalate_q_gap=0.5)
    state = _state(results=[_res(_ORIG_WDL), _res(_ESC_WDL)], game=game)
    rec = state.samples_per_game[0][0]

    assert submit_async_curriculum_move_queries(state, [0]) == 1
    assert submit_async_sf_labels_from_curriculum_moves(state, [0]) == 1
    assert state.pending_sf_labels[0].query_fen == chess.STARTING_FEN
    attached, failed = flush_async_sf_labels_for_records(state, [rec])

    assert (attached, failed) == (1, 0)
    assert len(state.stockfish.calls) == 2  # move future reused + escalation
    assert state.stockfish.calls[1]["fresh"] is True
    assert rec.sf_wdl_original is not None


def test_count_label_escalations_moved_threshold() -> None:
    """Finalize telemetry: escalated = marker present; moved = |dq| >= 0.2."""
    plain = _record()  # no marker

    moved = _record()
    moved.sf_wdl = np.array([0.9, 0.05, 0.05], dtype=np.float32)  # q=+0.85
    moved.sf_wdl_original = np.array([0.05, 0.05, 0.9], dtype=np.float32)  # q=-0.85

    # Binary-exact fractions so the |dq| comparison is float-fuzz-free.
    unmoved = _record()
    unmoved.sf_wdl = np.array([0.5625, 0.25, 0.1875], dtype=np.float32)  # q=+0.375
    unmoved.sf_wdl_original = np.array([0.5, 0.25, 0.25], dtype=np.float32)  # q=+0.25

    assert _count_label_escalations([plain]) == (0, 0)
    assert _count_label_escalations([plain, moved, unmoved]) == (2, 1)


def test_harvester_sees_the_original_label() -> None:
    """The harvester's severity input is the PRE-escalation label, so
    escalation cannot hide the positions the harvester mines."""
    rec = _record()
    assert _harvest_sf_wdl(rec) is None

    rec.sf_wdl = np.array([0.9, 0.05, 0.05], dtype=np.float32)
    assert _harvest_sf_wdl(rec) is rec.sf_wdl  # no escalation -> live label

    rec.sf_wdl_original = np.array([0.05, 0.05, 0.9], dtype=np.float32)
    assert _harvest_sf_wdl(rec) is rec.sf_wdl_original
