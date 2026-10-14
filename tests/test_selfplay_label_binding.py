from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.selfplay import manager, network_turn, stockfish_turn
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.state import SelfplayState, _NetRecord
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishResult
from tests.selfplay_helpers import FakeStockfish, UniformPolicyValueModel


FEN = "k7/8/1K6/8/8/8/4PPPP/2R5 w - - 0 1"


class HeldPool(StockfishPool):
    def __init__(self):  # pyright: ignore[reportMissingSuperCall]
        self.nodes = 100
        self.queries: list[tuple[str, dict[str, Any], Future]] = []

    def submit(self, fen: str, **kwargs) -> Future:
        future = Future()
        self.queries.append((fen, kwargs, future))
        return future


class RecordingSync(FakeStockfish):
    def __init__(self):
        super().__init__([0.9, 0.0, 0.1])
        self.queries: list[str] = []

    def search(self, fen: str, **kwargs) -> StockfishResult:
        self.queries.append(fen)
        return super().search(fen, **kwargs)


def _state(stockfish, *, c_ply: bool = True, full: bool = True) -> SelfplayState:
    state = SelfplayState.create(
        model=UniformPolicyValueModel().eval(), device="cpu",
        rng=np.random.default_rng(0), stockfish=stockfish, evaluator=None,
        batch_size=1, continuous=False, target=1,
        opponent=OpponentConfig(), temp=TemperatureConfig(temperature=0.0),
        search=SearchConfig(simulations=1, fast_simulations=1,
                            playout_cap_fraction=float(full)),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(selfplay_fraction=1.0, sf_label_nodes_cap=50),
    )
    state.boards[0] = chess.Board(FEN)
    state.cboards[0] = CBoard.from_board(chess.Board(FEN))
    state.has_c_ply = c_ply
    return state


def _step(state: SelfplayState, *, opponent: bool = False) -> None:
    manager._run_step(
        state, net_idxs=[] if opponent else [0],
        sp_opp_idxs=[0] if opponent else [], cur_opp_idxs=[],
        stockfish=state.stockfish,
    )


def _choose(monkeypatch, uci: str, turn: bool) -> None:
    # Control only the legal action selector; use the real root evaluation,
    # search, record creation, forced turn, scheduler and label functions.
    action = uci_to_policy_index(uci, turn)

    def select(probs, actions, temps, rng, **kwargs):
        del temps, rng, kwargs
        assert probs[0][action] >= 0.0
        actions[0] = action

    monkeypatch.setattr(network_turn, "_resample_actions_with_temperature", select)


def _result(fen: str, wdl: list[float]) -> StockfishResult:
    board = chess.Board(fen)
    return StockfishResult(
        bestmove_uci=next(iter(board.legal_moves)).uci(),
        wdl=np.asarray(wdl, dtype=np.float32), pvs=[],
    )


def _assert_wdl(rec: _NetRecord, expected: list[float]) -> None:
    assert rec.sf_wdl is not None
    np.testing.assert_allclose(rec.sf_wdl, expected)


@pytest.mark.parametrize("fen", ["k7/8/1K6/8/8/8/8/2R5 w - - 0 1", FEN])
def test_forced_reply_fixture_is_legal_and_nonterminal(fen):
    board = chess.Board(fen)
    cb = CBoard.from_board(chess.Board(fen))
    assert board.is_valid()
    for uci in ("c1a1", "a8b8"):
        if not board.turn:
            assert [m.uci() for m in board.legal_moves] == [uci]
            assert cb.legal_move_indices().size == 1
        cb.push_index(uci_to_policy_index(uci, board.turn))
        board.push_uci(uci)
        assert cb.fen() == board.fen()
        assert not board.is_game_over()
        assert not cb.is_game_over()
    if fen == FEN:
        assert len(board.piece_map()) == 7


@pytest.mark.parametrize("c_ply", [True, False])
def test_forced_turn_keeps_original_pending_record_and_position(monkeypatch, c_ply):
    pool = HeldPool()
    state = _state(pool, c_ply=c_ply)
    _choose(monkeypatch, "c1a1", chess.WHITE)
    _step(state)
    rec = state.samples_per_game[0][-1]
    p1, budget, original = pool.queries[0]
    assert chess.Board(p1).turn == chess.BLACK
    assert rec.pov_color == chess.WHITE
    assert budget["nodes"] == 50
    assert state.pending_sf_labels[0].record is rec
    assert state.pending_sf_labels[0].query_fen == p1

    _step(state, opponent=True)  # real forced move; no new record
    assert state.samples_per_game[0] == [rec]
    assert chess.Board(state.cboards[0].fen()).turn == chess.WHITE
    assert len(pool.queries) == 1
    assert stockfish_turn.poll_async_sf_labels(state) == (0, 0)
    assert rec.sf_wdl is None
    original.set_result(_result(p1, [0.9, 0.0, 0.1]))
    assert stockfish_turn.poll_async_sf_labels(state) == (1, 0)
    _assert_wdl(rec, [0.1, 0.0, 0.9])
    np.testing.assert_array_equal(
        np.flatnonzero(rec.sf_legal_mask), CBoard.from_board(chess.Board(p1)).legal_move_indices(),
    )


def test_omitted_original_query_does_not_rebind_after_forced_move(monkeypatch):
    pool = HeldPool()
    state = _state(pool)
    original = stockfish_turn.submit_async_sf_label_queries
    suppressed: list[list[int]] = []

    def omit_original_label(_state, idxs):
        suppressed.append(list(idxs))
        return 0

    # Model missing P1 admission independently of the corrected queue cap.
    # Actual cap spill and unrelated-future custody have separate controls.
    _choose(monkeypatch, "c1a1", chess.WHITE)
    with monkeypatch.context() as admission:
        admission.setattr(stockfish_turn, "submit_async_sf_label_queries", omit_original_label)
        if getattr(manager, "submit_async_sf_label_queries", None) is original:
            admission.setattr(manager, "submit_async_sf_label_queries", omit_original_label)
        _step(state)
    assert suppressed == [[0]]
    assert stockfish_turn.submit_async_sf_label_queries is original
    rec = state.samples_per_game[0][-1]
    assert not pool.queries
    assert not state.pending_sf_labels

    # Restored admission at the forced reply must reject the earlier row.
    _step(state, opponent=True)
    assert not pool.queries
    assert state.samples_per_game[0] == [rec]
    assert rec.sf_wdl is None

    # A later valid NET record still admits its own original-position query.
    _choose(monkeypatch, "a1a2", chess.WHITE)
    _step(state)
    later = state.samples_per_game[0][-1]
    assert later is not rec
    assert len(pool.queries) == 1
    fen, _, future = pool.queries[0]
    future.set_result(_result(fen, [0.2, 0.0, 0.8]))
    assert stockfish_turn.poll_async_sf_labels(state) == (1, 0)
    _assert_wdl(later, [0.8, 0.0, 0.2])
    assert rec.sf_wdl is None
    assert rec.sf_move_index is None


def test_distinct_records_receive_out_of_order_labels_for_their_own_positions(monkeypatch):
    pool = HeldPool()
    state = _state(pool)
    _choose(monkeypatch, "c1a1", chess.WHITE)
    _step(state)
    first = state.samples_per_game[0][-1]
    _step(state, opponent=True)
    _choose(monkeypatch, "a1a2", chess.WHITE)
    _step(state)
    second = state.samples_per_game[0][-1]
    assert first is not second
    assert len(pool.queries) == 2
    p1, _, f1 = pool.queries[0]
    p3, _, f3 = pool.queries[1]
    f3.set_result(_result(p3, [0.2, 0.0, 0.8]))
    assert stockfish_turn.poll_async_sf_labels(state) == (1, 0)
    assert first.sf_wdl is None
    _assert_wdl(second, [0.8, 0.0, 0.2])
    f1.set_result(_result(p1, [0.9, 0.0, 0.1]))
    assert stockfish_turn.poll_async_sf_labels(state) == (1, 0)
    _assert_wdl(first, [0.1, 0.0, 0.9])
    _assert_wdl(second, [0.8, 0.0, 0.2])


def test_fast_record_does_not_trigger_label_query(monkeypatch):
    pool = HeldPool()
    state = _state(pool, full=False)
    _choose(monkeypatch, "c1a1", chess.WHITE)
    _step(state)
    assert not state.samples_per_game[0][-1].has_policy
    _step(state, opponent=True)
    assert not pool.queries


def test_synchronous_forced_turn_does_not_label_old_unlabeled_record(monkeypatch):
    sf = RecordingSync()
    state = _state(sf)
    _choose(monkeypatch, "c1a1", chess.WHITE)
    _step(state)
    rec = state.samples_per_game[0][-1]
    assert len(sf.queries) == 1
    _assert_wdl(rec, [0.1, 0.0, 0.9])
    # Simulate a failed/missing original label in the synchronous path.
    rec.sf_policy_target = rec.sf_move_index = rec.sf_wdl = None
    _step(state, opponent=True)
    assert len(sf.queries) == 1
    assert rec.sf_wdl is None


def test_seven_piece_fixture_passes_six_man_adjudication_prefilter(monkeypatch):
    state = _state(HeldPool())
    state.game = replace(state.game, syzygy_path="unused-six-man-tables")
    state.tb_probe = cast(Any, SimpleNamespace(max_pieces=6))
    state.tb_adj_roll_arr[0] = 1

    def unexpected_probe(*args):
        del args
        pytest.fail("Seven-piece fixture must not access six-man tables")

    monkeypatch.setattr(manager, "tb_adjudicate_result", unexpected_probe)
    assert manager._tb_adjudicate_active_games(state) == 0
    _choose(monkeypatch, "c1a1", chess.WHITE)
    _step(state)
    assert manager._tb_adjudicate_active_games(state) == 0
    _step(state, opponent=True)
    assert manager._tb_adjudicate_active_games(state) == 0
    assert not state.done_arr[0]
