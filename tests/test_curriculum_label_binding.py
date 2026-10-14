from __future__ import annotations


import chess
import numpy as np
import pytest

from chess_anti_engine.selfplay import manager, stockfish_turn
from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.state import SelfplayState
from chess_anti_engine.stockfish.uci import StockfishResult
from tests.selfplay_helpers import UniformPolicyValueModel
from tests.test_selfplay_label_binding import HeldPool, _assert_wdl, _choose


FEN = "k7/7p/1K6/7P/8/8/2P1PP2/2R5 b - - 0 1"


def _state(tmp_path, *, refute=False, c_ply=True, mode="move", opp_rows=False, full=True):
    seed = tmp_path / "curriculum-seed.fen"
    seed.write_text(FEN + "\n")
    pool = HeldPool()
    pool.nodes = 5000
    state = SelfplayState.create(
        model=UniformPolicyValueModel().eval(), device="cpu",
        rng=np.random.default_rng(0), stockfish=pool, evaluator=None,
        batch_size=1, continuous=False, target=1,
        opponent=OpponentConfig(), temp=TemperatureConfig(temperature=0.0),
        search=SearchConfig(simulations=1, fast_simulations=1, playout_cap_fraction=float(full)),
        opening=OpeningConfig(
            random_start_plies=0, opening_fen_list_path=str(seed), opening_fen_prob=1.0,
            opening_fen_sf_refute_frac=1.0 if refute else 0.0,
            opening_fen_sf_refute_plies=3, sf_refute_record_opp_rows=opp_rows,
        ),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(
            selfplay_fraction=float(refute),
            sf_move_nodes=5000 if mode == "move" else 0,
            sf_label_nodes_floor=10000 if mode == "floor" else 0,
        ),
        fen_sf_refute_queue=[FEN] if refute else None,
    )
    state.has_c_ply = c_ply
    assert state.cboards[0].fen() == FEN
    assert state.net_color_arr[0] == 0
    assert bool(state.selfplay_arr[0]) == refute
    if refute:
        assert state.sf_refute_opp_plies_left[0] == 3
    return state, pool


def _step(state: SelfplayState):
    net, sp, cur, done = state.classify_active_slots()
    assert not done
    cur = [i for i in cur if i not in state.pending_sf_moves]
    manager._run_step(state, net_idxs=net, sp_opp_idxs=sp, cur_opp_idxs=cur,
                      stockfish=state.stockfish)
    return net, sp, cur


def _result(fen: str, uci: str, wdl: list[float]) -> StockfishResult:
    board = chess.Board(fen)
    assert chess.Move.from_uci(uci) in board.legal_moves
    return StockfishResult(bestmove_uci=uci, wdl=np.asarray(wdl, dtype=np.float32), pvs=[])


def _finish_move(state: SelfplayState, pool: HeldPool, uci: str):
    future = state.pending_sf_moves[0]
    fen = next(fen for fen, _, f in pool.queries if f is future)
    future.set_result(_result(fen, uci, [0.9, 0.0, 0.1]))
    # The real driver polls labels first, then applies ready moves. A separate
    # held label does not complete merely because this move has completed.
    stockfish_turn.poll_async_sf_labels(state)
    assert stockfish_turn.finish_pending_curriculum_moves(state) == 1


def _through_forced_turn(state, pool, monkeypatch):
    _choose(monkeypatch, "h7h6", chess.BLACK)
    assert _step(state) == ([0], [], [])
    rec = state.samples_per_game[0][-1]
    assert rec.ply_index == 1
    assert rec.pov_color == chess.BLACK
    p1 = state.cboards[0].fen()
    assert _step(state) == ([], [], [0])
    correct = [p for p in state.pending_sf_labels if p.record is rec]
    _finish_move(state, pool, "c1g1")
    board = chess.Board(state.cboards[0].fen())
    assert board.is_valid()
    assert len(board.piece_map()) == 8
    assert [m.uci() for m in board.legal_moves] == ["a8b8"]
    assert _step(state) == ([0], [], [])
    assert not state.done_arr[0]
    assert state.cboards[0].ply == 4
    assert chess.Board(state.cboards[0].fen()).turn == chess.WHITE
    return rec, p1, correct


@pytest.mark.parametrize("refute", [False, True])
@pytest.mark.parametrize("c_ply", [False, True])
@pytest.mark.parametrize("mode", ["move", "floor"])
def test_delayed_curriculum_label_is_not_rebound_after_forced_net_turn(
    tmp_path, monkeypatch, refute, c_ply, mode,
):
    state, pool = _state(tmp_path, refute=refute, c_ply=c_ply, mode=mode)
    rec, p1, correct = _through_forced_turn(state, pool, monkeypatch)
    assert correct
    assert all(p.query_fen == p1 for p in correct)
    assert state.samples_per_game[0] == [rec]
    assert rec.sf_wdl is None
    assert _step(state) == ([], [], [0])
    bound_to_rec = [p for p in state.pending_sf_labels if p.record is rec]
    assert bound_to_rec == correct
    assert all(p.query_fen == p1 for p in bound_to_rec)

    # Playing the next SF reply must remain independent of old label latency.
    _finish_move(state, pool, "g1g8")
    for pending in correct:
        pending.future.set_result(_result(p1, "c1g1", [0.9, 0.0, 0.1]))
    stockfish_turn.poll_async_sf_labels(state)
    _assert_wdl(rec, [0.1, 0.0, 0.9])


@pytest.mark.parametrize("c_ply", [False, True])
def test_shared_move_future_keeps_correct_ordinary_label(tmp_path, monkeypatch, c_ply):
    state, pool = _state(tmp_path, c_ply=c_ply, mode="reuse")
    rec, _, correct = _through_forced_turn(state, pool, monkeypatch)
    assert len(correct) == 1
    _assert_wdl(rec, [0.1, 0.0, 0.9])
    assert not state.pending_sf_labels
    _step(state)
    assert not state.pending_sf_labels


def test_omitted_label_cannot_bind_old_record_to_later_shared_move(tmp_path, monkeypatch):
    state, pool = _state(tmp_path, mode="reuse")
    _choose(monkeypatch, "h7h6", chess.BLACK)
    _step(state)
    rec = state.samples_per_game[0][-1]
    suppressed: list[list[int]] = []
    original = stockfish_turn.submit_async_sf_labels_from_curriculum_moves

    def omit_original_label(_state, idxs):
        suppressed.append(list(idxs))
        return 0

    # Model intentional non-admission at P1 independently of queue capacity.
    # The real reuse spill/barrier remains covered by the label gating suite.
    with monkeypatch.context() as admission:
        admission.setattr(stockfish_turn, "submit_async_sf_labels_from_curriculum_moves", omit_original_label)
        if getattr(manager, "submit_async_sf_labels_from_curriculum_moves", None) is original:
            admission.setattr(manager, "submit_async_sf_labels_from_curriculum_moves", omit_original_label)
        _step(state)
    assert suppressed == [[0]]
    assert not state.pending_sf_labels
    _finish_move(state, pool, "c1g1")
    _step(state)
    # Restored admission sees the later position and must reject the old row.
    _step(state)
    assert not state.pending_sf_labels
    _finish_move(state, pool, "g1g8")
    assert rec.sf_wdl is None
    assert rec.sf_move_index is None


def test_refute_opponent_rows_keep_existing_exclusion(tmp_path, monkeypatch):
    state, pool = _state(tmp_path, refute=True, opp_rows=True)
    rec, _, correct = _through_forced_turn(state, pool, monkeypatch)
    assert state.samples_per_game[0][-1].is_sf_refute_opp
    first_opp = state.samples_per_game[0][-1]
    _step(state)
    assert [p for p in state.pending_sf_labels if p.record is rec] == correct
    _finish_move(state, pool, "g1g8")
    assert first_opp.sf_played_move_index == uci_to_policy_index("c1g1", chess.WHITE)
    assert state.samples_per_game[0][-1].sf_played_move_index == uci_to_policy_index("g1g8", chess.WHITE)


@pytest.mark.parametrize("full", [False, True])
def test_forced_turn_does_not_overwrite_original_played_reply(tmp_path, monkeypatch, full):
    state, pool = _state(tmp_path, mode="reuse", full=full)
    rec, _, _ = _through_forced_turn(state, pool, monkeypatch)
    original = rec.sf_played_move_index
    assert original == uci_to_policy_index("c1g1", chess.WHITE)
    _step(state)
    _finish_move(state, pool, "g1g8")
    assert rec.sf_played_move_index == original


def test_synchronous_curriculum_result_cannot_label_old_record(tmp_path, monkeypatch):
    state, pool = _state(tmp_path)
    rec, _, correct = _through_forced_turn(state, pool, monkeypatch)
    for pending in correct:
        pending.future.set_exception(RuntimeError("first label failed"))
    stockfish_turn.poll_async_sf_labels(state)
    assert not state.pending_sf_labels
    assert rec.sf_wdl is None
    original = rec.sf_played_move_index
    p3 = state.cboards[0].fen()
    stockfish_turn._process_sf_results(
        state, [0], results={0: _result(p3, "g1g8", [0.2, 0.0, 0.8])},
        play_curriculum_moves=True,
    )
    assert state.cboards[0].ply == 5
    assert state.done_arr[0]
    assert rec.sf_wdl is None
    assert rec.sf_move_index is None
    assert rec.sf_played_move_index == original
