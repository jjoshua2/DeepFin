from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.selfplay.config import GameConfig
from chess_anti_engine.selfplay.config import OpponentConfig
from chess_anti_engine.selfplay.state import _NetRecord
from chess_anti_engine.selfplay.stockfish_turn import (
    flush_async_sf_labels_for_records,
    finish_pending_curriculum_moves,
    submit_async_curriculum_move_queries,
    submit_async_sf_labels_from_curriculum_moves,
    submit_async_sf_label_queries,
    submit_sf_queries,
)
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishResult


class _FakeCBoard:
    turn = True
    occ_white = 0
    occ_black = 0
    castling = 0

    def __init__(self) -> None:
        self.pushed: list[int] = []

    def fen(self) -> str:
        return chess.STARTING_FEN

    def legal_move_indices(self) -> np.ndarray:
        return np.array([0], dtype=np.int64)

    def push_index(self, idx: int) -> None:
        self.pushed.append(int(idx))
        self.turn = not self.turn

    def is_game_over(self) -> bool:
        return False


class _FakePool(StockfishPool):
    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.calls: list[dict] = []

    def submit(self, fen: str, *, nodes=None, syzygy_path=None):  # noqa: ANN001
        self.calls.append({"fen": fen, "nodes": nodes, "syzygy_path": syzygy_path})
        fut: Future = Future()
        fut.set_result(StockfishResult(bestmove_uci="a2a3", wdl=np.array([0.0, 1.0, 0.0]), pvs=[]))
        return fut


def _record(*, has_policy: bool) -> _NetRecord:
    return _NetRecord(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_probs=np.zeros((POLICY_SIZE,), dtype=np.float32),
        net_wdl_est=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        search_wdl_est=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        pov_color=chess.WHITE,
        ply_index=0,
        has_policy=has_policy,
        priority=1.0,
        sample_weight=1.0,
        keep_prob=1.0,
        legal_mask=np.zeros((POLICY_SIZE,), dtype=np.uint8),
    )


def _state(*, has_policy: bool, sf_move_nodes: int = 0) -> Any:
    cboard = _FakeCBoard()
    return SimpleNamespace(
        batch_size=1,
        pending_sf_labels=[],
        pending_sf_moves={},
        stockfish=_FakePool(),
        samples_per_game=[[_record(has_policy=has_policy)]],
        cboards=[cboard],
        base_nodes=100,
        last_net_full=[True],
        tb_adj_roll_arr=[True],
        game=GameConfig(sf_move_nodes=sf_move_nodes),
        rng=np.random.default_rng(0),
        opponent=OpponentConfig(),
        move_idx_history=[[]],
        mcts_tree=None,
        root_ids=[-1],
        done_arr=np.zeros((1,), dtype=np.int8),
        finalized_arr=np.zeros((1,), dtype=np.int8),
        selfplay_arr=np.zeros((1,), dtype=np.int8),
    )


def test_async_sf_labels_skip_records_discarded_by_finalize() -> None:
    state = _state(has_policy=False)

    submitted = submit_async_sf_label_queries(state, [0])

    assert submitted == 0
    assert state.pending_sf_labels == []
    assert state.stockfish.calls == []


def test_async_sf_labels_submit_for_replay_kept_records() -> None:
    state = _state(has_policy=True)

    submitted = submit_async_sf_label_queries(state, [0])

    assert submitted == 1
    assert len(state.pending_sf_labels) == 1
    assert state.stockfish.calls[0]["nodes"] == 100


def test_finalize_flush_waits_for_pending_replay_kept_labels() -> None:
    state = _state(has_policy=True)
    rec = state.samples_per_game[0][0]
    submitted = submit_async_sf_label_queries(state, [0])

    attached, failed = flush_async_sf_labels_for_records(state, [rec])

    assert submitted == 1
    assert (attached, failed) == (1, 0)
    assert state.pending_sf_labels == []
    assert rec.sf_policy_target is not None
    assert rec.sf_move_index is not None
    assert rec.sf_wdl is not None


def test_curriculum_move_queries_can_use_separate_low_node_budget() -> None:
    state = _state(has_policy=True, sf_move_nodes=10)

    submit_sf_queries(state, [0], for_move=True)

    assert state.stockfish.calls[0]["nodes"] == 10


def test_pending_curriculum_move_applies_without_stamping_low_node_label() -> None:
    state = _state(has_policy=True, sf_move_nodes=10)
    rec = state.samples_per_game[0][0]

    submitted = submit_async_curriculum_move_queries(state, [0])
    completed = finish_pending_curriculum_moves(state, block=False)

    assert submitted == 1
    assert completed == 1
    assert state.pending_sf_moves == {}
    assert state.cboards[0].pushed == [0]
    assert state.move_idx_history[0] == [0]
    assert rec.sf_policy_target is None
    assert rec.sf_move_index is None


def test_curriculum_move_future_reused_for_label_when_nodes_are_shared() -> None:
    state = _state(has_policy=True, sf_move_nodes=0)
    rec = state.samples_per_game[0][0]

    submitted_moves = submit_async_curriculum_move_queries(state, [0])
    submitted_labels = submit_async_sf_labels_from_curriculum_moves(state, [0])
    attached, failed = flush_async_sf_labels_for_records(state, [rec])

    assert submitted_moves == 1
    assert submitted_labels == 1
    assert (attached, failed) == (1, 0)
    assert len(state.stockfish.calls) == 1
    assert rec.sf_policy_target is not None


def test_curriculum_label_uses_separate_query_when_move_nodes_are_low() -> None:
    state = _state(has_policy=True, sf_move_nodes=10)
    rec = state.samples_per_game[0][0]

    submitted_moves = submit_async_curriculum_move_queries(state, [0])
    submitted_labels = submit_async_sf_labels_from_curriculum_moves(state, [0])
    attached, failed = flush_async_sf_labels_for_records(state, [rec])

    assert submitted_moves == 1
    assert submitted_labels == 1
    assert (attached, failed) == (1, 0)
    assert [call["nodes"] for call in state.stockfish.calls] == [10, 100]
