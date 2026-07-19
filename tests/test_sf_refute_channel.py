"""SF-refutation channel: short SF-as-opponent seed games + handoff to selfplay."""

from __future__ import annotations

import chess
import numpy as np
from unittest.mock import MagicMock

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.opening import (
    OpeningConfig,
    sample_sf_refute_batch,
    try_take_sf_refute_opening,
)
from chess_anti_engine.selfplay.state import SelfplayState
from chess_anti_engine.selfplay.stockfish_turn import _push_curriculum_opponent_move


def test_sample_sf_refute_batch_round_robin_covers_all() -> None:
    seeds = [f"s{i}" for i in range(10)]
    # frac=0.5 → K=5; two consecutive iters should cover all 10 without overlap.
    a = sample_sf_refute_batch(seeds, frac=0.5, training_iteration=0)
    b = sample_sf_refute_batch(seeds, frac=0.5, training_iteration=1)
    assert len(a) == 5
    assert len(b) == 5
    assert set(a) | set(b) == set(seeds)
    assert set(a).isdisjoint(set(b))
    # Same iter is deterministic.
    assert sample_sf_refute_batch(seeds, frac=0.5, training_iteration=0) == a
    assert sample_sf_refute_batch(seeds, frac=0.0, training_iteration=0) == []
    assert sample_sf_refute_batch([], frac=0.5, training_iteration=0) == []


def test_sample_sf_refute_batch_full_frac() -> None:
    seeds = ["a", "b", "c"]
    assert sample_sf_refute_batch(seeds, frac=1.0, training_iteration=7) == seeds


def test_try_take_sf_refute_opening_pops_and_tags() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    q = [fen, fen]
    cfg = OpeningConfig(
        opening_fen_list_path="/nonexistent/list.txt",
        opening_fen_sf_refute_frac=0.5,
        opening_fen_sf_refute_plies=5,
    )
    start = try_take_sf_refute_opening(cfg=cfg, fen_sf_refute_queue=q)
    assert start is not None
    assert start.source.startswith("fenlist_sf_refute")
    assert len(q) == 1
    # Channel off — must not consume.
    cfg_off = OpeningConfig(
        opening_fen_list_path="/nonexistent/list.txt",
        opening_fen_sf_refute_frac=0.0,
        opening_fen_sf_refute_plies=5,
    )
    assert try_take_sf_refute_opening(cfg=cfg_off, fen_sf_refute_queue=q) is None
    assert len(q) == 1


def test_try_take_sf_refute_opening_empty_queue() -> None:
    cfg = OpeningConfig(
        opening_fen_list_path="/nonexistent/list.txt",
        opening_fen_sf_refute_frac=0.5,
        opening_fen_sf_refute_plies=5,
    )
    assert try_take_sf_refute_opening(cfg=cfg, fen_sf_refute_queue=[]) is None
    assert try_take_sf_refute_opening(cfg=cfg, fen_sf_refute_queue=None) is None


def test_sf_refute_handoff_flips_to_selfplay_after_m_plies() -> None:
    """After M curriculum SF pushes, selfplay_arr becomes 1 (net-net handoff)."""
    n = 1
    board = chess.Board()
    # Minimal SelfplayState: only fields used by _push_curriculum_opponent_move.
    state = SelfplayState(
        device="cpu",
        rng=np.random.default_rng(0),
        stockfish=MagicMock(),
        evaluator=MagicMock(),
        model=None,
        opponent=OpponentConfig(),
        temp=TemperatureConfig(),
        search=SearchConfig(),
        opening=OpeningConfig(opening_fen_sf_refute_plies=3),
        diff_focus=DiffFocusConfig(),
        game=GameConfig(),
        batch_size=n,
        continuous=False,
        target=1,
        volatility_source="raw",
        base_nodes=1000,
        terminal_eval_nodes=1000,
        done_arr=np.zeros(n, dtype=np.int8),
        finalized_arr=np.zeros(n, dtype=np.int8),
        net_color_arr=np.ones(n, dtype=np.int8),
        selfplay_arr=np.zeros(n, dtype=np.int8),  # curriculum
        opening_source_arr=["fenlist_sf_refute"],
        boards=[board],
        cboards=[CBoard.from_board(board)],
        starting_boards=None,
        starting_ply_arr=np.zeros(n, dtype=np.int32),
        move_idx_history=[[] for _ in range(n)],
        samples_per_game=[[] for _ in range(n)],
        consecutive_low_winrate=[0] * n,
        last_net_full=[True] * n,
        force_full_next=[False] * n,
        root_ids=[-1] * n,
        pending_sf_labels=[],
        pending_sf_moves={},
        sf_label_escalations=[0] * n,
        tb_probe=None,
        tb_result_arr=[None] * n,
        tb_adj_roll_arr=np.zeros(n, dtype=np.int8),
        fen_dole_queue=None,
        fen_sf_refute_queue=None,
        sf_refute_opp_plies_left=np.array([3], dtype=np.int32),
        games_started=1,
        games_completed=0,
        mcts_tree=None,
    )
    legal = state.cboards[0].legal_move_indices()
    assert legal.size > 0
    for expect_left in (2, 1, 0):
        legal = state.cboards[0].legal_move_indices()
        assert legal.size > 0
        _push_curriculum_opponent_move(
            state, 0,
            legal_indices=legal,
            cand_idxs=[int(legal[0])],
            cand_scores=[0.0],
            regret_limit=float("inf"),
        )
        assert int(state.sf_refute_opp_plies_left[0]) == expect_left
    assert int(state.selfplay_arr[0]) == 1  # handed off to selfplay
