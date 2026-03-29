from __future__ import annotations

import chess
import numpy as np

from chess_anti_engine.selfplay.manager import (
    _choose_curriculum_opponent_move,
    _effective_curriculum_topk,
    _effective_curriculum_wdl_regret,
)


def test_effective_curriculum_topk_shrinks_as_random_stage_burns_off() -> None:
    assert _effective_curriculum_topk(random_move_prob=0.8, stage_end=0.5, topk_max=12) == 12
    assert _effective_curriculum_topk(random_move_prob=0.5, stage_end=0.5, topk_max=12) == 12
    assert _effective_curriculum_topk(random_move_prob=0.4, stage_end=0.5, topk_max=12) == 7
    assert _effective_curriculum_topk(random_move_prob=0.2, stage_end=0.5, topk_max=12) == 3
    assert _effective_curriculum_topk(random_move_prob=0.0, stage_end=0.5, topk_max=12) == 1


def test_effective_curriculum_topk_respects_topk_floor() -> None:
    assert _effective_curriculum_topk(random_move_prob=0.10, stage_end=0.5, topk_max=12, topk_min=2) == 3
    assert _effective_curriculum_topk(random_move_prob=0.05, stage_end=0.5, topk_max=12, topk_min=2) == 2
    assert _effective_curriculum_topk(random_move_prob=0.0, stage_end=0.5, topk_max=12, topk_min=2) == 2


def test_effective_curriculum_wdl_regret_tightens_with_pid() -> None:
    assert _effective_curriculum_wdl_regret(
        random_move_prob=0.32,
        random_move_prob_start=0.32,
        random_move_prob_floor=0.05,
        regret_max=0.50,
        regret_min=0.01,
    ) == 0.50
    assert _effective_curriculum_wdl_regret(
        random_move_prob=0.05,
        random_move_prob_start=0.32,
        random_move_prob_floor=0.05,
        regret_max=0.50,
        regret_min=0.01,
    ) == 0.01


def test_choose_curriculum_opponent_move_regret_filter_non_random() -> None:
    """Non-random path picks uniformly among moves within the regret band."""
    board = chess.Board()
    legal_moves = list(board.legal_moves)
    cand_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4"), chess.Move.from_uci("g1f3")]
    cand_scores = [0.80, 0.795, 0.60]
    rng = np.random.default_rng(0)

    picked = {
        _choose_curriculum_opponent_move(
            rng=rng,
            legal_moves=legal_moves,
            cand_moves=cand_moves,
            cand_scores=cand_scores,
            curriculum_topk=3,
            random_move_prob=0.0,  # no random corruption
            regret_limit=0.01,
        ).uci()
        for _ in range(200)
    }

    # Only e2e4 and d2d4 are within 0.01 regret of the best
    assert picked <= {"e2e4", "d2d4"}
    assert "g1f3" not in picked
    # Should sample both, not just the best
    assert "e2e4" in picked
    assert "d2d4" in picked


def test_choose_curriculum_opponent_move_random_is_truly_random() -> None:
    """Random corruption picks a truly random legal move (not just acceptable ones)."""
    board = chess.Board()
    legal_moves = list(board.legal_moves)
    cand_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    cand_scores = [0.80, 0.795]
    rng = np.random.default_rng(42)

    picked = {
        _choose_curriculum_opponent_move(
            rng=rng,
            legal_moves=legal_moves,
            cand_moves=cand_moves,
            cand_scores=cand_scores,
            curriculum_topk=2,
            random_move_prob=1.0,  # always random
            regret_limit=0.01,
        ).uci()
        for _ in range(500)
    }

    # Should pick from ALL legal moves, not just candidates
    assert len(picked) > 2


def test_choose_curriculum_opponent_move_regret_pv_order_not_wdl_order() -> None:
    """When PV order differs from WDL score order, regret uses max WDL score."""
    board = chess.Board()
    legal_moves = list(board.legal_moves)
    # Simulate SF PV order ≠ WDL order: PV1 has LOWER WDL than PV3
    cand_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4"), chess.Move.from_uci("g1f3")]
    cand_scores = [0.60, 0.595, 0.80]  # PV3 has highest WDL
    rng = np.random.default_rng(0)

    picked = {
        _choose_curriculum_opponent_move(
            rng=rng,
            legal_moves=legal_moves,
            cand_moves=cand_moves,
            cand_scores=cand_scores,
            curriculum_topk=3,
            random_move_prob=0.0,
            regret_limit=0.01,
        ).uci()
        for _ in range(200)
    }

    # Only g1f3 (0.80) is within 0.01 of max score (0.80)
    assert picked == {"g1f3"}
