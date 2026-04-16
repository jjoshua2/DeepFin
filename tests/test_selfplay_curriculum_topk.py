from __future__ import annotations

import numpy as np

from chess_anti_engine.moves.encode import uci_to_policy_index
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


# Helper: build legal_indices array for the starting position (white to move).
def _starting_legal_indices() -> np.ndarray:
    import chess
    from chess_anti_engine.encoding._lc0_ext import CBoard
    cb = CBoard.from_board(chess.Board())
    return cb.legal_move_indices()


def test_choose_curriculum_opponent_move_regret_filter_non_random() -> None:
    """Non-random path picks uniformly among moves within the regret band."""
    legal_indices = _starting_legal_indices()
    turn = True  # white
    cand_indices = [
        uci_to_policy_index("e2e4", turn),
        uci_to_policy_index("d2d4", turn),
        uci_to_policy_index("g1f3", turn),
    ]
    cand_scores = [0.80, 0.795, 0.60]
    rng = np.random.default_rng(0)

    picked = {
        _choose_curriculum_opponent_move(
            rng=rng,
            legal_indices=legal_indices,
            cand_indices=cand_indices,
            cand_scores=cand_scores,
            curriculum_topk=3,
            random_move_prob=0.0,
            regret_limit=0.01,
        )
        for _ in range(200)
    }

    e2e4 = uci_to_policy_index("e2e4", turn)
    d2d4 = uci_to_policy_index("d2d4", turn)
    g1f3 = uci_to_policy_index("g1f3", turn)

    assert picked <= {e2e4, d2d4}
    assert g1f3 not in picked
    assert e2e4 in picked
    assert d2d4 in picked


def test_choose_curriculum_opponent_move_random_is_truly_random() -> None:
    """Random corruption picks a truly random legal move (not just acceptable ones)."""
    legal_indices = _starting_legal_indices()
    turn = True
    cand_indices = [
        uci_to_policy_index("e2e4", turn),
        uci_to_policy_index("d2d4", turn),
    ]
    cand_scores = [0.80, 0.795]
    rng = np.random.default_rng(42)

    picked = {
        _choose_curriculum_opponent_move(
            rng=rng,
            legal_indices=legal_indices,
            cand_indices=cand_indices,
            cand_scores=cand_scores,
            curriculum_topk=2,
            random_move_prob=1.0,
            regret_limit=0.01,
        )
        for _ in range(500)
    }

    # Should pick from ALL legal moves, not just candidates
    assert len(picked) > 2


def test_choose_curriculum_opponent_move_regret_pv_order_not_wdl_order() -> None:
    """When PV order differs from WDL score order, regret uses max WDL score."""
    legal_indices = _starting_legal_indices()
    turn = True
    cand_indices = [
        uci_to_policy_index("e2e4", turn),
        uci_to_policy_index("d2d4", turn),
        uci_to_policy_index("g1f3", turn),
    ]
    cand_scores = [0.60, 0.595, 0.80]  # PV3 has highest WDL
    rng = np.random.default_rng(0)

    picked = {
        _choose_curriculum_opponent_move(
            rng=rng,
            legal_indices=legal_indices,
            cand_indices=cand_indices,
            cand_scores=cand_scores,
            curriculum_topk=3,
            random_move_prob=0.0,
            regret_limit=0.01,
        )
        for _ in range(200)
    }

    g1f3 = uci_to_policy_index("g1f3", turn)
    assert picked == {g1f3}
