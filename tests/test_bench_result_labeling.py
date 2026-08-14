from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.bench.play_batch_timing import (
    _result_to_wdl,
    _surprise_priority,
)
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask
from chess_anti_engine.selfplay.network_turn import _policy_kl


def test_bench_result_to_wdl_treats_truncated_as_draw():
    assert _result_to_wdl("*", pov_white=True) == 1
    assert _result_to_wdl("*", pov_white=False) == 1


def test_bench_result_to_wdl_standard_results():
    assert _result_to_wdl("1/2-1/2", pov_white=True) == 1
    assert _result_to_wdl("1/2-1/2", pov_white=False) == 1

    assert _result_to_wdl("1-0", pov_white=True) == 0
    assert _result_to_wdl("1-0", pov_white=False) == 2

    assert _result_to_wdl("0-1", pov_white=True) == 2
    assert _result_to_wdl("0-1", pov_white=False) == 0


# ── task #173: one KL convention, not three ─────────────────────────────────
#
# `_mcts_tree.c` and `network_turn._policy_kl` were unified on the
# support-restricted convention (skip a term when either side underflows).
# `_surprise_priority` kept the pre-fix formula — floor BOTH sides at 1e-12 and
# sum over all 4672 entries — for four more months. It is not a rounding
# difference: the extra mass is dominated by `-log(1e-12) = 27.63` over moves
# the search never visited, so the statistic's magnitude is a free parameter of
# its own implementation. Measured on the 12 newest live shards (2026-08-12,
# sims 100 / topk 16): 11,710 of 22,265 policy rows (52.6%) carry at least one
# legal move at zero visit mass, which is exactly when the two disagree.


def _sparse_case(seed: int) -> tuple[np.ndarray, np.ndarray, chess.Board]:
    """A real position with a Gumbel-shaped visit distribution: most legal moves
    get exactly zero visit mass."""
    rng = np.random.default_rng(seed)
    board = chess.Board()
    for _ in range(6):
        moves = list(board.legal_moves)
        board.push(moves[int(rng.integers(len(moves)))])
    logits = (rng.standard_normal(POLICY_SIZE) * 2.0).astype(np.float32)
    legal = np.flatnonzero(legal_move_mask(board))
    probs = np.zeros(POLICY_SIZE, dtype=np.float32)
    chosen = rng.choice(legal, size=max(2, len(legal) // 4), replace=False)
    visits = rng.integers(1, 40, size=chosen.size).astype(np.float32)
    probs[chosen] = visits / visits.sum()
    return logits, probs, board


def _prior(logits: np.ndarray, board: chess.Board) -> np.ndarray:
    mask = legal_move_mask(board)
    lg = logits.astype(np.float64, copy=True)
    lg[~mask] = -1e9
    lg -= float(np.max(lg))
    e = np.exp(lg)
    e[~mask] = 0.0
    return (e / e.sum()).astype(np.float32)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_bench_surprise_priority_uses_the_production_kl(seed: int) -> None:
    """The bench must rank positions by the statistic production stores, not by
    a second one that happens to share its name."""
    logits, probs, board = _sparse_case(seed)
    expected = _policy_kl(_prior(logits, board), probs, legal_move_mask(board))
    assert _surprise_priority(logits, probs, board) == pytest.approx(
        expected, rel=1e-6, abs=1e-9,
    )


def test_the_old_floored_bench_kl_really_did_diverge() -> None:
    """Guards the test above from being vacuous: if the pre-fix formula agreed
    anyway, the parity assertion would pass on a broken implementation too."""
    diverged = 0
    seeds = range(1, 33)
    for seed in seeds:
        logits, probs, board = _sparse_case(seed)
        prior = _prior(logits, board).astype(np.float64)
        imp = np.maximum(probs.astype(np.float64), 1e-12)
        pc = np.maximum(prior, 1e-12)
        old = float(np.sum(pc * (np.log(pc) - np.log(imp))))
        if abs(old - _surprise_priority(logits, probs, board)) > 1e-3:
            diverged += 1
    assert diverged > len(list(seeds)) // 2, (
        f"only {diverged} of {len(list(seeds))} fixtures diverged under the OLD "
        "floored formula; the sparse fixture has stopped reproducing the defect, "
        "so the parity test above proves nothing"
    )
