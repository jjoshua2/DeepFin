"""``GumbelConfig.root_expand_all``: one root eval for every legal move.

The flag exists because an unvisited legal move enters the improved-policy softmax
with ``completed_q = root_q``, an IDENTICAL constant for every such move, so the
target's ranking over that tail is purely the net prior -- and CE then trains the
net toward its own prior there, a fixed point a wrongly-low prior cannot leave.

The load-bearing property is the NO-OP one. Whenever every legal move is already a
Gumbel candidate the flag must change precisely nothing, bit for bit; if it does not,
the implementation is perturbing the search rather than only widening round 1, and no
number measured with it can be trusted.
"""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import GumbelConfig, run_gumbel_root_many
from chess_anti_engine.moves import POLICY_SIZE

# 20 legal moves (<= topk 32, flag must be inert) vs 45 (> topk, flag must act).
# Neither position has a legal move leading to a terminal node -- asserted in the
# premise test -- so every tail move costs exactly one NET EVALUATION and the
# eval-count assertions below can demand EXACT equality rather than a bound.
NARROW = chess.Board()
WIDE = chess.Board("r3k2r/pppq1ppp/2npbn2/1B2p1B1/3PP1b1/2N2N2/PPPQ1PPP/R3K2R w KQkq - 0 1")
# Root candidates are m = max(2, min(topk, n_legal, m_cap)) with
# m_cap = (sims+1)//2 (gumbel.py:710), so WHICH of the three binds depends on the
# sim budget. Only the two production budgets are tested:
#
#   sims 256 (FULL plies, the ones we train on): m_cap 128, so topk(32) binds
#   sims  32 (FAST plies):                       m_cap  16, so m_cap binds and the
#                                                prior-only tail is LARGER
#
# Intermediate budgets are deliberately not used. At sims=64 m_cap == topk == 32
# and the two constraints coincide, which makes "widen m to legal.size" (variant A)
# indistinguishable from correct behaviour -- a mutation survived exactly that way.
SIMS = 256
SIMS_FAST = 32
TOPK = 32


class _DetEval:
    """Deterministic evaluator keyed on the ENCODED position bytes.

    Keying on the input rather than on call order is what makes the bit-identity
    assertion meaningful: the extra round-0 evaluations change how many times and in
    what order positions are submitted, so an order-dependent stub would report a
    difference that came from the harness rather than from the flag.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.rows = 0

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        n = int(x.shape[0])
        self.calls += 1
        self.rows += n
        pol = np.empty((n, POLICY_SIZE), dtype=np.float32)
        wdl = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            seed = int(np.frombuffer(
                np.ascontiguousarray(x[i]).view(np.uint8).tobytes()[:8].ljust(8, b"\0"),
                dtype=np.uint64,
            )[0] % (2**32))
            rng = np.random.default_rng(seed)
            pol[i] = rng.normal(0.0, 2.0, POLICY_SIZE).astype(np.float32)
            wdl[i] = rng.normal(0.0, 1.0, 3).astype(np.float32)
        return pol, wdl


def _run(board: chess.Board, *, expand: bool, sims: int = SIMS):
    ev = _DetEval()
    cfg = GumbelConfig(
        simulations=sims, topk=TOPK, add_noise=False, root_expand_all=expand,
    )
    probs, actions, values, _masks = run_gumbel_root_many(
        None, [board.copy()], device="cpu", rng=np.random.default_rng(0),
        cfg=cfg, evaluator=ev,
    )[:4]
    return probs[0], int(actions[0]), float(values[0]), ev


def test_premise_the_two_production_budgets_bind_differently() -> None:
    """Pin which term of the three-way min binds, per budget.

    Without this the behaviour tests could pass vacuously, and a future change to
    topk or the budgets could silently move the tests into the degenerate regime
    where m_cap == topk and variant A stops being detectable.
    """
    assert (SIMS + 1) // 2 > TOPK, "at 256 sims topk must bind, not m_cap"
    assert (SIMS_FAST + 1) // 2 < TOPK, "at 32 sims m_cap must bind, not topk"
    assert NARROW.legal_moves.count() <= min(TOPK, (SIMS + 1) // 2)
    assert WIDE.legal_moves.count() > min(TOPK, (SIMS + 1) // 2)
    # At the fast budget even the narrow position has a prior-only tail.
    assert NARROW.legal_moves.count() > min(TOPK, (SIMS_FAST + 1) // 2)
    # No legal move ends the game, so tail moves are never satisfied by a terminal
    # backprop instead of a net eval. This is what licenses `extra == tail`; drop it
    # and the counts below would have to become bounds, which a PARTIAL expansion
    # silently passes (a mutation truncating the tail survived exactly that way).
    for board in (NARROW, WIDE):
        for mv in board.legal_moves:
            board.push(mv)
            over = board.is_game_over()
            board.pop()
            assert not over


def test_noop_when_every_legal_move_is_already_a_candidate() -> None:
    """THE CONTROL: n_legal <= topk => bit-identical, not merely close."""
    off_p, off_a, off_v, off_ev = _run(NARROW, expand=False)
    on_p, on_a, on_v, on_ev = _run(NARROW, expand=True)
    assert np.array_equal(off_p, on_p), (
        "root_expand_all perturbed a position where it is a no-op; "
        f"max|delta|={np.abs(off_p - on_p).max():.3e}"
    )
    assert (off_a, off_v) == (on_a, on_v)
    assert off_ev.rows == on_ev.rows, "no-op position still cost extra evaluations"


def test_tail_stops_being_a_constant_q_block_when_n_legal_exceeds_topk() -> None:
    """Where the flag acts, the target must actually change.

    Deliberately not asserting a per-move direction: a real Q can be worse than
    root_q, and correctly SUPPRESSING a bad tail move is as much the point as
    promoting a good one.
    """
    off_p, _, _, _ = _run(WIDE, expand=False)
    on_p, _, _, _ = _run(WIDE, expand=True)
    assert not np.array_equal(off_p, on_p), "flag did nothing where it should act"
    assert np.count_nonzero(off_p != on_p) >= 2


def test_extra_evaluations_are_additive_and_bounded_by_the_tail() -> None:
    """Variant B, not A: the tail is paid for on top of the halving budget.

    If the implementation had instead widened ``m`` to ``legal.size``, the eval count
    would be unchanged and the survivors would simply get fewer visits each.
    """
    _, _, _, off_ev = _run(WIDE, expand=False)
    _, _, _, on_ev = _run(WIDE, expand=True)
    tail = WIDE.legal_moves.count() - min(TOPK, (SIMS + 1) // 2)
    extra = on_ev.rows - off_ev.rows
    assert extra > 0, "variant A detected: no additional evaluations were spent"
    assert extra == tail, (
        f"expected EXACTLY one eval per prior-only move ({tail}), got {extra}: "
        "fewer means the tail is being truncated and some legal moves still have "
        "no search opinion; more means candidates are being re-walked"
    )


def test_flag_off_reproduces_itself_exactly() -> None:
    """Regression guard: the default path must be untouched and deterministic."""
    a_p, a_a, a_v, _ = _run(WIDE, expand=False)
    b_p, b_a, b_v, _ = _run(WIDE, expand=False)
    assert np.array_equal(a_p, b_p)
    assert (a_a, a_v) == (b_a, b_v)


def test_probs_stay_a_normalised_distribution() -> None:
    """Extra visits must not leak mass onto illegal moves or break normalisation."""
    for expand in (False, True):
        p, _, _, _ = _run(WIDE, expand=expand)
        assert p.min() >= 0.0
        assert float(p.sum()) == pytest.approx(1.0, abs=1e-5)


def test_fast_ply_budget_is_the_regime_where_m_cap_binds() -> None:
    """At 32 sims m_cap(16) binds below topk(32), so even NARROW has a tail.

    Production fast plies run here. They are not trained on directly, but they
    still build the tree the full-ply target is read from, so the flag must behave
    sanely in this regime rather than only in the one we happen to measure.
    """
    off_p, _, _, off_ev = _run(NARROW, expand=False, sims=SIMS_FAST)
    on_p, _, _, on_ev = _run(NARROW, expand=True, sims=SIMS_FAST)
    assert not np.array_equal(off_p, on_p), (
        "at the fast budget m_cap binds, so NARROW has non-candidates and the "
        "flag must act -- if it did not, the tail is being computed off topk alone"
    )
    tail = NARROW.legal_moves.count() - min(TOPK, (SIMS_FAST + 1) // 2)
    extra = on_ev.rows - off_ev.rows
    assert extra == tail, f"expected exactly {tail} extra evals, got {extra}"
