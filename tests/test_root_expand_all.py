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

# 20 legal moves (<= topk 32, flag must be inert) vs 48 (> topk, flag must act).
NARROW = chess.Board()
WIDE = chess.Board("r3k2r/pppq1ppp/2npbn2/1B2p1B1/3PP1b1/2N2N2/PPPQ1PPP/R3K2R w KQkq - 0 1")
# SIMS must keep m_cap = (SIMS+1)//2 STRICTLY ABOVE topk, or topk and the cap
# coincide and "widen m to legal.size" (variant A) becomes indistinguishable from
# the real thing -- a mutation that survived exactly this way at SIMS=64.
# Production runs 256 (m_cap 128); 128 keeps the same ordering at half the cost.
SIMS = 128
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


def _run(board: chess.Board, *, expand: bool):
    ev = _DetEval()
    cfg = GumbelConfig(
        simulations=SIMS, topk=TOPK, add_noise=False, root_expand_all=expand,
    )
    probs, actions, values, _masks = run_gumbel_root_many(
        None, [board.copy()], device="cpu", rng=np.random.default_rng(0),
        cfg=cfg, evaluator=ev,
    )[:4]
    return probs[0], int(actions[0]), float(values[0]), ev


def test_premise_narrow_fits_in_topk_and_wide_does_not() -> None:
    """Without this both behaviour tests below could pass vacuously."""
    assert NARROW.legal_moves.count() <= min(TOPK, (SIMS + 1) // 2)
    assert WIDE.legal_moves.count() > min(TOPK, (SIMS + 1) // 2)
    assert (SIMS + 1) // 2 > TOPK, (
        "m_cap must not coincide with topk, or variant A is untestable here"
    )


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
    assert extra <= tail, f"spent {extra} extra evals for a tail of only {tail}"


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
