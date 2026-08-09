"""``GumbelConfig.root_expand_all``: one root eval for every legal move.

The flag exists because an unvisited legal move enters the improved-policy softmax
with ``completed_q = root_q``, an IDENTICAL constant for every such move, so the
target's ranking over that tail is purely the net prior -- and CE then trains the
net toward its own prior there, a fixed point a wrongly-low prior cannot leave.

Two properties are load-bearing, and the first is the one that is easy to fake:

* the tail's target order must STOP being the prior order. A test that only checks
  "the numbers changed" passes for an implementation that spends every extra
  evaluation and then writes back a constant, which is precisely the defect.
* whenever every legal move is already a Gumbel candidate the flag must change
  nothing, bit for bit. If it does, the flag is perturbing the search rather than
  only widening the target, and no number measured with it can be trusted.
"""
from __future__ import annotations

import hashlib

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts import gumbel as gumbel_mod
from chess_anti_engine.mcts.gumbel import (
    GumbelConfig,
    immediate_terminal_draw_indices,
    run_gumbel_root_many,
)
from chess_anti_engine.moves import POLICY_SIZE

# 20 legal moves (<= topk 32, flag must be inert) vs 45 (> topk, flag must act).
# Neither position has a legal move leading to a terminal node -- asserted in the
# premise test -- so every tail move costs exactly one NET EVALUATION and the
# eval-count assertions below can demand EXACT equality rather than a bound.
NARROW = chess.Board()
WIDE = chess.Board("r3k2r/pppq1ppp/2npbn2/1B2p1B1/3PP1b1/2N2N2/PPPQ1PPP/R3K2R w KQkq - 0 1")
# Same position at halfmove clock 99: 33 of the 45 legal moves now end the game as
# a 50-move draw, and `_init_board_search_state` zeroes them out of `priors` when
# root_q > 0. The legal set the flag must walk is therefore the 12 survivors, NOT
# the 45 children hanging off the root node -- see test_terminal_draw_moves_*.
DRAWISH = chess.Board("r3k2r/pppq1ppp/2npbn2/1B2p1B1/3PP1b1/2N2N2/PPPQ1PPP/R3K2R w KQkq - 99 60")
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


def _row_key(row: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=8).digest()


class _DetEval:
    """Deterministic evaluator keyed on the WHOLE encoded position.

    Hashing the full row matters and is not pedantry: the encoding's leading bytes
    are shared by every position reachable in one ply from a common root (the first
    byte that differs across the root and all 20 children of the start position is
    at index 46), so a stub keyed on a prefix is a CONSTANT FUNCTION. Under such a
    stub every tail move gets the same Q, the flag cannot change the tail's ranking
    even when it is working perfectly, and the tests below pass vacuously.

    Keying on the input rather than on call order is what makes the bit-identity
    assertion meaningful: the extra evaluations change how many times and in what
    order positions are submitted, so an order-dependent stub would report a
    difference that came from the harness rather than from the flag.
    """

    def __init__(self, wdl_overrides: dict[bytes, tuple[float, float, float]] | None = None) -> None:
        self.calls = 0
        self.rows = 0
        self.batches: list[int] = []
        self.batch_keys: list[list[bytes]] = []
        self._wdl_overrides = dict(wdl_overrides or {})
        self.overrides_hit = 0

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        n = int(x.shape[0])
        self.calls += 1
        self.rows += n
        self.batches.append(n)
        keys: list[bytes] = []
        pol = np.empty((n, POLICY_SIZE), dtype=np.float32)
        wdl = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            key = _row_key(x[i])
            keys.append(key)
            rng = np.random.default_rng(np.frombuffer(key, dtype=np.uint64)[0] % (2**32))
            pol[i] = rng.normal(0.0, 2.0, POLICY_SIZE).astype(np.float32)
            override = self._wdl_overrides.get(key)
            if override is not None:
                self.overrides_hit += 1
                wdl[i] = np.asarray(override, dtype=np.float32)
            else:
                wdl[i] = rng.normal(0.0, 1.0, 3).astype(np.float32)
        self.batch_keys.append(keys)
        return pol, wdl


class _Capture:
    """Root prior + Gumbel candidate set, lifted out of the search as it runs.

    The tail is not reconstructible from the outside: it is the legal set AFTER
    terminal-draw masking minus the candidates the Gumbel noise happened to pick.
    Guessing it (say, "the lowest-probability legal moves") would silently test a
    different set than the one the flag acts on.
    """

    def __init__(self) -> None:
        self.priors: np.ndarray | None = None
        self.candidates: np.ndarray | None = None

    def tail(self) -> np.ndarray:
        assert self.priors is not None
        assert self.candidates is not None
        legal = np.nonzero(self.priors > 0.0)[0]
        cand = {int(a) for a in self.candidates}
        return np.array([a for a in legal if int(a) not in cand], dtype=np.int64)


def _run(
    board: chess.Board,
    *,
    expand: bool,
    sims: int = SIMS,
    monkeypatch: pytest.MonkeyPatch | None = None,
    wdl_overrides: dict[bytes, tuple[float, float, float]] | None = None,
):
    capture = _Capture()
    if monkeypatch is not None:
        real = gumbel_mod._build_improved_policy_for_board

        def spy(st, **kwargs):
            capture.priors = np.array(st.priors, copy=True)
            capture.candidates = np.array(st.candidates, copy=True)
            return real(st, **kwargs)

        monkeypatch.setattr(gumbel_mod, "_build_improved_policy_for_board", spy)
    ev = _DetEval(wdl_overrides)
    cfg = GumbelConfig(
        simulations=sims, topk=TOPK, add_noise=False, root_expand_all=expand,
    )
    probs, actions, values, _masks = run_gumbel_root_many(
        None, [board.copy()], device="cpu", rng=np.random.default_rng(0),
        cfg=cfg, evaluator=ev,
    )[:4]
    return probs[0], int(actions[0]), float(values[0]), ev, capture


def _order(values: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """Indices of `subset` sorted by `values`, descending -- a pure ranking."""
    return subset[np.argsort(-values[subset], kind="stable")]


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


def test_premise_the_evaluator_stub_is_not_a_constant_function() -> None:
    """Guard the instrument itself: a prefix-keyed stub made every test vacuous.

    Positions one ply apart share a long encoded prefix, so this is not a
    hypothetical -- the first version of this file keyed on 8 bytes and every
    board in a subtree hashed to the same seed.
    """
    from chess_anti_engine.encoding import encode_positions_batch

    boards = [WIDE.copy()]
    for mv in WIDE.legal_moves:
        child = WIDE.copy()
        child.push(mv)
        boards.append(child)
    encoded = encode_positions_batch(boards)
    x = np.asarray(encoded[0] if isinstance(encoded, tuple) else encoded)
    keys = {_row_key(x[i]) for i in range(x.shape[0])}
    assert len(keys) == x.shape[0], (
        f"stub key collides: {x.shape[0]} distinct positions -> {len(keys)} keys"
    )


def test_noop_when_every_legal_move_is_already_a_candidate() -> None:
    """THE CONTROL: n_legal <= topk => bit-identical, not merely close."""
    off_p, off_a, off_v, off_ev, _ = _run(NARROW, expand=False)
    on_p, on_a, on_v, on_ev, _ = _run(NARROW, expand=True)
    assert np.array_equal(off_p, on_p), (
        "root_expand_all perturbed a position where it is a no-op; "
        f"max|delta|={np.abs(off_p - on_p).max():.3e}"
    )
    assert (off_a, off_v) == (on_a, on_v)
    assert off_ev.rows == on_ev.rows, "no-op position still cost extra evaluations"


def test_tail_target_order_stops_being_the_prior_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE POINT OF THE FLAG, stated as an assertion.

    Flag off, every tail move carries the same completed_q, so its target logit is
    log(prior) + <the same constant> and the tail's target order is EXACTLY the
    prior order -- the fixed point. Flag on, each tail move carries its own Q and
    the order must move. Without this, an implementation that spends the evals and
    then writes a constant back is indistinguishable from a correct one.
    """
    off_p, _, _, _, off_cap = _run(WIDE, expand=False, monkeypatch=monkeypatch)
    on_p, _, _, _, on_cap = _run(WIDE, expand=True, monkeypatch=monkeypatch)
    off_tail, on_tail = off_cap.tail(), on_cap.tail()
    assert off_tail.size >= 3, "fixture must have a tail to rank"
    assert np.array_equal(off_tail, on_tail), "flag changed the candidate set"
    assert off_cap.priors is not None
    prior_order = _order(off_cap.priors, off_tail)
    assert np.array_equal(_order(off_p, off_tail), prior_order), (
        "premise broken: with the flag OFF the tail should already rank by prior"
    )
    assert not np.array_equal(_order(on_p, on_tail), prior_order), (
        "flag spent evaluations but the tail still ranks purely by the net prior"
    )


def test_a_tail_move_into_a_lost_position_is_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direction, not just movement: the tail's Q must enter with the right sign.

    Order-changed alone is satisfied by an implementation that ranks the tail by
    the NEGATION of the search opinion, which is strictly worse than the prior-only
    fixed point it replaces. Here one tail move is made to lead to a position the
    opponent wins outright, so its target mass must fall.
    """
    plain_p, _, _, plain_ev, cap = _run(WIDE, expand=True, monkeypatch=monkeypatch)
    tail = cap.tail()
    assert tail.size >= 3
    # The expansion is the LAST eval batch (pinned by the placement test) and walks
    # the tail in ascending action order, so batch row j is tail[j]. Taking the key
    # from what the evaluator actually SAW avoids re-deriving the search's encoding
    # -- an earlier version encoded the child itself, never matched, and the test
    # passed a null through as a result.
    tail_keys = plain_ev.batch_keys[-1]
    assert len(tail_keys) == tail.size, "last batch is not the tail expansion"
    j = tail.size // 2
    victim = int(tail[j])
    # WDL is from the perspective of the side to move AT THAT NODE -- the opponent,
    # after our move. A win for them is a disaster for us.
    overrides = {tail_keys[j]: (12.0, 0.0, -12.0)}

    hurt_p, _, _, hurt_ev, hurt_cap = _run(
        WIDE, expand=True, monkeypatch=monkeypatch, wdl_overrides=overrides,
    )
    assert hurt_ev.overrides_hit == 1, (
        f"override never reached the evaluator ({hurt_ev.overrides_hit} hits); "
        "the two runs are identical and the assertion below would be vacuous"
    )
    assert np.array_equal(hurt_cap.tail(), tail), "override perturbed the tail set"
    assert hurt_p[victim] < plain_p[victim], (
        f"tail move {victim} leads to a position the opponent wins, but its target "
        f"mass rose ({plain_p[victim]:.6f} -> {hurt_p[victim]:.6f}): the tail's Q "
        "is entering the improved policy with the wrong sign"
    )


def test_expansion_runs_after_halving_and_leaves_it_bit_identical() -> None:
    """Pin the PLACEMENT, which is a search-vs-target decision, not a detail.

    `_completed_q_transform` min-max renormalises over all legal moves, so a tail
    carrying real Q values BEFORE halving changes which candidates survive each
    round -- a strength change wearing the costume of a target change. Running it
    after halving leaves every halving batch untouched and confines the effect to
    the phase-4 target. The eval batch sequence is the observable: identical
    prefix, tail paid for exactly once, last.
    """
    _, _, _, off_ev, _ = _run(WIDE, expand=False)
    _, _, _, on_ev, _ = _run(WIDE, expand=True)
    tail = WIDE.legal_moves.count() - min(TOPK, (SIMS + 1) // 2)
    assert on_ev.batches[:-1] == off_ev.batches, (
        "halving's evaluation batches changed shape; the expansion is running "
        f"inside or before sequential halving: {on_ev.batches} vs {off_ev.batches}"
    )
    assert on_ev.batches[-1] == tail, (
        f"expected the tail ({tail}) to be the LAST batch, got {on_ev.batches[-1]}"
    )


def test_extra_evaluations_are_additive_and_bounded_by_the_tail() -> None:
    """Variant B, not A: the tail is paid for on top of the halving budget.

    If the implementation had instead widened ``m`` to ``legal.size``, the eval count
    would be unchanged and the survivors would simply get fewer visits each.
    """
    _, _, _, off_ev, _ = _run(WIDE, expand=False)
    _, _, _, on_ev, _ = _run(WIDE, expand=True)
    tail = WIDE.legal_moves.count() - min(TOPK, (SIMS + 1) // 2)
    extra = on_ev.rows - off_ev.rows
    assert extra > 0, "variant A detected: no additional evaluations were spent"
    assert extra == tail, (
        f"expected EXACTLY one eval per prior-only move ({tail}), got {extra}: "
        "fewer means the tail is being truncated and some legal moves still have "
        "no search opinion; more means candidates are being re-walked"
    )


def test_terminal_draw_moves_are_not_part_of_the_legal_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tail must be read off `priors`, not off the root's children.

    A move that ends the game as a draw is zeroed out of `priors` when root_q > 0,
    but its child node still hangs off the root. Walking `root.children` instead
    would expand 33 moves the improved policy will never score -- pure waste, and
    a legal set that disagrees with the one the target is built from.
    """
    draws = immediate_terminal_draw_indices(DRAWISH)
    assert len(draws) == 33, f"fixture drifted: {len(draws)} drawing moves"
    survivors = DRAWISH.legal_moves.count() - len(draws)
    assert 0 < survivors <= TOPK, "survivors must fit inside topk for this to be a no-op"
    # Force root_q > 0 so the masking branch is taken at all. The key is taken from
    # a probe run rather than encoded here: the search's root encoding does NOT
    # match a standalone `encode_positions_batch` call, so a locally-derived key
    # never matches and the override silently does nothing.
    _, _, _, probe_ev, _ = _run(DRAWISH, expand=False, monkeypatch=monkeypatch)
    assert len(probe_ev.batch_keys[0]) == 1, "first batch should be the root alone"
    overrides = {probe_ev.batch_keys[0][0]: (12.0, 0.0, -12.0)}

    off_p, _, _, off_ev, off_cap = _run(
        DRAWISH, expand=False, monkeypatch=monkeypatch, wdl_overrides=overrides,
    )
    on_p, _, _, on_ev, _ = _run(
        DRAWISH, expand=True, monkeypatch=monkeypatch, wdl_overrides=overrides,
    )
    assert off_ev.overrides_hit >= 1, "root WDL override never reached the evaluator"
    assert off_cap.priors is not None
    assert int(np.count_nonzero(off_cap.priors > 0.0)) == survivors, (
        "terminal-draw masking did not fire; root_q may not be positive"
    )
    assert on_ev.rows == off_ev.rows, (
        f"flag spent {on_ev.rows - off_ev.rows} extra evaluations on a position "
        "whose entire non-drawing legal set is already a candidate -- the tail is "
        "being read off root.children rather than priors"
    )
    assert np.array_equal(off_p, on_p)


def test_flag_off_reproduces_itself_exactly() -> None:
    """Regression guard: the default path must be untouched and deterministic."""
    a_p, a_a, a_v, _, _ = _run(WIDE, expand=False)
    b_p, b_a, b_v, _, _ = _run(WIDE, expand=False)
    assert np.array_equal(a_p, b_p)
    assert (a_a, a_v) == (b_a, b_v)


def test_probs_stay_a_normalised_distribution() -> None:
    """Extra visits must not leak mass onto illegal moves or break normalisation."""
    for expand in (False, True):
        p, _, _, _, _ = _run(WIDE, expand=expand)
        assert p.min() >= 0.0
        assert float(p.sum()) == pytest.approx(1.0, abs=1e-5)


def test_fast_ply_budget_is_the_regime_where_m_cap_binds() -> None:
    """At 32 sims m_cap(16) binds below topk(32), so even NARROW has a tail.

    Production fast plies run here. They are not trained on directly, but they
    still build the tree the full-ply target is read from, so the flag must behave
    sanely in this regime rather than only in the one we happen to measure. This
    is also where the cost is worst: the tail is most of the legal set.
    """
    off_p, _, _, off_ev, _ = _run(NARROW, expand=False, sims=SIMS_FAST)
    on_p, _, _, on_ev, _ = _run(NARROW, expand=True, sims=SIMS_FAST)
    assert not np.array_equal(off_p, on_p), (
        "at the fast budget m_cap binds, so NARROW has non-candidates and the "
        "flag must act -- if it did not, the tail is being computed off topk alone"
    )
    tail = NARROW.legal_moves.count() - min(TOPK, (SIMS_FAST + 1) // 2)
    extra = on_ev.rows - off_ev.rows
    assert extra == tail, f"expected exactly {tail} extra evals, got {extra}"
    # And pin the cost claim in GumbelConfig.root_expand_all, on the wide position
    # where it was measured: the fast budget is where this flag is expensive, so a
    # production wiring that fires on fast plies is buying a large cost for plies
    # that emit no policy training row.
    _, _, _, w_off_ev, _ = _run(WIDE, expand=False, sims=SIMS_FAST)
    _, _, _, w_on_ev, _ = _run(WIDE, expand=True, sims=SIMS_FAST)
    w_extra = w_on_ev.rows - w_off_ev.rows
    assert w_extra == WIDE.legal_moves.count() - min(TOPK, (SIMS_FAST + 1) // 2)
    assert w_extra / w_off_ev.rows > 0.5, (
        "the docstring calls the fast-ply regime ~+88%; measured "
        f"+{100 * w_extra / w_off_ev.rows:.0f}%"
    )
