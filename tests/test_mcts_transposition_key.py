"""Transposition-table correctness for the C Gumbel search (audit W1 + W2).

The Gumbel path is the only consumer of the tree's transposition table, and a
hit does not merely reuse a value: it copies the donor node's child ACTION LIST
onto the new leaf and marks it expanded forever. So the key must imply "same
legal move set". ``CBoard.zobrist_hash`` does not — it is the repetition key and
deliberately ignores en passant — which injected illegal pawn captures into
production trees and dropped legal ones.

W1 fixes that structural identity. A second boundary is equally important:
production evaluates `lc0_root_legacy_meta`, whose input includes the halfmove
clock, raw EP metadata, seven historical positions and repetition context. A
structural transposition may therefore share every legal move while requiring a
different policy/value evaluation. The TT may copy donor priors and W/N only
when that evaluation/search context matches too.

The W1 tests fail on the pre-fix build (67 illegal + 3 missing children over the
same 32k-node sample). The context tests are mutation-style: removing the new
context check turns a required real evaluation back into a donor reuse.
"""
from __future__ import annotations

import dataclasses

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.mcts import _mcts_tree
from chess_anti_engine.mcts import gumbel_c as gumbel_c_mod
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import (
    root_coverage_miss_count,
    root_support_narrowed_count,
    run_gumbel_root_many_c,
)
from chess_anti_engine.moves.encode import move_to_index

POLICY_SIZE = 4672

# Same placement/castling/side-to-move, differing only in the ep right. White
# has a pawn on d5, so e5-e6 en passant is a real extra legal move.
_EP_FEN = "rnbqkbnr/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3"
_NO_EP_FEN = "rnbqkbnr/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq - 0 3"

# (fen, sims, seed) triples that are each individually RED on the pre-fix build.
# A parametrisation that cannot fail is decoration, so the search shape is pinned
# per position rather than shared: the same FEN at a different seed produces a
# clean tree. Trailing number = corrupt nodes observed on `main` at that shape,
# re-derivable with the walk below against an un-patched extension.
_EP_RICH_CASES = (
    ("rnbqkb1r/pp2pppp/3p1n2/2pP4/8/2N2N2/PPP1PPPP/R1BQKB1R b KQkq - 0 5", 512, 2),   # 22
    ("r1bqkb1r/pp1ppppp/2n2n2/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4", 1024, 1),  # 2
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", 512, 3),       # 1
)


class _Ev:
    """Deterministic pseudo-random evaluator keyed on the encoded row."""

    def __init__(self, seed: int = 20260803) -> None:
        rng = np.random.default_rng(seed)
        self._pol = rng.standard_normal((1024, POLICY_SIZE)).astype(np.float32)
        self._wdl = rng.standard_normal((1024, 3)).astype(np.float32)

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        n = int(x.shape[0])
        flat = np.ascontiguousarray(x, dtype=np.float32).reshape(n, -1)
        idx = np.array(
            [int.from_bytes(flat[i].tobytes()[:6], "little") % 1024 for i in range(n)],
            dtype=np.int64,
        )
        return self._pol[idx], self._wdl[idx]


def _cb(fen: str) -> CBoard:
    return CBoard.from_board(chess.Board(fen))


def _legal_set(board: chess.Board) -> set[int]:
    return {int(x) for x in CBoard.from_board(board).legal_move_indices()}


def test_zobrist_hash_stays_ep_blind() -> None:
    """The repetition hash is NOT what changed — it is persisted in resume
    records and matches python-chess is_repetition() semantics."""
    assert _cb(_EP_FEN).zobrist_hash == _cb(_NO_EP_FEN).zobrist_hash


def test_transposition_key_separates_positions_with_different_legal_moves() -> None:
    ep, no_ep = _cb(_EP_FEN), _cb(_NO_EP_FEN)
    # The two positions genuinely differ: d5xe6 e.p. is legal in one only.
    assert _legal_set(chess.Board(_EP_FEN)) != _legal_set(chess.Board(_NO_EP_FEN))
    assert ep.transposition_key != no_ep.transposition_key


def test_transposition_key_ignores_an_unusable_ep_right() -> None:
    """An ep square no pawn can capture on does not change the legal move set,
    so it must not split the table (that is the hit rate the exclusion bought)."""
    # 1. e4: ep square e3 is set, but Black has no pawn on d4/f4.
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    cb = CBoard.from_board(board)
    assert cb.transposition_key == cb.zobrist_hash


def test_transposition_key_is_stable_across_construction_paths() -> None:
    board = chess.Board(_NO_EP_FEN)
    board.push(chess.Move.from_uci("e2e4"))  # irrelevant push; recompute path
    pushed = CBoard.from_board(chess.Board(_EP_FEN))
    fresh = CBoard.from_board(chess.Board(_EP_FEN))
    assert pushed.transposition_key == fresh.transposition_key


_HISTORY_SEQ_A = ("g1f3", "g8f6", "g2g3", "g7g6")
_HISTORY_SEQ_B = ("g2g3", "g7g6", "g1f3", "g8f6")


def _board_after(moves: tuple[str, ...]) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return board


def test_structural_key_does_not_imply_legacy_history_network_input() -> None:
    """Same legal position, different move-order history => different net input.

    This is the premise the old TT violated: its key/action guard proved only
    structural chess identity, then reused a policy and Q as though it had
    proved evaluator identity too.
    """
    a = _board_after(_HISTORY_SEQ_A)
    b = _board_after(_HISTORY_SEQ_B)
    assert a.board_fen() == b.board_fen()
    assert a.turn == b.turn
    assert a.castling_rights == b.castling_rights
    assert a.halfmove_clock == b.halfmove_clock

    ca, cb = CBoard.from_board(a), CBoard.from_board(b)
    assert ca.transposition_key == cb.transposition_key
    assert set(ca.legal_move_indices()) == set(cb.legal_move_indices())

    xa = encode_cboard(
        ca,
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    )
    xb = encode_cboard(
        cb,
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    )
    # Current-position planes agree; the historical slots do not.
    np.testing.assert_array_equal(xa[:13], xb[:13])
    assert np.any(xa[13:104] != xb[13:104]), (
        "fixture lost its history distinction, so it can no longer test TT "
        "evaluation identity"
    )
    assert not np.array_equal(xa, xb)


def test_unusable_ep_is_structurally_ignored_but_legacy_meta_still_encodes_it() -> None:
    """A second proof that legal-set identity is weaker than evaluator identity."""
    ep = chess.Board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )
    no_ep = chess.Board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    )
    ce, cn = CBoard.from_board(ep), CBoard.from_board(no_ep)
    assert ce.transposition_key == cn.transposition_key
    assert set(ce.legal_move_indices()) == set(cn.legal_move_indices())

    xe = encode_cboard(
        ce,
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    )
    xn = encode_cboard(
        cn,
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    )
    assert np.any(xe[110] != xn[110])
    assert not np.array_equal(xe, xn)


def _search_tree(fen: str, sims: int, seed: int):
    cfg = GumbelConfig(
        simulations=sims, topk=16, c_scale=0.1, temperature=0.0, add_noise=False
    )
    res = run_gumbel_root_many_c(
        None,
        [chess.Board(fen)],
        device="cpu",
        rng=np.random.default_rng(seed),
        cfg=cfg,
        evaluator=_Ev(seed),
        target_batch=0,
        vloss_weight=1,
    )
    return res[4], int(res[5][0])


def _walk_violations(tree, root_id: int, root_board: chess.Board):
    """Every expanded node's child-action set must equal the legal move set of
    the position on its action path. Returns (nodes_checked, violations)."""
    violations: list[str] = []
    checked = 0
    stack = [(root_id, root_board)]
    while stack:
        node, board = stack.pop()
        if not tree.is_expanded(node):
            continue
        actions, _visits = tree.get_children_visits(node)
        got = {int(a) for a in actions}
        if not got:
            continue
        checked += 1
        legal = _legal_set(board)
        if got != legal:
            violations.append(
                f"node={node} fen={board.fen()} "
                f"extra={sorted(got - legal)} missing={sorted(legal - got)}"
            )
            continue  # an illegal action cannot be replayed to recurse
        i2m = {int(move_to_index(mv, board)): mv for mv in board.legal_moves}
        for a in got:
            child = tree.find_child(node, a)
            if child < 0:
                continue
            nb = board.copy(stack=True)
            nb.push(i2m[a])
            stack.append((child, nb))
    return checked, violations


@pytest.mark.parametrize(("fen", "sims", "seed"), _EP_RICH_CASES)
def test_expanded_nodes_carry_their_own_legal_moves(
    monkeypatch, fen: str, sims: int, seed: int
) -> None:
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    tree, rid = _search_tree(fen, sims=sims, seed=seed)
    checked, violations = _walk_violations(tree, rid, chess.Board(fen))
    assert checked > 100, "search produced too few expanded nodes to be a test"
    assert not violations, "\n".join(violations[:5])


def test_transposition_guard_runs_and_passes(monkeypatch) -> None:
    """Every structural probe is classified, and W1 legal-set rejects stay zero."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _mcts_tree.tt_stats(reset=True)
    fen, sims, seed = _EP_RICH_CASES[0]
    _search_tree(fen, sims=sims, seed=seed)
    stats = _mcts_tree.tt_stats()
    assert stats["probe_hits"] > 0, "no transposition hit — test is not exercising the guard"
    assert stats["reject"] == 0
    assert stats["probe_hits"] == (
        stats["reuse"] + stats["reject"] + stats["context_reject"]
    ), stats


class _CountingEv:
    def __init__(self) -> None:
        self.rows = 0

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        n = int(x.shape[0])
        self.rows += n
        return (
            np.zeros((n, POLICY_SIZE), dtype=np.float32),
            np.zeros((n, 3), dtype=np.float32),
        )


def _one_sim_on_existing_tree(
    tree, board: chess.Board, evaluator: _CountingEv,
) -> None:
    """One deterministic root candidate; root eval is supplied, so any evaluator
    row is a LEAF that TT reuse did not satisfy."""
    legal = sorted(_legal_set(board))
    assert legal
    root_pol = np.full((1, POLICY_SIZE), -20.0, dtype=np.float32)
    root_pol[0, legal[0]] = 20.0
    root_wdl = np.zeros((1, 3), dtype=np.float32)
    cfg = GumbelConfig(
        simulations=1,
        topk=2,
        c_scale=0.1,
        temperature=0.0,
        add_noise=False,
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    )
    run_gumbel_root_many_c(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=cfg,
        evaluator=evaluator,
        pre_pol_logits=root_pol,
        pre_wdl_logits=root_wdl,
        tree=tree,
        root_node_ids=None,
        target_batch=1,
        vloss_weight=1,
    )


def _prime_tt_then_search(
    first: chess.Board, second: chess.Board,
) -> tuple[dict[str, int], int]:
    """Prime one deterministic leaf, reset COUNTERS only, then search a fresh
    root in the same tree. The TT table itself deliberately survives."""
    tree = _mcts_tree.MCTSTree()
    _one_sim_on_existing_tree(tree, first, _CountingEv())
    _mcts_tree.tt_stats(reset=True)
    second_ev = _CountingEv()
    _one_sim_on_existing_tree(tree, second, second_ev)
    return _mcts_tree.tt_stats(), second_ev.rows


def test_tt_still_reuses_an_exact_history_context(monkeypatch) -> None:
    """Positive control: the safety fix must not simply disable the TT."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    board = _board_after(_HISTORY_SEQ_A)
    stats, rows = _prime_tt_then_search(board, board.copy(stack=True))
    assert stats["probe_hits"] > 0, stats
    assert stats["reuse"] > 0, stats
    assert stats["context_reject"] == 0, stats
    assert stats["reject"] == 0, stats
    assert rows == 0, (
        "an exact-context donor was not sufficient; the second one-sim search "
        "should need no leaf evaluation"
    )


def test_tt_re_evaluates_a_structural_twin_with_different_history(monkeypatch) -> None:
    """The production-path regression: same key/actions, different net context.

    MUTANT: removing `tt_donor_context_match` turns context_reject back to 0,
    reuse back on, and rows to 0 — exactly the silent stale-Q/prior behavior.
    """
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    first = _board_after(_HISTORY_SEQ_A)
    second = _board_after(_HISTORY_SEQ_B)
    assert CBoard.from_board(first).transposition_key == CBoard.from_board(second).transposition_key

    stats, rows = _prime_tt_then_search(first, second)
    assert stats["probe_hits"] > 0, stats
    assert stats["reject"] == 0, stats
    assert stats["context_reject"] > 0, stats
    assert stats["probe_hits"] == (
        stats["reuse"] + stats["reject"] + stats["context_reject"]
    ), stats
    assert rows > 0, (
        "different-history recipient skipped the evaluator and inherited the "
        "donor's policy/Q"
    )


class _CountingHistoryEv:
    """Leaf evaluator that records the exact encoded rows the C search requested."""

    def __init__(self) -> None:
        self.rows = 0
        self.seen: list[np.ndarray] = []

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        arr = np.asarray(x, dtype=np.float32)
        self.rows += int(arr.shape[0])
        self.seen.extend(np.array(row, copy=True) for row in arr)
        n = int(arr.shape[0])
        pol = np.zeros((n, POLICY_SIZE), dtype=np.float32)
        wdl = np.tile(np.array([[0.8, 0.1, -0.3]], dtype=np.float32), (n, 1))
        return pol, wdl


def _board_after(*moves: str) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)
    return board


def _two_action_root_logits(
    board: chess.Board, desired_uci: str,
) -> tuple[np.ndarray, set[int]]:
    """Make desired_uci the deterministic one-sim Gumbel candidate.

    The support deliberately contains TWO legal actions: a singleton support is
    an early-finished root and would never exercise a leaf / TT probe.
    """
    desired_move = chess.Move.from_uci(desired_uci)
    desired = int(move_to_index(desired_move, board))
    legal = [int(move_to_index(mv, board)) for mv in board.legal_moves]
    other = next(action for action in legal if action != desired)

    logits = np.full((1, POLICY_SIZE), -20.0, dtype=np.float32)
    logits[0, other] = 0.0
    logits[0, desired] = 20.0
    return logits, {desired, other}


def test_structural_transposition_with_different_history_is_re_evaluated(
    monkeypatch,
) -> None:
    """A structural TT hit must not reuse another history's NN Q / priors.

    These two move orders reach the same current chess position:

        Nf3 Nf6 g3 g6
        g3  g6  Nf3 Nf6

    but the second history has a different seven-slot LC0 history and a
    different rule-50 clock. cboard_transposition_key intentionally aliases
    them because their current legal chess state is identical. That is valid for
    structural storage, but NOT for the C Gumbel shortcut: the old code copied
    the first leaf's priors and W/N into the second leaf and skipped its network
    evaluation entirely.

    One simulation is enough to make the test discriminating. Each root exposes
    two legal actions (to avoid the single-support early finish), while the
    supplied root logits force the action that completes the transposition.
    Therefore the correct path performs exactly two leaf evaluations. Restoring
    the old structural-only reuse performs one and fails this test.
    """
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _mcts_tree.tt_stats(reset=True)

    board_a = _board_after("g1f3", "g8f6", "g2g3")
    logits_a, support_a = _two_action_root_logits(board_a, "g7g6")
    final_a = board_a.copy(stack=True)
    final_a.push_uci("g7g6")

    board_b = _board_after("g2g3", "g7g6", "g1f3")
    logits_b, support_b = _two_action_root_logits(board_b, "g8f6")
    final_b = board_b.copy(stack=True)
    final_b.push_uci("g8f6")

    cb_a = CBoard.from_board(final_a)
    cb_b = CBoard.from_board(final_b)
    assert cb_a.transposition_key == cb_b.transposition_key
    assert _legal_set(final_a) == _legal_set(final_b)
    assert final_a.halfmove_clock != final_b.halfmove_clock

    enc_a = encode_cboard(
        cb_a,
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    )
    enc_b = encode_cboard(
        cb_b,
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    )
    assert not np.array_equal(enc_a, enc_b), (
        "fixture histories encode identically; the TT-context test is vacuous"
    )

    tree = _mcts_tree.MCTSTree()
    evaluator = _CountingHistoryEv()
    cfg = GumbelConfig(
        simulations=1, topk=2, c_scale=0.1, temperature=0.0, add_noise=False,
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
    )
    root_wdl = np.zeros((1, 3), dtype=np.float32)

    run_gumbel_root_many_c(
        None, [board_a], device="cpu", rng=np.random.default_rng(1), cfg=cfg,
        evaluator=evaluator, pre_pol_logits=logits_a, pre_wdl_logits=root_wdl,
        tree=tree, allowed_root_indices_batch=[support_a],
        target_batch=1, vloss_weight=1,
    )
    assert evaluator.rows == 1, "the donor leaf was not actually evaluated"
    first_stats = _mcts_tree.tt_stats()
    assert first_stats["probe_hits"] == 0

    run_gumbel_root_many_c(
        None, [board_b], device="cpu", rng=np.random.default_rng(2), cfg=cfg,
        evaluator=evaluator, pre_pol_logits=logits_b, pre_wdl_logits=root_wdl,
        tree=tree, allowed_root_indices_batch=[support_b],
        target_batch=1, vloss_weight=1,
    )
    stats = _mcts_tree.tt_stats()
    assert stats["probe_hits"] == 1, stats
    assert stats.get("context_reject", 0) == 1, stats
    assert stats["reuse"] == 0, stats
    assert stats["reject"] == 0, stats
    assert evaluator.rows == 2, (
        "the second history was structurally aliased and skipped its NN eval"
    )
    assert len(evaluator.seen) == 2
    assert not np.array_equal(evaluator.seen[0], evaluator.seen[1]), (
        "the evaluator did not receive the two distinct history encodings"
    )


def test_reused_root_coverage_check_runs_without_allowed_indices(monkeypatch) -> None:
    """Audit W2: the support check used to be short-circuited whenever
    ``allowed_root_indices_batch is None``, which selfplay — the only caller
    that carries a tree across plies — always passes.

    Also the ALARM half of the counter split: a root MISSING an action means the
    tree disagrees with the rules, and that must stay on
    ``root_coverage_miss_count()`` — never on the routine narrowed counter."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    board = chess.Board(_EP_FEN)
    legal = np.array(sorted(_legal_set(board)), dtype=np.int32)
    assert legal.size > 2

    tree = _mcts_tree.MCTSTree()
    stale_root = tree.add_root(1, 0.0)
    # A root expanded with a child set that is missing a legal action — exactly
    # what a stale/transposed node looks like.
    subset = legal[:-1]
    tree.expand(
        stale_root,
        subset,
        np.full(subset.size, 1.0 / subset.size, dtype=np.float64),
    )

    before = root_coverage_miss_count()
    narrowed_before = root_support_narrowed_count()
    cfg = GumbelConfig(
        simulations=8, topk=4, c_scale=0.1, temperature=0.0, add_noise=False
    )
    res = run_gumbel_root_many_c(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=cfg,
        evaluator=_Ev(1),
        tree=tree,
        root_node_ids=[stale_root],
        allowed_root_indices_batch=None,
        target_batch=0,
        vloss_weight=1,
    )
    new_root = int(res[5][0])
    assert new_root != stale_root, "deficient root was reused"
    assert root_coverage_miss_count() == before + 1
    assert root_support_narrowed_count() == narrowed_before, (
        "a missing action is the W2 alarm — it must not be filed as routine "
        "support narrowing"
    )
    actions, _ = tree.get_children_visits(new_root)
    assert {int(a) for a in actions} == set(legal.tolist())


# --- W2b: the gate is support EQUALITY, not coverage ---------------------
#
# A carried root may be a strict SUPERSET of the support the CURRENT search is
# allowed to touch: `allowed_root_indices_batch` (UCI `searchmoves`) and the
# winning-root terminal-draw prune both narrow `legal_idx` below the previous
# ply's expansion. `gss_score_and_halve` reads max_visit / total_visits /
# weighted_q / mixed_value / min_q / max_q off ALL of the root's children, so an
# excluded child still moves the Q transform every INCLUDED candidate is scored
# through — while the improved policy on top is built over the current support
# only. The superset gate accepted exactly that.

# Position with 37 legal moves; the first 10 policy indices are the support and
# the 11th is the carried-but-excluded child. Search shapes pinned per case
# (the house rule: a parametrisation that cannot fail is decoration) — the
# selfplay shape is the live yaml's `c_scale=0.1` linear root, the play shape is
# `gumbel.PLAY_SEARCH_DEFAULTS`' root-log split, which is what the UCI path that
# actually carries trees across moves runs.
_NARROW_FEN = "r1bqkb1r/pp1ppppp/2n2n2/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4"
_NARROW_SHAPES = ("selfplay_linear_root", "play_log_root")


def _narrow_cfg(shape: str) -> GumbelConfig:
    """Written as ``dataclasses.replace`` on one base rather than ``**dict``:
    a ``dict[str, float]`` splat widens every keyword to ``float`` and
    basedpyright rejects the ``int``/``bool``/``str`` fields it lands on."""
    base = GumbelConfig(
        simulations=128, topk=16, c_scale=0.1, temperature=0.0, add_noise=False
    )
    if shape == "selfplay_linear_root":
        return base
    return dataclasses.replace(
        base, c_scale=0.025, c_visit=50.0, c_visit_root=900.0,
        c_scale_root=7.0, q_visit_exp_root=-1.0,
    )


def _narrowed_support_tree(carried: list[int], excluded: int, visits: int):
    """A carried root over ``carried`` whose ``excluded`` child holds ``visits``
    visits at an extreme Q — a heavily-searched move the current ply then drops.

    ⚑ Sign, stated exactly because it is easy to get backwards. ``backprop``
    negates on the way up, so ``visits`` × ``backprop(path, +1.0)`` leaves
    ``W[child] = +visits`` and ``W[root] = -visits``. ``gss_score_and_halve``
    reads the child PARENT-POV as ``q = -W/N``, so the excluded child presents as
    ``q = -1.0`` (a proven LOSS from the root's side) and the carried root's own
    ``root_Q`` as ``≈ -1.0``. That is the pollution this test injects: an extreme
    ``min_q`` plus ``max_visit = visits``. The opposite sign (``-1.0``, giving a
    proven WIN child) was swept and is red pre-fix on the same configurations;
    the magnitude, not the direction, is what makes the transform move.

    Built by tree surgery rather than a first search so the two arms differ ONLY
    in whether the root is reused: the tree handed to both is byte-for-byte the
    same shape, and no transposition-table entry exists to make node ids matter.
    """
    tree = _mcts_tree.MCTSTree()
    rid = tree.add_root(1, 0.0)
    acts = np.array(sorted(carried), dtype=np.int32)
    tree.expand(rid, acts, np.full(acts.size, 1.0 / acts.size, dtype=np.float64))
    cid = tree.find_child(rid, excluded)
    assert cid >= 0
    path = np.array([rid, cid], dtype=np.int32)
    for _ in range(visits):
        tree.backprop(path, 1.0)
    return tree, rid


@pytest.mark.parametrize("shape", _NARROW_SHAPES)
def test_a_narrowed_root_does_not_inherit_the_excluded_child(
    monkeypatch, shape: str
) -> None:
    """RED on the pre-fix (superset) gate, in both search shapes.

    Arm: carried root over support+{excluded}, reused. Control: the SAME tree,
    same seed, same evaluator, same narrowed support — but ``root_node_ids=None``
    so the root is built over the current support alone, which is the Python
    reference's (``gumbel.py``, no tree reuse) semantic. The two must agree
    exactly. Pre-fix they do not: the played move flips between two INCLUDED
    moves, and the root's visit split across the included moves shifts with it —
    both decided by a child the current search says does not exist.

    Control chosen as the same populated tree rather than an empty one so the
    single manipulated variable is the reuse decision itself — post-fix the arm
    rebuilds into the identical node slot, so equality is exact, not approximate.

    ⚑ Two classes of assertion here, and they are not equally durable. The
    behavioural block (played move, visit split, policy, value) states the
    REQUIREMENT: a narrowed search must not be steered by an excluded child. The
    counter and ``arm_root != stale_root`` asserts pin THIS FIX'S MECHANISM —
    reject and rebuild. A future implementation that instead taught
    ``gss_score_and_halve`` to restrict its statistics to the current support
    would satisfy the requirement while legitimately failing those two; treat
    them as a signal to re-read the mechanism, not as a regression.
    """
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    board = chess.Board(_NARROW_FEN)
    legal = sorted(_legal_set(board))
    assert len(legal) > 11, "position must be wide enough to narrow"
    support = legal[:10]
    excluded = legal[10]
    carried = [*support, excluded]

    cfg = _narrow_cfg(shape)

    def _search(tree, root_ids):
        return run_gumbel_root_many_c(
            None, [chess.Board(_NARROW_FEN)], device="cpu",
            rng=np.random.default_rng(0), cfg=cfg, evaluator=_Ev(1),
            tree=tree, root_node_ids=root_ids,
            allowed_root_indices_batch=[set(support)],
            target_batch=0, vloss_weight=1,
        )

    arm_tree, stale_root = _narrowed_support_tree(carried, excluded, visits=4000)
    misses_before = root_coverage_miss_count()
    narrowed_before = root_support_narrowed_count()
    arm = _search(arm_tree, [stale_root])
    arm_root = int(arm[5][0])

    assert root_support_narrowed_count() == narrowed_before + 1, (
        "the wider carried root was reused instead of rebuilt"
    )
    assert root_coverage_miss_count() == misses_before, (
        "a narrowed support is routine — it must not raise the W2 alarm counter"
    )
    assert arm_root != stale_root
    arm_actions, arm_visits = arm_tree.get_children_visits(arm_root)
    assert {int(a) for a in arm_actions} == set(support), (
        "the rebuilt root must hold exactly the current support"
    )

    ctl_tree, _ = _narrowed_support_tree(carried, excluded, visits=4000)
    ctl = _search(ctl_tree, None)
    ctl_actions, ctl_visits = ctl_tree.get_children_visits(int(ctl[5][0]))

    # Played move first: it is the assertion with a chess meaning, and the one
    # the superset gate flips.
    assert int(arm[1][0]) == int(ctl[1][0]), (
        f"reused wide root played {int(arm[1][0])}, support-only root played "
        f"{int(ctl[1][0])} — the excluded child decided the halving"
    )
    assert dict(
        zip((int(a) for a in arm_actions), (int(v) for v in arm_visits), strict=True)
    ) == dict(
        zip((int(a) for a in ctl_actions), (int(v) for v in ctl_visits), strict=True)
    ), "the excluded child moved the visit split over the included moves"
    assert np.array_equal(arm[0][0], ctl[0][0]), "improved policy diverged"
    assert float(arm[2][0]) == float(ctl[2][0]), "root value diverged"


# The PRODUCTION narrowing path. `searchmoves` above is the directly-drivable
# one, but nothing in this repo emits it — selfplay always passes
# `allowed_root_indices_batch=None`. What selfplay DOES hit is the winning-root
# terminal-draw prune, and that is the shape that decides whether this fix
# touches training data at all.
#
# White is winning and the halfmove clock is at 99, so every move that is not a
# pawn push leaves a position at 100 that is a claimable 50-move draw and gets
# pruned. Two cases, so the production path is pinned in BOTH search shapes and
# at two prune ratios; each was chosen because the played move FLIPS on the
# pre-fix gate at that exact shape:
#   (fen, search shape, evaluator seed, heavy visits, legal, support)
_DRAW_PRUNE_CASES = (
    ("selfplay_linear_root", "4k3/8/8/8/8/8/PPP1P3/3QK3 w - - 99 60", 3, 200, 21, 8),
    ("play_log_root", "4k3/8/8/8/8/8/3P4/3QK3 w - - 99 60", 1, 4000, 15, 2),
)


class _WinningRootEv(_Ev):
    """``_Ev`` with the WDL of its FIRST batch pinned to a decisive win.

    Two things have to be true at once and they pull in opposite directions.
    The ROOT must evaluate as winning or `want_draws` is False, the prune never
    runs, and the test passes vacuously against a root that was never narrowed.
    The LEAVES must keep varied values, or every candidate carries the same
    ``q_hat``, the Q transform becomes a constant across candidates, and the
    halving degenerates to a pure log-prior ranking that no amount of root
    pollution can move — measured: with a uniform winning WDL the played move
    does not flip at any (position, shape, sign, visit-count) combination tried.

    The root batch is the first ``evaluate_encoded`` call on this path
    (``pre_pol_logits`` is None, so the root is evaluated before any leaf), and
    each arm builds its own evaluator, so the flag is per-search and
    deterministic.
    """

    def __init__(self, seed: int = 20260803) -> None:
        super().__init__(seed)
        self._root_batch_done = False

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pol, wdl = super().evaluate_encoded(x, relations)
        if not self._root_batch_done:
            self._root_batch_done = True
            wdl = np.tile(
                np.array([8.0, 0.0, -8.0], dtype=np.float32), (pol.shape[0], 1)
            )
        return pol, wdl


@pytest.mark.parametrize(
    ("shape", "fen", "ev_seed", "heavy_visits", "n_legal", "n_support"),
    _DRAW_PRUNE_CASES,
    ids=[c[0] for c in _DRAW_PRUNE_CASES],
)
def test_the_winning_root_draw_prune_rebuilds_a_selfplay_carried_root(
    monkeypatch, shape: str, fen: str, ev_seed: int,
    heavy_visits: int, n_legal: int, n_support: int,
) -> None:
    """The narrowing path production actually reaches, in selfplay's call shape:
    persistent tree, root carried from the previous ply with its FULL legal
    expansion, ``allowed_root_indices_batch=None``.

    RED on the pre-fix gate, which reused the wide root for a search over the
    post-prune support and let the pruned draw children skew its halving
    transform — the played move differs from the support-only control in both
    cases.

    This is the case that makes the fix DATA-AFFECTING rather than merely
    search-shaped: on a narrowed ply the stored policy row and ``values_out``
    are now built from a rebuilt root, so training rows written on winning
    positions change.
    """
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    board = chess.Board(fen)
    legal = sorted(_legal_set(board))
    assert len(legal) == n_legal, "the position's legal count is part of the fixture"
    cfg = _narrow_cfg(shape)

    def _search(tree, root_ids):
        return run_gumbel_root_many_c(
            None, [chess.Board(fen)], device="cpu",
            rng=np.random.default_rng(0), cfg=cfg,
            evaluator=_WinningRootEv(ev_seed),
            tree=tree, root_node_ids=root_ids,
            allowed_root_indices_batch=None,  # selfplay never sets this
            target_batch=0, vloss_weight=1,
        )

    # Read the post-prune support off a search rather than hardcoding it, so a
    # rules or terminal-detection change makes this fail loudly instead of
    # quietly testing an un-narrowed root.
    probe_tree = _mcts_tree.MCTSTree()
    probe = _search(probe_tree, None)
    support = {int(a) for a in probe_tree.get_children_visits(int(probe[5][0]))[0]}
    assert len(support) == n_support, (
        f"draw prune did not narrow the root as expected: {sorted(support)}"
    )
    assert support < set(legal)

    def _carried_tree():
        """The previous ply's root — every legal move — with one of the moves
        this ply will prune carrying a heavy, extreme-Q subtree. See
        _narrowed_support_tree for the backprop sign: parent-POV q = -1.0."""
        tree = _mcts_tree.MCTSTree()
        rid = tree.add_root(1, 0.0)
        acts = np.array(legal, dtype=np.int32)
        tree.expand(rid, acts, np.full(acts.size, 1.0 / acts.size, dtype=np.float64))
        cid = tree.find_child(rid, next(a for a in legal if a not in support))
        assert cid >= 0
        path = np.array([rid, cid], dtype=np.int32)
        for _ in range(heavy_visits):
            tree.backprop(path, 1.0)
        return tree, rid

    arm_tree, stale_root = _carried_tree()
    misses_before = root_coverage_miss_count()
    narrowed_before = root_support_narrowed_count()
    arm = _search(arm_tree, [stale_root])
    arm_root = int(arm[5][0])

    assert root_support_narrowed_count() == narrowed_before + 1, (
        f"the {n_legal}-child carried root was reused for a "
        f"{n_support}-move search"
    )
    assert root_coverage_miss_count() == misses_before, (
        "the draw prune is routine — it must not raise the W2 alarm counter"
    )
    assert arm_root != stale_root
    arm_actions = {int(a) for a in arm_tree.get_children_visits(arm_root)[0]}
    assert arm_actions == support, (
        "the rebuilt root must hold exactly the post-prune support"
    )

    # Same tree contents, no reuse: the semantic a fresh root gives.
    ctl_tree, _ = _carried_tree()
    ctl = _search(ctl_tree, None)
    assert int(arm[1][0]) == int(ctl[1][0]), (
        f"reused wide root played {int(arm[1][0])}, support-only root played "
        f"{int(ctl[1][0])} — pruned draw children decided the halving"
    )
    assert np.array_equal(arm[0][0], ctl[0][0]), "stored policy row diverged"
    assert float(arm[2][0]) == float(ctl[2][0]), "stored root value diverged"
    assert set(np.nonzero(arm[0][0])[0].tolist()) <= support


def test_the_root_support_gate_is_equality_not_coverage() -> None:
    """The classifier itself, in every direction, asserted on the VERDICT rather
    than on a bool — because which verdict comes back decides which counter
    moves, and conflating the alarm with the routine case is the thing this
    split exists to prevent.

    The old superset reading (``isin(actions, child_actions).all()`` behind a
    ``child_actions.size < actions.size`` early-out) agrees on the equal,
    reordered, wider and same-size-but-different rows — and disagrees on the two
    that matter: a NARROWED support, and the empty support its
    ``actions.size == 0`` early-out waved through.
    """
    tree = _mcts_tree.MCTSTree()
    rid = tree.add_root(1, 0.0)
    acts = np.array([7, 11, 23], dtype=np.int32)
    tree.expand(rid, acts, np.full(3, 1.0 / 3.0, dtype=np.float64))
    classify = gumbel_c_mod._classify_expanded_root_support
    equal = gumbel_c_mod._ROOT_SUPPORT_EQUAL
    missing = gumbel_c_mod._ROOT_SUPPORT_MISSING_ACTION
    narrowed = gumbel_c_mod._ROOT_SUPPORT_NARROWED

    assert classify(tree, rid, np.array([7, 11, 23], dtype=np.int32)) == equal
    assert classify(tree, rid, np.array([23, 7, 11], dtype=np.int64)) == equal, (
        "order-blind"
    )
    assert classify(tree, rid, np.array([7, 11], dtype=np.int32)) == narrowed, (
        "narrowed support: the ROOT is a superset — this is the bug, and it is "
        "ROUTINE, so it must not read as the W2 alarm"
    )
    assert classify(tree, rid, np.array([7, 11, 23, 40], dtype=np.int32)) == missing, (
        "the root is missing an action we are about to search — the W2 alarm"
    )
    assert classify(tree, rid, np.array([7, 11, 40], dtype=np.int32)) == missing, (
        "same size, different members — the size early-out must not decide it, "
        "and a missing action outranks the extra one"
    )
    assert classify(tree, rid, np.empty(0, dtype=np.int32)) == narrowed, (
        "an empty support is narrowing taken to its limit, not a missing action"
    )


def test_selfplay_shaped_tree_carry_still_reuses_its_root(monkeypatch) -> None:
    """Positive direction of W2. Making the support check unconditional — and,
    later, tightening it from coverage to equality — must not quietly disable
    tree carry: if reuse stopped happening altogether every other test here would
    still pass, and the only symptom would be a slower search that throws away
    its tree every ply.

    This walks a selfplay-shaped carry — persistent tree, root advanced with
    ``find_child`` on the played action, ``allowed_root_indices_batch=None``.
    None of these plies narrows its support, so BOTH rejection counters must stay
    flat: equality is not an excuse to rebuild an equal root.
    """
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    board = chess.Board(
        "r1bqkb1r/pp1ppppp/2n2n2/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4"
    )
    tree = _mcts_tree.MCTSTree()
    cfg = GumbelConfig(
        simulations=64, topk=8, c_scale=0.1, temperature=0.0, add_noise=False
    )

    root_ids: list[int] | None = None
    misses_before = root_coverage_miss_count()
    narrowed_before = root_support_narrowed_count()
    reused_plies = 0
    for ply in range(5):
        res = run_gumbel_root_many_c(
            None,
            [board],
            device="cpu",
            rng=np.random.default_rng(7),
            cfg=cfg,
            evaluator=_Ev(7),
            tree=tree,
            root_node_ids=root_ids,
            allowed_root_indices_batch=None,
            target_batch=0,
            vloss_weight=1,
        )
        rid = int(res[5][0])
        if root_ids is not None:
            reused_plies += int(rid == root_ids[0])
        action = int(res[1][0])
        move = {int(move_to_index(mv, board)): mv for mv in board.legal_moves}[action]
        board.push(move)
        child = tree.find_child(rid, action)
        assert child >= 0, f"played action has no child node at ply {ply}"
        root_ids = [child]

    assert reused_plies == 4, (
        f"tree carry stopped reusing roots: only {reused_plies}/4 carried plies "
        "reused their root"
    )
    assert root_coverage_miss_count() == misses_before, (
        "the support check raised the W2 alarm on a legitimately carried root"
    )
    assert root_support_narrowed_count() == narrowed_before, (
        "the equality gate rebuilt a root whose support did not narrow"
    )


def _reset_guard_reporter(
    monkeypatch, *, misses: int = 0, narrowed: int = 0,
) -> None:
    """Clear the reporter's rate limiter and all four reported counters."""
    monkeypatch.setattr(gumbel_c_mod, "_tt_health_next_check", 0.0)
    monkeypatch.setattr(gumbel_c_mod, "_tt_health_reported", (0, 0, 0, 0))
    monkeypatch.setattr(gumbel_c_mod, "_ROOT_COVERAGE_MISSES", misses)
    monkeypatch.setattr(gumbel_c_mod, "_ROOT_NARROWED_REBUILDS", narrowed)
    _mcts_tree.tt_stats(reset=True)


def _tiny_search() -> None:
    cfg = GumbelConfig(
        simulations=2, topk=2, c_scale=0.1, temperature=0.0, add_noise=False
    )
    run_gumbel_root_many_c(
        None,
        [chess.Board()],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=cfg,
        evaluator=_Ev(0),
        target_batch=0,
        vloss_weight=1,
    )


def test_guard_health_is_silent_when_nothing_fired(monkeypatch, capsys) -> None:
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _reset_guard_reporter(monkeypatch)
    _tiny_search()
    assert "[mcts]" not in capsys.readouterr().err


def test_root_coverage_miss_is_reported_on_the_production_path(
    monkeypatch, capsys
) -> None:
    """A counter no production path reads is the defect these guards exist to
    fix. The shared C-path entry point must actually emit the line."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _reset_guard_reporter(monkeypatch)

    board = chess.Board(_EP_FEN)
    legal = np.array(sorted(_legal_set(board)), dtype=np.int32)
    tree = _mcts_tree.MCTSTree()
    stale_root = tree.add_root(1, 0.0)
    subset = legal[:-1]
    tree.expand(
        stale_root,
        subset,
        np.full(subset.size, 1.0 / subset.size, dtype=np.float64),
    )
    cfg = GumbelConfig(
        simulations=8, topk=4, c_scale=0.1, temperature=0.0, add_noise=False
    )
    run_gumbel_root_many_c(
        None, [board], device="cpu", rng=np.random.default_rng(0), cfg=cfg,
        evaluator=_Ev(1), tree=tree, root_node_ids=[stale_root],
        allowed_root_indices_batch=None, target_batch=0, vloss_weight=1,
    )
    assert root_coverage_miss_count() == 1
    capsys.readouterr()

    # The reporter runs at search ENTRY, so the miss above is reported by the
    # NEXT search; clear the rate limiter the way 60s of wall clock would.
    monkeypatch.setattr(gumbel_c_mod, "_tt_health_next_check", 0.0)
    _tiny_search()
    err = capsys.readouterr().err
    assert "[mcts] search guards FIRED" in err, err
    assert "root_coverage_miss=1" in err, err
    # The routine counter rides along on the same line, so an operator reading an
    # alarm can see whether narrowing was also happening.
    assert "root_support_narrowed=0" in err, err


def test_routine_support_narrowing_is_announced_once_not_every_minute(
    monkeypatch, capsys
) -> None:
    """The narrowed counter is ROUTINE and monotonically increasing on winning
    selfplay positions. If it re-triggered the operator line the way the alarms
    do, a worker would emit this message every 60 s forever and the alarms it
    shares a line with would be trained into invisibility. So: announced on the
    first occurrence, silent after — while a real alarm still prints, carrying
    the updated narrowed count with it."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _reset_guard_reporter(monkeypatch, narrowed=7)

    _tiny_search()
    first = capsys.readouterr().err
    assert "root_support_narrowed=7" in first, first
    assert "ROUTINE" in first, first

    monkeypatch.setattr(gumbel_c_mod, "_ROOT_NARROWED_REBUILDS", 4321)
    monkeypatch.setattr(gumbel_c_mod, "_tt_health_next_check", 0.0)
    _tiny_search()
    assert "[mcts]" not in capsys.readouterr().err, (
        "routine narrowing re-triggered the operator line"
    )

    # An actual alarm must still get through, and carry the current count.
    monkeypatch.setattr(gumbel_c_mod, "_ROOT_COVERAGE_MISSES", 1)
    monkeypatch.setattr(gumbel_c_mod, "_tt_health_next_check", 0.0)
    _tiny_search()
    err = capsys.readouterr().err
    assert "root_coverage_miss=1" in err, err
    assert "root_support_narrowed=4321" in err, err


def test_tt_context_reject_is_routine_and_announced_once(monkeypatch, capsys) -> None:
    """Expected history misses are observable without becoming a recurring alarm."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _reset_guard_reporter(monkeypatch)
    monkeypatch.setattr(
        gumbel_c_mod._mcts_tree_ext,
        "tt_stats",
        lambda *a, **k: {
            "probe_hits": 9,
            "reuse": 4,
            "reject": 0,
            "context_reject": 5,
        },
    )
    _tiny_search()
    first = capsys.readouterr().err
    assert "tt_context_reject=5" in first, first
    assert "ROUTINE" in first, first
    assert "tt_donor_reject=0" in first, first

    monkeypatch.setattr(gumbel_c_mod, "_tt_health_next_check", 0.0)
    _tiny_search()
    assert "[mcts]" not in capsys.readouterr().err, (
        "routine TT history rejection re-triggered the operator line"
    )


def test_tt_reject_is_reported_on_the_production_path(monkeypatch, capsys) -> None:
    """The reject counter reaches the operator too. Driven by patching the
    counter SOURCE the production reporter reads, so this pins the wiring — the
    ep-aware key means no real search can produce a reject to observe."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _reset_guard_reporter(monkeypatch)
    monkeypatch.setattr(
        gumbel_c_mod._mcts_tree_ext,
        "tt_stats",
        lambda *a, **k: {"probe_hits": 9, "reuse": 4, "reject": 5},
    )
    _tiny_search()
    err = capsys.readouterr().err
    assert "[mcts] search guards FIRED" in err, err
    assert "tt_donor_reject=5" in err, err


def test_guard_health_reports_once_per_change(monkeypatch, capsys) -> None:
    """Static counters must not spam every search."""
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    _reset_guard_reporter(monkeypatch, misses=3)
    _tiny_search()
    assert "[mcts] search guards FIRED" in capsys.readouterr().err

    monkeypatch.setattr(gumbel_c_mod, "_tt_health_next_check", 0.0)
    _tiny_search()
    assert "[mcts]" not in capsys.readouterr().err
