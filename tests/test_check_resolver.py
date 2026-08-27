"""Tests for the recursive check resolver and the two race arms.

⚑ WHAT THE RESOLVER IS FOR. The NNUE evaluation is undefined in check — the
evaluator refuses such a position outright — so a caller has to resolve check
nodes before asking for a number. Resolution is RECURSIVE, because an evasion
can itself give check, and it backs up by MINIMAX, because the side to move has
no choice about being in check and must survive every reply.

⚑ WHAT IT IS NOT. It is not a search. On a position that is NOT in check it
evaluates immediately, even if a mate is available in one move. Several tests
below pin that, because "the resolver should have found that mate" is the
natural wrong expectation and it would justify a large and unwanted change.

⚑ THE SYNTHETIC PACK MAKES MINIMAX READABLE. ``bucket_pack`` is all zeros except
for the final bias of each layer stack, so an evaluation collapses to
``(bucket + 1) * 100`` with ``bucket = (piece_count - 1) // 4``. Every leaf value
in these tests is therefore a hand-computable function of piece count, which is
what makes it possible to assert the BACKUP rather than just its plausibility.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from pathlib import Path

import chess
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.nnue import _nnue_ext
from scripts import nnue_parse
from tests.test_nnue_native_eval import cboard, write_synthetic_pack

ARMS = ("nnue-static", "nnue-qsearch")


@pytest.fixture(scope="module")
def bucket_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """fc2 bias = (bucket + 1) * 1600, everything else zero.

    The same construction test_nnue_native_eval.py uses, built here rather than
    imported: a fixture is resolved by NAME within a module, so importing one
    across files shadows the import and reads as dead code to every linter that
    looks. See the module docstring for why this pack makes minimax readable.
    """
    path = tmp_path_factory.mktemp("resolver") / "bucket.pack"
    write_synthetic_pack(
        path, {"fc2_bias": [(b, (b + 1) * 1600) for b in range(nnue_parse.PSQT_BUCKETS)]}
    )
    return path


def arm(name: str, pack: Path, fens: list[str]) -> tuple[list[int], dict[str, int]]:
    boards = [CBoard.from_board(chess.Board(f)) for f in fens]
    return _nnue_ext.arm_eval(name, str(pack), boards)


def arm_board(name: str, pack: Path, board: chess.Board) -> tuple[int, dict[str, int]]:
    """Evaluate a python-chess board WITH ITS MOVE STACK.

    ⚑ CBoard.from_board carries the game history, and the repetition tests depend
    on it: a position that has occurred before is a draw, and that fact lives in
    the history rather than in the placement.
    """
    values, stats = _nnue_ext.arm_eval(name, str(pack), [CBoard.from_board(board)])
    return values[0], stats


@pytest.fixture(autouse=True)
def _default_arm_config() -> None:
    """Every test starts from the shipped configuration and leaves it there.

    The knobs are process-global (they are read at context construction), so a
    test that changed one and returned would silently retune the rest of the
    file — and the failure would land on whichever test ran next, not on the one
    that caused it.
    """
    _nnue_ext.set_arm_config(
        _nnue_ext.RESOLVER_MAX_DEPTH, _nnue_ext.QSEARCH_MAX_PLY, _nnue_ext.QSEARCH_CHECK_PLIES
    )


# ===========================================================================
# Positions
# ===========================================================================

# Real positions from seeded random playouts, chosen because the resolver's
# recursion reaches the stated depth on each: an evasion gives check, and that
# check's evasion gives check again. Hand-constructing these is unreliable —
# these were FOUND and then cross-checked against python-chess (see
# test_recursion_depth_matches_python_chess).
CHAIN_FENS: dict[int, str] = {
    2: "4k2r/5n1p/6p1/1NP1R3/pP1B4/P4K1R/8/6N1 b - - 0 45",
    3: "4kb1r/pq2rppp/3Q2bn/2p3B1/1Pp2P2/p1nP2P1/3R3P/1N2KBR1 w - - 2 32",
    4: "2r3b1/3k3r/3b3n/1P1Q4/P2p1RP1/R2P4/1NK1B2P/2B5 w - - 4 50",
    5: "1r1n1b1r/1p2p2p/3k1p2/3b3P/pPQq1Pp1/P2p4/1P1NN2R/1R3B1K w - - 1 27",
}

#: The side to move is in check and mates the OPPONENT with one of its evasions,
#: so the resolver's minimax must return a mate score from depth 1.
MATE_AT_DEPTH_1 = "2b1kr2/p3rn1p/p5p1/3P4/P2p3B/Q1N3P1/7R/3BK3 w - - 3 36"

#: Back-rank mate. Used with two different halfmove clocks.
MATED_ROOT = "R5k1/5ppp/8/8/8/8/8/6K1 b - - {} 80"

#: Black king g8 in check from Rg2 on the g-file; every evasion is a king move,
#: so at clock 99 every child crosses the fifty-move line.
FIFTY_MOVE_CHECK = "6k1/8/8/8/8/8/6R1/6K1 b - - {} 80"

#: Perpetual check. From this start, 1.Qe8+ Kh7 2.Qe4+ Kh8 3.Qe8+ repeats.
PERPETUAL_START = "7k/6p1/8/8/4Q3/8/8/6K1 w - - 0 1"

#: Scholar's mate, one move short: Qxf7# is a CAPTURE, so quiescence searches it
#: at any check-ply budget and reaches a mate the static leaf cannot see.
SCHOLAR_MATE = "r1bqk2r/pppp1ppp/5n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 5 4"

#: No captures, no pawns, and rook moves onto the eighth rank — which POLICY_LUT
#: labels PROMO_MAYBE_QUEEN. Quiet checks (Rd8+, Re2+) are available too.
ROOK_TO_BACK_RANK = "4k3/8/8/8/8/8/3R4/4K3 w - - 0 1"


def perpetual(*sans: str) -> chess.Board:
    board = chess.Board(PERPETUAL_START)
    for san in sans:
        board.push_san(san)
    return board


# ===========================================================================
# A value-level mirror of the resolver, over python-chess
# ===========================================================================
#
# ⚑ WHY A VALUE MIRROR AND NOT JUST A SHAPE CHECK. The depth and terminal-counter
# assertions elsewhere in this file constrain the tree the resolver WALKS. They
# say nothing about the number it hands back, and a reviewer demonstrated the gap
# with a mutant that negates the backup only at depth >= 2:
#
#     int32_t from_our_side = (depth >= 2) ? child_value : -child_value;
#
# It survived all 46 tests while returning +700 where the truth is -600. Every
# counter still matched, because the mutant changes no control flow at all.
#
# ⚑ WHAT IS AND IS NOT INDEPENDENT HERE. Move generation, check detection, mate
# and stalemate, repetition and the fifty-move clock all come from python-chess —
# a third party to our C. The synthetic pack's evaluation is arithmetic this file
# already documents ((bucket + 1) * 100), not a second copy of the evaluator.
# `_mirror_insufficient_material` is TRANSCRIBED from cboard_insufficient_material
# rather than taken from python-chess, because the two implement different rules
# (python-chess is stricter about KNN vs K) and the resolver's job is to match the
# board layer it actually calls. That one predicate is therefore not independent,
# and it is the only one; the backup arithmetic the mutant broke is fully checked.

_MIRROR_CLAMP = 32000
_MIRROR_MATE_BASE = 100000
_MIRROR_MATE_STEP = 100


def mirror_eval(board: chess.Board) -> int:
    """The synthetic pack's evaluation: (bucket + 1) * 100, clamped."""
    bucket = (bin(board.occupied).count("1") - 1) // 4
    return max(-_MIRROR_CLAMP, min(_MIRROR_CLAMP, (bucket + 1) * 100))


def _mirror_insufficient_material(board: chess.Board) -> bool:
    """Transcribed from cboard_insufficient_material — see the note above."""
    if board.pawns or board.rooks or board.queens:
        return False
    total = bin(board.occupied).count("1")
    if total <= 2:
        return True
    if total == 3 and (board.knights or board.bishops):
        return True
    if total == 4 and bin(board.bishops).count("1") == 2:
        light = 0x55AA55AA55AA55AA
        on_light = bin(board.bishops & light).count("1")
        return on_light in (0, 2)
    return False


def mirror_drawn(board: chess.Board) -> bool:
    """cboard_search_terminal's boolean: game over, or LC0-style two-fold."""
    if board.halfmove_clock >= 100:
        return True
    if board.is_repetition(3):
        return True
    if _mirror_insufficient_material(board):
        return True
    if not any(board.legal_moves):
        return True
    return board.is_repetition(2)


MirrorLeaf = Callable[[chess.Board, int], int]


def mirror_resolve(board: chess.Board, depth: int, max_depth: int = 32) -> int:
    """cae_resolve_node, in python-chess. Leaf = the static evaluation."""
    return _mirror_resolve(board, depth, max_depth, leaf=lambda b, _d: mirror_eval(b))


def _mirror_resolve(
    board: chess.Board, depth: int, max_depth: int, leaf: MirrorLeaf
) -> int:
    if not board.is_check():
        if mirror_drawn(board):
            return 0
        value = leaf(board, depth)
        # Mate-band values pass through; only an evaluation is clamped.
        if abs(value) >= _MIRROR_MATE_BASE - 256 * _MIRROR_MATE_STEP:
            return value
        return max(-_MIRROR_CLAMP, min(_MIRROR_CLAMP, value))

    moves = list(board.legal_moves)
    # ⚑ Mate before the draw rules, mirroring the C for the same reason.
    if not moves:
        return -(_MIRROR_MATE_BASE - min(depth, 256) * _MIRROR_MATE_STEP)
    if mirror_drawn(board):
        return 0
    if depth >= max_depth:
        return 0

    # `moves` is non-empty here, so the sentinel is always replaced; it is an int
    # rather than None so the return type is one thing.
    best = -(1 << 30)
    for move in moves:
        board.push(move)
        child = _mirror_resolve(board, depth + 1, max_depth, leaf)
        board.pop()
        best = max(best, -child)
    return best


def mirror_qsearch_arm(
    board: chess.Board, max_ply: int, check_plies: int, max_depth: int = 32
) -> int:
    """The qsearch arm end to end: resolver, with quiescence at its leaves.

    Plain negamax with no window. Fail-soft alpha-beta at a full root window
    returns the exact minimax value, so the C's pruning must not change the
    answer — which is the point of comparing against an unpruned mirror.

    ⚑ `handoff` is the CARRIED quiescence ply. Restarting it at 0 inside an
    excursion refunds the budget and does not terminate; using the resolver's
    depth makes an in-check root start over budget. Both were real, and both are
    what this mirror pins.
    """

    def q_leaf(b: chess.Board, depth: int, handoff: int) -> int:
        return _q_node(b, handoff, depth)

    def _q_node(b: chess.Board, qply: int, depth: int) -> int:
        stand_pat = mirror_eval(b)
        if qply >= max_ply or depth >= max_depth:
            return stand_pat
        best = stand_pat
        for move in b.legal_moves:
            tactical = b.is_capture(move) or move.promotion is not None
            if not tactical and qply >= check_plies:
                continue
            b.push(move)
            try:
                if not tactical and not b.is_check():
                    continue
                if b.is_check():
                    child = _mirror_resolve(
                        b, depth + 1, max_depth,
                        leaf=lambda bb, dd, h=qply + 1: q_leaf(bb, dd, h),
                    )
                elif mirror_drawn(b):
                    child = 0
                else:
                    child = _q_node(b, qply + 1, depth + 1)
            finally:
                b.pop()
            best = max(best, -child)
        return best

    return _mirror_resolve(board, 0, max_depth, leaf=lambda b, d: q_leaf(b, d, 0))


def python_chess_chain_depth(board: chess.Board, depth: int = 0) -> int:
    """The resolver's recursion depth, reimplemented over python-chess.

    ⚑ A DELIBERATELY INDEPENDENT IMPLEMENTATION. It shares no code with the C
    resolver — different move generator, different repetition and fifty-move
    rules, different terminal detection. Internal equivalence between our C and
    our own reimplementation of the same rule would be worth nothing; the value
    here is that python-chess is a THIRD party to the rule.
    """
    if not board.is_check():
        return depth
    if board.is_repetition(2) or board.is_fifty_moves() or board.is_insufficient_material():
        return depth
    deepest = depth
    for move in board.legal_moves:
        board.push(move)
        deepest = max(deepest, python_chess_chain_depth(board, depth + 1))
        board.pop()
    return deepest


# ===========================================================================
# The invariant: the evaluator is never asked about a position in check
# ===========================================================================


@pytest.mark.parametrize("fen", [CHAIN_FENS[2], MATE_AT_DEPTH_1, MATED_ROOT.format(0)])
def test_the_raw_evaluator_still_refuses_what_the_arms_now_answer(
    fen: str, bucket_pack: Path
) -> None:
    """The backstop is unchanged; the arms are what make check positions askable.

    ⚑ If a future edit ever "fixed" the evaluator to return something for a
    position in check, this test would go green while the whole design lost its
    enforcement — so it asserts the refusal, not just that the arms work.
    """
    with pytest.raises(_nnue_ext.InCheckError):
        _nnue_ext.provider_eval("nnue", str(bucket_pack), cboard(fen))
    for name in ARMS:
        values, _stats = arm(name, bucket_pack, [fen])
        assert isinstance(values[0], int)


def test_the_arms_agree_with_the_raw_evaluator_on_a_quiet_position(bucket_pack: Path) -> None:
    """Resolution must be a no-op where there is nothing to resolve."""
    fen = chess.STARTING_FEN
    raw = _nnue_ext.provider_eval("nnue", str(bucket_pack), cboard(fen))
    for name in ARMS:
        values, stats = arm(name, bucket_pack, [fen])
        assert values[0] == raw
        assert stats["nodes"] == 1
        assert stats["resolved_leaves"] == 1
        assert stats["calls_in_check"] == 0


# ===========================================================================
# Recursion
# ===========================================================================


@pytest.mark.parametrize("expected_depth", sorted(CHAIN_FENS))
def test_recursion_depth_matches_python_chess(expected_depth: int, bucket_pack: Path) -> None:
    """Evasion-gives-check chains, 2 to 5 plies deep, against a third party."""
    fen = CHAIN_FENS[expected_depth]
    assert python_chess_chain_depth(chess.Board(fen)) == expected_depth
    _values, stats = arm("nnue-static", bucket_pack, [fen])
    assert stats["max_depth_seen"] == expected_depth
    assert stats["depth_cutoffs"] == 0
    # A chain deeper than one ply is the whole point: "extend one ply out of
    # check" would stop at 1 and evaluate a position that is still in check.
    assert expected_depth >= 2


def test_a_deep_chain_visits_more_nodes_than_it_resolves(bucket_pack: Path) -> None:
    """The expansion factor is a real ratio, not a constant 1."""
    _values, stats = arm("nnue-static", bucket_pack, [CHAIN_FENS[5]])
    assert stats["nodes"] > stats["resolved_leaves"] > 0
    assert stats["calls"] == 1
    assert stats["calls_in_check"] == 1


# ===========================================================================
# Terminals
# ===========================================================================


def test_a_mated_root_scores_in_the_mate_band(bucket_pack: Path) -> None:
    for name in ARMS:
        values, stats = arm(name, bucket_pack, [MATED_ROOT.format(0)])
        assert values[0] == -_nnue_ext.RESOLVER_MATE_BASE
        assert stats["terminal_mate"] == 1
        assert stats["resolved_leaves"] == 0


def test_checkmate_outranks_a_fifty_move_claim(bucket_pack: Path) -> None:
    """⚑ ORDERING, AND IT IS NOT COSMETIC.

    ``cboard_is_game_over`` tests the halfmove clock FIRST and never asks whether
    the side to move is in check. So a resolver that consulted the draw rules
    before counting evasions would score this checkmate as a draw — and a draw is
    a perfectly plausible-looking 0 that nothing downstream could distinguish
    from a real one.
    """
    board = chess.Board(MATED_ROOT.format(100))
    assert board.is_checkmate()
    assert board.halfmove_clock >= 100
    for name in ARMS:
        value, stats = arm_board(name, bucket_pack, board)
        assert value == -_nnue_ext.RESOLVER_MATE_BASE
        assert stats["terminal_mate"] == 1
        assert stats["terminal_draw"] == 0


def test_the_fifty_move_rule_fires_inside_the_resolution(bucket_pack: Path) -> None:
    """Depth 0 is in check; every evasion is a king move that crosses the clock."""
    board = chess.Board(FIFTY_MOVE_CHECK.format(99))
    assert board.is_check()
    assert not board.is_checkmate()
    value, stats = arm_board("nnue-static", bucket_pack, board)
    assert value == 0
    assert stats["max_depth_seen"] == 1
    assert stats["terminal_draw"] == board.legal_moves.count()
    assert stats["resolved_leaves"] == 0  # nothing was ever evaluated


def test_a_root_already_over_the_clock_is_a_draw_without_searching(bucket_pack: Path) -> None:
    value, stats = arm("nnue-static", bucket_pack, [FIFTY_MOVE_CHECK.format(100)])
    assert value[0] == 0
    assert stats["nodes"] == 1
    assert stats["terminal_draw"] == 1


def test_stalemate_scores_zero_rather_than_an_evaluation(bucket_pack: Path) -> None:
    fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    assert chess.Board(fen).is_stalemate()
    for name in ARMS:
        values, stats = arm(name, bucket_pack, [fen])
        assert values[0] == 0
        assert stats["terminal_draw"] == 1
        assert stats["resolved_leaves"] == 0


# ===========================================================================
# Repetition — the reason the recursion is finite
# ===========================================================================


def test_perpetual_check_is_a_draw_at_the_repeated_root(bucket_pack: Path) -> None:
    board = perpetual("Qe8+", "Kh7", "Qe4+", "Kh8", "Qe8+")
    assert board.is_check()
    assert board.is_repetition(2)
    for name in ARMS:
        value, stats = arm_board(name, bucket_pack, board)
        assert value == 0
        assert stats["nodes"] == 1
        assert stats["terminal_draw"] == 1


def test_a_repetition_created_by_the_resolvers_own_push_is_detected(bucket_pack: Path) -> None:
    """⚑ THE HISTORY IS THE WHOLE GAME, NOT A WINDOW.

    Black is in check and one evasion (…Kh8) returns to the position the game
    started from — three plies back, through the resolver's OWN push. Nothing in
    the placement says "draw"; only the history does. And the minimax then PICKS
    that draw over the evaluations, which is what makes the assertion on the
    returned value an assertion about the backup and not just the terminal.
    """
    board = perpetual("Qe8+", "Kh7", "Qe4+")
    assert board.is_check()
    assert not board.is_repetition(2)

    value, stats = arm_board("nnue-static", bucket_pack, board)
    assert stats["terminal_draw"] >= 1
    assert stats["max_depth_seen"] == 1

    # Three pieces => bucket 0 => every leaf evaluates to 100 for the side to
    # move there, i.e. -100 for the side to move at the root. The repetition is
    # worth 0, which is strictly better, so minimax must return 0.
    assert stats["resolved_leaves"] > 0
    assert value == 0


# ===========================================================================
# Minimax backup
# ===========================================================================


def test_the_backup_maximises_for_the_side_to_move(bucket_pack: Path) -> None:
    """⚑ THE MUTANT THIS EXISTS FOR IS A MINIMISING BACKUP.

    Every evasion here leads to a leaf worth 100 to the OPPONENT (-100 to us) and
    one leads to a repetition worth 0. Maximising returns 0; minimising returns
    -100. Both are legal-looking integers in the evaluation band, which is why a
    test that only asserted "some number came back" would not notice.
    """
    board = perpetual("Qe8+", "Kh7", "Qe4+")
    value, _stats = arm_board("nnue-static", bucket_pack, board)
    assert value == 0, "a minimising backup would return -100 here"


def test_a_shorter_mate_scores_higher_than_a_longer_one() -> None:
    """The depth term, read off the C constants rather than restated."""
    base, step = _nnue_ext.RESOLVER_MATE_BASE, _nnue_ext.RESOLVER_MATE_PLY_STEP
    scores = [-(base - d * step) for d in range(6)]
    # From the mated side's own point of view: a mate further away is less bad.
    assert scores == sorted(scores)
    # And once negated for the mating side, sooner is better.
    assert [-s for s in scores] == sorted((-s for s in scores), reverse=True)


def test_a_mate_delivered_by_an_evasion_backs_up_as_a_win(bucket_pack: Path) -> None:
    board = chess.Board(MATE_AT_DEPTH_1)
    assert board.is_check()
    value, stats = arm_board("nnue-static", bucket_pack, board)
    expected = _nnue_ext.RESOLVER_MATE_BASE - 1 * _nnue_ext.RESOLVER_MATE_PLY_STEP
    assert value == expected
    assert stats["terminal_mate"] == 1
    assert stats["max_depth_seen"] == 1


def test_the_resolver_does_not_search_a_quiet_position_for_mate(bucket_pack: Path) -> None:
    """⚑ Pinning what this is NOT, so nobody "fixes" it into a search.

    White mates in one here and is not in check. The resolver evaluates and
    returns a plain evaluation, because resolving check is its entire job.
    """
    fen = "6k1/1R6/8/8/8/8/R7/6K1 w - - 0 1"   # Ra8# is available
    board = chess.Board(fen)
    assert not board.is_check()
    assert any(child.is_checkmate() for child in _pushed(board))
    values, stats = arm("nnue-static", bucket_pack, [fen])
    assert stats["nodes"] == 1
    assert stats["resolved_leaves"] == 1
    assert abs(values[0]) < _nnue_ext.RESOLVER_EVAL_CLAMP


def _pushed(board: chess.Board) -> list[chess.Board]:
    out = []
    for move in board.legal_moves:
        board.push(move)
        out.append(board.copy(stack=False))
        board.pop()
    return out


# ===========================================================================
# Score scale
# ===========================================================================


def test_the_mate_band_cannot_overlap_the_evaluation_band() -> None:
    """⚑ AUDIT N1'S DEFECT CLASS, IN THIS SCALE.

    The codebase once carried two mate->cp formulas, one of which folded mates
    into a range real evaluations reach, so on 1.34% of live rows a mate scored
    BELOW a plain evaluation. The fix was total band separation, and
    tests/test_mate_score_single_home.py pins it for the cp mapping. The resolver
    needs the same property in its own units, and this is where it is pinned.
    """
    clamp = _nnue_ext.RESOLVER_EVAL_CLAMP
    floor = _nnue_ext.RESOLVER_MATE_BASE - _nnue_ext.RESOLVER_MAX_PLIES * (
        _nnue_ext.RESOLVER_MATE_PLY_STEP
    )
    assert floor > clamp, "the deepest mate must outrank the largest evaluation"
    # Every representable mate, at every depth the resolver can reach.
    for depth in range(_nnue_ext.RESOLVER_MAX_PLIES + 1):
        score = _nnue_ext.RESOLVER_MATE_BASE - depth * _nnue_ext.RESOLVER_MATE_PLY_STEP
        assert score > clamp
        assert -score < -clamp


def test_the_recursion_cap_is_reachable_within_the_mate_band() -> None:
    """A mate at the cap must still be scored as a mate, not fall out of band."""
    assert _nnue_ext.RESOLVER_MAX_DEPTH <= _nnue_ext.RESOLVER_MAX_PLIES


def test_evaluations_are_clamped_into_the_evaluation_band(bucket_pack: Path) -> None:
    fens = [chess.STARTING_FEN, "8/5pk1/6p1/8/1P6/5PKP/8/8 w - - 0 40"]
    for name in ARMS:
        values, _stats = arm(name, bucket_pack, fens)
        for value in values:
            assert abs(value) <= _nnue_ext.RESOLVER_EVAL_CLAMP


# ===========================================================================
# Arm fairness
# ===========================================================================


def test_the_qsearch_arm_cannot_bypass_the_resolver(bucket_pack: Path) -> None:
    """⚑⚑ THE FAIRNESS INVARIANT, AS AN OBSERVATION.

    Check resolution is mandatory correctness work. If the qsearch arm could
    reach the evaluator on a position in check it would be both wrong and
    cheaper, and the race would read the saving as a win. So on a position in
    check the qsearch arm must show the resolver ran: the call counted as
    in-check, and the resolver visited more than the one node.
    """
    for fen in (CHAIN_FENS[3], CHAIN_FENS[5]):
        _values, stats = arm("nnue-qsearch", bucket_pack, [fen])
        assert stats["calls"] == 1
        assert stats["calls_in_check"] == 1
        assert stats["nodes"] > 1, "the resolver did not expand the check"
        assert stats["depth_cutoffs"] == 0


def test_both_arms_resolve_the_same_check_the_caller_asked_about(bucket_pack: Path) -> None:
    """The shared component does identical work in both arms.

    ⚑ The arms' TOTAL resolver counts differ, and that is correct: quiescence
    walks into check nodes of its own and resolving those is its cost. What must
    match is the resolution of the check the CALLER handed in — so this compares
    the arms with quiescence turned off, which is the same code with the same
    leaf, and it must agree exactly.
    """
    _nnue_ext.set_arm_config(_nnue_ext.RESOLVER_MAX_DEPTH, 0, 0)
    for fen in CHAIN_FENS.values():
        static_values, static_stats = arm("nnue-static", bucket_pack, [fen])
        qs_values, qs_stats = arm("nnue-qsearch", bucket_pack, [fen])
        assert static_values == qs_values
        for key in ("nodes", "resolved_leaves", "terminal_mate", "terminal_draw",
                    "max_depth_seen", "calls_in_check"):
            assert static_stats[key] == qs_stats[key], key


def test_quiescence_off_makes_the_qsearch_arm_the_static_arm(bucket_pack: Path) -> None:
    """The arm's own negative control, and the proof the ply knob is live.

    ⚑ SCHOLAR_MATE IS IN THE LIST FOR A REASON. On the synthetic pack an
    evaluation depends only on piece count, so it is the same for both sides and
    a capture can never beat standing pat — quiescence can only move a value by
    finding a MATE or a draw. Without a position where it finds one, "quiescence
    changed nothing" would be indistinguishable from "quiescence is not running",
    and this test would pass with the arm switched off.
    """
    fens = [chess.STARTING_FEN, SCHOLAR_MATE, *CHAIN_FENS.values()]
    baseline, _stats = arm("nnue-static", bucket_pack, fens)

    _nnue_ext.set_arm_config(_nnue_ext.RESOLVER_MAX_DEPTH, 0, 0)
    off_values, off_stats = arm("nnue-qsearch", bucket_pack, fens)
    assert off_values == baseline
    assert off_stats["qsearch_max_ply"] == 0
    assert off_stats["qnodes"] == off_stats["resolved_leaves"]

    _nnue_ext.set_arm_config(_nnue_ext.RESOLVER_MAX_DEPTH, 4, 1)
    on_values, on_stats = arm("nnue-qsearch", bucket_pack, fens)
    assert on_stats["qsearch_max_ply"] == 4
    assert on_stats["qnodes"] > off_stats["qnodes"]
    assert on_values != baseline, "quiescence changed nothing, so it is not running"


def test_quiescence_finds_a_mate_a_static_leaf_cannot_see(bucket_pack: Path) -> None:
    """⚑ AND THE MATE SURVIVES THE RESOLVER'S CLAMP.

    Qxf7# is a capture, so quiescence searches it at any check-ply budget; the
    child is in check and goes back through the shared resolver, which scores the
    mate. The resolver then clamps its leaf hook's return value into the
    evaluation band UNLESS it is a mate score — and an unconditional clamp would
    turn this 99,900 into 32,000, demoting a forced mate to a merely good
    position with nothing downstream able to tell.
    """
    static_value, _s = arm("nnue-static", bucket_pack, [SCHOLAR_MATE])
    _nnue_ext.set_arm_config(32, 1, 0)
    q_value, stats = arm("nnue-qsearch", bucket_pack, [SCHOLAR_MATE])

    expected = _nnue_ext.RESOLVER_MATE_BASE - _nnue_ext.RESOLVER_MATE_PLY_STEP
    assert q_value[0] == expected
    assert q_value[0] > _nnue_ext.RESOLVER_EVAL_CLAMP
    assert abs(static_value[0]) <= _nnue_ext.RESOLVER_EVAL_CLAMP
    assert stats["terminal_mate"] == 1


def test_a_back_rank_rook_lift_is_not_a_promotion(bucket_pack: Path) -> None:
    """⚑ POLICY_LUT's `promotion` FIELD IS SPECULATIVE, NOT A FACT.

    init_policy_lut stamps PROMO_MAYBE_QUEEN on every entry landing on rank 0 or
    7 regardless of the moving piece, leaving cboard_push to resolve it. Reading
    it as "this move promotes" made quiescence classify every back-rank rook and
    queen move as tactical, so it searched them at ANY check-ply budget and the
    budget silently did not govern what it was named for.

    Here White has no capture and no pawn, so with checks off quiescence must do
    nothing at all: exactly one node, the root's own stand-pat.
    """
    _nnue_ext.set_arm_config(32, 4, 0)
    _values, stats = arm("nnue-qsearch", bucket_pack, [ROOK_TO_BACK_RANK])
    assert stats["qnodes"] == 1, "a quiet rook move was searched as if it promoted"
    assert stats["nodes"] == 1


# ===========================================================================
# The knobs reach the code that runs
# ===========================================================================


def test_the_backed_up_value_matches_an_independent_minimax(bucket_pack: Path) -> None:
    """⚑⚑ THE VALUE AT EVERY DEPTH, against a python-chess mirror.

    The mutant this exists for negates the backup only at depth >= 2:

        int32_t from_our_side = (depth >= 2) ? child_value : -child_value;

    It changes no control flow, so every depth and terminal counter in this file
    still matched and all 46 tests passed — while CHAIN_FENS[3] came back +700
    instead of -600. Only a value-level comparison can see it.
    """
    for depth, fen in sorted(CHAIN_FENS.items()):
        values, _stats = arm("nnue-static", bucket_pack, [fen])
        assert values[0] == mirror_resolve(chess.Board(fen), 0), f"chain depth {depth}"
    # The mirror is only worth something if it disagrees with a wrong answer, and
    # these positions are the ones the surviving mutant got wrong.
    assert mirror_resolve(chess.Board(CHAIN_FENS[3]), 0) == -600
    assert mirror_resolve(chess.Board(CHAIN_FENS[4]), 0) == 400
    assert mirror_resolve(chess.Board(CHAIN_FENS[5]), 0) == -600


def test_the_qsearch_arm_value_matches_an_independent_minimax(bucket_pack: Path) -> None:
    """The whole composed mechanism — resolver, quiescence, and the carried ply.

    The mirror is unpruned negamax; the C is fail-soft alpha-beta. At a full root
    window those must agree exactly, so this also pins that the pruning is sound.
    """
    _nnue_ext.set_arm_config(32, 2, 1)
    fens = [chess.STARTING_FEN, SCHOLAR_MATE, *CHAIN_FENS.values()]
    values, _stats = arm("nnue-qsearch", bucket_pack, fens)
    for fen, value in zip(fens, values, strict=True):
        assert value == mirror_qsearch_arm(chess.Board(fen), 2, 1), fen


# ===========================================================================
# The quiescence budgets, exercised where they were broken
# ===========================================================================


def test_the_check_budget_governs_from_an_IN_CHECK_root(bucket_pack: Path) -> None:
    """⚑⚑ THE BUG THIS EXISTS FOR: BOTH BUDGETS WERE THE RESOLVER'S DEPTH.

    Quiescence used to be handed the resolver's depth-from-root as its ply. The
    resolver only calls its leaf hook at a NON-CHECK node, so an in-check root
    entered quiescence at ply >= 1 — and `try_checks = (ply < check_plies)` at the
    default check_plies=1 was therefore FALSE on exactly the positions the check
    budget exists for. Every knob test in this file used a quiet root, where
    ply == 0 and the bug is invisible.

    So: from IN-CHECK roots, raising the check budget must do more work.
    """
    qnodes: list[int] = []
    for check_plies in (0, 1, 2):
        _nnue_ext.set_arm_config(32, 4, check_plies)
        _values, stats = arm("nnue-qsearch", bucket_pack, list(CHAIN_FENS.values()))
        assert stats["qsearch_check_plies"] == check_plies
        qnodes.append(stats["qnodes"])

    assert qnodes[1] > qnodes[0], (
        f"check_plies=1 did no more work than 0 ({qnodes[1]} vs {qnodes[0]}); "
        "the budget is not reaching quiescence from an in-check root"
    )
    assert qnodes[2] > qnodes[1], (
        f"check_plies=2 did no more work than 1 ({qnodes[2]} vs {qnodes[1]})"
    )


def test_the_ply_budget_governs_from_an_IN_CHECK_root(bucket_pack: Path) -> None:
    """And the ply counter reported back is QUIESCENCE plies, not resolver depth.

    CHAIN_FENS[5] resolves 5 plies deep, so under the old conflation
    `qmax_ply_seen` would have reported the resolver's depth and the budget would
    have cut quiescence off before it made a single move.
    """
    previous: int = 0
    for max_ply in (0, 1, 2, 3, 4):
        _nnue_ext.set_arm_config(32, max_ply, min(1, max_ply))
        _values, stats = arm("nnue-qsearch", bucket_pack, [CHAIN_FENS[5]])
        assert stats["qsearch_max_ply"] == max_ply
        # ⚑ Exactly the budget, not merely bounded by it: a counter that tracked
        # resolver depth would exceed it (this chain resolves 5 deep).
        assert stats["qmax_ply_seen"] == max_ply
        assert stats["qnodes"] > previous
        assert stats["depth_cutoffs"] == 0
        previous = stats["qnodes"]


def test_a_check_excursion_does_not_refund_the_quiescence_budget(
    bucket_pack: Path,
) -> None:
    """⚑ The budget is CARRIED through the resolver, not restarted at its leaves.

    Restarting it at 0 is the obvious reading of "quiescence gets its own
    counter", and it does not terminate: quiescence plays a checking move, the
    resolver resolves it, and the resolver's leaves start quiescence again with a
    full allowance. Measured — it ran for minutes on CHAIN_FENS[5] at
    qsearch_max_ply=1 before being killed.

    The observable consequence of carrying it is that the work stays bounded and
    the reported quiescence ply never exceeds the budget, even though the
    resolver recursion around it goes far deeper.
    """
    _nnue_ext.set_arm_config(32, 2, 2)
    _values, stats = arm("nnue-qsearch", bucket_pack, list(CHAIN_FENS.values()))
    assert stats["qmax_ply_seen"] <= 2
    assert stats["max_depth_seen"] > stats["qmax_ply_seen"], (
        "the resolver never went deeper than quiescence, so this position cannot "
        "distinguish a carried budget from a restarted one"
    )
    assert stats["depth_cutoffs"] == 0


def test_quiescence_work_saturates_as_the_recursion_cap_rises(
    bucket_pack: Path,
) -> None:
    """⚑⚑ THE OBSERVATION THAT CATCHES A REFUNDED BUDGET.

    A refund leaves every counter looking obedient — `qmax_ply_seen` still caps at
    max_ply, because each restart caps at max_ply all over again — and on this
    pack it does not even change the returned values, since a side-symmetric
    evaluation means only a mate or a draw can move a number. Both of the obvious
    assertions therefore pass against the bug.

    What a refund cannot hide is that it makes quiescence's cost a function of the
    RESOLVER's cap. With the budget carried, quiescence is bounded by its own
    budget and the work SATURATES: raising the cap past the point where the check
    chains end changes nothing. Measured on these four positions at
    max_ply=2, check_plies=2:

        cap    carried (shipped)    refunded
          6            5,468           8,378
         12            8,446      15,327,851
         32            8,446         (does not finish)

    The order below matters: the cheap comparison runs first, so the mutant dies
    at cap 12 and never reaches the cap-32 case it would hang on. A mutant killed
    by a timeout is reported as inconclusive, not dead.
    """
    fens = list(CHAIN_FENS.values())

    _nnue_ext.set_arm_config(6, 2, 2)
    _v6, s6 = arm("nnue-qsearch", bucket_pack, fens)
    _nnue_ext.set_arm_config(12, 2, 2)
    v12, s12 = arm("nnue-qsearch", bucket_pack, fens)

    assert s12["qnodes"] < 3 * s6["qnodes"], (
        f"doubling the recursion cap multiplied quiescence work "
        f"{s6['qnodes']} -> {s12['qnodes']}; the quiescence budget is being "
        "refunded across check excursions instead of carried"
    )

    _nnue_ext.set_arm_config(32, 2, 2)
    v32, s32 = arm("nnue-qsearch", bucket_pack, fens)
    assert s32["qnodes"] == s12["qnodes"], "quiescence work did not saturate"
    assert v32 == v12

    # And the values are right, not merely stable.
    for fen, value in zip(fens, v32, strict=True):
        assert value == mirror_qsearch_arm(chess.Board(fen), 2, 2, max_depth=32), fen


def test_the_configuration_is_reported_from_the_context_that_ran(bucket_pack: Path) -> None:
    """⚑ ANNOUNCED FROM THE CONSUMER'S OWN PARAMETER.

    A context snapshots its configuration when it is built, so after a
    ``set_arm_config`` the globals and an existing context can legitimately
    disagree. The stats therefore report the CONTEXT's fields. Reading them back
    out of the setter would prove only that the setter remembers its argument.
    """
    _nnue_ext.set_arm_config(9, 3, 2)
    _values, stats = arm("nnue-qsearch", bucket_pack, [chess.STARTING_FEN])
    assert stats["resolver_max_depth"] == 9
    assert stats["qsearch_max_ply"] == 3
    assert stats["qsearch_check_plies"] == 2


def test_a_live_context_keeps_the_configuration_it_was_built_with(bucket_pack: Path) -> None:
    """⚑⚑ THE DISCRIMINATING OBSERVATION, AND arm_eval CANNOT MAKE IT.

    arm_eval builds and drops a context inside one call, so its context and the
    module globals can never disagree — a stats dict that reported the GLOBALS
    would pass every arm_eval-based assertion in this file. A context that
    OUTLIVES a set_arm_config() is the only place the two can differ, and this
    pins which one is reported.

    It also pins the semantics themselves: the setter takes effect at the next
    init(), never on a context already running. A setter that appeared to retune
    a live consumer would be this repo's signature defect wearing a fix's
    clothes.
    """
    _nnue_ext.set_arm_config(32, 4, 1)
    handle = _nnue_ext.arm_open("nnue-qsearch", str(bucket_pack))
    _nnue_ext.arm_handle_eval(handle, [cboard(SCHOLAR_MATE)])
    before = _nnue_ext.arm_stats(handle)

    _nnue_ext.set_arm_config(32, 0, 0)
    fresh_values, fresh_stats = arm("nnue-qsearch", bucket_pack, [SCHOLAR_MATE])
    after = _nnue_ext.arm_stats(handle)

    assert fresh_stats["qsearch_max_ply"] == 0        # the new context obeys
    assert after["qsearch_max_ply"] == 4              # the live one does not
    assert after["resolver_max_depth"] == 32

    # And it is not only the reported number: the live context still SEARCHES.
    _nnue_ext.arm_handle_eval(handle, [cboard(SCHOLAR_MATE)])
    latest = _nnue_ext.arm_stats(handle)
    assert latest["qnodes"] > after["qnodes"] > 1
    assert latest["calls"] == before["calls"] + 1
    # The freshly-built context, with quiescence off, cannot see the mate.
    assert abs(fresh_values[0]) <= _nnue_ext.RESOLVER_EVAL_CLAMP


def test_an_arm_handle_accumulates_across_calls(bucket_pack: Path) -> None:
    """Counters are a RUN's, not a batch's — that is what the handle is for."""
    handle = _nnue_ext.arm_open("nnue-static", str(bucket_pack))
    assert _nnue_ext.arm_stats(handle)["calls"] == 0
    _nnue_ext.arm_handle_eval(handle, [cboard(f) for f in CHAIN_FENS.values()])
    first = _nnue_ext.arm_stats(handle)
    assert first["calls"] == len(CHAIN_FENS)
    assert first["calls_in_check"] == len(CHAIN_FENS)

    _nnue_ext.arm_handle_eval(handle, [cboard(chess.STARTING_FEN)])
    second = _nnue_ext.arm_stats(handle)
    assert second["calls"] == first["calls"] + 1
    assert second["calls_in_check"] == first["calls_in_check"]
    assert second["nodes"] == first["nodes"] + 1
    assert second["max_depth_seen"] == first["max_depth_seen"]  # a max, not a sum


def test_arm_open_refuses_the_raw_evaluator(bucket_pack: Path) -> None:
    with pytest.raises(ValueError, match="not a resolver-backed arm"):
        _nnue_ext.arm_open("nnue", str(bucket_pack))


def test_the_depth_cap_binds_and_is_counted(bucket_pack: Path) -> None:
    """⚑ A COUNTER THAT NEVER FIRES CANNOT SAY THE CAP WAS NEVER HIT.

    ``depth_cutoffs`` is 0 on every position in the bench, which is the reading
    we want — but only if the counter is capable of being nonzero. Capping the
    recursion at 1 ply on a chain that needs 5 makes it fire.
    """
    _nnue_ext.set_arm_config(1, 0, 0)
    _values, stats = arm("nnue-static", bucket_pack, [CHAIN_FENS[5]])
    assert stats["depth_cutoffs"] > 0
    assert stats["max_depth_seen"] == 1


def test_the_check_ply_budget_reaches_quiescence(bucket_pack: Path) -> None:
    """Checking moves cost a push each to find, so the budget must be real."""
    fens = [ROOK_TO_BACK_RANK]
    _nnue_ext.set_arm_config(32, 4, 0)
    _values, no_checks = arm("nnue-qsearch", bucket_pack, fens)
    _nnue_ext.set_arm_config(32, 4, 2)
    _values, with_checks = arm("nnue-qsearch", bucket_pack, fens)
    assert no_checks["qsearch_check_plies"] == 0
    assert with_checks["qsearch_check_plies"] == 2
    assert with_checks["qnodes"] > no_checks["qnodes"]
    # The quiet checks are what the resolver then has to expand, so the SHARED
    # component sees them too — quiescence's own cost, not a surcharge on it.
    assert with_checks["nodes"] > no_checks["nodes"]


@pytest.mark.parametrize(
    ("depth", "max_ply", "check_plies"),
    [
        (0, 0, 0),               # depth below 1
        (300, 0, 0),             # depth beyond the mate band's ply span
        (8, 9, 0),               # quiescence deeper than the recursion cap
        (8, 4, 5),               # more check plies than quiescence plies
        (8, -1, 0),
        (8, 4, -1),
    ],
)
def test_set_arm_config_rejects_an_incoherent_configuration(
    depth: int, max_ply: int, check_plies: int
) -> None:
    with pytest.raises(ValueError, match="must be in"):
        _nnue_ext.set_arm_config(depth, max_ply, check_plies)


# ===========================================================================
# The registry and the tree
# ===========================================================================


def test_the_registry_lists_every_arm_alongside_the_raw_evaluator() -> None:
    """SIX providers, and the tuple is exact rather than a membership check.

    ⚑ THE COUNT IS THE POINT, WHICH IS WHY THIS ASSERTION IS AN EQUALITY. The
    registry is the single answer to "which providers exist", so a provider
    added without a deliberate edit here — or one silently dropped — has to
    break a test rather than be absorbed. This tuple was left at three when
    ``nnue-qsearch-refresh`` landed and the file went red; that is the assertion
    working, not the assertion being wrong.

    What the six are, and why each earns a row:

      nnue                  the raw evaluator, which REFUSES a position in check
      nnue-static           check resolution + a static NNUE leaf
      nnue-qsearch          the same resolution + tactical quiescence: PRODUCTION
      nnue-qsearch-refresh  that same quiescence over a full NNUE refresh at
                            every node — the substrate oracle for the
                            incremental accumulator
      nnue-qsearch-dag      that same quiescence over a canonical position DAG —
                            one evaluation per structural position, persisted
                            across calls
      nnue-fastq            FastQ-4+ (docs/fastq_design.md): a DIFFERENT search
                            on that same DAG — capture/promotion-only moves, SEE
                            and delta pruning, owned evasion recursion, a node
                            budget. ~6 evaluations per call against qsearch's
                            ~72.

    ⚑ THE LAST TWO ARE NOT THE SAME KIND OF THING, WHICH IS THE DISTINCTION THIS
    LIST EXISTS TO KEEP STRAIGHT. `nnue-qsearch-refresh` and `nnue-qsearch-dag`
    are SUBSTRATES: the same search with a different source for the stand-pat
    number, which is what lets a substrate change be proved not to have altered
    the search. `nnue-fastq` is a different SEARCH that happens to share the DAG
    substrate — it is deliberately NOT a fourth CaeQsearchSubstrate, because that
    enum means "where the number comes from" and nothing else.

    Only the first three are installable in MCTSTree — see ARMS,
    test_the_dag_arm_is_not_installable_in_the_tree, and
    test_every_shipped_provider_declares_whether_it_is_reentrant.
    """
    assert _nnue_ext.provider_names() == (
        "nnue",
        "nnue-static",
        "nnue-qsearch",
        "nnue-qsearch-refresh",
        "nnue-qsearch-dag",
        "nnue-fastq",
    )


def test_each_name_installs_a_DIFFERENT_provider_in_the_tree(bucket_pack: Path) -> None:
    """⚑ THE MUTANT IS A HARD-WIRED CAPSULE ATTRIBUTE.

    One module publishes three capsules. The tree's name table therefore carries
    the ATTRIBUTE as well as the module; a table that named only the module and
    read ``value_provider_capsule`` for every entry would accept all three names
    and install the raw evaluator for all three — a value accepted and silently
    ignored, and invisible unless something asks the tree which provider it is
    holding.
    """
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    for name in ("nnue", *ARMS):
        tree = MCTSTree()
        tree.set_value_provider(name, str(bucket_pack))
        try:
            assert tree.value_provider_name() == name
        finally:
            tree.clear_value_provider()


def test_the_tree_evaluates_a_check_position_through_an_arm(bucket_pack: Path) -> None:
    """The seam carries the resolver end to end, not just inside _nnue_ext."""
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    board = cboard(CHAIN_FENS[3])
    tree = MCTSTree()
    tree.set_value_provider("nnue", str(bucket_pack))
    try:
        with pytest.raises(_nnue_ext.InCheckError):
            tree.value_provider_eval(board)
    finally:
        tree.clear_value_provider()

    expected, _stats = arm("nnue-static", bucket_pack, [CHAIN_FENS[3]])
    tree = MCTSTree()
    tree.set_value_provider("nnue-static", str(bucket_pack))
    try:
        assert tree.value_provider_eval(board) == expected[0]
    finally:
        tree.clear_value_provider()


def test_an_arm_can_be_installed_from_its_capsule_directly(bucket_pack: Path) -> None:
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    tree.set_value_provider(_nnue_ext.qsearch_arm_capsule, str(bucket_pack))
    try:
        assert tree.value_provider_name() == "nnue-qsearch"
    finally:
        tree.clear_value_provider()


class _CaeValueProvider(ctypes.Structure):
    """Mirrors CaeValueProvider in mcts/_value_provider.h, field for field."""

    _fields_ = (
        ("name", ctypes.c_char_p),
        ("init", ctypes.c_void_p),
        ("eval", ctypes.c_void_p),
        ("retain", ctypes.c_void_p),
        ("destroy", ctypes.c_void_p),
        ("kernel_name", ctypes.c_void_p),
        ("requires_gil", ctypes.c_int),
    )


class _CaeValueProviderExport(ctypes.Structure):
    """Mirrors CaeValueProviderExport — the capsule payload."""

    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("provider", ctypes.POINTER(_CaeValueProvider)),
        ("in_check_error", ctypes.c_void_p),
    )


def _clone_export_with_requires_gil(source_capsule: object, requires_gil: int):
    """A capsule identical to a WORKING provider's but for ``requires_gil``.

    ⚑ ONE BIT DIFFERS, AND THAT IS WHAT MAKES THE MUTANT LEGIBLE. The vtable is
    copied from the live ``nnue-qsearch`` export, so every function pointer is
    real: with the install guard dropped, the tree installs it and runs, and the
    test fails on "did not raise" rather than dying in a segfault that proves
    nothing. Building the struct from scratch with dummy pointers would make the
    mutant run a crash instead of a failing assertion.

    The layout is asserted against the extension's own numbers below, so a C
    struct change breaks this loudly instead of silently reading the wrong
    offsets — the hazard of describing a C ABI from Python.
    """
    pythonapi = ctypes.pythonapi
    pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    pythonapi.PyCapsule_GetPointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
    pythonapi.PyCapsule_New.restype = ctypes.py_object
    pythonapi.PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)

    raw = pythonapi.PyCapsule_GetPointer(source_capsule, _CAPSULE_NAME)
    assert raw, "could not read the source capsule"
    source = ctypes.cast(raw, ctypes.POINTER(_CaeValueProviderExport)).contents

    # The mirror is only trustworthy if the real capsule reads back correctly.
    assert source.struct_size == ctypes.sizeof(_CaeValueProviderExport), (
        "CaeValueProviderExport layout drifted from this test's mirror"
    )
    assert source.provider.contents.name == b"nnue-qsearch"
    assert source.provider.contents.requires_gil == 0

    # Module-lifetime storage: the capsule hands out a pointer, so these must
    # outlive every use. Kept in a module-level list rather than a local.
    vtable = _CaeValueProvider()
    ctypes.memmove(ctypes.byref(vtable), ctypes.byref(source.provider.contents),
                   ctypes.sizeof(_CaeValueProvider))
    vtable.name = b"fake-non-reentrant"
    vtable.requires_gil = requires_gil

    export = _CaeValueProviderExport()
    export.abi_version = source.abi_version
    export.struct_size = source.struct_size
    export.provider = ctypes.pointer(vtable)
    export.in_check_error = source.in_check_error

    _KEEPALIVE.extend((vtable, export))
    return pythonapi.PyCapsule_New(ctypes.byref(export), _CAPSULE_NAME, None)


_CAPSULE_NAME = b"cae.value_provider.v1"
_KEEPALIVE: list[object] = []


def test_the_tree_refuses_a_provider_that_declares_eval_non_reentrant(
    bucket_pack: Path,
) -> None:
    """⚑⚑ THE GUARD IS AT THE DOOR, NOT THE ABSENCE OF A DOORWAY.

    ``MCTSTree`` evaluates from several threads with the GIL released, so a
    provider whose ``eval`` is not reentrant — the position-DAG arm — must not be
    installable. It is tempting to call that settled by not exporting a capsule
    and not listing the name, and it is NOT: ``resolve_provider_export`` accepts
    a capsule handed in directly, and the name table's own comment invites
    exactly that. An exclusion that rests on unreachability breaks the moment
    someone exports the capsule symmetrically with the other arms, and the rule
    the docs state would never have run.

    So the refusal is tested through a capsule the name table has NEVER heard
    of, carrying a real vtable with ``requires_gil`` set. Nothing about this
    capsule's route resembles the DAG arm's, which is the point: the guard keys
    on the declaration, not on who is asking.
    """
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    hostile = _clone_export_with_requires_gil(_nnue_ext.qsearch_arm_capsule, 1)

    tree = MCTSTree()
    with pytest.raises(ValueError, match="non-reentrant"):
        tree.set_value_provider(hostile, str(bucket_pack))
    # And nothing was installed on the way to refusing.
    assert tree.value_provider_name() is None


def test_the_refusal_keys_on_the_flag_and_not_on_the_capsule_being_unfamiliar(
    bucket_pack: Path,
) -> None:
    """The control the test above needs to mean anything.

    Same construction, same unfamiliar capsule, ``requires_gil`` CLEARED — and
    it installs. Without this, "an unknown capsule is rejected" would explain the
    refusal just as well as the flag does, and the flag would be untested.
    """
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    benign = _clone_export_with_requires_gil(_nnue_ext.qsearch_arm_capsule, 0)

    tree = MCTSTree()
    tree.set_value_provider(benign, str(bucket_pack))
    try:
        assert tree.value_provider_name() == "fake-non-reentrant"
        assert tree.value_provider_eval(cboard(chess.STARTING_FEN)) == arm(
            "nnue-qsearch", bucket_pack, [chess.STARTING_FEN]
        )[0][0]
    finally:
        tree.clear_value_provider()


def test_every_shipped_provider_declares_whether_it_is_reentrant(
    bucket_pack: Path,
) -> None:
    """The three tree-installable providers are reentrant; the DAG-backed two are not.

    ⚑ Read through the capsule ABI — the same field ``MCTSTree`` gates on — so
    this cannot drift from what the guard sees. Neither `nnue-qsearch-dag` nor
    `nnue-fastq` publishes a capsule (deliberately), so from Python their
    declaration is only observable as a refusal. ⚑ THAT REFUSAL IS THE NAME
    TABLE'S, NOT THE FLAG'S — the ergonomic layer, not the enforcement. The flag
    is what stops a capsule handed over directly, and
    tests/test_nnue_incremental.py pins its VALUE by parsing the vtable
    initializer, which is the reading that cannot be satisfied by omission.
    """
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    pythonapi = ctypes.pythonapi
    pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    pythonapi.PyCapsule_GetPointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
    for capsule in (
        _nnue_ext.value_provider_capsule,
        _nnue_ext.static_arm_capsule,
        _nnue_ext.qsearch_arm_capsule,
    ):
        raw = pythonapi.PyCapsule_GetPointer(capsule, _CAPSULE_NAME)
        export = ctypes.cast(raw, ctypes.POINTER(_CaeValueProviderExport)).contents
        assert export.provider.contents.requires_gil == 0, export.provider.contents.name

    # The DAG arm is not in the name table, so this is the table's refusal — the
    # ergonomic layer. The flag is what stops it when a capsule IS supplied,
    # which the two tests above cover.
    tree = MCTSTree()
    for name in ("nnue-qsearch-dag", "nnue-fastq"):
        with pytest.raises(ValueError, match="no value provider named"):
            tree.set_value_provider(name, str(bucket_pack))


def test_arm_eval_refuses_a_provider_that_keeps_no_resolver_statistics(
    bucket_pack: Path,
) -> None:
    """⚑ The ctx cast is guarded by the VTABLE, not by the name string.

    ``arm_eval`` reads a CaeArmCtx out of the ctx the provider returned. The raw
    evaluator's ctx is a CaeNnueWeights, and reading one as the other would be
    out of bounds — so the guard is identity against the two arm vtables, and it
    has to reject rather than guess.
    """
    with pytest.raises(ValueError, match="not a resolver-backed arm"):
        _nnue_ext.arm_eval("nnue", str(bucket_pack), [cboard(chess.STARTING_FEN)])


# ===========================================================================
# Randomised cross-check
# ===========================================================================


def test_randomised_cross_check_against_python_chess(bucket_pack: Path) -> None:
    """The resolver's legality and terminal decisions, against a third party.

    Over seeded random playouts, every in-check position is scored by the C
    resolver and independently classified by python-chess. Three claims are
    checked on each: the recursion depth, that a mated root scores in the mate
    band and nothing else does at depth 0, and that a position python-chess calls
    drawn scores exactly 0.
    """
    import random

    rng = random.Random(20260824)
    checked = 0
    mated = 0
    drawn = 0
    deep = 0
    # ⚑ 250 plies, not 90. At 90 the walk produced 215 in-check positions and ZERO
    # in-check draws, so `drawn > 0` below would have been asserting something the
    # generator cannot deliver — which is how the tautological `drawn >= 0` it
    # replaced got written in the first place. Measured: 60 games x 250 plies gives
    # ~900 in-check roots, 4 mates and 2 draws. The fix for a population that is
    # too thin is a bigger population, not a weaker assertion.
    for _ in range(60):
        board = chess.Board()
        for _ply in range(250):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if not board.is_check():
                continue
            checked += 1
            value, stats = arm_board("nnue-static", bucket_pack, board)

            assert stats["max_depth_seen"] == python_chess_chain_depth(board.copy(stack=True))
            assert stats["depth_cutoffs"] == 0

            # ⚑ THE VALUE, not just the shape. An earlier version of this test
            # asserted the recursion depth and the terminal counters and nothing
            # about what came back, and a reviewer's mutant that flips the backup
            # sign only at depth >= 2 survived all 46 tests in the file. Depth is
            # a property of the tree walked; the value is the thing consumers use.
            assert value == mirror_resolve(board, 0), board.fen()

            if board.is_checkmate():
                mated += 1
                assert value == -_nnue_ext.RESOLVER_MATE_BASE
                assert stats["terminal_mate"] == 1
            elif mirror_drawn(board):
                # ⚑ mirror_drawn, not a hand-picked subset of it. An earlier
                # version tested only repetition and the fifty-move clock, and an
                # in-check K+N vs K — drawn by INSUFFICIENT MATERIAL — fell
                # through to the else branch and was asserted to expand. The
                # draw branch has to be the same predicate the resolver uses.
                drawn += 1
                assert value == 0
                assert stats["terminal_draw"] == 1
            else:
                # A live in-check root that is neither mate nor an immediate draw
                # MUST have been expanded — one node would mean it was scored as
                # a terminal it is not.
                assert stats["nodes"] > 1
                assert stats["max_depth_seen"] >= 1
                deep += 1

    # ⚑ THE POPULATION IS ASSERTED, NOT ASSUMED — and every branch's counter gets
    # a real bound. `assert drawn >= 0` stood here once under this same comment,
    # which is true of any count and therefore says nothing: the draw branch could
    # have gone unexercised for the whole run and this would still have passed.
    assert checked > 700, f"only {checked} in-check positions reached the resolver"
    assert mated > 0, "no checkmate was reached; the mate branch went unexercised"
    assert deep > 500, f"only {deep} roots were actually expanded"
    assert drawn > 0, "no in-check draw was reached; the draw branch went unexercised"


def test_randomised_cross_check_of_the_qsearch_arm(bucket_pack: Path) -> None:
    """The same check, driving the arm that composes quiescence with the resolver.

    ⚑ THE STATIC ARM'S CROSS-CHECK CANNOT COVER THIS. It never enters quiescence,
    so it exercises neither budget, neither the carried handoff ply, nor the
    re-entry from quiescence back into the resolver — the whole mechanism the
    ply-conflation bug lived in. A smaller budget and fewer roots, because the
    mirror is unpruned and Python.
    """
    import random

    _nnue_ext.set_arm_config(32, 2, 1)
    rng = random.Random(987654321)
    checked = 0
    disagreed_with_static = 0
    for _ in range(25):
        board = chess.Board()
        for _ply in range(80):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if not board.is_check():
                continue
            checked += 1
            value, stats = arm_board("nnue-qsearch", bucket_pack, board)
            assert value == mirror_qsearch_arm(board, 2, 1), board.fen()
            assert stats["qmax_ply_seen"] <= 2
            assert stats["depth_cutoffs"] == 0
            static_value, _s = arm_board("nnue-static", bucket_pack, board)
            if static_value != value:
                disagreed_with_static += 1

    assert checked > 60, f"only {checked} in-check positions reached the qsearch arm"
    # ⚑ Quiescence has to have CHANGED something, or this is the static arm's
    # cross-check wearing a different name and it would pass with the arm off.
    assert disagreed_with_static > 0, "quiescence never changed a value"
