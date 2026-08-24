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


def test_the_registry_lists_both_arms_alongside_the_raw_evaluator() -> None:
    assert _nnue_ext.provider_names() == ("nnue", "nnue-static", "nnue-qsearch")


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
    for _ in range(60):
        board = chess.Board()
        for _ply in range(90):
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

            if board.is_checkmate():
                mated += 1
                assert value == -_nnue_ext.RESOLVER_MATE_BASE
                assert stats["terminal_mate"] == 1
            elif board.is_repetition(2) or board.is_fifty_moves():
                drawn += 1
                assert value == 0
                assert stats["terminal_draw"] == 1
            else:
                assert stats["terminal_mate"] == 0 or stats["max_depth_seen"] > 0

    # ⚑ The population is asserted, not assumed. A cross-check that happened to
    # draw no in-check positions would pass while testing nothing.
    assert checked > 200, f"only {checked} in-check positions reached the resolver"
    assert mated > 0, "no checkmate was reached; the mate branch went unexercised"
    assert drawn >= 0
