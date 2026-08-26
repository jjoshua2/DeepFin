"""FastQ-4+ — the §8 verification plan of docs/fastq_design.md.

Every test here is paired with a NAMED MUTANT that was run, watched to fail, and
reverted; the commit message carries the transcripts. A test that has not killed
its mutant does not count as done, and several in this file were rewritten after
their first mutant walked straight through them.

⚑ WHAT THE ARM IS. `nnue-fastq` answers one question per leaf — "is the static
value tactically unstable here, and if so what is the corrected value?" — over
the canonical position DAG, with capture/promotion-only move generation, SEE and
delta pruning, owned evasion recursion and a node budget. It is NOT a fourth
`CaeQsearchSubstrate`: see the ⚑⚑ block at the top of _fastq_search.h for why
that distinction is load-bearing rather than stylistic.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.nnue import _nnue_ext
from scripts import nnue_parse
from tests.test_nnue_native_eval import write_synthetic_pack

ARM = "nnue-fastq"
REFERENCE_ARM = "nnue-qsearch"

#: Dense PSQT so every active feature moves the value. Same reasoning as
#: tests/test_qsearch_dag_parity.py's fixture: a ±32 pack once made a whole
#: module's value assertions near-vacuous.
_PSQT_MAGNITUDE = 2000


@pytest.fixture(scope="module")
def synthetic_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(20260826)
    halfka = rng.integers(
        -_PSQT_MAGNITUDE,
        _PSQT_MAGNITUDE + 1,
        size=nnue_parse.HALFKA_DIMS * nnue_parse.PSQT_BUCKETS,
        dtype=np.int32,
    )
    threats = rng.integers(
        -_PSQT_MAGNITUDE,
        _PSQT_MAGNITUDE + 1,
        size=nnue_parse.THREAT_DIMS * nnue_parse.PSQT_BUCKETS,
        dtype=np.int32,
    )
    path = tmp_path_factory.mktemp("fastq") / "dense-psqt.pack"
    write_synthetic_pack(
        path, blobs={"ft_psqt": [(0, halfka)], "threat_psqt": [(0, threats)]}
    )
    return path


@pytest.fixture(scope="module", params=["synthetic", "real"])
def eval_pack(request: pytest.FixtureRequest, synthetic_pack: Path) -> Path:
    """Both nets from one body; the real one skips when CAE_NNUE_TEST_PACK is unset.

    The synthetic run is mandatory because CI has no 111 MB net. It is also the
    weaker instrument — a PSQT-only pack leaves the accumulator, the transformer
    and every FC activation unexercised — so the real arm is not decoration.
    """
    if request.param == "synthetic":
        return synthetic_pack
    env = os.environ.get("CAE_NNUE_TEST_PACK")
    if not env:
        pytest.skip("needs the real NNUE pack (set CAE_NNUE_TEST_PACK)")
    path = Path(env)
    if not path.is_file():
        pytest.skip(f"CAE_NNUE_TEST_PACK does not exist: {path}")
    return path


@pytest.fixture(autouse=True)
def _fastq_defaults() -> Iterator[None]:
    """Every test starts from the §6 defaults and puts them back.

    ⚑ Explicit rather than relying on the module globals being untouched: the
    knob-threading tests below deliberately set extremes, and a test that
    inherited one of those would be asserting the wrong configuration by
    accident.
    """
    _nnue_ext.fastq_set_config(
        _nnue_ext.FASTQ_MAX_QPLY,
        _nnue_ext.FASTQ_NODE_CAP,
        _nnue_ext.FASTQ_DELTA_MARGIN,
        _nnue_ext.FASTQ_RECAPTURE_EXEMPT,
    )
    yield
    _nnue_ext.fastq_set_config(
        _nnue_ext.FASTQ_MAX_QPLY,
        _nnue_ext.FASTQ_NODE_CAP,
        _nnue_ext.FASTQ_DELTA_MARGIN,
        _nnue_ext.FASTQ_RECAPTURE_EXEMPT,
    )


def _open(pack: Path, **knobs: int) -> object:
    """Open a FastQ arm, optionally setting knobs FIRST.

    ⚑ ORDER MATTERS AND IS THE POINT. A context snapshots the knobs at init(),
    so setting them after arm_open() would change nothing — which is exactly the
    class of defect §6 asks to be proved absent, and is why every helper here
    configures before opening.
    """
    if knobs:
        _nnue_ext.fastq_set_config(
            knobs.get("max_qply", _nnue_ext.FASTQ_MAX_QPLY),
            knobs.get("node_cap", _nnue_ext.FASTQ_NODE_CAP),
            knobs.get("delta_margin", _nnue_ext.FASTQ_DELTA_MARGIN),
            knobs.get("see_recapture_exempt", _nnue_ext.FASTQ_RECAPTURE_EXEMPT),
        )
    return _nnue_ext.arm_open(ARM, str(pack))


def _eval(handle: object, boards: list[chess.Board]) -> list[int]:
    return _nnue_ext.arm_handle_eval(handle, [CBoard.from_board(b) for b in boards])


def _run(pack: Path, boards: list[chess.Board], **knobs: int):
    handle = _open(pack, **knobs)
    values = _eval(handle, boards)
    return values, _nnue_ext.fastq_stats(handle), handle


# ===========================================================================
# §8.7 / §7 — the counter identity, asserted rather than believed
# ===========================================================================


def _assert_counter_identity(stats: dict[str, int]) -> None:
    """§7's evaluate-once identity, in the exact form that is actually true.

    ⚑ THE SPEC SAYS "NNUE evaluations must equal nodes created". That holds only
    where every created node has a static value, and an in-check node
    deliberately has none — the store publishes it with value_valid = 0 because
    the NNUE evaluation is undefined in check. Asserting the spec's wording
    verbatim would fail on any position with a check in it; asserting it with the
    in-check term is the same claim, stated so it can be checked.
    """
    assert (
        stats["nnue_evals"] + stats["nodes_created_in_check"] == stats["nodes_created"]
    ), (
        f"evaluate-once broken: {stats['nnue_evals']} evals + "
        f"{stats['nodes_created_in_check']} in-check != "
        f"{stats['nodes_created']} created"
    )


# ===========================================================================
# §8.1 — the quiet certificate is window-independent
# ===========================================================================

CERT_LOUD = (
    _nnue_ext.CERT_IN_CHECK | _nnue_ext.CERT_PROMOTION | _nnue_ext.CERT_GOOD_CAP
)

#: Two move orders reaching one structural position. Colour-asymmetric so the
#: two orders are genuinely the same node rather than mirror images.
TRANSPOSITION = (
    ("e2e4", "d7d5", "g1f3", "g8f6"),
    ("g1f3", "g8f6", "e2e4", "d7d5"),
)


def _line(moves: tuple[str, ...]) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return board


def test_the_certificate_is_a_function_of_the_position_and_nothing_else() -> None:
    """§3.1: computable from the structural position alone.

    Two move orders, and a third board differing only in halfmove clock — which
    the DAG's identity excludes — must all certify identically.
    """
    a, b = _line(TRANSPOSITION[0]), _line(TRANSPOSITION[1])
    assert a.board_fen() == b.board_fen()
    assert a.turn == b.turn
    cert_a = _nnue_ext.fastq_certificate(CBoard.from_board(a))
    cert_b = _nnue_ext.fastq_certificate(CBoard.from_board(b))
    assert cert_a == cert_b

    clocked = a.copy(stack=False)
    clocked.halfmove_clock = 37
    assert _nnue_ext.fastq_certificate(CBoard.from_board(clocked)) == cert_a


def test_the_certificate_bits_mean_what_they_say() -> None:
    """Each loud bit, isolated, so "quiet" cannot be quiet for the wrong reason."""
    cases = [
        # In check.
        ("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", _nnue_ext.CERT_IN_CHECK),
        # A promotion is available and nothing hangs.
        ("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", _nnue_ext.CERT_PROMOTION),
        # A capture with SEE >= 0 (a free pawn).
        ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", _nnue_ext.CERT_GOOD_CAP),
        # Genuinely quiet: no check, no promotion, and the only capture loses.
        ("4k3/2n5/8/3p4/8/4N3/8/4K3 w - - 0 1", 0),
    ]
    for fen, expected_loud in cases:
        bits = _nnue_ext.fastq_certificate(CBoard.from_board(chess.Board(fen)))
        assert bits & _nnue_ext.CERT_COMPUTED, fen
        assert (bits & CERT_LOUD) == expected_loud, (
            f"{fen}: loud bits {bits & CERT_LOUD:#x}, expected {expected_loud:#x}"
        )


def test_the_stored_certificate_is_the_one_the_position_earns(eval_pack: Path) -> None:
    """⚑⚑ §8 MUTANT 1's TARGET: the STORED bits, compared against a standalone
    computation that has no window at all.

    The mutant folds delta pruning — which reads `alpha` — into the certificate.
    Under it a node first reached beneath a tight window gets stored as quiet
    while the windowless computation calls it loud, and this comparison fails.
    Testing only `fastq_certificate` would miss it entirely, because the mutant
    lives on the path that WRITES the cache, not the one that computes it.
    """
    handle = _open(eval_pack)
    _eval(handle, CORPUS)

    checked = 0
    for board in CORPUS:
        for move in list(board.legal_moves):
            child = board.copy(stack=False)
            child.push(move)
            stored = _nnue_ext.fastq_stored_certificate(
                handle, CBoard.from_board(child)
            )
            if stored is None:
                continue
            assert stored == _nnue_ext.fastq_certificate(CBoard.from_board(child)), (
                f"stored certificate disagrees with the position: {child.fen()}"
            )
            checked += 1
    # Measured: 355 on the synthetic pack, 309 on the real one. The floor is set
    # well below both so a shift in search shape does not fail the test, but far
    # enough above zero that a certificate cache which stopped being written
    # could not slip through as "nothing to compare".
    assert checked >= 150, f"only {checked} stored certificates were compared"


# ===========================================================================
# Corpora
# ===========================================================================

QUIET_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "8/5pk1/6p1/8/1P6/5PKP/8/8 w - - 0 40",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
]

TACTICAL_FENS = [
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r2q1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N1BP2/PP1QN1PP/R3KB1R w KQ - 0 10",
    "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]


def _scripted_corpus(games: int = 24, seed: int = 20260826) -> list[chess.Board]:
    out: list[chess.Board] = []
    rng = random.Random(seed)
    for _game in range(games):
        board = chess.Board()
        for ply in range(40):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if ply >= 6 and ply % 2 == 0:
                out.append(board.copy(stack=False))
    return out


CORPUS = (
    [chess.Board(f) for f in QUIET_FENS + TACTICAL_FENS] + _scripted_corpus()
)


# ===========================================================================
# §8.2 — evaluate once per canonical position
# ===========================================================================


def test_a_transposition_is_evaluated_once(eval_pack: Path) -> None:
    """§8 MUTANT 2: skip the canonical lookup and this counts 2 instead of 1."""
    a, b = _line(TRANSPOSITION[0]), _line(TRANSPOSITION[1])
    handle = _open(eval_pack, max_qply=0, node_cap=0)

    _eval(handle, [a])
    after_first = _nnue_ext.fastq_stats(handle)
    _eval(handle, [b])
    after_second = _nnue_ext.fastq_stats(handle)

    assert after_first["nodes_created"] >= 1
    assert after_second["nodes_created"] == after_first["nodes_created"], (
        "the second move order created a node, so the two orders are not sharing "
        "a canonical position"
    )
    assert after_second["hits_cross_call"] == after_first["hits_cross_call"] + 1
    _assert_counter_identity(after_second)


def test_transpositions_inside_one_search_are_evaluated_once(eval_pack: Path) -> None:
    """⚑ THE ROOT TEST ABOVE DOES NOT COVER THE CHILD PATH, AND ITS MUTANT PROVED IT.

    `test_a_transposition_is_evaluated_once` runs at max_qply 0, so the search
    never expands a child and `cae_nnue_dag_intern_child` is never called.
    Mutant 2 — the canonical lookup skipped — walked straight through it. Two
    capture orders converging on one position INSIDE a single search is what
    exercises the child path, and a within-call hit is the observable.
    """
    _values, stats, _handle = _run(eval_pack, CORPUS, node_cap=0)
    assert stats["hits_within_call"] > 0, (
        "no position was reached twice within a single search, so the child "
        "path's canonical lookup is untested here"
    )
    _assert_counter_identity(stats)


def test_evaluations_never_exceed_the_nodes_that_were_created(eval_pack: Path) -> None:
    """The §7 identity over the whole corpus, both nets."""
    _values, stats, _handle = _run(eval_pack, CORPUS)
    _assert_counter_identity(stats)
    assert stats["nodes_created"] > 100, "corpus too small to mean anything"


# ===========================================================================
# §8.3 — no backed-up search value ever reaches the DAG
# ===========================================================================


def test_the_dag_holds_static_values_only(eval_pack: Path) -> None:
    """⚑⚑ §8 MUTANT 3: cache a fail-high value in the node payload.

    Every stored value is compared against a STANDALONE static evaluation of the
    same position — a different code path that never sees a window. A search
    value differs from the static one exactly when the search found something,
    which is precisely the node a caching mutant would poison.

    The corpus spans all the tactical FENs rather than one position: the first
    version of this used a single fixture, and the mutant poisoned nodes
    elsewhere while leaving that one's values untouched.
    """
    weights = _nnue_ext.load(str(eval_pack))
    handle = _open(eval_pack)
    _eval(handle, CORPUS)

    compared = 0
    for board in CORPUS:
        for move in list(board.legal_moves):
            child = board.copy(stack=False)
            child.push(move)
            if child.is_check():
                continue  # published with no static value, by design
            node = _nnue_ext.arm_dag_lookup(handle, CBoard.from_board(child))
            if node is None:
                continue
            stored = _nnue_ext.arm_dag_value(handle, node)
            if stored is None:
                continue
            static = _nnue_ext.evaluate(weights, CBoard.from_board(child))
            assert stored == static, (
                f"a searched value reached the DAG at {child.fen()}: "
                f"stored {stored} != static {static}"
            )
            compared += 1
    # Measured: 797 synthetic, 566 real.
    assert compared >= 300, f"only {compared} stored values were compared"


def test_the_same_node_answers_the_same_under_two_different_windows(
    eval_pack: Path,
) -> None:
    """A two-window fixture: nodes are reached deep first, then asked directly.

    ⚑ IF A FAIL-HIGH BOUND WERE CACHED AS A NODE FACT, this is where it would
    show: the deep visit arrives under a narrow (-beta, -alpha) window inherited
    from its parent, while the direct visit gets the full window. A value that
    moved between the two is a window that got stored.
    """
    handle = _open(eval_pack)
    _eval(handle, CORPUS)

    deep: dict[str, int] = {}
    for board in CORPUS:
        for move in list(board.legal_moves):
            child = board.copy(stack=False)
            child.push(move)
            if child.is_check():
                continue
            node = _nnue_ext.arm_dag_lookup(handle, CBoard.from_board(child))
            if node is None:
                continue
            value = _nnue_ext.arm_dag_value(handle, node)
            if value is not None:
                deep[child.fen()] = value
    assert len(deep) >= 300, f"only {len(deep)} nodes were reached deep"

    roots = [chess.Board(fen) for fen in deep]
    _eval(handle, roots)
    for board in roots:
        node = _nnue_ext.arm_dag_lookup(handle, CBoard.from_board(board))
        assert node is not None
        assert _nnue_ext.arm_dag_value(handle, node) == deep[board.fen()], (
            f"the stored value changed with the window at {board.fen()}"
        )


# ===========================================================================
# §8.4 — the cycle guard
# ===========================================================================

#: ⚑⚑ §8.4 ASKS FOR A CRAFTED REPETITION FIXTURE. IT COULD NOT BE BUILT, AND THE
#: REASON IS A PROPERTY OF THE MOVE POLICY RATHER THAN A GAP IN THE SEARCH.
#:
#: A cycle needs a node to repeat on the DFS path, which needs the piece multiset
#: to return to its starting state. Captures and promotions both change it, and
#: they are the ONLY moves FastQ generates outside check (§3.2). So every edge in
#: a cycle must be an evasion, and therefore every node in it must be in check —
#: an unbroken alternating chain of checks each delivered by a quiet evasion.
#:
#: MEASURED FOUR WAYS, all zero:
#:   * 14,546 scripted positions at qply 12 / 20 / 40, budget off: 0 cycle draws,
#:     0 path ceilings, max_ply_seen saturating at 18 — the search terminates on
#:     its own.
#:   * a direct DFS over exactly that edge class (quiet moves only, in-check
#:     nodes only, depth 12) from 13,989 in-check positions: no cycle exists.
#:   * a CONSTRUCTIVE hunt rather than a corpus sweep, because a mutual-perpetual-
#:     check loop is a composed position and would never appear in game rows:
#:     1,857,031 random sparse positions (2 kings + 2-5 pieces a side), of which
#:     831,189 were in check, each DFS'd to depth 8 requiring every edge to both
#:     evade and give check. Zero cycles.
#:   * before the speculative-promotion fix this was NOT true — back-rank king
#:     and rook moves were being generated as tactical, which handed the search
#:     reversible non-check edges and produced 806,823 cycle draws at qply 40.
#:     The cycles were an artifact of that bug.
#:
#: The guard is KEPT rather than deleted, unlike the SEE king-legality guard which
#: was removed for being unobservable. Two things separate them: §4.3 mandates
#: this one because the DAG explicitly admits back-edges, and §9 defers quiet-
#: check generation — the moment that lands, quiet non-check edges exist and
#: cycles become reachable immediately. The test below is written so it FAILS if
#: that ever changes, which is what turns "unreachable today" into a claim the
#: suite maintains rather than a comment that rots.


def test_no_cycle_is_reachable_under_the_current_move_policy(eval_pack: Path) -> None:
    """⚑ THIS TEST EXISTS TO FAIL WHEN THE MOVE POLICY WIDENS.

    It asserts the measured property above: at a qply far beyond the default and
    with the budget off, the search terminates without ever repeating a node and
    without hitting its structural ceiling. If a later change (quiet checks, §9)
    makes cycles reachable, `cycle_draws` goes positive and this fails — at which
    point §8.4's crafted repetition fixture becomes both constructible and
    required, and the guard it tests is already here waiting for it.
    """
    _values, stats, _handle = _run(eval_pack, CORPUS, max_qply=16, node_cap=0)
    assert stats["nodes"] > 500, "corpus too small for this to mean anything"
    assert stats["path_ceilings"] == 0, (
        "the recursion hit its structural ceiling — something is not terminating"
    )
    assert stats["cycle_draws"] == 0, (
        "a cycle is now reachable, so §8.4's crafted repetition fixture is "
        "constructible: build it and assert the draw score directly"
    )
    _assert_counter_identity(stats)


# ===========================================================================
# §8.5 — the budget counts every node, evasions included
# ===========================================================================

#: ⚑ RE-DERIVED AFTER THE SPECULATIVE-PROMOTION FIX, NOT GUESSED. Kiwipete was
#: the obvious choice and is far too small once back-rank king and rook moves
#: stop being generated as tactical: it runs 8 evaluations, so a cap of 8 never
#: binds and the test passed for the wrong reason. This is the largest FastQ tree
#: found over 9,557 scripted positions — 110 evaluations at qply 4, unbounded.
BUDGET_FEN = "5bnr/p3k1p1/5q2/1pp1p2n/1B3Pp1/5N1B/PP2P1KP/Rq5R w - - 0 32"


def test_the_budget_bounds_the_evaluations_it_is_named_for(eval_pack: Path) -> None:
    """⚑⚑ THE ASSERTION IS ABOUT NNUE EVALUATIONS, NOT ABOUT A COUNTER.

    The first implementation checked the cap on ENTRY to a node — matching the
    shape of §3.3's pseudocode — and bounded nothing: interning happens in the
    parent, before the recursive call, and interning is what evaluates. A call
    with node_cap 32 reached 65 evaluations. So this asserts the cost, and would
    have failed against that version.
    """
    for cap in (4, 8, 16, 32, 64):
        _values, stats, _handle = _run(
            eval_pack, [chess.Board(BUDGET_FEN)], node_cap=cap
        )
        assert stats["budget_trips"] > 0, f"cap {cap} never bound"
        # +1 for the root, which is the position being asked about rather than
        # work the search chose to do.
        assert stats["nnue_evals"] <= cap + 1, (
            f"cap {cap} allowed {stats['nnue_evals']} NNUE evaluations"
        )
        _assert_counter_identity(stats)


#: A check storm: the side to move is in check, so every child of the root is an
#: evasion and a budget that exempted evasions would run the whole tree. Chosen
#: by sweeping for the most evasion nodes among in-check positions (21).
CHECK_STORM_FEN = "r4b1r/pbnp1p1n/4Qk2/2p1q1pp/1N1PpB1P/PP3PP1/4PKBR/R5N1 b - - 2 20"


def test_evasions_are_charged_to_the_budget_too(eval_pack: Path) -> None:
    """§8 MUTANT 5: exempt evasions from the budget.

    ⚑ THE FIXTURE HAS TO BE IN CHECK AT THE ROOT, which is what makes evasion
    expansion the ONLY thing the budget can be charging. On an ordinary position
    the tactical loop's own budget check would hold the line and the mutant would
    survive.
    """
    board = chess.Board(CHECK_STORM_FEN)
    assert board.is_check(), "fixture must be in check for this test to mean anything"

    _values, stats, _handle = _run(eval_pack, [board], node_cap=2, max_qply=8)
    assert stats["evasion_nodes"] > 0
    assert stats["budget_trips"] > 0, "the cap never bound on an all-evasion tree"
    assert stats["nnue_evals"] <= 3, (
        f"a cap of 2 allowed {stats['nnue_evals']} evaluations on an evasion tree"
    )


# ---------------------------------------------------------------------------
# What a budget trip RETURNS, not just that it happened
# ---------------------------------------------------------------------------

#: Rxe7+ is the only tactical move and it gives check, so at node_cap 1 the root
#: spends the whole budget reaching the in-check child and that child's evasion
#: loop trips on iteration 0 — the exact first-iteration case.
TRIP_IN_CHECK_FEN = CAPTURE_CHECK_FEN = "4k3/4r3/8/8/8/8/4R3/4K2R w K - 0 1"


def test_an_in_check_budget_trip_returns_a_value_not_a_supermate(
    eval_pack: Path,
) -> None:
    """⚑⚑ THE BUG THIS CATCHES SHIPPED, AND ITS SYMPTOM LOOKED LIKE SUCCESS.

    An in-check node has no stand-pat — the NNUE evaluation is undefined in
    check — so the evasion loop seeds `best` at -CAE_FASTQ_INF. If the budget
    refuses the FIRST evasion, that seed is what left the function: -200000,
    which the parent negated to +200000. That is twice CAE_RESOLVER_MATE_BASE, a
    score better than mate-in-0, produced by a node the search declined to look
    at — and the §8 harness classifies anything past the eval clamp as a mate, so
    it would have been banked as FastQ finding a forced win.

    ⚑ EVERY PRIOR BUDGET TEST ASSERTED ONLY `budget_trips > 0` AND AN EVALUATION
    COUNT. All of them passed with the bug in place, because none of them looked
    at the number that came back. That is why this asserts the VALUE.

    ⚑ The obvious repair — "return alpha, like the path-ceiling branch" — does
    NOT fix it, which the assertions below would catch: cae_arm_fastq_eval enters
    the root with beta = +CAE_FASTQ_INF, so a first-generation child's alpha IS
    -CAE_FASTQ_INF and returning it reproduces -200000 exactly.
    """
    board = chess.Board(TRIP_IN_CHECK_FEN)
    capture = chess.Move.from_uci("e2e7")
    assert board.gives_check(capture)

    uncapped, _uncapped_stats, _h = _run(eval_pack, [board], node_cap=0)
    values, stats, _handle = _run(eval_pack, [board], node_cap=1)

    # Anti-vacuity: the trip must have happened, in the IN-CHECK branch, on
    # iteration 0. nodes == 2 is root + the in-check child and nothing below it,
    # so the evasion loop provably broke before evaluating a single evasion.
    assert stats["budget_trips"] > 0, "the cap never bound"
    assert stats["evasion_nodes"] == 1, "no in-check node was reached"
    assert stats["nodes"] == 2, (
        f"expected root + the in-check child only, got {stats['nodes']} nodes; "
        "the evasion loop did not trip on its first iteration"
    )

    value = values[0]
    assert abs(value) <= _nnue_ext.RESOLVER_EVAL_CLAMP, (
        f"a budget trip returned {value}, outside the evaluation range "
        f"(±{_nnue_ext.RESOLVER_EVAL_CLAMP}); pre-fix this was ±200000"
    )
    assert abs(value) < _nnue_ext.RESOLVER_MATE_BASE, (
        f"a budget trip returned the mate-band score {value} from a node the "
        "search never looked at"
    )
    # The trip really changed the answer, so the fallback path is what produced
    # the value asserted above rather than the search having finished anyway.
    assert value != uncapped[0]
    _assert_counter_identity(stats)


def test_a_normal_node_budget_trip_returns_at_least_its_stand_pat(
    eval_pack: Path,
) -> None:
    """The other branch: a node that is NOT in check has a stand-pat to fall back
    on, and fail-soft means its answer can only be at or above it.

    Paired with the in-check test above because the two branches reach the budget
    by different routes and only one of them had a value to return.
    """
    board = chess.Board(BUDGET_FEN)
    assert not board.is_check(), "this test is about the non-check branch"

    weights = _nnue_ext.load(str(eval_pack))
    stand_pat = _nnue_ext.evaluate(weights, CBoard.from_board(board))
    values, stats, _handle = _run(eval_pack, [board], node_cap=2)

    assert stats["budget_trips"] > 0, "the cap never bound"
    assert values[0] >= stand_pat, (
        f"a budget trip returned {values[0]}, below the node's own stand-pat "
        f"{stand_pat}; fail-soft cannot go below the bound it started from"
    )
    assert abs(values[0]) <= _nnue_ext.RESOLVER_EVAL_CLAMP
    _assert_counter_identity(stats)


#: Black is in check from Nb6 and EVERY legal evasion gives discovered check back
#: (the Ra4 battery on Ke4 opens the moment the black king leaves rank 4). Crafted,
#: because the corpus has no such row: Kb4/Kd4 stay on the rank, so a3 and the
#: white king's own square are what remove them, leaving Kc5/Kb5/Kc3/Kb3.
ROOT_IN_CHECK_ALL_EVASIONS_CHECK_FEN = "8/8/1N6/8/r1k1K3/P7/8/8 b - - 0 1"


def test_a_trip_below_an_in_check_ROOT_stays_in_range_too(eval_pack: Path) -> None:
    """⚑⚑ THE CLAMP IS NOT BELT-AND-BRACES; THIS IS THE CASE THAT NEEDS IT.

    Returning `beta` instead of `alpha` stops a budget trip from PROMOTING an
    unsearched move, but beta is -alpha_parent — and an in-check parent's alpha
    starts at -CAE_FASTQ_INF as well. So when the parent is itself in check with
    an untouched window, beta IS +CAE_FASTQ_INF and the escape simply moves up a
    level: the child returns +200000, the parent negates it to -200000, and that
    is what leaves the arm.

    ⚑ NO CORPUS ROW REACHES THIS. Swept at node_cap 1/2/3/4 with the clamp
    removed: zero escapes, on all 4xN rows. It needs a root that is in check whose
    evasions give check back, which is a composed shape — so the mutant for the
    clamp survived every corpus-based test until this fixture existed. That is the
    difference between "we could not find it" and "it cannot happen", and only
    one of those is a reason to drop a guard.

    node_cap=1 puts the trip on the root's first evasion, and every evasion here
    gives check, so the assertion does not depend on move ORDER.
    """
    board = chess.Board(ROOT_IN_CHECK_ALL_EVASIONS_CHECK_FEN)
    assert board.is_check(), "the ROOT must be in check"
    moves = list(board.legal_moves)
    assert moves, "fixture is mate; there is no evasion to trip on"
    assert all(board.gives_check(m) for m in moves), (
        "every evasion must give check, or the child may not be an in-check node"
    )

    values, stats, _handle = _run(eval_pack, [board], node_cap=1)
    assert stats["budget_trips"] > 0
    assert stats["evasion_nodes"] >= 2, "root and child must both be in check"
    assert abs(values[0]) <= _nnue_ext.RESOLVER_EVAL_CLAMP, (
        f"a trip under an in-check root returned {values[0]}, outside "
        f"±{_nnue_ext.RESOLVER_EVAL_CLAMP}"
    )


def test_no_value_leaves_the_evaluation_range_anywhere_on_the_corpus(
    eval_pack: Path,
) -> None:
    """The invariant as a sweep, at a cap tight enough to trip constantly.

    A crafted fixture proves the fix on one shape; this proves no OTHER shape
    reaches the same escape. Mate scores are the one legitimate way past the
    clamp, so they are allowed by name rather than by magnitude.
    """
    values, stats, _handle = _run(eval_pack, CORPUS, node_cap=2)
    assert stats["budget_trips"] > 100, "the cap barely bound; sweep is too easy"

    def _legal(v: int) -> bool:
        if abs(v) <= _nnue_ext.RESOLVER_EVAL_CLAMP:
            return True
        # A real mate score: within MATE_BASE, ply-discounted downward.
        floor = _nnue_ext.RESOLVER_MATE_BASE - (
            _nnue_ext.RESOLVER_MATE_PLY_STEP * _nnue_ext.RESOLVER_MAX_PLIES
        )
        return floor <= abs(v) <= _nnue_ext.RESOLVER_MATE_BASE

    bad = [(b.fen(), v) for b, v in zip(CORPUS, values, strict=True) if not _legal(v)]
    assert not bad, f"{len(bad)} values escaped the evaluation range: {bad[:3]}"


#: Raw NNUE output past CAE_RESOLVER_EVAL_CLAMP is unreachable with the
#: production net (max |v| = 4546 over the corpus), so clamp parity with
#: cae_qsearch_node can only be tested against a pack built to exceed it.
#: Measured at this magnitude: 147 of 444 non-check corpus rows evaluate past the
#: clamp, reaching 89044 — which is inside the mate band the §8 harness
#: classifies on.
_HUGE_PSQT_MAGNITUDE = 200_000


@pytest.fixture(scope="module")
def huge_psqt_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(20260826)
    path = tmp_path_factory.mktemp("fastq-huge") / "huge-psqt.pack"
    write_synthetic_pack(
        path,
        blobs={
            "ft_psqt": [
                (
                    0,
                    rng.integers(
                        -_HUGE_PSQT_MAGNITUDE,
                        _HUGE_PSQT_MAGNITUDE + 1,
                        size=nnue_parse.HALFKA_DIMS * nnue_parse.PSQT_BUCKETS,
                        dtype=np.int32,
                    ),
                )
            ],
            "threat_psqt": [
                (
                    0,
                    rng.integers(
                        -_HUGE_PSQT_MAGNITUDE,
                        _HUGE_PSQT_MAGNITUDE + 1,
                        size=nnue_parse.THREAT_DIMS * nnue_parse.PSQT_BUCKETS,
                        dtype=np.int32,
                    ),
                )
            ],
        },
    )
    return path


def test_the_stand_pat_is_clamped_the_way_qsearch_clamps_it(
    huge_psqt_pack: Path,
) -> None:
    """cae_qsearch_node clamps its stand-pat; FastQ reads the same DAG and did not.

    ⚑ THE DAG STORES THE RAW NNUE VALUE ON PURPOSE — that is the store's contract
    and the qsearch-dag arm depends on it — so the clamp belongs at every reader,
    and a reader that skips it is a reader whose values can leave the evaluation
    range without any budget trip being involved.

    ⚑ ANTI-VACUITY IS THE WHOLE DIFFICULTY HERE. With the production net the
    largest raw evaluation on this corpus is 4546 against a clamp of 32000, so
    the clamp is unobservable and a test using it would pass whether or not the
    clamp existed. The first assertion below fails the test if the pack ever
    stops exceeding the clamp, which is what stops this from quietly becoming
    that test again.
    """
    weights = _nnue_ext.load(str(huge_psqt_pack))
    quiet_rows = [b for b in CORPUS if not b.is_check()]
    raw = [
        _nnue_ext.evaluate(weights, CBoard.from_board(b)) for b in quiet_rows
    ]
    over = [v for v in raw if abs(v) > _nnue_ext.RESOLVER_EVAL_CLAMP]
    assert len(over) > 50, (
        f"only {len(over)} raw evaluations exceed the clamp; this pack no longer "
        "exercises clamping and the test below would be vacuous"
    )

    values, _stats, _handle = _run(huge_psqt_pack, quiet_rows, node_cap=0)
    escaped = [
        (b.fen(), v)
        for b, v in zip(quiet_rows, values, strict=True)
        if abs(v) > _nnue_ext.RESOLVER_EVAL_CLAMP
        and abs(v) < _nnue_ext.RESOLVER_MATE_BASE
        - _nnue_ext.RESOLVER_MATE_PLY_STEP * _nnue_ext.RESOLVER_MAX_PLIES
    ]
    assert not escaped, (
        f"{len(escaped)} unclamped static values reached the search output: "
        f"{escaped[:3]}"
    )


# ===========================================================================
# §8.6 — check policy
# ===========================================================================

#: The only way to make progress is a QUIET check. FastQ must not generate it and
#: must return the stand-pat.
QUIET_CHECK_ONLY_FEN = "7k/8/8/8/8/8/6Q1/K7 w - - 0 1"


def test_an_available_quiet_check_does_not_make_a_position_loud(
    eval_pack: Path,
) -> None:
    """§8.6, first half, CERTIFICATE side: "gives check" is not a loud bit.

    The position has no capture and no promotion, so the certificate reads quiet
    and the answer is the static value — even though Qg7+ and Qb7+ are available
    and forcing.

    ⚑⚑ THIS TEST ALONE DOES NOT COVER THE MOVE POLICY, AND ITS MUTANT PROVED IT.
    The §8.6 mutant that generates quiet checks in cae_fastq_tactical_moves WALKED
    THROUGH this body: the certificate short-circuits at line 437 of
    _fastq_search.h and returns BEFORE the move policy is ever consulted, so a
    position that certifies quiet cannot observe what the generator would have
    done. The claim is split in two here for that reason —
    test_the_move_policy_never_generates_a_quiet_check below runs the generator by
    picking positions the certificate calls loud, and is what actually kills it.
    """
    board = chess.Board(QUIET_CHECK_ONLY_FEN)
    assert not board.is_check()
    assert any(
        board.gives_check(m) for m in board.legal_moves
    ), "fixture no longer offers a quiet check"
    assert not any(
        board.is_capture(m) or m.promotion for m in board.legal_moves
    ), "fixture must have no tactical move at all"

    assert not (
        _nnue_ext.fastq_certificate(CBoard.from_board(board)) & CERT_LOUD
    ), "an available quiet check set a loud certificate bit"

    weights = _nnue_ext.load(str(eval_pack))
    values, stats, _handle = _run(eval_pack, [board])
    assert values[0] == _nnue_ext.evaluate(weights, CBoard.from_board(board))
    assert stats["nodes"] == 1, (
        f"FastQ expanded {stats['nodes']} nodes in a position whose only forcing "
        "moves are quiet checks"
    )
    assert stats["quiet_returns"] == 1


#: Positions that are LOUD (so the move generator really runs) and that also offer
#: a quiet check the generator must decline. Swept out of the parity corpus under
#: three constraints that make the expected node count exact rather than
#: approximate: no legal move lands on rank 1 or 8 (which would draw in the
#: speculative-promotion widening documented in _fastq_search.h), no SEE-passing
#: capture gives check (so no child is in check for a legitimate reason), and at
#: least one quiet check has SEE >= 0 (so the SEE gate is not what declines it).
QUIET_CHECK_POLICY_FENS = (
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/pppp1ppp/8/8/3pP3/5N2/PPP2PPP/RNBQKB1R b KQkq e3 0 3",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/8/2p5/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    "rnbqkbnr/2pp1pp1/8/Pp2p2p/5P2/P6P/2PPP1P1/RNBQKBNR b KQkq - 0 5",
)


@pytest.mark.parametrize("fen", QUIET_CHECK_POLICY_FENS)
def test_the_move_policy_never_generates_a_quiet_check(
    eval_pack: Path, fen: str
) -> None:
    """§8.6, first half, GENERATOR side: the mutant-killing half.

    Run at max_qply=1 with the budget disabled and delta pruning switched off by
    an absurd margin, so the expanded set is exactly {captures with SEE >= 0} and
    the node count is a PREDICTION rather than an observation: 1 root + one child
    per SEE-passing capture. All five fixtures were checked against that formula
    before the test was written.

    Two independent readings of one fact, because a bare node count could in
    principle be matched by a different miscount:
      - the exact node total, and
      - evasion_nodes == 0. A generated quiet check produces a child that IS in
        check, so the evasion counter is a direct read on the policy — and under
        the mutant that child also recurses its evasions, which qply does not
        bound (§3.2), so the count blows well past the prediction too.
    """
    board = chess.Board(fen)
    assert not board.is_check()
    captures = [
        m
        for m in board.legal_moves
        if board.is_capture(m)
        and _nnue_ext.see(
            CBoard.from_board(board), m.from_square, m.to_square, m.promotion or 0
        )
        >= 0
    ]
    quiet_checks = [
        m
        for m in board.legal_moves
        if not board.is_capture(m)
        and not m.promotion
        and board.gives_check(m)
        and _nnue_ext.see(
            CBoard.from_board(board), m.from_square, m.to_square, m.promotion or 0
        )
        >= 0
    ]
    # Anti-vacuity: the position must be loud (else the certificate returns first
    # and the generator never runs) and must really offer a declinable check.
    assert captures, "fixture is no longer loud; the generator would not run"
    assert quiet_checks, "fixture no longer offers a SEE-passing quiet check"
    assert not any(board.gives_check(m) for m in captures)
    assert _nnue_ext.fastq_certificate(CBoard.from_board(board)) & CERT_LOUD

    _values, stats, _handle = _run(
        eval_pack, [board], max_qply=1, node_cap=0, delta_margin=1_000_000
    )
    assert stats["nodes"] == 1 + len(captures), (
        f"expanded {stats['nodes']} nodes where the {len(captures)} SEE-passing "
        f"capture(s) predict {1 + len(captures)}; the extra move(s) can only be "
        f"the quiet check(s) {[m.uci() for m in quiet_checks]}"
    )
    assert stats["evasion_nodes"] == 0, (
        "a child was in check, so a check was GENERATED rather than resolved"
    )
    _assert_counter_identity(stats)


def test_a_capture_that_gives_check_gets_exact_evasion_resolution(
    eval_pack: Path,
) -> None:
    """§8.6, second half: checks are RESOLVED, never generated.

    Rxe7+ is a capture, so the move policy yields it; the reply is in check, so
    that node resolves its evasions exactly. Seeing an evasion node at all is the
    proof that the resolution happened rather than the child being evaluated as
    if it were quiet.
    """
    board = chess.Board(CAPTURE_CHECK_FEN)
    capture = chess.Move.from_uci("e2e7")
    assert capture in board.legal_moves
    assert board.is_capture(capture)
    assert board.gives_check(capture)

    _values, stats, _handle = _run(eval_pack, [board])
    assert stats["evasion_nodes"] > 0, (
        "the capture's reply was never treated as a check node"
    )
    _assert_counter_identity(stats)


# ===========================================================================
# §8.8 — every knob provably reaches the C search
# ===========================================================================

#: ⚑⚑ THE MUTATION IS AN EXTREME VALUE, NOT A PLAUSIBLE ONE. §6's variants
#: (max_qply 2/4/6/8) are the experiment matrix; proving a knob is WIRED needs a
#: value whose effect cannot be confused with noise. A knob that is parsed,
#: validated, stored and then never read is this repo's signature defect, and
#: only a measured change in the search's own counters rules it out.
KNOB_FEN = "5bnr/p3k1p1/5q2/1pp1p2n/1B3Pp1/5N1B/PP2P1KP/Rq5R w - - 0 32"

#: The recapture exemption only fires where a SEE-negative capture sits on the
#: square the parent just captured on. Swept for: 11 exemptions, the most found.
RECAPTURE_FEN = "1n1q1b1r/2rk3p/1pp1p2n/3P1pp1/p3PPP1/N6P/PPPBK3/R1Q2B1R w - - 1 23"


def test_max_qply_reaches_the_search(eval_pack: Path) -> None:
    board = [chess.Board(KNOB_FEN)]
    shallow = _run(eval_pack, board, max_qply=0, node_cap=0)[1]
    deep = _run(eval_pack, board, max_qply=6, node_cap=0)[1]
    assert shallow["nodes"] == 1, "max_qply=0 still expanded a child"
    assert deep["nodes"] > shallow["nodes"] * 5
    assert deep["max_ply_seen"] > shallow["max_ply_seen"]


def test_node_cap_reaches_the_search(eval_pack: Path) -> None:
    board = [chess.Board(KNOB_FEN)]
    tight = _run(eval_pack, board, node_cap=2)[1]
    loose = _run(eval_pack, board, node_cap=0)[1]
    assert tight["nnue_evals"] <= 3
    assert loose["nnue_evals"] > tight["nnue_evals"] * 3
    assert tight["budget_trips"] > 0
    assert loose["budget_trips"] == 0


def test_delta_margin_reaches_the_search(eval_pack: Path) -> None:
    """A huge margin disables delta pruning; a zero margin prunes hardest.

    ⚑ The counter alone would not do: `delta_prunes` could move because the
    search shape changed. The node count has to move with it.
    """
    board = [chess.Board(KNOB_FEN)]
    strict = _run(eval_pack, board, delta_margin=0, node_cap=0)[1]
    off = _run(eval_pack, board, delta_margin=1_000_000, node_cap=0)[1]
    assert strict["delta_prunes"] > 0
    assert off["delta_prunes"] == 0, "a margin of a million still pruned something"
    # ⚑ NOT `off["nodes"] > strict["nodes"]`. Searching MORE moves at a node can
    # produce a tighter bound and therefore an EARLIER beta cutoff deeper down,
    # so the node count is not monotone in the margin — measured, it went the
    # other way. Inequality is the honest claim: the knob reaches the search.
    assert off["nodes"] != strict["nodes"], "the margin changed nothing at all"


def test_see_recapture_exempt_reaches_the_search(eval_pack: Path) -> None:
    """The exemption keeps SEE-negative captures on the recapture square.

    The fixture is chosen so the exemption actually fires: with it off, those
    moves are pruned and the tree is smaller.
    """
    # ⚑ THE WHOLE CORPUS, NOT THE SWEPT SINGLE FIXTURE. That fixture fires 11
    # exemptions on the real net and ZERO on the synthetic one, whose values
    # cause an immediate beta cutoff before the move loop is reached — so a
    # single-position version of this test passed on the real arm and was
    # vacuous on the mandatory one.
    on = _run(eval_pack, CORPUS, see_recapture_exempt=1, node_cap=0)[1]
    off = _run(eval_pack, CORPUS, see_recapture_exempt=0, node_cap=0)[1]
    assert on["recapture_exemptions"] > 0, "the exemption never fired"
    assert off["recapture_exemptions"] == 0
    assert off["see_prunes"] > on["see_prunes"], (
        "turning the exemption off did not prune more"
    )
    assert on["nodes"] != off["nodes"]


def test_the_stats_report_the_context_snapshot_not_the_globals(
    eval_pack: Path,
) -> None:
    """⚑⚑ THE ANNOUNCEMENT COMES FROM THE CONSUMER'S OWN PARAMETER.

    A context snapshots the knobs at init(). Changing the globals afterwards must
    not change what this context runs OR what it reports — otherwise a reader
    who set a knob and read it back would be seeing their own write echoed,
    which is exactly how a dead knob passes for a live one.
    """
    handle = _open(eval_pack, max_qply=2, node_cap=7, delta_margin=13)
    before = _nnue_ext.fastq_stats(handle)
    assert (before["max_qply"], before["node_cap"], before["delta_margin"]) == (2, 7, 13)

    _nnue_ext.fastq_set_config(6, 61, 999, 0)
    after = _nnue_ext.fastq_stats(handle)
    assert (after["max_qply"], after["node_cap"], after["delta_margin"]) == (2, 7, 13), (
        "the context's reported configuration followed the globals"
    )

    fresh = _nnue_ext.arm_open(ARM, str(eval_pack))
    fresh_stats = _nnue_ext.fastq_stats(fresh)
    assert (fresh_stats["max_qply"], fresh_stats["node_cap"]) == (6, 61)


@pytest.mark.parametrize(
    ("knobs", "message"),
    [
        ((-1, 32, 200, 1), "max_qply"),
        ((4, -1, 200, 1), "node_cap"),
        ((4, 32, -1, 1), "delta_margin"),
        ((4, 32, 200, 2), "see_recapture_exempt"),
    ],
)
def test_incoherent_knobs_are_rejected(knobs: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _nnue_ext.fastq_set_config(*knobs)


# ===========================================================================
# The arm's place in the module
# ===========================================================================


def test_fastq_is_registered_and_declares_itself_non_reentrant() -> None:
    """⚑⚑ THE REFUSAL YOU SEE HERE IS THE NAME TABLE'S, NOT THE FLAG'S.

    FastQ drives the same non-atomic probe -> evaluate -> publish -> link path as
    the DAG arm AND writes the quiet certificate into the node payload, so it
    must never run on tree threads with the GIL released. It declares
    requires_gil = 1, and MCTSTree's resolve_provider_export refuses any provider
    that does — but that check is not what fires below, because `nnue-fastq` is
    absent from the tree's name table and lookup fails first.

    So this test asserts what is actually true here: the arm is registered, and
    the tree refuses it. The FLAG's value is pinned where it is observable —
    tests/test_nnue_incremental.py parses the vtable initializer — and the flag's
    ENFORCEMENT is pinned in tests/test_check_resolver.py, which forges an
    ABI-valid capsule and watches the install refuse it. Asserting a
    "non-reentrant" message here would be claiming a mechanism that did not run.
    """
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    assert ARM in _nnue_ext.provider_names()
    tree = MCTSTree()
    with pytest.raises(ValueError, match="no value provider named"):
        tree.set_value_provider(ARM, "unused-path")


def test_arm_stats_refuses_a_fastq_handle_instead_of_reporting_zeros(
    eval_pack: Path,
) -> None:
    """⚑⚑ THE NUMBER IT WOULD HAVE RETURNED IS A PLAUSIBLE-LOOKING ZERO.

    `arm_stats` reports the check RESOLVER's counters. FastQ runs its own search
    and touches none of them, so on a FastQ handle every key — `nnue_evals` most
    dangerously — would read 0 no matter how much work the arm did. Zero there is
    indistinguishable from an idle handle, and "0 NNUE evaluations" reads as
    astonishing efficiency rather than as the wrong counter.

    This is not hypothetical: scripts/fastq_reference_arm.py was written against
    `arm_stats` first and reported `nnue-fastq mean 0.00 median 0 max 0` across
    all 467 rows, which is exactly what a perfect result would look like.

    The refusal follows the convention `arm_dag_stats` already set — it raises on
    an arm that owns no store rather than reporting a zero that could be mistaken
    for an idle one.
    """
    handle = _open(eval_pack)
    _eval(handle, [chess.Board(KNOB_FEN)])

    with pytest.raises(ValueError, match="fastq_stats"):
        _nnue_ext.arm_stats(handle)

    # Anti-vacuity: the counter the caller wanted is non-zero, so the refusal
    # replaced a WRONG answer rather than a missing one.
    assert _nnue_ext.fastq_stats(handle)["nnue_evals"] > 0

    # The refusal is arm-specific, not a blanket break of arm_stats.
    reference = _nnue_ext.arm_open(REFERENCE_ARM, str(eval_pack))
    _nnue_ext.arm_handle_eval(reference, [CBoard.from_board(chess.Board(KNOB_FEN))])
    assert _nnue_ext.arm_stats(reference)["nnue_evals"] > 0


def test_fastq_stats_refuses_a_non_fastq_handle_instead_of_reporting_zeros(
    eval_pack: Path,
) -> None:
    """⚑⚑ THE MIRROR OF THE TEST ABOVE, AND THE HALF THAT WAS MISSING.

    Refusing arm_stats() on a FastQ handle closed one direction and left the
    other wide open: only cae_arm_fastq_eval writes ctx->fastq_totals, so
    fastq_stats() on a qsearch handle read a zeroed struct and reported
    nnue_evals = 0 for an arm that had just done thousands of evaluations. Same
    silent-wrongness shape, same plausible-looking zero, opposite direction.

    Fixing one direction of a defect and leaving the other is this codebase's
    documented failure mode for exactly this class, which is why the pair is
    asserted together.
    """
    reference = _nnue_ext.arm_open(REFERENCE_ARM, str(eval_pack))
    _nnue_ext.arm_handle_eval(reference, [CBoard.from_board(chess.Board(KNOB_FEN))])

    with pytest.raises(ValueError, match="arm_stats"):
        _nnue_ext.fastq_stats(reference)

    # Anti-vacuity: the handle really did work, so the zeroed struct would have
    # been a wrong answer rather than an honest "nothing happened".
    assert _nnue_ext.arm_stats(reference)["nnue_evals"] > 0

    # And the refusal is arm-specific: a FastQ handle still answers.
    handle = _open(eval_pack)
    _eval(handle, [chess.Board(KNOB_FEN)])
    assert _nnue_ext.fastq_stats(handle)["nnue_evals"] > 0


def test_a_fastq_context_always_owns_a_dag_store(eval_pack: Path) -> None:
    """⚑ RENAMED, BECAUSE THE OLD NAME PROMISED SOMETHING THE BODY NEVER DID.

    This was `test_the_arm_refuses_to_run_without_its_store`, and its whole body
    was `assert provider_names().count(ARM) == 1` — a registration check wearing
    a refusal check's name. The refusal it claimed (cae_arm_fastq_eval returning
    CAE_VALUE_ERR_NOT_LOADED on a store-less context) is UNREACHABLE from Python
    by construction: the vtable pairs cae_arm_init_fastq with cae_arm_fastq_eval,
    so every context reaching the eval was built by an init that makes a store.
    A test cannot assert an unreachable branch, and pretending otherwise is worse
    than not testing it.

    What IS assertable is the property that makes the branch unreachable, so that
    is what this asserts: every FastQ handle owns a DAG. arm_dag_lookup raises on
    an arm with no store, so a successful lookup is the store's existence
    observed rather than assumed. The init/eval PAIRING itself is pinned in
    tests/test_nnue_incremental.py, which parses the vtable initializer.
    """
    assert _nnue_ext.provider_names().count(ARM) == 1

    handle = _open(eval_pack)
    board = chess.Board()
    _eval(handle, [board])
    # Raises ValueError on an arm that owns no store, so reaching a node id at
    # all is the assertion.
    assert _nnue_ext.arm_dag_lookup(handle, CBoard.from_board(board)) is not None
    assert _nnue_ext.arm_dag_stats(handle)["node_count"] > 0


# ===========================================================================
# §5 move ordering — SEE descending, MVV-LVA as the tiebreak
# ===========================================================================

#: Two captures with IDENTICAL SEE (both 0) and very different victims: Qxd4
#: wins a queen and is recaptured by a queen, axb5 wins a pawn and is recaptured
#: by a pawn. SEE alone cannot separate them; MVV-LVA puts the queen first.
#: Crafted, because the corpus has exactly one two-capture tied-SEE row and its
#: two victims are both pawns (one via en passant), so it cannot discriminate.
MVV_LVA_TIEBREAK_FEN = "4k3/8/p7/1p2p3/P2q4/8/8/3QK3 w - - 0 1"


def test_an_equal_see_tie_is_broken_by_mvv_lva(eval_pack: Path) -> None:
    """§5: "MVV-LVA exists only as the pre-SEE tiebreak" — asserted, not assumed.

    ⚑ ORDERING IS NORMALLY UNOBSERVABLE, WHICH IS WHY THIS USES node_cap=1. With
    a budget of one node the root expands its FIRST move and nothing else, so the
    DAG afterwards contains exactly one child — and which one it is IS the
    ordering decision, read directly rather than inferred from a node count.

    ⚑ Ties are not a corner case here: 233 of the 348 corpus nodes with two or
    more tactical moves (67.0%) contain an equal-SEE tie, and MVV-LVA reorders a
    tie group at 120 of them (34.5%). Leaving ties to move-generation order was
    a silent dependence on an unrelated implementation detail across a third of
    all nodes.
    """
    board = chess.Board(MVV_LVA_TIEBREAK_FEN)
    captures = [m for m in board.legal_moves if board.is_capture(m)]
    assert len(captures) == 2, "fixture must offer exactly two captures"
    by_uci = {m.uci(): m for m in captures}
    preferred, other = by_uci["d1d4"], by_uci["a4b5"]

    # Anti-vacuity: the tie is real and MVV-LVA is the only thing separating them.
    scored = [
        _nnue_ext.see(CBoard.from_board(board), m.from_square, m.to_square, 0)
        for m in captures
    ]
    assert scored[0] == scored[1], "the fixture's SEEs are no longer tied"
    assert board.piece_type_at(preferred.to_square) == chess.QUEEN
    assert board.piece_type_at(other.to_square) == chess.PAWN

    handle = _open(eval_pack, max_qply=1, node_cap=1)
    _eval(handle, [board])

    def child(move: chess.Move) -> object:
        after = board.copy(stack=False)
        after.push(move)
        return _nnue_ext.arm_dag_lookup(handle, CBoard.from_board(after))

    assert child(preferred) is not None, (
        "the higher-victim capture was not the move the search picked first"
    )
    assert child(other) is None, (
        "both children were expanded; node_cap=1 did not isolate the first move"
    )
    assert _nnue_ext.fastq_stats(handle)["nodes"] == 2
