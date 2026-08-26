"""The qsearch-on-DAG retrofit: same search, different evaluation substrate.

``nnue-qsearch-dag`` runs the SAME quiescence as ``nnue-qsearch`` — same move
policy, same ordering, same ply and depth budgets, same fail-soft arithmetic —
and changes only where a node's stand-pat number comes from: one evaluation per
canonical structural position, held in a ``CaePositionDag`` that survives across
calls.

That makes ``nnue-qsearch`` an ORACLE rather than a similar arm, and it is the
oracle this file spends: with the node budget off (its default), every value the
DAG arm returns must be bitwise the value the oracle returns, on a corpus of
several hundred positions, while performing STRICTLY FEWER NNUE evaluations.
Both halves are needed. Parity alone is satisfied by a DAG that caches nothing,
and a lower evaluation count alone is satisfied by a DAG that returns the wrong
number quickly.

⚑ THE PACK'S PSQT MAGNITUDE IS LOAD-BEARING. At the ±32 an earlier fixture used,
the whole graph evaluated inside a couple of internal units and a node returning
some OTHER node's value was indistinguishable from the right one — every value
assertion was near-vacuous. ±2000 separates a single ply by ~100 units, so a
substrate that confused two positions would be caught rather than tolerated.

⚑ THE CORPUS IS DELIBERATELY COLOUR-ASYMMETRIC. Symmetric lines (1.Nf3 Nf6 2.g3
g6) evaluate to 0 from both sides on a PSQT-only net, which made a whole module
of value assertions pass under a mutant that always returned node 0's value. The
scripted lines below are asymmetric and the crafted FENs are material-imbalanced.
"""

from __future__ import annotations

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

ORACLE = "nnue-qsearch"
DAG_ARM = "nnue-qsearch-dag"

#: See the ⚑ in the module docstring — not arbitrary.
_PSQT_MAGNITUDE = 2000

#: Quiescence deep enough to walk real capture chains and cheap enough for CI.
#: One check ply, because a quiet check is the only non-capture this search
#: generates and the DAG must handle those edges too.
_RESOLVER_DEPTH = 12
_QSEARCH_MAX_PLY = 3
_QSEARCH_CHECK_PLIES = 1

#: Hand-picked positions covering the shapes the substrate has to survive:
#: quiet openings, heavy middlegames with live tactics, endgames (small piece
#: counts hit a different NNUE bucket), and positions already in check — which
#: this arm must never intern, because NNUE is undefined there.
QUIET_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pp2pppp/3p1n2/2pP4/4P3/2N5/PPP2PPP/R1BQKBNR b KQkq - 0 5",
    "8/5pk1/6p1/8/1P6/5PKP/8/8 w - - 0 40",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
]

TACTICAL_FENS = [
    # Kiwipete: the standard perft position, dense with captures both ways.
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r2q1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N1BP2/PP1QN1PP/R3KB1R w KQ - 0 10",
    "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 6 9",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
    "rnbq1b1r/ppp2kpp/3p1n2/8/3PP3/8/PPP2PPP/RNBQKB1R w KQ - 1 6",
]

IN_CHECK_FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 3",
    "4k3/8/8/8/8/8/4R3/4K3 b - - 0 1",
]

#: Two move orders reaching one structural position, twice over. Both lines are
#: colour-asymmetric on purpose (see the module docstring), and the corpus
#: containing BOTH orders is what puts real transpositions in front of the DAG.
TRANSPOSITION_LINES = [
    (("e2e4", "d7d5", "g1f3", "g8f6"), ("g1f3", "g8f6", "e2e4", "d7d5")),
    (("d2d4", "e7e6", "c2c4", "c7c5"), ("c2c4", "c7c5", "d2d4", "e7e6")),
]


@pytest.fixture(scope="module")
def dag_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Dense PSQT so every active feature moves the value. See the module ⚑."""
    rng = np.random.default_rng(20260825)
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
    path = tmp_path_factory.mktemp("qsearch-dag") / "dense-psqt.pack"
    write_synthetic_pack(
        path,
        blobs={"ft_psqt": [(0, halfka)], "threat_psqt": [(0, threats)]},
    )
    return path


@pytest.fixture(autouse=True)
def _arm_config() -> Iterator[None]:
    """One configuration for the whole module, restored to the C defaults after.

    ⚑ The fourth argument is passed EXPLICITLY. ``set_arm_config`` resets the
    node cap when it is omitted, and a test that relied on that would be
    asserting the default by accident; the budget tests set it and put it back.
    """
    _nnue_ext.set_arm_config(
        _RESOLVER_DEPTH, _QSEARCH_MAX_PLY, _QSEARCH_CHECK_PLIES, 0
    )
    yield
    _nnue_ext.set_arm_config(
        _nnue_ext.RESOLVER_MAX_DEPTH,
        _nnue_ext.QSEARCH_MAX_PLY,
        _nnue_ext.QSEARCH_CHECK_PLIES,
        _nnue_ext.QSEARCH_DAG_NODE_CAP,
    )


def _scripted_games() -> list[chess.Board]:
    """Deterministic natural positions from seeded legal-move play.

    Sampling every second ply from ply 6 onwards keeps openings, middlegames and
    the odd endgame in one list, and captures/checks arrive on their own.
    """
    out: list[chess.Board] = []
    rng = random.Random(20260825)
    for _game in range(24):
        board = chess.Board()
        for ply in range(40):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if ply >= 6 and ply % 2 == 0:
                out.append(board.copy(stack=True))
    return out


def _transposition_pairs() -> list[chess.Board]:
    """Both move orders of each line, and every position along both."""
    out: list[chess.Board] = []
    for seq_a, seq_b in TRANSPOSITION_LINES:
        for seq in (seq_a, seq_b):
            board = chess.Board()
            for uci in seq:
                board.push(chess.Move.from_uci(uci))
                out.append(board.copy(stack=True))
    return out


def _tactical_children() -> list[chess.Board]:
    """Every capture child of the tactical FENs, alongside its parent.

    ⚑ This is what makes the corpus SHARE SUBTREES rather than merely repeat
    positions: a capture child is a node the parent's own quiescence searches, so
    interning it during the parent's call and hitting it during the child's call
    is exactly the cross-call reuse being measured.
    """
    out: list[chess.Board] = []
    for fen in TACTICAL_FENS:
        parent = chess.Board(fen)
        out.append(parent)
        for move in parent.legal_moves:
            if not parent.is_capture(move):
                continue
            child = parent.copy(stack=True)
            child.push(move)
            out.append(child)
    return out


def _corpus() -> list[chess.Board]:
    boards = [chess.Board(fen) for fen in (*QUIET_FENS, *IN_CHECK_FENS)]
    boards.extend(_tactical_children())
    boards.extend(_transposition_pairs())
    boards.extend(_scripted_games())
    return boards


CORPUS = _corpus()

#: Counters that describe the SEARCH TREE rather than how evaluations were
#: obtained. The whole claim of this PR is that changing the substrate leaves
#: every one of them alone, so they are compared as a block.
SEARCH_SHAPE_KEYS = (
    "calls",
    "calls_in_check",
    "nodes",
    "resolved_leaves",
    "terminal_mate",
    "terminal_draw",
    "depth_cutoffs",
    "max_depth_seen",
    "qnodes",
    "qterminal_draw",
    "qply_cutoffs",
    "qmax_ply_seen",
)


def _cboards(boards: list[chess.Board]) -> list[CBoard]:
    return [CBoard.from_board(board) for board in boards]


def _run(provider: str, pack: Path, boards: list[chess.Board]):
    return _nnue_ext.arm_eval(provider, str(pack), _cboards(boards))


def _assert_store_identity(stats: dict[str, int]) -> None:
    """The DAG's headline invariant, applied to a store the C search drove.

    ⚑ ``state_inits + state_makes == node_count`` and NOT ``nnue_evals <=
    node_count``: the latter holds by construction on every path and gets MORE
    slack as nodes are duplicated, so it is blind to the failure it looks like
    it is watching. The identity is what catches a node published without the
    accounted NNUE work behind it.
    """
    assert stats["state_inits"] + stats["state_makes"] == stats["node_count"]


def test_the_corpus_is_large_and_really_spans_what_it_claims_to() -> None:
    """The acceptance corpus is a premise; assert it rather than assume it.

    A parity test over 12 quiet positions would pass under substrates that a
    tactical position separates immediately, so "several hundred rows, with
    checks and captures in them" has to be a checked property of the fixture.
    """
    assert len(CORPUS) >= 300
    assert sum(1 for b in CORPUS if b.is_check()) >= 3
    assert sum(1 for b in CORPUS if any(b.is_capture(m) for m in b.legal_moves)) >= 100
    # Transpositions: the same STRUCTURAL position under more than one row.
    #
    # ⚑ The key canonicalises en passant the way the DAG does — an ep square is
    # part of identity only when a capture onto it is actually available.
    # Keying on ``board.ep_square`` raw reports 457 distinct positions in 457
    # rows and the assertion passes vacuously, because 1.e4 d5 2.Nf3 Nf6 leaves
    # no ep square while 1.Nf3 Nf6 2.e4 d5 leaves an unusable d6 — one node in
    # the graph, two keys in a test that had not read _position_dag.h.
    keys = [
        (
            b.board_fen(),
            b.turn,
            b.castling_rights,
            b.ep_square if b.has_legal_en_passant() else None,
        )
        for b in CORPUS
    ]
    assert len(set(keys)) < len(keys)


def test_the_dag_arm_returns_the_oracles_value_on_every_corpus_row(dag_pack: Path) -> None:
    """THE headline: one variable changed, and no row moved.

    Bitwise equality on every row, and the search-shape counters equal as a
    block — the DAG arm must not merely agree on values but walk the identical
    tree, because a substrate that pruned differently could agree by luck on a
    corpus and diverge on the next one.
    """
    oracle_values, oracle_stats = _run(ORACLE, dag_pack, CORPUS)
    dag_values, dag_stats = _run(DAG_ARM, dag_pack, CORPUS)

    assert dag_values == oracle_values
    assert {k: dag_stats[k] for k in SEARCH_SHAPE_KEYS} == {
        k: oracle_stats[k] for k in SEARCH_SHAPE_KEYS
    }
    # The oracle owns no store, and says so rather than reporting a bare 0.
    assert oracle_stats["dag_enabled"] == 0
    assert dag_stats["dag_enabled"] == 1


def test_the_dag_arm_evaluates_strictly_fewer_positions_than_the_oracle(
    dag_pack: Path,
) -> None:
    """The evaluate-once win, read off the counters rather than assumed.

    ⚑ ``nnue_evals`` is counted inside the code that calls the evaluator, in
    both arms, so this is a comparison between two consumers' own numbers and
    not between a measurement and a model of one.
    """
    _oracle_values, oracle_stats = _run(ORACLE, dag_pack, CORPUS)
    _dag_values, dag_stats = _run(DAG_ARM, dag_pack, CORPUS)

    # The oracle evaluates once per quiescence node, by construction.
    assert oracle_stats["nnue_evals"] == oracle_stats["qnodes"]
    # The DAG arm walks the same nodes and evaluates strictly fewer of them.
    assert dag_stats["qnodes"] == oracle_stats["qnodes"]
    assert dag_stats["nnue_evals"] < oracle_stats["nnue_evals"]

    # Evaluate-once as an identity, not an aspiration: this arm interns only
    # non-check positions, so every node it created cost exactly one evaluation.
    assert dag_stats["nnue_evals"] == dag_stats["dag_nodes_interned"]
    # And the reuse is where the saving came from.
    assert dag_stats["dag_hits_within_call"] + dag_stats["dag_hits_cross_call"] > 0
    assert (
        dag_stats["nnue_evals"]
        + dag_stats["dag_hits_within_call"]
        + dag_stats["dag_hits_cross_call"]
        == dag_stats["qnodes"]
    )
    assert dag_stats["dag_memory_bytes"] > 0


def test_the_stores_own_identity_holds_through_the_arms_interning(dag_pack: Path) -> None:
    handle = _nnue_ext.arm_open(DAG_ARM, str(dag_pack))
    _nnue_ext.arm_handle_eval(handle, _cboards(CORPUS))
    store = _nnue_ext.arm_dag_stats(handle)
    _assert_store_identity(store)
    arm = _nnue_ext.arm_stats(handle)
    # The two surfaces report ONE store: the arm's created-node count is the
    # store's node count, and the store's evaluation count is the arm's.
    assert store["node_count"] == arm["dag_nodes_interned"]
    assert store["nnue_evals"] == arm["nnue_evals"]
    assert store["node_count"] == arm["dag_node_count"]


def test_a_second_call_over_a_shared_subtree_hits_nodes_the_first_call_created(
    dag_pack: Path,
) -> None:
    """Cross-call reuse: the number FastQ's case rests on, measured here.

    The second call's board is a capture CHILD of the first's, so the two
    searches overlap. ⚑ THEY DO NOT NEST: as a top-level root the child gets the
    full ply budget and the full window, so its tree is LARGER than the pruned,
    budget-reduced version the parent's search walked. The honest claim is
    therefore not "zero new evaluations" — measured, it is 133 — but "strictly
    fewer than the same board costs a store that has never seen the parent",
    which is the saving, priced against the control that makes it a saving.

    Re-running the FIRST board is the case that IS exact: a deterministic search
    over an unchanged store walks the identical tree, so it must cost nothing.
    """
    parent = chess.Board(TACTICAL_FENS[0])
    move = next(m for m in parent.legal_moves if parent.is_capture(m))
    child = parent.copy(stack=True)
    child.push(move)

    handle = _nnue_ext.arm_open(DAG_ARM, str(dag_pack))
    _nnue_ext.arm_handle_eval(handle, _cboards([parent]))
    after_first = _nnue_ext.arm_stats(handle)
    assert after_first["dag_hits_cross_call"] == 0  # nothing existed before it

    # The child really is inside the parent's tree — asserted, not assumed.
    child_node = _nnue_ext.arm_dag_lookup(handle, CBoard.from_board(child))
    assert child_node is not None

    _nnue_ext.arm_handle_eval(handle, _cboards([child]))
    after_second = _nnue_ext.arm_stats(handle)
    warm_evals = after_second["nnue_evals"] - after_first["nnue_evals"]

    # The control: the same board through a store that never saw the parent.
    cold_handle = _nnue_ext.arm_open(DAG_ARM, str(dag_pack))
    _nnue_ext.arm_handle_eval(cold_handle, _cboards([child]))
    cold_evals = _nnue_ext.arm_stats(cold_handle)["nnue_evals"]

    assert after_second["dag_hits_cross_call"] > 0
    assert warm_evals < cold_evals

    # Re-running the FIRST board: an identical tree over an unchanged store.
    _nnue_ext.arm_handle_eval(handle, _cboards([parent]))
    after_third = _nnue_ext.arm_stats(handle)
    assert after_third["nnue_evals"] == after_second["nnue_evals"]
    assert after_third["dag_nodes_interned"] == after_second["dag_nodes_interned"]
    assert after_third["dag_hits_cross_call"] > after_second["dag_hits_cross_call"]
    _assert_store_identity(_nnue_ext.arm_dag_stats(handle))


def test_the_dag_holds_the_static_evaluation_and_never_a_searched_one(
    dag_pack: Path,
) -> None:
    """⚑⚑ THE FORBIDDEN THING, ASSERTED DIRECTLY AT THE NODE.

    A node's stored value must be the STATIC NNUE evaluation of that position —
    what ``evaluate()`` returns — and never the quiescence result backed up
    through it. The test first PROVES the two are distinguishable on this
    position (the qsearch value differs from the static one), because on a quiet
    position they coincide and the assertion would be vacuous.
    """
    weights = _nnue_ext.load(str(dag_pack))
    board = chess.Board(TACTICAL_FENS[0])
    cboard = CBoard.from_board(board)

    static_value = _nnue_ext.evaluate(weights, cboard)
    handle = _nnue_ext.arm_open(DAG_ARM, str(dag_pack))
    searched_value = _nnue_ext.arm_handle_eval(handle, [cboard])[0]
    assert searched_value != static_value, "fixture is quiet; it cannot discriminate"

    node = _nnue_ext.arm_dag_lookup(handle, cboard)
    assert node is not None
    assert _nnue_ext.arm_dag_value(handle, node) == static_value

    # And the same for every capture child the search interned.
    checked = 0
    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
        child = board.copy(stack=True)
        child.push(move)
        child_cb = CBoard.from_board(child)
        child_node = _nnue_ext.arm_dag_lookup(handle, child_cb)
        if child_node is None or child.is_check():
            continue
        assert _nnue_ext.arm_dag_value(handle, child_node) == _nnue_ext.evaluate(
            weights, child_cb
        )
        checked += 1
    assert checked > 0


def test_a_window_bounded_subtree_never_poisons_a_later_full_window_search(
    dag_pack: Path,
) -> None:
    """The two-window fixture: one position, two windows, two correct answers.

    Inside the parent's quiescence every child is searched under a NARROW
    (alpha, beta) and some of them fail high or low — the value that comes back
    is a bound, not the position's value. Asked about the same position at top
    level, the arm searches it under the full window and must produce the exact
    value. Both answers come out of ONE persistent store, so a bound written
    into a node during the first search would be returned as a stand-pat during
    the second.

    Every capture child is checked rather than one hand-picked node, so the test
    does not depend on knowing which child happened to fail high.
    """
    parent = chess.Board(TACTICAL_FENS[0])
    children = []
    for move in parent.legal_moves:
        if not parent.is_capture(move):
            continue
        child = parent.copy(stack=True)
        child.push(move)
        children.append(child)
    assert len(children) >= 4

    exact, _stats = _run(ORACLE, dag_pack, children)

    shared = _nnue_ext.arm_open(DAG_ARM, str(dag_pack))
    # The narrow-window pass: the parent's own search visits every one of them.
    _nnue_ext.arm_handle_eval(shared, _cboards([parent]))
    interned = [
        _nnue_ext.arm_dag_lookup(shared, CBoard.from_board(child)) for child in children
    ]
    assert all(node is not None for node in interned), (
        "the parent's search did not reach these children; the fixture cannot "
        "put a bound on them and the test would be vacuous"
    )

    # The full-window pass, through the SAME store.
    after = _nnue_ext.arm_handle_eval(shared, _cboards(children))
    assert after == exact

    # …and a store that never saw the parent agrees, so the shared-store answer
    # is not merely self-consistent.
    fresh = _nnue_ext.arm_open(DAG_ARM, str(dag_pack))
    assert _nnue_ext.arm_handle_eval(fresh, _cboards(children)) == exact


def test_the_node_budget_ships_off(dag_pack: Path) -> None:
    """Default OFF, pinned against the C constant rather than restated here."""
    assert _nnue_ext.QSEARCH_DAG_NODE_CAP == 0
    _nnue_ext.set_arm_config(_RESOLVER_DEPTH, _QSEARCH_MAX_PLY, _QSEARCH_CHECK_PLIES)
    _values, stats = _run(DAG_ARM, dag_pack, [chess.Board(TACTICAL_FENS[0])])
    assert stats["dag_node_cap"] == 0
    assert stats["dag_budget_trips"] == 0


def test_the_node_budget_binds_the_dag_arm_and_changes_its_answer(
    dag_pack: Path,
) -> None:
    """⚑ THE KNOB-THREADING MUTANT'S TEST. A C path that ignored ``dag_node_cap``
    would leave every one of these assertions at its uncapped value.

    Both halves are asserted: the trip counter fires AND the search really was
    cut short — fewer nodes and, on this fixture, a different answer. A counter
    that fires without changing the search would be a counter, not a mechanism.
    """
    board = chess.Board(TACTICAL_FENS[1])

    _nnue_ext.set_arm_config(_RESOLVER_DEPTH, _QSEARCH_MAX_PLY, _QSEARCH_CHECK_PLIES, 0)
    uncapped, uncapped_stats = _run(DAG_ARM, dag_pack, [board])
    assert uncapped_stats["qnodes"] > 1
    assert uncapped_stats["dag_budget_trips"] == 0

    _nnue_ext.set_arm_config(_RESOLVER_DEPTH, _QSEARCH_MAX_PLY, _QSEARCH_CHECK_PLIES, 1)
    capped, capped_stats = _run(DAG_ARM, dag_pack, [board])

    assert capped_stats["dag_node_cap"] == 1
    assert capped_stats["dag_budget_trips"] > 0
    assert capped_stats["qnodes"] < uncapped_stats["qnodes"]
    assert capped != uncapped

    # Default-off bit-identity is untouched by the cap having been set: this is
    # a per-CONTEXT snapshot, so a context built with 0 keeps 0.
    _nnue_ext.set_arm_config(_RESOLVER_DEPTH, _QSEARCH_MAX_PLY, _QSEARCH_CHECK_PLIES, 0)
    reopened, reopened_stats = _run(DAG_ARM, dag_pack, [board])
    oracle, _oracle_stats = _run(ORACLE, dag_pack, [board])
    assert reopened == oracle == uncapped
    assert reopened_stats["dag_budget_trips"] == 0


def test_the_node_budget_is_not_consulted_by_any_other_arm(dag_pack: Path) -> None:
    """A knob that reaches only one arm has to SAY so from the others' stats.

    ⚑ Reported from each context's own field, so an arm that cannot use the
    budget reports 0 rather than echoing the global that was just written —
    which is the difference between a knob and a knob that looks applied.
    """
    _nnue_ext.set_arm_config(_RESOLVER_DEPTH, _QSEARCH_MAX_PLY, _QSEARCH_CHECK_PLIES, 1)
    boards = [chess.Board(TACTICAL_FENS[1])]
    for name in ("nnue-static", ORACLE, "nnue-qsearch-refresh"):
        _values, stats = _run(name, dag_pack, boards)
        assert stats["dag_node_cap"] == 0, name
        assert stats["dag_budget_trips"] == 0, name
        assert stats["dag_enabled"] == 0, name
    # The oracle is unchanged by a cap that is in force globally.
    with_cap, _ = _run(ORACLE, dag_pack, boards)
    _nnue_ext.set_arm_config(_RESOLVER_DEPTH, _QSEARCH_MAX_PLY, _QSEARCH_CHECK_PLIES, 0)
    without_cap, _ = _run(ORACLE, dag_pack, boards)
    assert with_cap == without_cap


def test_a_live_context_keeps_the_node_cap_it_was_built_with(dag_pack: Path) -> None:
    """The snapshot rule, for the new knob, on the only surface that can see it.

    ``arm_eval`` builds and drops a context inside one call, so its context and
    the globals can never disagree there. A context that OUTLIVES a
    ``set_arm_config`` is the only place they can.
    """
    _nnue_ext.set_arm_config(_RESOLVER_DEPTH, _QSEARCH_MAX_PLY, _QSEARCH_CHECK_PLIES, 4)
    handle = _nnue_ext.arm_open(DAG_ARM, str(dag_pack))
    _nnue_ext.set_arm_config(_RESOLVER_DEPTH, _QSEARCH_MAX_PLY, _QSEARCH_CHECK_PLIES, 0)

    assert _nnue_ext.set_arm_config(
        _RESOLVER_DEPTH, _QSEARCH_MAX_PLY, _QSEARCH_CHECK_PLIES, 0
    )["dag_node_cap"] == 0
    assert _nnue_ext.arm_stats(handle)["dag_node_cap"] == 4


def test_set_arm_config_rejects_a_negative_node_cap() -> None:
    with pytest.raises(ValueError, match="dag_node_cap must be >= 0"):
        _nnue_ext.set_arm_config(
            _RESOLVER_DEPTH, _QSEARCH_MAX_PLY, _QSEARCH_CHECK_PLIES, -1
        )


def test_the_store_persists_until_a_reset_is_explicitly_asked_for(
    dag_pack: Path,
) -> None:
    board = chess.Board(TACTICAL_FENS[0])
    handle = _nnue_ext.arm_open(DAG_ARM, str(dag_pack))
    first = _nnue_ext.arm_handle_eval(handle, _cboards([board]))
    before = _nnue_ext.arm_dag_stats(handle)
    assert before["node_count"] > 1
    # ⚑ Rerooting is the persistence mechanism: the call left the root on the
    # node for the position it was asked about, and every node stayed alive.
    assert before["root_id"] == _nnue_ext.arm_dag_lookup(
        handle, CBoard.from_board(board)
    )

    _nnue_ext.arm_dag_reset(handle)
    cleared = _nnue_ext.arm_dag_stats(handle)
    assert cleared["node_count"] == 0
    assert cleared["root_id"] == -1
    assert cleared["state_inits"] == cleared["state_makes"] == cleared["nnue_evals"] == 0
    # The allocations survive a reset; only the graph semantics are dropped.
    assert cleared["payload_capacity"] == before["payload_capacity"]
    assert _nnue_ext.arm_dag_lookup(handle, CBoard.from_board(board)) is None

    # A cleared store re-derives the same answer from scratch.
    assert _nnue_ext.arm_handle_eval(handle, _cboards([board])) == first
    after = _nnue_ext.arm_dag_stats(handle)
    _assert_store_identity(after)
    assert after["node_count"] == before["node_count"]


def test_an_in_check_root_is_resolved_and_never_interned(dag_pack: Path) -> None:
    """NNUE is undefined in check, so this arm stores no such node.

    The resolver walks the evasions and hands their non-check leaves to
    quiescence; those are what get interned. So every node the arm creates owns
    a real static value, which is why ``nnue_evals == dag_nodes_interned`` is an
    identity for this arm rather than a coincidence.
    """
    boards = [chess.Board(fen) for fen in IN_CHECK_FENS]
    oracle, oracle_stats = _run(ORACLE, dag_pack, boards)

    handle = _nnue_ext.arm_open(DAG_ARM, str(dag_pack))
    values = _nnue_ext.arm_handle_eval(handle, _cboards(boards))
    stats = _nnue_ext.arm_stats(handle)
    store = _nnue_ext.arm_dag_stats(handle)

    assert values == oracle
    assert oracle_stats["calls_in_check"] == len(boards)
    assert stats["nnue_evals"] == stats["dag_nodes_interned"]
    _assert_store_identity(store)
    # Every published node has a value: none of them was in check.
    assert store["nnue_evals"] == store["node_count"]
    for board in boards:
        assert _nnue_ext.arm_dag_lookup(handle, CBoard.from_board(board)) is None


def test_the_arm_dag_accessors_refuse_an_arm_that_owns_no_store(
    dag_pack: Path,
) -> None:
    """A zero from a store that does not exist is not a reading. Refuse instead."""
    handle = _nnue_ext.arm_open(ORACLE, str(dag_pack))
    for call in (
        lambda: _nnue_ext.arm_dag_stats(handle),
        lambda: _nnue_ext.arm_dag_lookup(handle, CBoard.from_board(chess.Board())),
        lambda: _nnue_ext.arm_dag_value(handle, 0),
        lambda: _nnue_ext.arm_dag_reset(handle),
    ):
        with pytest.raises(ValueError, match="not DAG-backed"):
            call()


def test_the_dag_arm_is_not_installable_in_the_tree(dag_pack: Path) -> None:
    """⚑ DELIBERATE, AND THE REASON IS THREADS.

    MCTSTree drives its value provider from several search threads, and this
    store's probe/evaluate/publish path is not atomic — a concurrent publish can
    free the accumulator array another thread is reading. So the arm is reachable
    only through the single-threaded ``arm_open``/``arm_eval`` surface, and the
    tree does not know the name. When that changes, it must change because
    synchronization was added, not because a name was added to a table.
    """
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    # ⚑ Matched on the message, because "raises ValueError" would also be
    # satisfied by the pack failing to load — which would make the test pass for
    # a reason that has nothing to do with the name table.
    with pytest.raises(ValueError, match="no value provider named"):
        tree.set_value_provider(DAG_ARM, str(dag_pack))
