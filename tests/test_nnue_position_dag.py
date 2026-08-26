"""Correctness gates for the reusable structural-position DAG + NNUE payload.

The DAG is intentionally a different abstraction from ``MCTSTree``:

* one structural chess position is one node;
* nodes have no parent pointer and can have multiple incoming edges;
* path-specific halfmove/repetition/history state is NOT node identity;
* each new NNUE node owns one incremental accumulator state and is evaluated at
  most once; a transposition hit reuses that state/value without make/evaluate.

The dense synthetic PSQT pack makes accumulator mistakes visible without needing
the real 100+ MB net in CI.

⚑ The counter gate every stats read here applies is ``state_inits +
state_makes == node_count``, NOT ``nnue_evals <= node_count``. The latter holds
by construction on every path and gets MORE slack as nodes are duplicated: under
a double-publish mutant it only loosens from ``5 <= 5`` to ``5 <= 9`` while the
identity fires. The identity is what catches a node published without the
accounted NNUE work behind it — duplication, or payload/graph desynchronisation.
"""

from __future__ import annotations

import os
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.moves import move_to_index
from chess_anti_engine.nnue import _nnue_ext
from scripts import nnue_parse
from tests.test_nnue_native_eval import write_synthetic_pack


_DAG_STAT_KEYS = {
    "root_id",
    "node_count",
    "edge_count",
    "payload_capacity",
    "probes",
    "hits",
    "inserts",
    "collision_steps",
    "edge_reuses",
    "state_inits",
    "state_makes",
    "nnue_evals",
    "node_reuses",
    "dag_memory_bytes",
    "nnue_payload_bytes",
    "memory_bytes",
}


#: PSQT weight magnitude. ⚑ Load-bearing, not arbitrary: at the original ±32 the
#: WHOLE graph evaluated inside ±2 internal units, so a node returning some other
#: node's value was indistinguishable from the right one and every value
#: assertion in this file was near-vacuous. ±2000 separates a single ply by ~100
#: units, which is what lets
#: ``test_transposition_returns_its_own_value_not_a_stale_one`` prove the merged
#: node's value is unique in the graph.
_PSQT_MAGNITUDE = 2000


@pytest.fixture(scope="module")
def dag_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
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
    path = tmp_path_factory.mktemp("nnue-dag") / "dense-psqt.pack"
    write_synthetic_pack(
        path,
        blobs={
            "ft_psqt": [(0, halfka)],
            "threat_psqt": [(0, threats)],
        },
    )
    return path


def _cboard(board: chess.Board) -> CBoard:
    return CBoard.from_board(board)


def _push(
    dag: object,
    weights: object,
    parent_id: int,
    board: chess.Board,
    uci: str,
) -> tuple[int, chess.Board, bool]:
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    action = move_to_index(move, board)
    child = board.copy(stack=True)
    child.push(move)
    child_cb = _cboard(child)
    node_id, value, created = _nnue_ext.dag_intern_child(
        dag,
        parent_id,
        action,
        child_cb,
    )
    if not child.is_check():
        # ``child.is_check()`` asks whether the NEW side to move is in check,
        # exactly the condition under which the static evaluator refuses.
        assert value == _nnue_ext.evaluate(weights, child_cb)
    else:
        assert value is None
    return node_id, child, created


def _assert_stats_schema(stats: dict[str, int]) -> None:
    assert set(stats) == _DAG_STAT_KEYS
    assert stats["memory_bytes"] == stats["dag_memory_bytes"] + stats["nnue_payload_bytes"]
    # ⚑ THE headline invariant, applied at every stats read in this file: one
    # canonical node <=> exactly one accounted NNUE state construction. A
    # duplicate publish, a node interned without a payload obligation, or a
    # published node whose work was never counted all break this identity —
    # unlike ``nnue_evals <= node_count``, which those same faults only make
    # more true.
    assert stats["state_inits"] + stats["state_makes"] == stats["node_count"]


def test_transposed_move_orders_share_one_real_node_and_one_evaluation(dag_pack: Path) -> None:
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights, 16)

    start = chess.Board()
    root, root_value, created = _nnue_ext.dag_intern_root(dag, _cboard(start))
    assert created is True
    assert root_value == _nnue_ext.evaluate(weights, _cboard(start))

    # Same final structural position through two independent move orders.
    seq_a = ("g1f3", "g8f6", "g2g3", "g7g6")
    seq_b = ("g2g3", "g7g6", "g1f3", "g8f6")

    board_a = start.copy(stack=True)
    node_a = root
    penultimate_a = -1
    for i, uci in enumerate(seq_a):
        if i == len(seq_a) - 1:
            penultimate_a = node_a
        node_a, board_a, made = _push(dag, weights, node_a, board_a, uci)
        assert made is True

    board_b = start.copy(stack=True)
    node_b = root
    penultimate_b = -1
    last_created = True
    for i, uci in enumerate(seq_b):
        if i == len(seq_b) - 1:
            penultimate_b = node_b
        node_b, board_b, last_created = _push(dag, weights, node_b, board_b, uci)

    assert board_a.board_fen() == board_b.board_fen()
    assert board_a.turn == board_b.turn
    assert board_a.castling_rights == board_b.castling_rights
    assert node_b == node_a
    assert last_created is False

    # This is the defining difference from the current MCTSTree transposition
    # helper: two parents point to ONE child id; there is no cloned recipient.
    children_a = dict(_nnue_ext.dag_children(dag, penultimate_a))
    children_b = dict(_nnue_ext.dag_children(dag, penultimate_b))
    assert node_a in children_a.values()
    assert node_a in children_b.values()

    stats = _nnue_ext.dag_stats(dag)
    _assert_stats_schema(stats)
    # root + 4 nodes on A + only 3 new nodes on B; B's fourth is the transposition.
    assert stats["node_count"] == 8
    assert stats["edge_count"] == 8
    assert stats["state_inits"] == 1
    assert stats["state_makes"] == 7
    assert stats["nnue_evals"] == 8
    assert stats["node_reuses"] >= 1
    # ``hits`` — a canonical-table probe that landed on an existing node — is the
    # transposition signal. ``node_reuses`` is not: it also counts a repeated
    # identical (parent, action) request, which never probes the table.
    assert stats["hits"] >= 1
    # Scenario fact, not an invariant: none of these 8 positions is in check, so
    # every node really was evaluated once. (The invariant is the
    # state_inits + state_makes identity checked in _assert_stats_schema.)
    assert stats["nnue_evals"] == stats["node_count"] == 8


def test_transposition_returns_its_own_value_not_a_stale_one(dag_pack: Path) -> None:
    """The transposition path must return the MERGED node's value.

    ⚑ This test exists because the obvious transposition test cannot see that
    bug. Its two move orders (1.Nf3 Nf6 2.g3 g6) reach a colour-SYMMETRIC
    position, and so does the start position, so every node in it evaluates to
    exactly 0: a mutant that returned node 0's value on the transposition path
    passed the whole module. So this line is deliberately asymmetric, and the
    test PROVES the separation it depends on before asserting anything.
    """
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights, 16)

    start = chess.Board()
    root, root_value, _ = _nnue_ext.dag_intern_root(dag, _cboard(start))

    # 1.e4 d5 2.Nf3 Nf6 and 1.Nf3 Nf6 2.e4 d5 reach one structural position, and
    # unlike the symmetric line above it is not equal in value to the root.
    seq_a = ("e2e4", "d7d5", "g1f3", "g8f6")
    seq_b = ("g1f3", "g8f6", "e2e4", "d7d5")

    board_a = start.copy(stack=True)
    node_a = root
    for uci in seq_a:
        node_a, board_a, created = _push(dag, weights, node_a, board_a, uci)
        assert created is True

    board_b = start.copy(stack=True)
    node_b = root
    for uci in seq_b[:-1]:
        node_b, board_b, created = _push(dag, weights, node_b, board_b, uci)
        assert created is True

    # The separation this test rests on, asserted rather than assumed: the
    # merged node's value is different from EVERY other node's value in the
    # graph, so returning any other node's value — node 0's included — is
    # detectable rather than a coincidence away from passing.
    merged_value = _nnue_ext.dag_value(dag, node_a)
    others = [
        _nnue_ext.dag_value(dag, node_id)
        for node_id in range(_nnue_ext.dag_stats(dag)["node_count"])
        if node_id != node_a
    ]
    assert merged_value is not None
    assert merged_value != root_value
    assert merged_value not in others

    # Now close the transposition and demand the merged node's OWN value back.
    closing = chess.Move.from_uci(seq_b[-1])
    action = move_to_index(closing, board_b)
    board_b = board_b.copy(stack=True)
    board_b.push(closing)
    merged_id, value, created = _nnue_ext.dag_intern_child(
        dag,
        node_b,
        action,
        _cboard(board_b),
    )
    assert created is False
    assert merged_id == node_a
    assert value == merged_value
    assert value == _nnue_ext.evaluate(weights, _cboard(board_b))
    assert _nnue_ext.dag_value(dag, merged_id) == merged_value

    stats = _nnue_ext.dag_stats(dag)
    _assert_stats_schema(stats)
    # 8 structural positions, one of which two parents share: no ninth node, and
    # no ninth evaluation.
    assert stats["node_count"] == 8
    assert stats["edge_count"] == 8
    assert stats["nnue_evals"] == 8


def test_a_repetition_cycle_closes_onto_the_existing_ancestor_node(dag_pack: Path) -> None:
    """1.Nf3 Nf6 2.Ng1 Ng8 returns to the root — a real back-edge, not a clone.

    Consumers have to know this: because repetition history is deliberately not
    node identity, the structural graph is NOT acyclic, and a path-aware overlay
    must adjudicate such an edge instead of following it recursively.
    """
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights, 16)

    start = chess.Board()
    root, root_value, _ = _nnue_ext.dag_intern_root(dag, _cboard(start))

    board = start.copy(stack=True)
    node = root
    for uci in ("g1f3", "g8f6", "f3g1"):
        node, board, created = _push(dag, weights, node, board, uci)
        assert created is True

    closing = chess.Move.from_uci("f6g8")
    action = move_to_index(closing, board)
    cycled = board.copy(stack=True)
    cycled.push(closing)
    # Structurally the start position again — only the (excluded) halfmove clock
    # and repetition history differ.
    assert cycled.board_fen() == start.board_fen()
    assert cycled.turn == start.turn
    assert cycled.castling_rights == start.castling_rights

    node_id, value, created = _nnue_ext.dag_intern_child(dag, node, action, _cboard(cycled))
    assert created is False
    assert node_id == root
    assert value == root_value
    # The back-edge is a real edge of the graph: the child id IS the ancestor id.
    assert (action, root) in _nnue_ext.dag_children(dag, node)

    stats = _nnue_ext.dag_stats(dag)
    _assert_stats_schema(stats)
    assert stats["node_count"] == 4   # root + 3, not 5: the cycle adds no node
    assert stats["edge_count"] == 4   # but it does add the closing edge
    assert stats["nnue_evals"] == 4


def test_growth_past_initial_caps_preserves_every_id_value_and_edge(dag_pack: Path) -> None:
    """Node array, hash table and NNUE payload all grow; nothing is lost.

    A rehash re-inserts by node index and every payload array is copied, so node
    ids, values and edges must survive all three growths — this is the test that
    would catch a realloc/rehash that silently renumbered or dropped nodes.
    """
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights, 16)   # node_cap 16, 64 hash slots, payload 16
    start = chess.Board()
    root, root_value, _ = _nnue_ext.dag_intern_root(dag, _cboard(start))
    initial = _nnue_ext.dag_stats(dag)
    _assert_stats_schema(initial)
    assert initial["payload_capacity"] == 16

    values: dict[int, int | None] = {root: root_value}
    boards: dict[int, chess.Board] = {root: start}
    links: list[tuple[int, int, int]] = []

    frontier: list[tuple[int, chess.Board]] = [(root, start)]
    while _nnue_ext.dag_stats(dag)["node_count"] < 80:
        parent_id, parent = frontier.pop(0)
        for move in parent.legal_moves:
            child = parent.copy(stack=True)
            child.push(move)
            child_cb = _cboard(child)
            action = move_to_index(move, parent)
            node_id, value, created = _nnue_ext.dag_intern_child(
                dag, parent_id, action, child_cb
            )
            expected = None if child.is_check() else _nnue_ext.evaluate(weights, child_cb)
            assert value == expected
            values[node_id] = value
            boards.setdefault(node_id, child)
            links.append((parent_id, action, node_id))
            if created:
                frontier.append((node_id, child))

    grown = _nnue_ext.dag_stats(dag)
    _assert_stats_schema(grown)
    # >45 nodes forces a rehash: the table starts at 64 slots and rehashes to
    # keep load under 70%.
    assert grown["node_count"] > 45
    assert grown["payload_capacity"] > initial["payload_capacity"]
    assert grown["dag_memory_bytes"] > initial["dag_memory_bytes"]
    assert grown["nnue_payload_bytes"] > initial["nnue_payload_bytes"]

    # Every id still resolves to ITS OWN value and ITS OWN position...
    for node_id, value in values.items():
        assert _nnue_ext.dag_value(dag, node_id) == value
        assert _nnue_ext.dag_lookup(dag, _cboard(boards[node_id])) == node_id
    # ...and every edge still points where it was put.
    children: dict[int, set[tuple[int, int]]] = {}
    for parent_id, action, node_id in links:
        edges = children.setdefault(parent_id, set(_nnue_ext.dag_children(dag, parent_id)))
        assert (action, node_id) in edges


def test_dag_lookup_is_a_graph_read_that_does_no_nnue_work(dag_pack: Path) -> None:
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights, 16)
    start = chess.Board()

    assert _nnue_ext.dag_lookup(dag, _cboard(start)) is None
    root, _, _ = _nnue_ext.dag_intern_root(dag, _cboard(start))
    assert _nnue_ext.dag_lookup(dag, _cboard(start)) == root
    child_id, child_board, _ = _push(dag, weights, root, start, "e2e4")
    assert _nnue_ext.dag_lookup(dag, _cboard(child_board)) == child_id

    absent = start.copy(stack=True)
    absent.push(chess.Move.from_uci("d2d4"))
    before = _nnue_ext.dag_stats(dag)
    assert _nnue_ext.dag_lookup(dag, _cboard(absent)) is None
    miss = _nnue_ext.dag_stats(dag)
    _assert_stats_schema(miss)
    # A miss costs one probe and moves nothing else — in particular no NNUE work
    # and no node.
    assert miss["probes"] == before["probes"] + 1
    assert miss["hits"] == before["hits"]
    assert miss["node_count"] == before["node_count"]
    assert miss["state_inits"] == before["state_inits"]
    assert miss["state_makes"] == before["state_makes"]
    assert miss["nnue_evals"] == before["nnue_evals"]

    # A hit raises `hits` — which is why `hits` is a canonical-table signal and
    # not a pure transposition count: this one came from a plain lookup.
    assert _nnue_ext.dag_lookup(dag, _cboard(child_board)) == child_id
    hit = _nnue_ext.dag_stats(dag)
    _assert_stats_schema(hit)
    assert hit["hits"] == miss["hits"] + 1
    assert hit["node_reuses"] == miss["node_reuses"]
    assert hit["nnue_evals"] == miss["nnue_evals"]


def test_structural_identity_excludes_draw_history_but_includes_usable_ep(dag_pack: Path) -> None:
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights)

    ordinary = chess.Board()
    draw_adjacent = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 99 50",
    )
    a, _, a_created = _nnue_ext.dag_intern_root(dag, _cboard(ordinary))
    b, _, b_created = _nnue_ext.dag_intern_root(dag, _cboard(draw_adjacent))
    assert a_created is True
    assert b_created is False
    assert a == b

    # But an actually exercisable EP right changes legal structure and therefore
    # must be a different canonical node.
    ep = chess.Board(
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    )
    no_ep = chess.Board(
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3",
    )
    assert ep.has_legal_en_passant()
    ep_id, _, _ = _nnue_ext.dag_intern_root(dag, _cboard(ep))
    no_ep_id, _, _ = _nnue_ext.dag_intern_root(dag, _cboard(no_ep))
    assert ep_id != no_ep_id


def test_action_child_mismatch_is_rejected_before_graph_or_nnue_mutates(dag_pack: Path) -> None:
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights)
    board = chess.Board()
    root, _, _ = _nnue_ext.dag_intern_root(dag, _cboard(board))
    before = _nnue_ext.dag_stats(dag)
    _assert_stats_schema(before)

    asked_move = chess.Move.from_uci("e2e4")
    wrong_child = board.copy(stack=True)
    wrong_child.push(chess.Move.from_uci("d2d4"))
    action = move_to_index(asked_move, board)

    with pytest.raises(ValueError, match="does not produce the supplied child"):
        _nnue_ext.dag_intern_child(dag, root, action, _cboard(wrong_child))

    # A push-only validator would have a subtler hole: cboard_push_index() is a
    # defensive no-op for malformed/illegal LUT entries, so an illegal action
    # paired with the UNCHANGED board could compare equal and become a fake edge.
    legal_actions = {move_to_index(move, board) for move in board.legal_moves}
    illegal_action = next(i for i in range(4672) if i not in legal_actions)
    with pytest.raises(ValueError, match="does not produce the supplied child"):
        _nnue_ext.dag_intern_child(dag, root, illegal_action, _cboard(board))

    after = _nnue_ext.dag_stats(dag)
    _assert_stats_schema(after)
    assert after["node_count"] == before["node_count"]
    assert after["edge_count"] == before["edge_count"]
    assert after["state_makes"] == before["state_makes"]
    assert after["nnue_evals"] == before["nnue_evals"]


def test_in_check_node_has_state_but_no_fake_static_value(dag_pack: Path) -> None:
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights)

    board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
    assert board.is_check()
    root, value, created = _nnue_ext.dag_intern_root(dag, _cboard(board))
    assert created is True
    assert value is None
    assert _nnue_ext.dag_value(dag, root) is None
    stats = _nnue_ext.dag_stats(dag)
    _assert_stats_schema(stats)
    assert stats["state_inits"] == 1
    assert stats["nnue_evals"] == 0

    # The accumulator state is still useful: a legal evasion derives its child
    # incrementally and the resolved non-check child gets a real static value.
    move = next(iter(board.legal_moves))
    action = move_to_index(move, board)
    child = board.copy(stack=True)
    child.push(move)
    child_cb = _cboard(child)
    child_id, child_value, child_created = _nnue_ext.dag_intern_child(
        dag,
        root,
        action,
        child_cb,
    )
    assert child_created is True
    assert child_value == _nnue_ext.evaluate(weights, child_cb)
    assert _nnue_ext.dag_value(dag, child_id) == child_value
    stats = _nnue_ext.dag_stats(dag)
    _assert_stats_schema(stats)
    assert stats["state_makes"] == 1
    assert stats["nnue_evals"] == 1


def test_reroot_and_reset_preserve_allocations_but_clear_graph_semantics(dag_pack: Path) -> None:
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights, 16)
    board = chess.Board()
    root, _, _ = _nnue_ext.dag_intern_root(dag, _cboard(board))
    child, child_board, _ = _push(dag, weights, root, board, "e2e4")

    _nnue_ext.dag_set_root(dag, child)
    before = _nnue_ext.dag_stats(dag)
    _assert_stats_schema(before)
    assert before["root_id"] == child
    allocated = before["memory_bytes"]

    _nnue_ext.dag_reset(dag)
    cleared = _nnue_ext.dag_stats(dag)
    _assert_stats_schema(cleared)
    assert cleared["root_id"] == -1
    assert cleared["node_count"] == 0
    assert cleared["edge_count"] == 0
    assert cleared["state_inits"] == 0
    assert cleared["state_makes"] == 0
    assert cleared["nnue_evals"] == 0
    assert cleared["memory_bytes"] == allocated

    new_root, new_value, created = _nnue_ext.dag_intern_root(dag, _cboard(child_board))
    assert new_root == 0
    assert created is True
    assert new_value == _nnue_ext.evaluate(weights, _cboard(child_board))


@pytest.mark.skipif(not os.environ.get("CAE_NNUE_TEST_PACK"), reason="needs real NNUE pack")
def test_real_net_dag_incremental_values_match_full_refresh() -> None:
    weights = _nnue_ext.load(os.environ["CAE_NNUE_TEST_PACK"])
    dag = _nnue_ext.dag_open(weights)
    board = chess.Board()
    node, value, _ = _nnue_ext.dag_intern_root(dag, _cboard(board))
    assert value == _nnue_ext.evaluate(weights, _cboard(board))
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"):
        node, board, _ = _push(dag, weights, node, board, uci)
        assert _nnue_ext.dag_value(dag, node) == _nnue_ext.evaluate(weights, _cboard(board))
