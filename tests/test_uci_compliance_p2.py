"""UCI protocol compliance (engine audit, P2 items).

1. ``go mate N`` always terminates: it stops once a mate in <= N is proven at
   the root, else at its other bound, else at ``GO_IMPLICIT_CAP_MS``.
2. A bad ``position`` is rejected as a whole, the previous position is kept,
   and every ``go`` answers ``bestmove 0000`` until a valid position arrives.
3. ``score mate N`` is reported where the search has PROVEN a mate distance
   (the solved lattice walked down to checkmate leaves), never invented.
4. ``Syzygy50MoveRule=false`` reaches the root DTZ shortcut too, not only the
   leaf probes.
5. Options / go tokens that are accepted and ignored say so.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, cast
from unittest.mock import MagicMock

import chess
import chess.syzygy
import numpy as np
import pytest
import torch

import chess_anti_engine.tablebase as tbmod
import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.moves import index_to_move, move_to_index
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import CmdGo, CmdPosition, CmdSetOption, GoArgs, parse_command
from chess_anti_engine.uci.search import (
    _MATE_WALK_NODE_LIMIT,
    SearchResult,
    SearchWorker,
    _MateWalk,
    _proven_line_mate,
    _proven_line_mates,
    _proven_mate_plies,
)
from chess_anti_engine.uci.time_manager import GO_IMPLICIT_CAP_MS, Deadline, limits_from_go
from chess_anti_engine.utils.syzygy import syzygy_roots

if TYPE_CHECKING:
    from chess_anti_engine.mcts._mcts_tree import MCTSTree


class _Out:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._cond = threading.Condition()

    def write(self, s: str) -> int:
        with self._cond:
            self.lines.extend(ln for ln in s.splitlines() if ln)
            self._cond.notify_all()
        return len(s)

    def flush(self) -> None:
        pass

    def bestmoves(self) -> list[str]:
        with self._cond:
            return [ln.split()[1] for ln in self.lines if ln.startswith("bestmove ")]

    def wait_bestmoves(self, n: int, timeout: float = 10.0) -> list[str]:
        with self._cond:
            self._cond.wait_for(
                lambda: sum(ln.startswith("bestmove ") for ln in self.lines) >= n, timeout,
            )
        return self.bestmoves()

    def text(self) -> str:
        with self._cond:
            return "\n".join(self.lines)


@pytest.fixture
def out(monkeypatch) -> _Out:
    capture = _Out()
    monkeypatch.setattr(engine_module, "sys", SimpleNamespace(stdout=capture, stderr=sys.stderr))
    return capture


def _worker() -> MagicMock:
    worker = MagicMock()
    worker._chunk_sims = 16
    worker._pucv_pool = worker._rpg_pool = None
    worker.advance_root.return_value = True
    worker.root_policy_bestmove.return_value = None
    worker.realized_search_path.return_value = "gumbel"
    worker.run.side_effect = lambda board, **kw: _ok(board)
    return worker


def _ok(board: chess.Board, move: str | None = None) -> SearchResult:
    move = move or next(iter(board.legal_moves)).uci()
    return SearchResult(move, None, 8, (move,), 0)


def _go_args(line: str) -> GoArgs:
    cmd = parse_command(line)
    assert isinstance(cmd, CmdGo)
    return cmd.args


def _position(line: str) -> CmdPosition:
    cmd = parse_command(line)
    assert isinstance(cmd, CmdPosition)
    return cmd


def _go(engine: Engine, line: str) -> None:
    engine.dispatch(parse_command(line))
    thread = engine._search_thread
    if thread is not None:
        thread.join(10)


# --- 1. go mate N -----------------------------------------------------------


def test_go_mate_is_parsed() -> None:
    assert _go_args("go mate 3").mate == 3


def test_go_mate_alone_is_capped_and_stops_on_proof() -> None:
    limits = limits_from_go(GoArgs(mate=3), side_to_move_is_white=True, move_overhead_ms=30)
    assert limits.deadline_ms == GO_IMPLICIT_CAP_MS
    assert limits.implicit_cap is True
    assert limits.mate_moves == 3
    assert limits.optimum_ms is None  # a mate hunt never banks on a settled move
    assert not limits.is_open_ended()


@pytest.mark.parametrize(
    "args",
    [GoArgs(mate=3, movetime_ms=500), GoArgs(mate=3, nodes=200), GoArgs(mate=3, depth=4),
     GoArgs(mate=3, wtime_ms=10_000, btime_ms=10_000)],
)
def test_go_mate_with_another_bound_uses_that_bound(args: GoArgs) -> None:
    limits = limits_from_go(args, side_to_move_is_white=True)
    assert limits.implicit_cap is False
    assert limits.deadline_ms != GO_IMPLICIT_CAP_MS
    assert limits.mate_moves == 3


@pytest.mark.parametrize("args", [GoArgs(mate=3, infinite=True), GoArgs(mate=3, ponder=True)])
def test_go_mate_never_stops_an_infinite_or_ponder_search(args: GoArgs) -> None:
    limits = limits_from_go(args, side_to_move_is_white=True)
    assert limits.mate_moves is None
    assert limits.is_open_ended()


@pytest.mark.parametrize("line", ["go mate abc", "go mate", "go mate 0", "go mate -2"])
def test_an_unusable_mate_value_with_no_other_bound_gets_the_implicit_cap(line: str) -> None:
    args = _go_args(line)
    assert args.mate is None
    assert args.errors
    limits = limits_from_go(args, side_to_move_is_white=True)
    assert limits.implicit_cap
    assert limits.deadline_ms == GO_IMPLICIT_CAP_MS
    assert limits.mate_moves is None


def test_mate_zero_is_announced_as_ignored(out) -> None:
    engine = Engine(_worker())
    try:
        _go(engine, "go mate 0 nodes 64")
        assert "go: ignored unusable value(s) 'mate 0' (mate needs N >= 1)" in out.text()
    finally:
        engine.close()


@pytest.mark.parametrize(
    ("line", "field", "value", "note"),
    [
        ("go movetime 999.9", "movetime_ms", 999, "movetime 999.9 -> 999"),
        ("go nodes 1.5", "nodes", 1, "nodes 1.5 -> 1"),
    ],
)
def test_a_non_integer_value_is_truncated_and_announced(out, line: str, field: str, value: int, note: str) -> None:
    args = _go_args(line)
    assert getattr(args, field) == value
    assert args.truncated == (note,)
    engine = Engine(_worker())
    try:
        _go(engine, line)
        assert f"go: non-integer value(s) rounded toward zero: {note}" in out.text()
    finally:
        engine.close()


def test_an_integral_float_stays_silent() -> None:
    args = _go_args("go movetime 1000.0 nodes 5e2")
    assert args.movetime_ms == 1000
    assert args.nodes == 500
    assert args.truncated == ()
    assert args.errors == ()


def test_go_mate_ponder_applies_the_mate_stop_after_ponderhit(out) -> None:
    worker = _worker()
    seen: list[dict] = []
    release = threading.Event()

    def run(board, **kw):
        seen.append(kw)
        if len(seen) == 1:  # the ponder phase: open-ended until ponderhit
            kw["stop_event"].wait(5)
        return _ok(board)

    worker.run.side_effect = run
    engine = Engine(worker, options=EngineOptions(ponder=True))
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder mate 3 wtime 5000 btime 5000"))
        assert "after ponderhit the search stops once a mate in <= 3 is proven" in out.text()
        engine.dispatch(parse_command("ponderhit"))
        assert out.wait_bestmoves(1)
        release.set()
        assert [kw["mate_moves"] for kw in seen] == [None, 3]
    finally:
        engine.close()


def test_the_engine_hands_mate_to_the_worker(out) -> None:
    worker = _worker()
    seen: list[dict] = []
    worker.run.side_effect = lambda board, **kw: (seen.append(kw), _ok(board))[1]
    engine = Engine(worker)
    try:
        _go(engine, "go mate 2")
        assert seen[-1]["mate_moves"] == 2
        assert seen[-1]["deadline"].remaining_ms() is not None  # capped, not infinite
        assert f"searching at most {GO_IMPLICIT_CAP_MS} ms" in out.text()
        _go(engine, "go nodes 64")
        assert seen[-1]["mate_moves"] is None
    finally:
        engine.close()


def _tiny_worker() -> SearchWorker:
    from chess_anti_engine.inference import DirectGPUEvaluator
    from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.model import ModelConfig, build_model

    torch.manual_seed(0)
    model = build_model(ModelConfig(input_extra_features="v1", embed_dim=16, num_layers=1,
                                    num_heads=2, ffn_mult=2.0))
    model.eval()
    evaluator = ThreadSafeGPUDispatcher(DirectGPUEvaluator(
        model, device="cpu", max_batch=64, use_amp=False, n_slots=2,
    ))
    return SearchWorker(
        evaluator, device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )


# Re8+ Rxe8 Rxe8# -- the reply is forced, so a small search proves it.
_MATE_IN_2 = "r5k1/5ppp/8/8/8/8/4RPPP/4R1K1 w - - 0 1"
# Every white move allows a mate in one.
_MATED_IN_1 = "8/5k2/8/5P2/8/7r/4q3/2K5 w - - 0 1"


def test_a_real_search_stops_on_a_proven_mate_and_reports_its_distance() -> None:
    worker = _tiny_worker()
    try:
        result = worker.run(
            chess.Board(_MATE_IN_2), stop_event=threading.Event(), deadline=Deadline(None),
            max_nodes=20_000, mate_moves=2,
        )
        assert result.nodes < 20_000
        assert result.bestmove_uci == "e2e8"
        assert result.score_mate == 2
        assert worker.last_stop_reason == "mate"
    finally:
        worker.close()


def test_mate_in_n_too_short_does_not_stop_the_search() -> None:
    """``go mate 1`` on a mate in 2: the proof never satisfies N, so the search
    runs to its node bound (here) rather than stopping on the longer mate."""
    worker = _tiny_worker()
    try:
        result = worker.run(
            chess.Board(_MATE_IN_2), stop_event=threading.Event(), deadline=Deadline(None),
            max_nodes=3000, mate_moves=1,
        )
        assert worker.last_stop_reason != "mate"
        assert result.score_mate == 2  # still reported: it was proven
    finally:
        worker.close()


def test_a_real_search_reports_being_mated() -> None:
    worker = _tiny_worker()
    try:
        result = worker.run(
            chess.Board(_MATED_IN_1), stop_event=threading.Event(), deadline=Deadline(None),
            max_nodes=8000,
        )
        # Exact, not the longer mate the proof found first (it read -3 before
        # the mate-in-one-on-the-board shortcut).
        assert result.score_mate == -1
    finally:
        worker.close()


def test_an_unproven_root_reports_no_mate() -> None:
    worker = _tiny_worker()
    try:
        result = worker.run(
            chess.Board(), stop_event=threading.Event(), deadline=Deadline(None), max_nodes=256,
        )
        assert result.score_mate is None
    finally:
        worker.close()


# --- 3. the mate-distance walk on a synthetic solved tree -------------------


class _Tree:
    """Just the three MCTSTree reads the walk uses. ``spec`` maps node id to
    (status, {uci: child id}); boards are replayed from the root FEN."""

    def __init__(self, root_fen: str, spec: dict[int, tuple[int, dict[str, int]]]) -> None:
        self.spec = spec
        self.boards = {0: chess.Board(root_fen)}
        self.edges: dict[int, dict[int, int]] = {}
        stack = [0]
        while stack:
            nid = stack.pop()
            board = self.boards[nid]
            self.edges[nid] = {}
            for uci, child in spec[nid][1].items():
                move = chess.Move.from_uci(uci)
                assert move in board.legal_moves, (uci, board.fen())
                self.edges[nid][int(move_to_index(move, board))] = child
                if child not in self.boards:
                    nxt = board.copy()
                    nxt.push(move)
                    self.boards[child] = nxt
                    stack.append(child)

    def get_solved_status(self, nid: int) -> int:
        return self.spec[nid][0] if nid >= 0 else 0

    def get_children_visits(self, nid: int):
        actions = np.array(sorted(self.edges[nid]), dtype=np.int32)
        return actions, np.ones(actions.size, dtype=np.int32)

    def find_child(self, nid: int, action: int) -> int:
        return self.edges[nid].get(int(action), -1)


def _mate(tree: _Tree, uci: str) -> int | None:
    board = tree.boards[0]
    action = int(move_to_index(chess.Move.from_uci(uci), board))
    return _proven_line_mate(cast("MCTSTree", cast(object, tree)), 0, board, action)


def test_walk_takes_the_shortest_win_and_the_longest_defence() -> None:
    # Root (white): e2e8 is a forced mate in 2 (black's only reply a8e8).
    tree = _Tree(_MATE_IN_2, {
        0: (1, {"e2e8": 1}),
        1: (-1, {"a8e8": 2}),
        2: (1, {"e1e8": 3, "f2f3": 4}),  # a proven mate and an unproven move
        3: (-1, {}),
        4: (0, {}),
    })
    assert _mate(tree, "e2e8") == 2


def test_walk_needs_every_defence_proven() -> None:
    """A LOSS node with a legal move missing from the proof gives no distance,
    even when the listed defence has one (Kh8 allows Ra8#; Kf8 escapes)."""
    fen = "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1"
    board = chess.Board(fen)
    tree = _Tree(fen, {0: (-1, {"g8h8": 1}), 1: (1, {})})  # only Kh8 listed
    as_tree = cast("MCTSTree", cast(object, tree))
    assert board.legal_moves.count() > 1
    assert _proven_mate_plies(as_tree, 1, tree.boards[1].copy(), _MateWalk(1000)) == 1
    assert _proven_mate_plies(as_tree, 0, board.copy(), _MateWalk(1000)) is None


def test_a_solved_leaf_that_is_not_checkmate_has_no_distance() -> None:
    """A tablebase-solved leaf: WIN/LOSS status, no checkmate on the board
    (KRvK, no mate in one anywhere along the line)."""
    fen = "7k/8/8/8/8/8/8/KR6 w - - 0 1"
    board = chess.Board(fen)
    assert not any(board.gives_check(m) and _mates(board, m) for m in board.legal_moves)
    tree = _Tree(fen, {0: (1, {"b1b7": 1}), 1: (-1, {})})  # LOSS leaf, not mate
    assert _mate(tree, "b1b7") is None


def _mates(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    try:
        return board.is_checkmate()
    finally:
        board.pop()


def test_a_transposition_cycle_proves_nothing() -> None:
    fen = "6k1/8/8/8/8/8/8/R5K1 w - - 0 1"
    tree = _Tree(fen, {0: (1, {"a1a2": 1}), 1: (-1, {"g8h8": 2}), 2: (1, {"a2a1": 3}),
                       3: (-1, {"h8g8": 0})})
    assert _mate(tree, "a1a2") is None


def _mate_in_one(board: chess.Board) -> chess.Move | None:
    for move in board.legal_moves:
        if _mates(board, move):
            return move
    return None


def _mate_in_two(board: chess.Board) -> chess.Move | None:
    for move in board.legal_moves:
        board.push(move)
        try:
            replies = list(board.legal_moves)
            if replies and all(_after(board, r, _mate_in_one) is not None for r in replies):
                return move
        finally:
            board.pop()
    return None


def _after(board: chess.Board, move: chess.Move, fn):
    board.push(move)
    try:
        return fn(board)
    finally:
        board.pop()


# Black to move, exactly two defences: Kb1 allows Rh1# (mate in one), Kd1 only
# a mate in two. A LOSS node's distance is its LONGEST proven defence.
_TWO_DEFENCES = "8/5K2/8/8/8/7R/4R3/2k5 b - - 0 1"


def _two_defences_tree() -> _Tree:
    board = chess.Board(_TWO_DEFENCES)
    spec: dict[int, tuple[int, dict[str, int]]] = {}
    nxt = [0]

    def new_id() -> int:
        nid = nxt[0]
        nxt[0] += 1
        return nid

    def win(b: chess.Board) -> int:
        nid = new_id()
        if _mate_in_one(b) is not None:
            spec[nid] = (1, {})  # the mate-in-one shortcut needs no children
            return nid
        move = _mate_in_two(b)
        assert move is not None
        spec[nid] = (1, {})
        spec[nid] = (1, {move.uci(): _after(b, move, loss)})
        return nid

    def loss(b: chess.Board) -> int:
        nid = new_id()
        spec[nid] = (-1, {})
        if b.is_checkmate():
            return nid
        spec[nid] = (-1, {m.uci(): _after(b, m, win) for m in list(b.legal_moves)})
        return nid

    assert loss(board) == 0
    return _Tree(_TWO_DEFENCES, spec)


def test_a_loss_node_takes_its_longest_defence() -> None:
    tree = _two_defences_tree()
    board = chess.Board(_TWO_DEFENCES)
    assert [_after(board, m, lambda b: _mate_in_one(b) is not None) for m in board.legal_moves] \
        .count(True) == 1
    plies = _proven_mate_plies(cast("MCTSTree", cast(object, tree)), 0, board.copy(), _MateWalk())
    # Kd1 (1) + mate in two (3) = 4 plies; the short defence (Kb1) would give 2.
    assert plies == 4


def test_the_walk_stops_at_its_budget_on_a_finite_tree() -> None:
    """The two-defences proof needs more than 3 node visits: a 3-visit walk is
    exhausted after exactly 3 and answers None; an ample one finishes."""
    tree = cast("MCTSTree", cast(object, _two_defences_tree()))
    board = chess.Board(_TWO_DEFENCES)
    walk = _MateWalk(budget=3)
    assert _proven_mate_plies(tree, 0, board.copy(), walk) is None
    assert walk.exhausted
    assert walk.visits == 3
    ample = _MateWalk()
    assert _proven_mate_plies(tree, 0, board.copy(), ample) == 4
    assert not ample.exhausted
    assert 3 < ample.visits < 50


class _LazySolvedTree:
    """A fully expanded proven tree generated on demand: every node solved
    (``root_status`` on even plies, its opposite on odd), every legal move a
    child, no checkmate anywhere -- the worst case for the mate walk. Counts
    status reads."""

    def __init__(self, fen: str, root_status: int = 1) -> None:
        self.root_status = root_status
        self.boards = {0: chess.Board(fen)}
        self.depth = {0: 0}
        self.kids: dict[int, dict[int, int]] = {}
        self.status_reads = 0

    def get_solved_status(self, nid: int) -> int:
        self.status_reads += 1
        return self.root_status if self.depth[nid] % 2 == 0 else -self.root_status

    def _expand(self, nid: int) -> dict[int, int]:
        if nid not in self.kids:
            board = self.boards[nid]
            self.kids[nid] = {}
            for move in board.legal_moves:
                child = len(self.boards)
                nb = board.copy(stack=False)
                nb.push(move)
                self.boards[child] = nb
                self.depth[child] = self.depth[nid] + 1
                self.kids[nid][int(move_to_index(move, board))] = child
        return self.kids[nid]

    def get_children_visits(self, nid: int):
        actions = np.array(sorted(self._expand(nid)), dtype=np.int32)
        return actions, np.ones(actions.size, dtype=np.int32)

    def find_child(self, nid: int, action: int) -> int:
        return self._expand(nid).get(int(action), -1)

    def node_count(self) -> int:
        return len(self.boards)


_DEEP_FEN = "4k3/8/8/8/8/8/8/R3K2R w - - 0 1"


def test_one_walk_bounds_the_cost_of_every_root_line() -> None:
    """One `_MateWalk` shared by all root lines: total node visits stay at the
    budget (per-line budgets did 24 x 4,000 here, ~10 s on the search thread),
    and once exhausted every line reads no distance."""
    tree = _LazySolvedTree(_DEEP_FEN)
    as_tree = cast("MCTSTree", cast(object, tree))
    board = chess.Board(_DEEP_FEN)
    actions, _ = tree.get_children_visits(0)
    assert actions.size == 24
    walk = _MateWalk()
    mates = [_proven_line_mate(as_tree, 0, board, int(a), walk) for a in actions]
    assert walk.exhausted
    assert walk.visits == _MATE_WALK_NODE_LIMIT
    assert mates == [None] * actions.size
    # Status reads: one per visit plus the per-child checks a visit makes
    # (measured ~16.7k here; per-line budgets read ~235k).
    assert tree.status_reads < 8 * _MATE_WALK_NODE_LIMIT


def test_one_info_emission_shares_one_budget_across_its_lines() -> None:
    tree = _LazySolvedTree(_DEEP_FEN)
    actions, _ = tree.get_children_visits(0)
    lines: list[int | None] = [int(a) for a in actions]
    lines.append(None)
    mates = _proven_line_mates(cast("MCTSTree", cast(object, tree)), 0, chess.Board(_DEEP_FEN), lines)
    assert mates == [None] * len(lines)
    assert tree.status_reads < 8 * _MATE_WALK_NODE_LIMIT


def test_one_info_emission_still_reports_a_proven_mate() -> None:
    tree = _Tree(_MATE_IN_2, {0: (1, {"e2e8": 1}), 1: (-1, {"a8e8": 2}), 2: (1, {"e1e8": 3}), 3: (-1, {})})
    board = tree.boards[0]
    action = int(move_to_index(chess.Move.from_uci("e2e8"), board))
    as_tree = cast("MCTSTree", cast(object, tree))
    assert _proven_line_mates(as_tree, 0, board, [action, None]) == [2, None]
    assert _proven_line_mates(None, None, board, [action]) == [None]


def _mate_check_host(tree: object) -> SearchWorker:
    """Just the attributes `SearchWorker._proven_mate_within` reads."""
    host = SimpleNamespace(_tree=tree, _root_id=0, _mate_check_key=None, last_mate_walk_visits=0)
    return cast("SearchWorker", cast(object, host))


class _StaticLazyTree(_LazySolvedTree):
    """A real tree does not grow while it is walked; this one only does so
    lazily, so report a fixed size (what the search would see between chunks)."""

    size = 1

    def node_count(self) -> int:
        return self.size


def test_the_go_mate_check_shares_one_budget_and_skips_an_unchanged_tree() -> None:
    tree = _StaticLazyTree(_DEEP_FEN)
    host = _mate_check_host(tree)
    board = chess.Board(_DEEP_FEN)
    within = SearchWorker._proven_mate_within
    assert within(host, board, 5, None) is None
    assert host.last_mate_walk_visits == _MATE_WALK_NODE_LIMIT
    assert tree.status_reads < 8 * _MATE_WALK_NODE_LIMIT
    # Same tree, root status and size: no second walk.
    reads = tree.status_reads
    assert within(host, board, 5, None) is None
    assert tree.status_reads - reads == 1  # the root status read only
    # The tree grew (a chunk ran): walk again, inside the same budget.
    tree.size = 2
    reads = tree.status_reads
    assert within(host, board, 5, None) is None
    assert 1 < tree.status_reads - reads < 8 * _MATE_WALK_NODE_LIMIT


class _CountingTree(_Tree):
    def node_count(self) -> int:
        return len(self.boards)


def test_an_exhausted_walk_reports_no_distance_rather_than_a_partial_one() -> None:
    """Mate in two through e2e8; with too small a budget the walk must answer
    None, never a distance from the part it managed to read."""
    spec = {
        0: (1, {"e2e8": 1}),
        1: (-1, {"a8e8": 2}),
        2: (1, {"e1e8": 3}),
        3: (-1, {}),
    }
    tree = _Tree(_MATE_IN_2, spec)
    as_tree = cast("MCTSTree", cast(object, tree))
    board = tree.boards[0]
    action = int(move_to_index(chess.Move.from_uci("e2e8"), board))
    assert _proven_line_mate(as_tree, 0, board, action, _MateWalk()) == 2
    starved = _MateWalk(budget=1)
    assert _proven_line_mate(as_tree, 0, board, action, starved) is None
    assert starved.exhausted


class _ProofPlusDeepSibling:
    """The `_MATE_IN_2` proof (Re8+ Rxe8 Rxe8#) at node 0, plus ONE more
    winning-looking child of node 0 -- a later action than e2e8 -- whose LOSS
    subtree is a fully expanded lazy tree with no mate in it (ids >= 1000)."""

    _OFFSET = 1000

    def __init__(self) -> None:
        self.proof = _Tree(_MATE_IN_2, {0: (1, {"e2e8": 1}), 1: (-1, {"a8e8": 2}),
                                        2: (1, {"e1e8": 3}), 3: (-1, {})})
        board = self.proof.boards[0]
        self.mate_action = int(move_to_index(chess.Move.from_uci("e2e8"), board))
        later = sorted(
            (int(move_to_index(m, board)), m) for m in board.legal_moves
            if int(move_to_index(m, board)) > self.mate_action and not board.gives_check(m)
        )
        self.deep_action, move = later[0]
        after = board.copy(stack=False)
        after.push(move)
        self.deep = _LazySolvedTree(after.fen(), root_status=-1)

    def get_solved_status(self, nid: int) -> int:
        if nid >= self._OFFSET:
            return self.deep.get_solved_status(nid - self._OFFSET)
        return self.proof.get_solved_status(nid)

    def get_children_visits(self, nid: int):
        if nid >= self._OFFSET:
            return self.deep.get_children_visits(nid - self._OFFSET)
        actions, visits = self.proof.get_children_visits(nid)
        if nid == 0:
            actions = np.array(sorted([*actions.tolist(), self.deep_action]), dtype=np.int32)
            visits = np.ones(actions.size, dtype=np.int32)
        return actions, visits

    def find_child(self, nid: int, action: int) -> int:
        if nid >= self._OFFSET:
            child = self.deep.find_child(nid - self._OFFSET, action)
            return -1 if child < 0 else child + self._OFFSET
        if nid == 0 and int(action) == self.deep_action:
            return self._OFFSET
        return self.proof.find_child(nid, action)


def test_a_node_whose_walk_ran_out_gives_no_distance_even_with_a_finished_child() -> None:
    """Node 0's first winning child (e2e8) is fully proven (mate in 3 plies);
    the later sibling's subtree then exhausts the budget. A distance read from
    the finished part would be an upper bound, not what was asked: None."""
    tree = _ProofPlusDeepSibling()
    as_tree = cast("MCTSTree", cast(object, tree))
    board = tree.proof.boards[0]
    assert tree.mate_action < tree.deep_action
    # With the sibling unsolved, the same walk finishes: mate in 3 plies.
    unsolved = _ProofPlusDeepSibling()
    unsolved.deep = _LazySolvedTree(tree.deep.boards[0].fen(), root_status=0)
    assert _proven_mate_plies(cast("MCTSTree", cast(object, unsolved)), 0, board.copy(), _MateWalk()) == 3
    starved = _MateWalk(budget=50)
    assert _proven_mate_plies(as_tree, 0, board.copy(), starved) is None
    assert starved.exhausted
    assert 0 not in starved.memo  # nothing half-walked left behind as an answer


def test_go_mate_searchmoves_ignores_a_mate_through_an_excluded_move() -> None:
    tree = _CountingTree(_MATE_IN_2, {
        0: (1, {"e2e8": 1, "h2h3": 4}),
        1: (-1, {"a8e8": 2}),
        2: (1, {"e1e8": 3}),
        3: (-1, {}),
        4: (0, {}),
    })
    board = tree.boards[0]
    mate_idx = int(move_to_index(chess.Move.from_uci("e2e8"), board))
    other_idx = int(move_to_index(chess.Move.from_uci("h2h3"), board))
    within = SearchWorker._proven_mate_within
    assert within(_mate_check_host(tree), board, 2, None) == 2
    assert within(_mate_check_host(tree), board, 2, {mate_idx}) == 2
    assert within(_mate_check_host(tree), board, 2, {other_idx}) is None
    assert within(_mate_check_host(tree), board, 1, None) is None


def test_go_mate_on_a_longer_proven_mate_stops_instead_of_idling() -> None:
    """`go mate 1` on a mate in two with no node bound: once the root is
    proven and the search cannot grow it, stop -- not idle to the cap."""
    import time

    worker = _tiny_worker()
    try:
        t0 = time.monotonic()
        result = worker.run(
            chess.Board(_MATE_IN_2), stop_event=threading.Event(),
            deadline=Deadline(30_000), max_nodes=None, mate_moves=1,
        )
        elapsed = time.monotonic() - t0
        assert worker.last_stop_reason == "mate_solved_root"
        assert elapsed < 15.0
        assert result.score_mate == 2
    finally:
        worker.close()


def test_a_solved_root_that_still_grows_does_not_stop_on_mate_solved_root(monkeypatch) -> None:
    """The `mate_solved_root` stop needs BOTH a solved root and a chunk that
    grew nothing. Here the root reads as solved (no mate in <= N) while every
    chunk still completes visits: the search must run to its node bound."""
    worker = _tiny_worker()
    try:
        monkeypatch.setattr(worker, "_root_solved_status", lambda: 1)
        monkeypatch.setattr(worker, "_proven_mate_within", lambda *_a, **_k: None)
        result = worker.run(
            chess.Board(), stop_event=threading.Event(), deadline=Deadline(30_000),
            max_nodes=256, mate_moves=1,
        )
        assert worker.last_stop_reason != "mate_solved_root"
        assert result.nodes >= 256
    finally:
        worker.close()


# --- 2. position rejection --------------------------------------------------

_BAD_POSITIONS = {
    "no keyword": "position foo",
    "too few FEN fields": "position fen 8/8/8/8 w",
    "too many FEN fields": "position fen 4k3/8/8/8/8/8/8/4K3 w - - 0 1 junk",
    "stray token after startpos": "position startpos e2e4",
    "unparseable FEN": "position fen not/a/fen w - - 0 1",
    "missing king": "position fen rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",
    "pawn on the back rank": "position fen rnbqkbnP/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQq - 0 1",
    "side not to move in check": "position fen 4k3/8/8/8/8/8/4R3/4K3 w - - 0 1",
    "bad castling field": "position fen 4k3/8/8/8/8/8/8/R3K3 w KQkqX - 0 1",
    "two white kings": "position fen 4k3/8/8/8/8/8/8/3KK3 w - - 0 1",
    "empty board": "position fen 8/8/8/8/8/8/8/8 w - - 0 1",
    "illegal move in the list": "position startpos moves e2e4 e7e5 e1e3 g8f6",
    "unparseable move in the list": "position startpos moves e2e4 xx",
}


@pytest.mark.parametrize("label", sorted(_BAD_POSITIONS))
def test_a_rejected_position_keeps_the_previous_one_and_go_answers_0000(label: str, out) -> None:
    worker = _worker()
    engine = Engine(worker)
    try:
        engine.dispatch(parse_command("position startpos moves d2d4"))
        previous = engine._board.copy()
        engine.dispatch(parse_command(_BAD_POSITIONS[label]))
        assert "info string position: rejected (" in out.text(), label
        assert engine._board == previous, label
        _go(engine, "go nodes 64")
        assert out.bestmoves() == ["0000"], label
        assert "go refused: the last position command was rejected" in out.text()
        worker.run.assert_not_called()
        # The next valid position is searched normally.
        engine.dispatch(parse_command("position startpos moves d2d4 d7d5"))
        _go(engine, "go nodes 64")
        assert len(out.bestmoves()) == 2
        assert out.bestmoves()[1] != "0000"
        assert worker.run.call_count == 1
    finally:
        engine.close()


def test_ucinewgame_clears_a_rejection(out) -> None:
    engine = Engine(_worker())
    try:
        engine.dispatch(parse_command("position foo"))
        engine.dispatch(parse_command("ucinewgame"))
        _go(engine, "go nodes 64")
        assert out.bestmoves()
        assert out.bestmoves()[0] != "0000"
    finally:
        engine.close()


def test_a_short_fen_is_still_padded_not_rejected(out) -> None:
    """Four / five FEN fields stay accepted (EPD-derived tooling sends them)."""
    engine = Engine(_worker())
    try:
        engine.dispatch(parse_command("position fen 4k3/8/8/8/8/8/4R3/3K4 b - -"))
        assert "rejected" not in out.text()
        assert engine._board == chess.Board("4k3/8/8/8/8/8/4R3/3K4 b - - 0 1")
    finally:
        engine.close()


def test_the_busy_worker_route_also_refuses(out) -> None:
    """When the previous search will not stop, `dispatch` answers through
    `_answer_failed_go`; with a rejected position that must be 0000 too, never
    a fallback move for the old board."""
    engine = Engine(_worker())
    try:
        engine.dispatch(parse_command("position foo"))
        engine._answer_failed_go(CmdGo(_go_args("go nodes 64")), worker_busy=True)
        assert out.bestmoves() == ["0000"]
        assert engine._bestmove_fallback_used == 0
    finally:
        engine.close()


@pytest.mark.parametrize("release", ["ponderhit", "stop"])
def test_a_go_ponder_after_a_rejected_position_holds_0000_until_released(out, release: str) -> None:
    """UCI: a pondering engine answers only after ponderhit / stop -- the
    refusal too."""
    engine = Engine(_worker(), options=EngineOptions(ponder=True))
    try:
        engine.dispatch(parse_command("position foo"))
        engine.dispatch(parse_command("go ponder wtime 1000 btime 1000"))
        assert "go refused" in out.text()
        assert out.wait_bestmoves(1, timeout=0.3) == []
        engine.dispatch(parse_command(release))
        assert out.wait_bestmoves(1) == ["0000"]
    finally:
        engine.close()


def test_a_go_infinite_after_a_rejected_position_holds_0000_until_stop(out) -> None:
    engine = Engine(_worker())
    try:
        engine.dispatch(parse_command("position foo"))
        engine.dispatch(parse_command("go infinite"))
        assert out.wait_bestmoves(1, timeout=0.3) == []
        engine.dispatch(parse_command("ponderhit"))  # not a release for infinite
        assert out.wait_bestmoves(1, timeout=0.3) == []
        engine.dispatch(parse_command("stop"))
        assert out.wait_bestmoves(1) == ["0000"]
    finally:
        engine.close()


def test_a_finite_go_after_a_rejected_position_answers_at_once(out) -> None:
    engine = Engine(_worker())
    try:
        engine.dispatch(parse_command("position foo"))
        engine.dispatch(parse_command("go movetime 50"))
        assert out.wait_bestmoves(1, timeout=1.0) == ["0000"]
    finally:
        engine.close()


def _ponder_worker(pv_for) -> MagicMock:
    """The ponder phase waits for its stop, then returns a result for the
    board it searched (the one BEFORE the predicted reply)."""
    worker = _worker()

    def run(board, **kw):
        kw["stop_event"].wait(5)
        pv = pv_for(board)
        return SearchResult(pv[0], pv[1] if len(pv) > 1 else None, 8, pv, 0)

    worker.run.side_effect = run
    return worker


@pytest.mark.parametrize(
    ("pv", "expect"),
    [
        (("e7e5", "g1f3"), "g1f3"),  # PV through the predicted reply: our planned answer
        (("b7b5", "a2a3"), None),    # PV elsewhere: a fallback move for the real board
    ],
)
def test_stop_on_a_ponder_miss_answers_for_the_gui_board(out, pv, expect) -> None:
    """`position startpos moves e2e4 e7e5` + `go ponder` searches the board
    after e2e4 (Black to move). On `stop` the bestmove must be legal in the
    GUI's position (White to move), not Black's search move."""
    engine = Engine(_ponder_worker(lambda _b: pv), options=EngineOptions(ponder=True))
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder wtime 5000 btime 5000"))
        engine.dispatch(parse_command("stop"))
        (move,) = out.wait_bestmoves(1)
        gui = chess.Board()
        gui.push_uci("e2e4")
        gui.push_uci("e7e5")
        assert chess.Move.from_uci(move) in gui.legal_moves, move
        if expect is not None:
            assert move == expect
        line = next(ln for ln in out.lines if ln.startswith("bestmove"))
        assert "ponder" not in line.split()[2:]
    finally:
        engine.close()


def test_a_ponder_miss_without_a_pv_reply_reads_the_ponder_tree(out) -> None:
    """PV not through the predicted reply: the answer comes from the ponder
    tree's subtree for the GUI position (visit counts), not a new evaluation
    and not the first legal move."""
    worker = _ponder_worker(lambda _b: ("b7b5", "a2a3"))
    worker.child_bestmove.return_value = "g1f3"
    engine = Engine(worker, options=EngineOptions(ponder=True))
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder wtime 5000 btime 5000"))
        engine.dispatch(parse_command("stop"))
        assert out.wait_bestmoves(1) == ["g1f3"]
        assert "source=ponder_tree move=g1f3" in out.text()
        (call,) = worker.child_bestmove.call_args_list
        board, popped = call.args
        assert board.turn == chess.BLACK
        assert popped == chess.Move.from_uci("e7e5")
        worker.fresh_root_policy_bestmove.assert_not_called()
    finally:
        engine.close()


def test_a_superseded_ponder_search_never_reads_the_worker_tree(out, monkeypatch) -> None:
    """A ponder search whose generation is no longer current (parked after
    `stop`, or a newer `go` owns the worker) must not compute a ponder-miss
    answer: that reads the worker's tree while another search may own it, and
    `_claim_answer` drops the answer anyway."""
    worker = _ponder_worker(lambda _b: ("b7b5", "a2a3"))
    engine = Engine(worker, options=EngineOptions(ponder=True))
    try:
        board = chess.Board()
        board.push_uci("e2e4")
        engine._popped_ponder_move = chess.Move.from_uci("e7e5")
        stale = SearchResult("b7b5", "a2a3", 8, ("b7b5", "a2a3"), 0)
        monkeypatch.setattr(engine, "_run_one_phase", lambda *a, **k: stale)
        misses: list[object] = []
        answer = SearchResult("g1f3", None, 8, ("g1f3",), 0)

        def miss(*a: object, **_k: object) -> SearchResult:
            misses.append(a)
            return answer

        monkeypatch.setattr(engine, "_ponder_miss_result", miss)
        release = threading.Event()
        release.set()
        limits = limits_from_go(GoArgs(ponder=True, wtime_ms=5000, btime_ms=5000),
                                side_to_move_is_white=False)
        engine._run_search(limits, engine._search_gen - 1, board, ponder_release=release)
        assert misses == []
        worker.child_bestmove.assert_not_called()
        assert out.bestmoves() == []
        # Control: the same call at the current generation does take the path.
        engine._popped_ponder_move = chess.Move.from_uci("e7e5")
        engine._run_search(limits, engine._search_gen, board, ponder_release=release,
                           cancel_event=release)
        assert len(misses) == 1
    finally:
        engine.close()


def test_child_bestmove_reads_the_most_visited_reply_from_a_real_tree() -> None:
    worker = _tiny_worker()
    try:
        board = chess.Board()
        board.push_uci("e2e4")
        worker.run(board, stop_event=threading.Event(), deadline=Deadline(None), max_nodes=512)
        tree, root = worker._tree, worker._root_id
        assert tree is not None
        assert root is not None
        actions, visits = tree.get_children_visits(root)
        reply = index_to_move(int(actions[int(np.argmax(visits))]), board)
        node = tree.find_child(root, int(move_to_index(reply, board)))
        kids, kid_visits = tree.get_children_visits(node)
        assert kids.size > 0
        after = board.copy()
        after.push(reply)
        want = index_to_move(int(kids[int(np.argmax(kid_visits))]), after).uci()
        assert worker.child_bestmove(board, reply) == want
        assert chess.Move.from_uci(want) in after.legal_moves
        # searchmoves restrict it; a board that is not the tree's root reads nothing.
        other = next(m.uci() for m in after.legal_moves if m.uci() != want)
        assert worker.child_bestmove(board, reply, searchmoves=(other,)) in (other, None)
        # A board that is not the tree's root reads nothing -- even a move whose
        # (side-relative) action id IS a root child: the mirror of the reply,
        # played from the start position, has the reply's id.
        mirror = chess.Move(chess.square_mirror(reply.from_square), chess.square_mirror(reply.to_square))
        assert mirror in chess.Board().legal_moves
        assert int(move_to_index(mirror, chess.Board())) == int(move_to_index(reply, board))
        assert worker.child_bestmove(chess.Board(), mirror) is None
    finally:
        worker.close()


def test_ponderhit_still_answers_from_the_real_phase(out) -> None:
    calls: list[chess.Board] = []

    def pv_for(board: chess.Board) -> tuple[str, ...]:
        calls.append(board.copy())
        return ("e7e5", "g1f3") if board.turn == chess.BLACK else ("g1f3", "b8c6")

    engine = Engine(_ponder_worker(pv_for), options=EngineOptions(ponder=True))
    try:
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder wtime 5000 btime 5000"))
        engine.dispatch(parse_command("ponderhit"))
        (move,) = out.wait_bestmoves(1)
        assert move == "g1f3"
        assert [b.turn for b in calls] == [chess.BLACK, chess.WHITE]
    finally:
        engine.close()


# --- 2b. status-invalid but answerable positions are normalised -------------

# Reviewer's regressions: main answered each with a legal move; a blanket
# status rejection forfeited them with 0000.
_NORMALISED = {
    "white rooks, phantom black rights": (
        "position fen 4k3/8/8/8/8/8/8/R3K2R w KQkq - 0 1", chess.BB_A1 | chess.BB_H1),
    "black rooks, phantom white rights": (
        "position fen r3k2r/8/8/8/8/8/8/4K3 w KQkq - 0 1", chess.BB_A8 | chess.BB_H8),
    "missing h1 rook": (
        "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq - 0 1",
        chess.BB_A1 | chess.BB_A8 | chess.BB_H8),
    "phantom rights then a move": (
        "position fen 4k3/8/8/8/8/8/8/R3K2R b KQkq - 0 1 moves e8d8", chess.BB_A1 | chess.BB_H1),
}


def _searched_boards(worker: MagicMock) -> list[chess.Board]:
    return [call.args[0] for call in worker.run.call_args_list]


@pytest.mark.parametrize("label", sorted(_NORMALISED))
def test_stale_castling_rights_are_normalised_everywhere(label: str, out) -> None:
    """One set of rights for python-chess, the board the search is handed (the
    C board and the net's planes are built from it) and every later rebuild of
    the root from `_pending_fen` (tree sync, ponder)."""
    from chess_anti_engine.encoding._lc0_ext import CBoard

    line, rights = _NORMALISED[label]
    worker = _worker()
    engine = Engine(worker)
    try:
        engine.dispatch(parse_command(line))
        text = out.text()
        assert "rejected" not in text, text
        assert "info string position: normalised castling rights KQkq -> " in text
        assert engine._board.castling_rights == rights
        assert engine._board.status() == chess.STATUS_VALID
        assert engine._pending_fen is not None
        assert chess.Board(engine._pending_fen).castling_rights == rights
        _go(engine, "go nodes 64")
        (searched,) = _searched_boards(worker)
        assert searched.castling_rights == rights
        assert searched.move_stack == engine._board.move_stack
        assert searched.root().castling_rights == chess.Board(engine._pending_fen).castling_rights
        c_bits = int(CBoard.from_board(searched).castling)
        want = (
            (1 if rights & chess.BB_H1 else 0) | (2 if rights & chess.BB_A1 else 0)
            | (4 if rights & chess.BB_H8 else 0) | (8 if rights & chess.BB_A8 else 0)
        )
        assert c_bits == want
        (move,) = out.bestmoves()
        assert chess.Move.from_uci(move) in engine._board.legal_moves
    finally:
        engine.close()


# Impossible ep squares that are real except for the push's occupied start
# square: (fen, ep square, the capture).
_KEPT_EP = (
    ("4k3/8/8/8/3pP3/8/4P3/4K3 b - e3 0 1", chess.E3, "d4e3"),
    ("4k3/3p4/8/3pP3/8/8/8/4K3 w - d6 0 1", chess.D6, "e5d6"),
)

# python-chess "captures" en passant with no pawn of the opponent behind the
# square too: nothing, a non-pawn, an own piece, the king.
_DROPPED_EP = (
    "4k3/8/8/8/3pN3/8/8/4K3 b - e3 0 1",
    "4k3/8/8/8/3pB3/8/8/4K3 b - e3 0 1",
    "4k3/8/8/3Pn3/8/8/8/4K3 w - e6 0 1",
    "4k3/8/8/8/3pK3/8/8/8 b - e3 0 1",
    "4k3/8/8/8/3pb3/8/8/4K3 b - e3 0 1",
    "4k3/8/8/8/3pp3/8/8/4K3 b - e3 0 1",
    "4k3/8/8/8/3pQ3/8/8/4K3 b - e3 0 1",
    "4k3/8/8/8/3pR3/8/8/4K3 b - e3 0 1",
    "4k3/8/8/8/3p4/8/8/4K3 b - e3 0 1",
)


def _c_matches_python_after_every_move(board: chess.Board) -> None:
    """Independent of the engine's gate: every field of the C board equals
    python-chess's at the root and after every legal move."""
    from chess_anti_engine.encoding._lc0_ext import CBoard

    def state(b: chess.Board) -> tuple:
        return (b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK], b.pawns,
                b.knights, b.bishops, b.rooks, b.queens, b.kings, b.ep_square, b.turn)

    def c_state(cb) -> tuple:
        return (cb.occ_white, cb.occ_black, cb.pawns, cb.knights, cb.bishops, cb.rooks,
                cb.queens, cb.kings, cb.ep_square, bool(cb.turn))

    cb = CBoard.from_board(board)
    assert c_state(cb) == state(board)
    assert sorted(int(i) for i in cb.legal_move_indices()) == sorted(
        move_to_index(mv, board) for mv in board.legal_moves)
    for mv in board.legal_moves:
        after = board.copy(stack=False)
        after.push(mv)
        c_after = CBoard.from_board(board)
        c_after.push_index(move_to_index(mv, board))
        assert c_state(c_after) == state(after), mv.uci()


@pytest.mark.parametrize(("fen", "square", "capture"), _KEPT_EP)
def test_an_unreachable_ep_square_with_its_pushed_pawn_is_kept_everywhere(
    fen: str, square: int, capture: str, out,
) -> None:
    """Dropping it would reject the capture, which python-chess accepts (main
    answered e1d1). Kept: python-chess, the searched board, `_pending_fen`, the
    C board and every position after a legal move all agree."""
    from chess_anti_engine.encoding._lc0_ext import CBoard

    assert chess.Board(fen).status() & chess.STATUS_INVALID_EP_SQUARE
    worker = _worker()
    engine = Engine(worker)
    try:
        engine.dispatch(parse_command(f"position fen {fen}"))
        assert f"kept en-passant square {chess.square_name(square)}" in out.text()
        assert engine._board.ep_square == square
        assert engine._pending_fen is not None
        assert chess.Board(engine._pending_fen).ep_square == square
        assert CBoard.from_board(engine._board).ep_square == square
        _c_matches_python_after_every_move(engine._board)
        engine.dispatch(parse_command(f"position fen {fen} moves {capture}"))
        assert "rejected" not in out.text()
        assert engine._board.piece_at(square) is not None
        _c_matches_python_after_every_move(engine._board)
        _go(engine, "go nodes 64")
        searched = _searched_boards(worker)[-1]
        assert searched.move_stack == [chess.Move.from_uci(capture)]
        (move,) = out.bestmoves()
        assert chess.Move.from_uci(move) in engine._board.legal_moves
    finally:
        engine.close()


@pytest.mark.parametrize("fen", _DROPPED_EP)
def test_an_ep_square_without_its_pushed_pawn_is_dropped_never_rejected(fen: str) -> None:
    cmd = parse_command(f"position fen {fen}")
    assert isinstance(cmd, CmdPosition)
    result = Engine._validated_position(cmd)
    assert not isinstance(result, str), result
    board, _moves, root_fen, notes = result
    assert board.ep_square is None
    assert root_fen is not None
    assert chess.Board(root_fen).ep_square is None
    assert any(n.startswith("normalised en-passant square") for n in notes), notes
    _c_matches_python_after_every_move(board)


@pytest.mark.parametrize("fen", _DROPPED_EP[:8])
def test_the_gate_catches_a_c_board_an_ep_capture_leaves_inconsistent(fen: str) -> None:
    """Called on the un-normalised board, the movegen gate must see what the
    C board does with these captures: a phantom knight/bishop/king/queen/rook
    type bit, or an own piece removed. Comparing only occupancy and pawns
    passed the first four."""
    reason = engine_module._movegen_disagreement(chess.Board(fen))
    assert reason is not None, fen


def test_an_impossible_en_passant_square_is_dropped(out) -> None:
    worker = _worker()
    engine = Engine(worker)
    try:
        engine.dispatch(parse_command(
            "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1"))
        assert "normalised en-passant square e6 is impossible -> none" in out.text()
        assert engine._board.ep_square is None
        assert engine._pending_fen is not None
        assert chess.Board(engine._pending_fen).ep_square is None
        _go(engine, "go nodes 64")
        (searched,) = _searched_boards(worker)
        assert searched.ep_square is None
    finally:
        engine.close()


_VALID_FENS = (
    chess.STARTING_FEN,
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # a real ep square
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 4 20",
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    "  4k3/8/8/8/8/8/8/4K3   w   -   -   0   1",
)


@pytest.mark.parametrize("fen", _VALID_FENS)
def test_a_valid_position_is_taken_exactly_as_sent(fen: str, out) -> None:
    """For a status-VALID root nothing changes vs main: the stored root FEN is
    the GUI's string itself and the board is python-chess's parse of it."""
    engine = Engine(_worker())
    try:
        cmd = _position(f"position fen {fen}")
        engine.dispatch(cmd)
        assert "normalised" not in out.text()
        assert "rejected" not in out.text()
        assert engine._pending_fen is cmd.fen
        parsed = chess.Board(cmd.fen)
        assert engine._board == parsed
        assert engine._board.castling_rights == parsed.castling_rights
        assert engine._board.ep_square == parsed.ep_square
    finally:
        engine.close()


_UNREACHABLE_FEN = "4k3/8/8/8/8/P7/PPPPPPPP/QQQQKQQQ w - - 0 1"


def test_an_unreachable_position_is_accepted_when_the_c_board_agrees(out) -> None:
    board = chess.Board(_UNREACHABLE_FEN)  # 9 white pawns, 17 white men
    assert board.status() & chess.STATUS_TOO_MANY_WHITE_PAWNS
    assert board.status() & chess.STATUS_TOO_MANY_WHITE_PIECES
    worker = _worker()
    engine = Engine(worker)
    try:
        engine.dispatch(parse_command(f"position fen {board.fen()}"))
        assert "rejected" not in out.text()
        _go(engine, "go nodes 64")
        (move,) = out.bestmoves()
        assert chess.Move.from_uci(move) in board.legal_moves
    finally:
        engine.close()


def test_an_unreachable_position_the_c_board_disagrees_on_is_rejected(monkeypatch, out) -> None:
    board = chess.Board(_UNREACHABLE_FEN)
    seen: list[str] = []

    def disagree(b: chess.Board) -> str:
        seen.append(b.fen())
        return "the C move generator disagrees with python-chess (1 vs 2 legal moves)"

    monkeypatch.setattr(engine_module, "_movegen_disagreement", disagree)
    engine = Engine(_worker())
    try:
        engine.dispatch(parse_command(f"position fen {board.fen()}"))
        assert "rejected" in out.text()
        assert "C move generator disagrees" in out.text()
        assert seen
        # A VALID root is never put through the comparison.
        seen.clear()
        engine.dispatch(parse_command("position startpos moves e2e4"))
        assert seen == []
        assert engine._position_error is None
    finally:
        engine.close()


def test_movegen_disagreement_reports_a_c_board_that_differs(monkeypatch) -> None:
    import chess_anti_engine.encoding._lc0_ext as lc0_ext

    real = lc0_ext.CBoard

    class _Short:
        """The real board, but its generator drops one move."""

        def __init__(self, board: chess.Board) -> None:
            self._cb = real.from_board(board)

        @classmethod
        def from_board(cls, board: chess.Board) -> _Short:
            return cls(board)

        def legal_move_indices(self):
            return self._cb.legal_move_indices()[1:]

        def __getattr__(self, name: str):
            return getattr(self._cb, name)

    monkeypatch.setattr(lc0_ext, "CBoard", _Short)
    reason = engine_module._movegen_disagreement(chess.Board())
    assert reason is not None
    assert "19 vs 20 legal moves" in reason


def test_movegen_disagreement_reads_both_generators() -> None:
    assert engine_module._movegen_disagreement(chess.Board(_UNREACHABLE_FEN)) is None
    assert engine_module._movegen_disagreement(chess.Board()) is None


# --- 4. Syzygy50MoveRule reaches the root shortcut --------------------------


class _CursedTable:
    """KRvK at a high counter: every quiet root move wins in 30 plies at a
    fresh counter (dtz from the mover's view after the move: -29)."""

    wdl: ClassVar[dict[str, None]] = {"KRvK": None}
    dtz: ClassVar[dict[str, None]] = {"KRvK": None}

    def probe_wdl(self, board: chess.Board) -> int:
        return 2 if board.turn == chess.WHITE else -2

    def probe_dtz(self, board: chess.Board) -> int:
        return 30 if board.turn == chess.WHITE else -29


_CURSED_FEN = "7k/8/8/8/8/8/8/KR6 w - - 90 120"


@pytest.mark.parametrize(("rule50", "wdl"), [(True, 1), (False, 2)])
def test_root_move_rule50_switch(monkeypatch, rule50: bool, wdl: int) -> None:
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: _CursedTable())
    picked = tbmod.try_tb_root_move(chess.Board(_CURSED_FEN), "stub", rule50=rule50)
    assert picked is not None
    assert picked[1] == wdl


def test_selfplay_default_is_unchanged(monkeypatch) -> None:
    """Callers that pass nothing (selfplay's `probe_best_move`, training labels)
    keep the 50-move-aware ranking."""
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: _CursedTable())
    board = chess.Board(_CURSED_FEN)
    assert tbmod.try_tb_root_move(board, "stub") == tbmod.try_tb_root_move(board, "stub", rule50=True)
    picked = tbmod.try_tb_root_move(board, "stub")
    assert picked is not None
    assert tbmod.probe_best_move(board, "stub") == picked[0]


class _FiftyBoundaryTable:
    """Probe stub for a root at halfmove 99 (`_FIFTY_FEN`). The pawn pushes zero
    the counter into a BLESSED loss for Black (wdl -1: a cursed win, dtz 101 for
    the mover); every quiet move lands on halfmove 100 where the table (fresh
    counter) still says White wins in 30. With the 50-move rule the quiet
    children are already drawn (`is_fifty_moves`), so the cursed-win pawn push
    outranks them; without it the quiet win in 30 is shortest."""

    def probe_wdl(self, _board: chess.Board) -> int:
        return -1

    def probe_dtz(self, _board: chess.Board) -> int:
        return -29


_FIFTY_FEN = "7k/8/8/8/8/8/P7/KR6 w - - 99 150"


def test_dtz_of_played_move_default_counts_the_fifty_move_draw() -> None:
    """Selfplay's `strict_tb` and the root ranking call this with the DEFAULT:
    a child whose counter has run out is a draw (0), whatever the table says."""
    board = chess.Board(_FIFTY_FEN)
    board.push(chess.Move.from_uci("b1b2"))
    assert board.halfmove_clock == 100
    assert board.is_fifty_moves()
    table = cast("chess.syzygy.Tablebase", cast(object, _FiftyBoundaryTable()))
    assert tbmod._dtz_of_played_move(board, table) == 0
    assert tbmod._dtz_of_played_move(board, table, rule50=True) == 0
    assert tbmod._dtz_of_played_move(board, table, rule50=False) == 30


def test_probe_best_move_default_follows_the_fifty_move_ranking(monkeypatch) -> None:
    """`probe_best_move` (training labels via selfplay/finalize.py) must keep the
    50-move-aware ranking: here it and plain DTZ pick DIFFERENT moves."""
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: _FiftyBoundaryTable())
    board = chess.Board(_FIFTY_FEN)
    rule50 = tbmod._select_dtz_move(board, "stub", rule50=True)
    plain = tbmod._select_dtz_move(board, "stub", rule50=False)
    assert rule50 is not None
    assert plain is not None
    assert board.piece_type_at(rule50[0].from_square) == chess.PAWN
    assert rule50[1] == 101
    assert board.piece_type_at(plain[0].from_square) != chess.PAWN
    assert plain[1] == 30
    assert tbmod.probe_best_move(board, "stub") == rule50[0]
    picked = tbmod.try_tb_root_move(board, "stub")
    assert picked is not None
    assert picked[0] == rule50[0]


@pytest.mark.parametrize(("value", "score"), [("true", 0), ("false", 41890)])
def test_syzygy50moverule_changes_the_engine_root_shortcut(monkeypatch, out, value: str, score: int) -> None:
    """The engine path: `setoption` -> the probe the worker holds -> the root
    shortcut -> the `score` the GUI sees. Read from the worker's own probe."""
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: _CursedTable())
    monkeypatch.setattr(engine_module, "get_tablebase", lambda _path: _CursedTable())
    worker = _tiny_worker()
    engine = Engine(worker)
    try:
        engine.dispatch(CmdSetOption(name="Syzygy50MoveRule", value=value))
        engine.dispatch(CmdSetOption(name="SyzygyPath", value="stub-tables"))
        probe = worker._tb_probe
        assert probe is not None
        assert probe.cursed_as_draw is (value == "true")
        engine.dispatch(parse_command(f"position fen {_CURSED_FEN}"))
        _go(engine, "go nodes 64")
        infos = [ln for ln in out.lines if ln.startswith("info depth")]
        assert infos
        assert f"score cp {score} " in infos[-1], infos
    finally:
        engine.close()
        worker.close()


def _smoke_syzygy() -> Path | None:
    for root in syzygy_roots():
        candidate = root / "data" / "syzygy_3-4man"
        if candidate.is_dir() and any(candidate.glob("*.rtbw")):
            return candidate
    return None


@pytest.mark.parametrize(("value", "score"), [("true", 0), ("false", 41890)])
def test_syzygy50moverule_on_the_real_smoke_tables(out, value: str, score: int) -> None:
    """KRvK with DTZ 19 at a fresh counter: at counter 90 the 50-move rule
    draws it, without the rule it is a win."""
    path = _smoke_syzygy()
    if path is None:
        pytest.skip("3-4 man smoke Syzygy set (data/syzygy_3-4man) not present")
    tbmod.get_tablebase(str(path))
    worker = _tiny_worker()
    engine = Engine(worker)
    try:
        engine.dispatch(CmdSetOption(name="SyzygyPath", value=str(path)))
        engine.dispatch(CmdSetOption(name="Syzygy50MoveRule", value=value))
        engine.dispatch(parse_command(f"position fen {_CURSED_FEN}"))
        _go(engine, "go nodes 64")
        infos = [ln for ln in out.lines if ln.startswith("info depth")]
        assert infos
        assert f"score cp {score} " in infos[-1], infos
    finally:
        engine.close()
        worker.close()


# --- 5. accepted-and-ignored inputs say so ----------------------------------


@pytest.mark.parametrize(
    ("name", "value", "expect"),
    [
        ("NoSuchOption", "3", "unknown option, ignored"),
        ("MultiPV", None, "no value given, ignored"),
        ("Ponder", "yes", "Ponder: expected true/false"),
        ("UCI_ShowWDL", "1", "UCI_ShowWDL: expected true/false"),
        ("Syzygy50MoveRule", "on", "Syzygy50MoveRule: expected true/false"),
        ("ClockBatchMarginSigmas", "abc", "ClockBatchMarginSigmas: 'abc' is not a number"),
        ("RPGOpenBudgetFrac", "2", "RPGOpenBudgetFrac: '2' is not a number in (0, 1]"),
        ("RPGOpenVPA", "8", "NO EFFECT on the live search (path=gumbel)"),
        ("LeafGather", "8", "NO EFFECT on the live search (path=gumbel)"),
    ],
)
def test_ignored_option_values_are_announced(out, name: str, value: str | None, expect: str) -> None:
    engine = Engine(_worker())
    try:
        engine.dispatch(CmdSetOption(name=name, value=value))
        assert expect in out.text(), out.text()
    finally:
        engine.close()


@pytest.mark.usefixtures("out")
def test_a_bad_check_value_keeps_the_old_setting() -> None:
    engine = Engine(_worker(), options=EngineOptions(ponder=True))
    try:
        engine.dispatch(CmdSetOption(name="Ponder", value="yes"))
        assert engine._options.ponder is True
    finally:
        engine.close()


def test_an_unparseable_go_bound_is_named_and_capped(out) -> None:
    worker = _worker()
    seen: list[dict] = []
    worker.run.side_effect = lambda board, **kw: (seen.append(kw), _ok(board))[1]
    engine = Engine(worker)
    try:
        _go(engine, "go movetime abc")
        assert "go: ignored unusable value(s) 'movetime abc'" in out.text()
        assert seen[-1]["deadline"].remaining_ms() is not None
    finally:
        engine.close()


def test_a_float_movetime_is_honoured() -> None:
    args = _go_args("go movetime 1000.0")
    assert args.movetime_ms == 1000
    assert args.errors == ()


def test_illegal_searchmoves_are_named(out) -> None:
    engine = Engine(_worker())
    try:
        engine.dispatch(parse_command("position startpos moves e2e4"))
        _go(engine, "go nodes 64 searchmoves e7e5 e2e4")
        assert "searchmoves e2e4 not legal here, ignored (1 of 2 kept)" in out.text()
    finally:
        engine.close()


@pytest.mark.parametrize("line", ["debug on", "register later"])
def test_unimplemented_spec_commands_say_so(out, line: str) -> None:
    engine = Engine(_worker())
    try:
        engine.dispatch(parse_command(line))
        assert "not implemented by this engine; ignored" in out.text()
    finally:
        engine.close()


def test_position_parse_errors() -> None:
    assert _position("position").error is not None
    too_many = _position("position fen 4k3/8/8/8/8/8/8/4K3 w - - 0 1 junk").error
    assert too_many is not None
    assert "at most 6" in too_many
    assert _position("position startpos moves e2e4").error is None
    padded = _position("position fen 4k3/8/8/8/8/8/8/4K3 w - -")
    assert padded.error is None
    assert padded.fen == "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


# --- combo options answer a value they do not have -----------------------------


@pytest.mark.parametrize(
    ("option", "field", "current"),
    [("PUCVPendingMode", "pucv_pending_mode", "legacy"),
     ("SearchParallel", "search_parallel", "pucv")],
)
def test_a_combo_value_the_option_does_not_have_is_answered_and_changes_nothing(
    out, option: str, field: str, current: str,
) -> None:
    worker = _worker()
    engine = Engine(worker)
    try:
        assert getattr(engine._options, field) == current
        engine.dispatch(parse_command(f"setoption name {option} value bogus"))
        assert f"info string {option}: 'bogus' is not one of" in out.text(), out.text()
        assert f"; keeping {current}" in out.text(), out.text()
        assert getattr(engine._options, field) == current
        worker.set_pucv_vloss_mode.assert_not_called()
    finally:
        engine.close()


def test_a_valid_combo_value_is_not_refused(out) -> None:
    worker = _worker()
    engine = Engine(worker)
    try:
        engine.dispatch(parse_command("setoption name PUCVPendingMode value virtual-mean"))
        assert "is not one of" not in out.text()
        assert engine._options.pucv_pending_mode == "virtual-mean"
        worker.set_pucv_vloss_mode.assert_called_once_with(1)
    finally:
        engine.close()
