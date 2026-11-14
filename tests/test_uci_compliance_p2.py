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
import numpy as np
import pytest
import torch

import chess_anti_engine.tablebase as tbmod
import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.moves import move_to_index
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import CmdGo, CmdPosition, CmdSetOption, GoArgs, parse_command
from chess_anti_engine.uci.search import (
    SearchResult,
    SearchWorker,
    _proven_line_mate,
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
    assert _proven_mate_plies(as_tree, 1, tree.boards[1].copy(), {}, [1000]) == 1
    assert _proven_mate_plies(as_tree, 0, board.copy(), {}, [1000]) is None


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
    "castling rights without rooks": "position fen 4k3/8/8/8/8/8/8/R3K3 w KQkq - 0 1",
    "bad castling field": "position fen 4k3/8/8/8/8/8/8/R3K3 w KQkqX - 0 1",
    "impossible en passant": "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1",
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


def test_a_go_ponder_after_a_rejected_position_still_answers(out) -> None:
    engine = Engine(_worker(), options=EngineOptions(ponder=True))
    try:
        engine.dispatch(parse_command("position foo"))
        _go(engine, "go ponder wtime 1000 btime 1000")
        assert out.bestmoves() == ["0000"]
    finally:
        engine.close()


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
        assert "go: ignored unparseable 'movetime abc'" in out.text()
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
