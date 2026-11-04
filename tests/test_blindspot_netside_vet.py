"""Terminal-only vet contracts, without importing model/engine startup code."""
from __future__ import annotations

import ast
from pathlib import Path

import chess

_SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "blindspot_netside_vet.py"


def _terminal_nodes() -> tuple[dict, dict[str, ast.expr]]:
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_terminal_q")
    namespace = {"chess": chess}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(_SOURCE), "exec"), namespace)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    nodes = {}
    for node in ast.walk(main):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"after", "d_after"}:
                assert isinstance(node.value, ast.IfExp)
                nodes[target.id] = node.value
        if (isinstance(node, ast.If)
                and ast.unparse(node.test) == "board.is_game_over()"
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Assign)):
            assignment = node.body[0]
            if ast.unparse(assignment.targets[0]) == "cq":
                nodes["cq"] = assignment.value
    assert set(nodes) == {"after", "cq", "d_after"}
    return namespace, nodes


class _ForbiddenRuntime:
    def __getattribute__(self, name):
        raise AssertionError(f"terminal evaluation accessed runtime: {name}")

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("terminal evaluation called engine/model code")


def _post_move_board(fen: str, uci: str) -> tuple[chess.Board, chess.Color]:
    board = chess.Board(fen)
    assert board.is_valid()
    assert not board.is_game_over()
    mover = board.turn
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    board.push(move)
    assert board.is_game_over()
    return board, mover


def _check_stages(board: chess.Board, mover: chess.Color, expected: float) -> None:
    namespace, nodes = _terminal_nodes()
    sentinel = _ForbiddenRuntime()
    namespace.update(board=board, mover=mover, eng=sentinel, _q=sentinel,
                     args=sentinel, ev=sentinel)
    fen, history = board.fen(), list(board.move_stack)
    for stage, node in nodes.items():
        actual = eval(compile(ast.Expression(node), str(_SOURCE), "eval"), namespace)
        assert actual == expected, stage
        assert (actual <= -0.5) == (expected <= -0.5), stage
        assert (actual >= 0.2) == (expected >= 0.2), stage
    assert board.fen() == fen
    assert board.move_stack == history
    helper = namespace["_terminal_q"]
    assert helper(board, mover) == expected
    assert helper(board, not mover) == -expected


def test_mate_is_a_win_for_each_saved_mover_at_all_three_stages() -> None:
    for fen, uci in [
        ("7k/8/5KQ1/8/8/8/8/8 w - - 0 1", "g6g7"),
        ("8/8/8/8/8/5kq1/8/7K b - - 0 1", "g3g2"),
    ]:
        board, mover = _post_move_board(fen, uci)
        assert board.is_checkmate()
        assert board.turn != mover
        _check_stages(board, mover, 1.0)


def test_automatic_draws_stay_neutral_at_all_three_stages() -> None:
    for fen, uci in [
        ("7k/5K2/8/6Q1/8/8/8/8 w - - 0 1", "g5g6"),
        ("7k/8/8/8/8/8/1r6/K7 w - - 0 1", "a1b2"),
        ("7k/8/8/8/8/8/R7/K7 w - - 149 1", "a2b2"),
    ]:
        board, mover = _post_move_board(fen, uci)
        assert not board.is_checkmate()
        _check_stages(board, mover, 0.0)


def test_fivefold_draw_uses_history_without_mutating_it() -> None:
    board = chess.Board()
    moves = ["g1f3", "g8f6", "f3g1", "f6g8"] * 4
    for uci in moves[:-1]:
        board.push_uci(uci)
    assert not board.is_game_over()
    mover = board.turn
    board.push_uci(moves[-1])
    assert board.is_fivefold_repetition()
    _check_stages(board, mover, 0.0)


def test_nonterminal_and_merely_claimable_draws_are_rejected() -> None:
    namespace, _ = _terminal_nodes()
    helper = namespace["_terminal_q"]
    claimable = chess.Board("7k/8/8/8/8/8/R7/K7 w - - 100 1")
    assert claimable.can_claim_fifty_moves()
    repetition = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        repetition.push_uci(uci)
    assert repetition.can_claim_threefold_repetition()
    for board in [chess.Board(), claimable, repetition]:
        assert not board.is_game_over()
        error = None
        try:
            helper(board, board.turn)
        except ValueError as exc:
            error = exc
        assert error is not None
        assert str(error) == "terminal score requires a game-over board"
