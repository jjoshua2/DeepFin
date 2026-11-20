"""The reply-forced 50-move draw in the shared forced-draw helpers (review lead
R1, 2026-11-19), and proof that it changes nothing else on the self-play path.

`forced_draw_actions` / `forced_draw_actions_cboard` (mcts/root_tactics.py) now
also count a move that leaves the fifty-move clock at 99 when every opponent
reply ends the game drawn and none mates. They feed the C runner's and the
Python reference's below-zero fallback, root-parallel Gumbel's root resolver
and the UCI drawing escape. Pinned here:

- a seeded, bounded port of the review's differential test against a
  python-chess oracle (cutechess/TCEC-style 100-ply adjudication, checkmate
  first) on every move of thousands of random roots: never a false draw, and
  no miss where the move leaves the clock at 99 (the oracle's wider "every
  reply draws some other way" set is out of scope); the CBoard helper agrees
  with the python-chess one;
- on every root with no clock-99 reply-forced move the helpers return exactly
  what the pre-change helpers returned, and the C runner (board and CBoard
  entries) and the Python reference produce bit-identical policy targets,
  moves and values with the old and new helper;
- a positive control: on a reply-forced root the runners now play the draw.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

import chess
import numpy as np
import pytest

import chess_anti_engine.mcts.gumbel as gumbel_module
import chess_anti_engine.mcts.gumbel_c as gumbel_c_module
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts.root_tactics import forced_draw_actions, forced_draw_actions_cboard
from chess_anti_engine.moves import move_to_index
from tests.test_root_draw_pruning import (
    BXA7_FEN,
    CLAIMABLE_ONLY_FEN,
    KXD7_FEN,
    SURVIVOR_WINS_FEN,
    _NeutralLeaves,
    _run_c,
    _run_py,
    _SideToMoveWins,
)

# Kd8 leaves the clock at 99 and every Black reply is reversible and not mate;
# a2a3 / a2a4 (the two pruning survivors) are mated next move.
REPLY_FORCED_FEN, REPLY_FORCED_DRAW = "2K5/8/1k6/1q6/8/8/P7/8 w - - 98 1", "c8d8"


# --- the helpers as they were before the clock-99 rule (verbatim) ------------

def _old_forced_draw_actions(board: chess.Board, actions: Iterable[int]) -> list[int]:
    wanted = {int(a) for a in actions}
    out: list[int] = []
    for move in board.legal_moves:
        action = int(move_to_index(move, board))
        if action not in wanted:
            continue
        board.push(move)
        try:
            if not board.is_checkmate() and (
                board.is_stalemate()
                or board.is_insufficient_material()
                or board.halfmove_clock >= 100
                or board.is_repetition(3)
            ):
                out.append(action)
        finally:
            board.pop()
    return out


def _old_forced_draw_actions_cboard(root_cb: CBoard, actions: Iterable[int]) -> list[int]:
    out: list[int] = []
    for action in actions:
        child = root_cb.copy()
        child.push_index(int(action))
        if child.is_game_over() and not child.is_checkmate():
            out.append(int(action))
    return out


# --- the review's oracle and position generator (seeded, bounded) ------------

_PIECES = "QRBNPqrbnp"


def _drawn_over(b: chess.Board) -> bool:
    if b.is_checkmate():
        return False
    o = b.outcome(claim_draw=False)  # stalemate, insufficient, 75-move, fivefold
    if o is not None:
        return o.winner is None
    return b.is_fifty_moves() or b.is_repetition(3)


def _oracle(b: chess.Board, m: chess.Move) -> bool:
    b.push(m)
    try:
        if _drawn_over(b):
            return True
        if b.is_checkmate():
            return False
        replies = list(b.legal_moves)
        if not replies:
            return False
        for r in replies:
            b.push(r)
            ok = _drawn_over(b)
            b.pop()
            if not ok:
                return False
        return True
    finally:
        b.pop()


def _rand_board(rng: random.Random, clock: int) -> chess.Board | None:
    b = chess.Board(None)
    sqs = rng.sample(range(64), 2 + rng.randint(0, 5))
    b.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE))
    b.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
    for s in sqs[2:]:
        p = chess.Piece.from_symbol(rng.choice(_PIECES))
        if p.piece_type == chess.PAWN and chess.square_rank(s) in (0, 7):
            continue
        b.set_piece_at(s, p)
    b.turn = rng.choice([chess.WHITE, chess.BLACK])
    b.halfmove_clock = clock
    if not b.is_valid() or b.is_game_over():
        return None
    return b


def _hist_board(rng: random.Random) -> chess.Board | None:
    """Reversible back-and-forth moves: a repetition-bearing history."""
    b = _rand_board(rng, rng.randint(80, 98))
    if b is None:
        return None
    for _ in range(rng.randint(0, 20)):
        if b.halfmove_clock >= 97 and rng.random() < 0.4:
            break
        rev = [m for m in b.legal_moves if not b.is_zeroing(m)]
        if not rev:
            return None
        if len(b.move_stack) >= 2 and rng.random() < 0.6:
            back = chess.Move(b.move_stack[-2].to_square, b.move_stack[-2].from_square)
            if back in rev:
                b.push(back)
                if b.is_game_over(claim_draw=False):
                    return None
                continue
        b.push(rng.choice(rev))
        if b.is_game_over(claim_draw=False):
            return None
    return b if 97 <= b.halfmove_clock <= 100 else None


def _positions(seed: int, n: int) -> list[chess.Board]:
    rng = random.Random(seed)
    out: list[chess.Board] = []
    for i in range(n):
        kind = i % 3
        if kind == 0:
            b = _rand_board(rng, rng.randint(97, 100))
        elif kind == 1:
            b = _hist_board(rng)
        else:
            b = _rand_board(rng, rng.randint(0, 96))
        if b is not None and not b.is_game_over(claim_draw=False):
            out.append(b)
    return out


_N_ROOTS = 5000  # ~2,000 usable roots, ~30k moves; the review ran ~360k moves


@pytest.mark.parametrize("seed", [1, 2])
def test_the_helpers_match_the_oracle_on_random_roots(seed: int) -> None:
    roots = _positions(seed, _N_ROOTS)
    assert len(roots) > 1500
    moves = forced = missed_off_scope = 0
    for b in roots:
        legal = list(b.legal_moves)
        acts = [int(move_to_index(m, b)) for m in legal]
        before = (b.fen(), len(b.move_stack))
        got = set(forced_draw_actions(b, acts))
        assert (b.fen(), len(b.move_stack)) == before
        cb = CBoard.from_board(b)
        cb_got = set(forced_draw_actions_cboard(cb, acts))
        # The clock-99 additions agree between the two helpers. (Their base
        # sets can differ on insufficient material: CBoard does not treat
        # same-coloured bishops as a dead position, python-chess does --
        # `4b3/8/8/8/6bk/1B6/2r5/2K5 w - - 97 1` Bxc2. That predates this rule.)
        assert (cb_got - set(_old_forced_draw_actions_cboard(cb, acts))) == (
            got - set(_old_forced_draw_actions(b, acts))
        ), b.fen()
        for m, a in zip(legal, acts, strict=True):
            want = _oracle(b, m)
            have = a in got
            # Never a false draw. A miss is allowed only outside the clock-99
            # rule's scope: a move after which every reply draws some OTHER
            # way (e.g. the only reply captures into insufficient material),
            # which the helpers deliberately do not scan for on every root.
            assert want or not have, ("false draw", b.fen(), m.uci())
            if want and not have:
                b.push(m)
                try:
                    assert b.halfmove_clock != 99, ("missed clock-99 draw", b.fen())
                finally:
                    b.pop()
                missed_off_scope += 1
            moves += 1
            forced += int(have)
    assert moves > 20_000
    assert forced > 0  # the sample includes forced draws
    assert missed_off_scope >= 0


@pytest.mark.parametrize("seed", [3, 4])
def test_roots_without_a_reply_forced_move_are_unchanged(seed: int) -> None:
    """Every root where no move leaves the clock at 99 with every reply drawn:
    the new helpers return exactly the old helpers' answer."""
    compared = with_draws = 0
    for b in _positions(seed, _N_ROOTS):
        acts = [int(move_to_index(m, b)) for m in b.legal_moves]
        old = _old_forced_draw_actions(b, acts)
        new = forced_draw_actions(b, acts)
        if set(new) - set(old):  # a reply-forced draw: the one intended change
            for a in set(new) - set(old):
                move = next(m for m in b.legal_moves if int(move_to_index(m, b)) == a)
                b.push(move)
                try:
                    assert b.halfmove_clock == 99
                finally:
                    b.pop()
            continue
        assert new == old, b.fen()
        cb = CBoard.from_board(b)
        assert forced_draw_actions_cboard(cb, acts) == _old_forced_draw_actions_cboard(cb, acts)
        compared += 1
        with_draws += int(bool(old))
    assert compared > 1500
    assert with_draws > 0


# --- the runners: bit-identical off the clock-99 case ------------------------

IDENTITY_ROOTS = [
    chess.Board().fen(),
    BXA7_FEN,             # pruned stalemate, below-zero fallback fires
    KXD7_FEN,             # would-be singleton, draws stay in
    SURVIVOR_WINS_FEN,
    CLAIMABLE_ONLY_FEN,   # clock 98: claims pruned, none forced
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "8/8/4k3/8/2K5/8/3Q4/8 w - - 40 80",
]


def _patch_old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gumbel_c_module, "forced_draw_actions_cboard", _old_forced_draw_actions_cboard)
    monkeypatch.setattr(gumbel_module, "forced_draw_actions", _old_forced_draw_actions)


def _both(run: Any, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    new = run()
    with monkeypatch.context() as m:
        _patch_old(m)
        old = run()
    return new, old


@pytest.mark.parametrize("fen", IDENTITY_ROOTS)
@pytest.mark.parametrize("evaluator", [_NeutralLeaves, _SideToMoveWins])
def test_runners_are_bit_identical_without_a_reply_forced_move(
    fen: str, evaluator: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = chess.Board(fen)
    for use_cboard in (False, True):
        new, old = _both(lambda u=use_cboard: _run_c(board, evaluator(), use_cboard=u), monkeypatch)
        assert np.array_equal(new[0], old[0]), (fen, use_cboard)  # policy target
        assert new[1] == old[1]  # move
        assert new[2] == old[2]  # value
    new, old = _both(lambda: _run_py(board, evaluator(), topk=16), monkeypatch)
    assert np.array_equal(new[0], old[0]), fen
    assert new[1:] == old[1:]


def test_runners_play_a_reply_forced_draw_on_a_refuted_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """The consumers receive the new rule: the C runner (both entries) plays Kd8
    where the old helper let it play a pawn move that is mated next move."""
    board = chess.Board(REPLY_FORCED_FEN)
    draw = int(move_to_index(chess.Move.from_uci(REPLY_FORCED_DRAW), board))
    for use_cboard in (False, True):
        new, old = _both(lambda u=use_cboard: _run_c(board, _SideToMoveWins(), use_cboard=u), monkeypatch)
        assert new[1] == draw
        assert new[2] == 0.0
        assert old[1] != draw
    # The Python reference: its 64-sim search does not drive both pawn moves
    # below 0 here, so the fallback does not fire; pin that its pruned-draw
    # set (what the fallback would play) now holds the draw.
    seen: list[list[int]] = []

    def spy(b: chess.Board, actions: Iterable[int]) -> list[int]:
        out = forced_draw_actions(b, actions)
        seen.append(out)
        return out

    monkeypatch.setattr(gumbel_module, "forced_draw_actions", spy)
    _run_py(board, _SideToMoveWins(), topk=16)
    assert seen == [[draw]]
