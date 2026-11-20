"""``board.outcome(claim_draw=True)`` without python-chess's legal-move sweep.

python-chess decides a claimable threefold by popping back to the last
irreversible move to count position keys, and then -- whether or not any
position has repeated -- pushing and popping EVERY legal move to see whether
one of them would make a third occurrence. That sweep is most of the cost of
``outcome(claim_draw=True)``, and selfplay/arena call it for every live game on
every ply, some of them twice (game-over test, then the rule-50 TB status).

A claim needs some position key to occur at least twice in the reversible
window already: the current position three times, or a reachable position
that is already there twice. So when no key repeats in the window, the
threefold claim is impossible and the answer is the claim-free outcome plus the
fifty-move claim; when one does, the library itself decides. The window
scanned here runs back to the last ZEROING move, which never stops earlier
than python-chess's last IRREVERSIBLE move (every zeroing move is
irreversible), so it can only see more repeats, never fewer: a missed repeat
would change an answer, an extra one only costs the library call. The result
is therefore identical to ``board.outcome(claim_draw=True)`` by construction;
``tests/test_claimable_outcome.py`` checks it position by position.

The board is temporarily popped and restored, exactly as python-chess's own
claim check does; callers must not share it with another thread meanwhile.
"""
from __future__ import annotations

import chess


def _reversible_window_repeats(board: chess.Board) -> bool:
    """Whether any position key occurs twice back to the last zeroing move."""
    seen = {board._transposition_key()}
    switchyard: list[chess.Move] = []
    try:
        while board.move_stack:
            move = board.pop()
            switchyard.append(move)
            if board.is_zeroing(move):
                return False
            key = board._transposition_key()
            if key in seen:
                return True
            seen.add(key)
        return False
    finally:
        while switchyard:
            board.push(switchyard.pop())


def outcome_claim_draw(board: chess.Board) -> chess.Outcome | None:
    """Same value as ``board.outcome(claim_draw=True)``, usually much faster."""
    if _reversible_window_repeats(board):
        return board.outcome(claim_draw=True)
    natural = board.outcome(claim_draw=False)
    if natural is not None:
        return natural
    if board.can_claim_fifty_moves():
        return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    return None


def is_game_over_claim_draw(board: chess.Board) -> bool:
    """Same value as ``board.is_game_over(claim_draw=True)``."""
    return outcome_claim_draw(board) is not None


def result_claim_draw(board: chess.Board) -> str:
    """Same value as ``board.result(claim_draw=True)``."""
    outcome = outcome_claim_draw(board)
    return outcome.result() if outcome is not None else "*"
