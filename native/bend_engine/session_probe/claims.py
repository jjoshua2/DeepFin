"""Optional claims with explicit evidence, not automatic terminal adjudication.

A prospective claim is made before its intended move. The move is a witness,
not a child to apply. This history-owning host is trusted by the Bend probe.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class Claim:
    reason: str
    intended_move: str | None = None


def claim_option(board: chess.Board) -> Claim | None:
    """Return one reproducible valid zero-valued option without changing history.

    Automatic endings (including mate/stalemate) have priority. Prefer a claim in
    the current position, then a prospective fifty-move or threefold witness in
    sorted UCI order. Multiple valid claims are equivalent zero-valued actions.
    """
    if board.chess960 or not board.is_valid():
        raise ValueError('claim evidence requires valid orthodox chess')
    if board.outcome(claim_draw=False) is not None:
        return None
    if board.is_fifty_moves():
        return Claim('fifty_moves')
    if board.is_repetition(3):
        return Claim('threefold_repetition')
    for reason in ('fifty_moves', 'threefold_repetition'):
        if reason == 'fifty_moves' and board.halfmove_clock < 99:
            continue
        if reason == 'threefold_repetition' and not board.can_claim_threefold_repetition():
            continue
        for move in sorted(board.legal_moves, key=lambda m: m.uci()):
            child = board.copy(stack=True)
            child.push(move)
            eligible = child.is_fifty_moves() if reason == 'fifty_moves' else child.is_repetition(3)
            if eligible:
                return Claim(reason, move.uci())
    return None
