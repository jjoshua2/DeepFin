"""Fail-closed E2 admission primitive; not yet wired into generator persistence.

Replay establishes internal consistency, NOT authenticated historical provenance.
Only complete standard-start histories and automatic natural terminals qualify.
Claimable draws, resignation, censoring and adjudication are deliberately excluded.
Capture BEFORE any tablebase rescore; never substitute a TB WDL for game outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from collections.abc import Sequence

import chess


@dataclass(frozen=True)
class NaturalOutcome:
    moves: tuple[str, ...]
    result: str
    termination: str
    final_fen: str
    trace_sha256: str


def capture_natural_outcome(
    board: chess.Board,
    *,
    resolved_result: str,
    adjudicated: bool,
    tb_adjudicated: bool,
    resigned: bool,
    censored: bool,
) -> NaturalOutcome:
    """Snapshot full board history before rescore; refuse unknown/altered histories."""
    flags = (adjudicated, tb_adjudicated, resigned, censored)
    if any(type(v) is not bool for v in flags) or any(flags):
        raise ValueError("non-natural or unknown termination flags")
    if board.chess960 or board.root().fen() != chess.STARTING_FEN:
        raise ValueError("complete standard-start history required")
    moves = tuple(m.uci() for m in board.move_stack)
    if not moves or len(moves) > 1600:
        raise ValueError("bounded complete history required")
    replay = chess.Board()
    for uci in moves:
        if replay.is_game_over(claim_draw=False):
            raise ValueError("move after automatic terminal")
        move = chess.Move.from_uci(uci)
        if move not in replay.legal_moves:
            raise ValueError("illegal history")
        replay.push(move)
    if replay.fen(en_passant="fen") != board.fen(en_passant="fen"):
        raise ValueError("history/final state mismatch")
    outcome = replay.outcome(claim_draw=False)
    if outcome is None or outcome.result() != resolved_result:
        raise ValueError("automatic natural result required")
    payload = {
        "start": chess.STARTING_FEN,
        "moves": moves,
        "result": outcome.result(),
        "termination": outcome.termination.name,
        "final_fen": replay.fen(en_passant="fen"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return NaturalOutcome(
        moves, outcome.result(), outcome.termination.name, replay.fen(en_passant="fen"), digest
    )


def admit_rows(
    receipt: NaturalOutcome,
    *,
    expected_trace_sha256: str,
    plies: Sequence[int],
    stm_white: Sequence[bool],
    outcomes: Sequence[int],
    tb_rescored: bool,
) -> tuple[int, ...]:
    """Bind ordered pre-move rows to replay; WDL class order W=0,D=1,L=2.

    The expected digest must come from independently pinned capture evidence.
    This API does not authenticate that external receipt or decode stored planes.
    Any TB-rescored bank is excluded, even if its labels happen to agree.
    """
    if type(tb_rescored) is not bool or tb_rescored:
        raise ValueError("TB-rescored or unknown outcome source")
    if (
        type(receipt.moves) is not tuple
        or not 1 <= len(receipt.moves) <= 1600
        or any(type(m) is not str or not 4 <= len(m) <= 5 for m in receipt.moves)
    ):
        raise ValueError("bounded immutable UCI history required")
    ps, turns, labels = tuple(plies), tuple(stm_white), tuple(outcomes)
    if not ps or len(ps) > 256 or not (len(ps) == len(turns) == len(labels)):
        raise ValueError("invalid bounded row cardinality")
    if any(type(p) is not int or not 0 <= p < len(receipt.moves) for p in ps):
        raise ValueError("invalid pre-move ply")
    if len(set(ps)) != len(ps):
        raise ValueError("duplicate row ply")
    replay = chess.Board()
    positions = []
    for uci in receipt.moves:
        positions.append(replay.turn)
        move = chess.Move.from_uci(uci)
        if move not in replay.legal_moves:
            raise ValueError("illegal receipt history")
        replay.push(move)
    verified = capture_natural_outcome(
        replay,
        resolved_result=receipt.result,
        adjudicated=False,
        tb_adjudicated=False,
        resigned=False,
        censored=False,
    )
    if verified != receipt or verified.trace_sha256 != expected_trace_sha256:
        raise ValueError("receipt identity mismatch")
    expected = []
    for p, turn, label in zip(ps, turns, labels):
        if type(turn) is not bool or turn != positions[p]:
            raise ValueError("row side-to-move mismatch")
        value = (
            1
            if receipt.result == "1/2-1/2"
            else (0 if (receipt.result == "1-0") == turn else 2)
        )
        if type(label) is not int or label != value:
            raise ValueError("stored outcome differs from natural result")
        expected.append(value)
    return tuple(expected)


def require_search_labels(
    labels: Sequence[Sequence[float]], present: Sequence[bool]
) -> None:
    """Reject missing/invalid sources before the production outcome fallback.

    Values must already be qualified normalized labels; this gate never repairs
    quantized values or changes the training mixture.
    """
    rows = tuple(tuple(row) for row in labels)
    masks = tuple(present)
    if not rows or len(rows) > 256 or len(rows) != len(masks):
        raise ValueError("invalid search cardinality")
    for row, mask in zip(rows, masks):
        if type(mask) is not bool or not mask or len(row) != 3:
            raise ValueError("search label missing")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in row):
            raise ValueError("invalid search distribution")
        if abs(math.fsum(row) - 1) > 1e-6:
            raise ValueError("search distribution not normalized")
