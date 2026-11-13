"""Opt-in finite selfplay TB ownership, history and outcome witnesses.

This board is for rules decisions and SF transport. It deliberately does not
replace SelfplayState.replay_board or change its encoding-history contract.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict

import chess
import chess.syzygy

from chess_anti_engine.tablebase import (
    MatchTablebaseError,
    _dtz_of_played_move,
    _rule50_dtz_key,
    open_strict_match_tablebase,
    rule50_match_status,
)

if TYPE_CHECKING:
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from .config import GameConfig


class _HistoryState(Protocol):
    """Only the read-only history/config surface needed by these helpers."""

    @property
    def starting_boards(self) -> Sequence[chess.Board] | None: ...

    @property
    def move_idx_history(self) -> Sequence[Sequence[int]]: ...

    @property
    def cboards(self) -> Sequence[CBoard]: ...

    @property
    def game(self) -> GameConfig: ...


@dataclass(frozen=True)
class TablebaseWitness:
    mode: str
    origin: str
    root_fen: str
    moves: tuple[str, ...]
    final_fen: str
    move_offset: int
    turn: bool
    halfmove_clock: int
    wdl: int | None
    dtz: int | None
    status: int | None
    result: str
    max_pieces: int
    asset_inventory_sha256: str
    # Inventory identity is names/stat metadata, not a claim of byte hashing.
    asset_identity_scope: str = "resolved-file-path,size,mtime_ns;not-content-hash"


def full_history_board(state: _HistoryState, slot: int) -> chess.Board:
    from chess_anti_engine.moves import index_to_move

    if state.starting_boards is None:
        raise ValueError("strict selfplay requires the original opening history")
    board = state.starting_boards[slot].copy(stack=True)
    if not board.is_valid() or not board.root().is_valid():
        raise ValueError("strict selfplay requires a valid original opening")
    for action in state.move_idx_history[slot]:
        move = index_to_move(int(action), board)
        if move not in board.legal_moves:
            raise ValueError("strict selfplay has an illegal played action")
        board.push(move)
    if board.fen() != chess.Board(state.cboards[slot].fen()).fen():
        raise ValueError("strict selfplay history/native position or clocks disagree")
    return board


class SfHistoryKwargs(TypedDict, total=False):
    root_fen: str
    root_moves: tuple[str, ...]


def sf_history_kwargs(state: _HistoryState, slot: int) -> SfHistoryKwargs:
    if getattr(state.game, "syzygy_outcome_mode", "theoretical_wdl") != "rule50_match_v1":
        return {}
    board = full_history_board(state, slot)
    return {"root_fen": board.root().fen(),
            "root_moves": tuple(move.uci() for move in board.move_stack)}


class StrictSelfplayTablebase:
    """One owned, no-eviction handle per finite call, never a global cache."""

    def __init__(self, path: str):
        self.tablebase = open_strict_match_tablebase(path, max_pieces=6)
        self._closed = False
        try:
            # Capacity alone is insufficient for the strict 3–6-man contract.
            expected = set(chess.syzygy.tablenames(piece_count=6))
            for kind, tables in (("WDL", self.tablebase.wdl), ("DTZ", self.tablebase.dtz)):
                present = {chess.syzygy.normalize_tablename(name) for name in tables}
                missing = expected - present
                if missing:
                    raise MatchTablebaseError(f"strict selfplay missing {kind} materials: {sorted(missing)}")
            inventory = []
            for tables in (self.tablebase.wdl, self.tablebase.dtz):
                for table_path in sorted({str(Path(table.path).resolve()) for table in tables.values()}):
                    stat = Path(table_path).stat()
                    inventory.append((table_path, stat.st_size, stat.st_mtime_ns))
            self.asset_inventory_sha256 = hashlib.sha256(
                json.dumps(inventory, separators=(",", ":")).encode()
            ).hexdigest()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.tablebase.close()

    def resolve(self, board: chess.Board, *, move_offset: int) -> TablebaseWitness:
        natural = board.outcome(claim_draw=True)
        status = rule50_match_status(board, self.tablebase, max_pieces=6)
        wdl = dtz = None
        if natural is not None:
            origin, result = "NATURAL", natural.result()
        else:
            eligible = chess.popcount(board.occupied) <= 6 and not board.castling_rights
            if eligible:
                # The shared primitive validated both probes. Retain raw values
                # in the witness as well as the conservative resolved status.
                wdl = int(self.tablebase.probe_wdl(board))
                dtz = int(self.tablebase.probe_dtz(board))
            if status is None:
                origin, result = "UNRESOLVED", "*"
            elif status == 2:
                origin, result = "TB_ADJUDICATED", "1/2-1/2"
            else:
                white_wins = (status == 1) == board.turn
                origin, result = "TB_ADJUDICATED", "1-0" if white_wins else "0-1"
        return TablebaseWitness(
            mode="rule50_match_v1", origin=origin,
            root_fen=board.root().fen(), moves=tuple(m.uci() for m in board.move_stack),
            final_fen=board.fen(), move_offset=move_offset, turn=board.turn,
            halfmove_clock=board.halfmove_clock, wdl=wdl, dtz=dtz,
            status=status, result=result, max_pieces=6,
            asset_inventory_sha256=self.asset_inventory_sha256,
        )

    def policy_move(self, board: chess.Board) -> chess.Move | None:
        """Strict policy-only DTZ ranking, abstaining on history-ambiguous roots/children.

        Probe every legal child before abstaining, so missing covered material
        cannot masquerade as an ordinary conservative abstention.
        """
        if board.outcome(claim_draw=True) is not None:
            return None
        if chess.popcount(board.occupied) > 6 or board.castling_rights:
            return None
        status = rule50_match_status(board, self.tablebase, max_pieces=6)
        candidates: list[tuple[tuple[int, int, int], chess.Move]] = []
        unresolved = status is None
        for move in list(board.legal_moves):
            child = board.copy(stack=True)
            child.push(move)
            child_status = rule50_match_status(child, self.tablebase, max_pieces=6)
            if child_status is None:
                unresolved = True
                continue
            try:
                dtz = _dtz_of_played_move(child, self.tablebase)
            except (KeyError, IndexError, chess.syzygy.MissingTableError) as exc:
                raise MatchTablebaseError("missing strict policy child") from exc
            rank = 0 if child_status == 2 else -child_status
            counter = 0 if child.halfmove_clock == 0 or child.is_checkmate() else board.halfmove_clock
            candidates.append(((rank, *_rule50_dtz_key(dtz, counter)), move))
        if unresolved or not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]


def validate_stockfish_contract(stockfish: object, path: str) -> None:
    """The actual instance, including every pooled replacement, must be explicit."""
    if (getattr(stockfish, "syzygy_path", None) != path
            or getattr(stockfish, "syzygy_50_move_rule", None) is not True
            or getattr(stockfish, "syzygy_probe_limit", None) != 6):
        raise ValueError("strict selfplay requires explicit SF full TB path, rule50=true, probe_limit=6")
    # Pool construction synchronously capability-checks all engines. The single
    # engine sets this flag only after its requested options' ready barrier.
    engines = getattr(stockfish, "_engines", (stockfish,))
    if not engines or any(not getattr(engine, "syzygy_ready_after_requests", False) for engine in engines):
        raise ValueError("strict selfplay SF capability/readiness contract was not realized")
