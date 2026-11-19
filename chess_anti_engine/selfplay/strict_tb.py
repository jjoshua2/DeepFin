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

from chess_anti_engine.utils.claimable_outcome import outcome_claim_draw
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


@dataclass
class _HistoryReplay:
    """One slot's replayed game, extended in place as the game grows.

    ``start`` is the exact starting-board OBJECT the replay began from, with
    the length and FEN it had then; ``actions`` the played action ids already
    pushed onto ``board`` (``ucis`` the same moves as UCI strings, ``root_fen``
    the replay's root). Reuse requires all of them to still describe the slot,
    so a recycled slot (new starting board), a resumed game or any rewritten
    history rebuilds from scratch. ``verified`` is the (action count, native
    FEN) pair the last native-board comparison passed for: the comparison is
    skipped only while both are unchanged, i.e. when it would compare the same
    two strings again.
    """

    start: chess.Board
    start_plies: int
    start_fen: str
    actions: list[int]
    board: chess.Board
    root_fen: str
    ucis: list[str]
    verified: tuple[int, str] | None = None


def _push_checked_actions(board: chess.Board, actions: Sequence[int]) -> None:
    from chess_anti_engine.moves import index_to_move

    for action in actions:
        move = index_to_move(int(action), board)
        if move not in board.legal_moves:
            raise ValueError("strict selfplay has an illegal played action")
        board.push(move)


def _reusable_replay(
    entry: _HistoryReplay | None, start: chess.Board, history: Sequence[int],
) -> _HistoryReplay | None:
    if entry is None:
        return None
    n = len(entry.actions)
    if (entry.start is not start or entry.start_plies != len(start.move_stack)
            or len(history) < n or list(history[:n]) != entry.actions
            or entry.start_fen != start.fen()):
        return None
    return entry


def full_history_board(state: _HistoryState, slot: int) -> chess.Board:
    """The slot's original opening plus every played action, legality-checked.

    Returns a fresh board the caller may mutate."""
    return _replayed_history(state, slot).board.copy(stack=True)


def _replayed_history(state: _HistoryState, slot: int) -> _HistoryReplay:
    """The checked replay behind ``full_history_board``; its board is READ-ONLY.

    When ``state`` carries a
    ``full_history_cache`` (``SelfplayState`` does) the replay is incremental:
    only actions appended since the previous call are decoded, checked and
    pushed, instead of re-validating the whole game on every step. The
    O(plies) full replay ran several times per slot per step in strict
    selfplay (TB adjudication, every SF label submit, finalize), i.e. O(plies^2)
    python-chess legal-move generation per game -- a third of the main
    thread's wall in a CPU profile of the collector shape. Checks are
    unchanged: each action is legality-checked once against the identical
    prefix position, and the native-board FEN comparison runs on every call.
    """
    if state.starting_boards is None:
        raise ValueError("strict selfplay requires the original opening history")
    start = state.starting_boards[slot]
    history = state.move_idx_history[slot]
    cache: dict[int, _HistoryReplay] | None = getattr(state, "full_history_cache", None)
    entry = _reusable_replay(cache.get(slot) if cache is not None else None, start, history)
    if entry is None:
        board = start.copy(stack=True)
        if not board.is_valid() or not board.root().is_valid():
            raise ValueError("strict selfplay requires a valid original opening")
        entry = _HistoryReplay(start=start, start_plies=len(start.move_stack),
                               start_fen=start.fen(), actions=[], board=board,
                               root_fen=board.root().fen(),
                               ucis=[move.uci() for move in board.move_stack])
    if cache is not None:
        # Dropped while extending: a failed check leaves the board part-pushed.
        cache.pop(slot, None)
    new_actions = [int(a) for a in history[len(entry.actions):]]
    _push_checked_actions(entry.board, new_actions)
    entry.actions.extend(new_actions)
    entry.ucis.extend(move.uci() for move in entry.board.move_stack[len(entry.ucis):])
    native_fen = state.cboards[slot].fen()
    if entry.verified != (len(entry.actions), native_fen):
        if entry.board.fen() != chess.Board(native_fen).fen():
            raise ValueError("strict selfplay history/native position or clocks disagree")
        entry.verified = (len(entry.actions), native_fen)
    if cache is not None:
        cache[slot] = entry
    return entry


class SfHistoryKwargs(TypedDict, total=False):
    root_fen: str
    root_moves: tuple[str, ...]


def sf_history_kwargs(state: _HistoryState, slot: int) -> SfHistoryKwargs:
    if getattr(state.game, "syzygy_outcome_mode", "theoretical_wdl") != "rule50_match_v1":
        return {}
    entry = _replayed_history(state, slot)
    return {"root_fen": entry.root_fen, "root_moves": tuple(entry.ucis)}


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
        natural = outcome_claim_draw(board)
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
        if outcome_claim_draw(board) is not None:
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
