"""Immutable, tensor-free observations for a finite collection call.

Ordinals identify starts within one ``play_batch`` call, not games across
workers or resumed sessions. Terminal provenance describes the producing path;
in particular, a tablebase result is not a claim of rule50 qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .state import SelfplayState
    from .strict_tb import TablebaseWitness

TerminalResolution = Literal[
    "natural", "tablebase_adjudication", "sf_adjudication", "sf_default_draw", "unresolved",
]

@dataclass(frozen=True)
class GameStartProvenance:
    start_ordinal: int
    slot: int
    root_fen: str
    opening_moves: tuple[str, ...]
    start_fen: str
    start_ply: int
    opening_source: str
    net_color: bool
    is_selfplay: bool


@dataclass(frozen=True)
class NetRecordProvenance:
    move_offset: int
    ply_index: int
    pov_color: bool
    has_policy: bool
    is_sf_refute_opp: bool


@dataclass(frozen=True)
class GameProvenance:
    """Raw producing paths, including separate final-label rescoring.

    In legacy mode ``natural_result`` is the native board's result (often
    ``*`` at a cap). Strict mode checks the full-history claim policy instead
    and preserves the separate native string in ``native_result``.
    ``resolved_result`` precedes optional TB rescoring; ``result`` labels replay.
    ``cap_reached`` counts plies since the opening, not absolute game ply.
    Played action IDs use the full4672 move space, independent of policy encoding.
    """

    start: GameStartProvenance
    game_id: int
    played_move_indices: tuple[int, ...]
    final_fen: str
    records: tuple[NetRecordProvenance, ...]
    result: str
    resolved_result: str
    natural_result: str
    resolution: TerminalResolution
    tablebase_rescore_result: str | None
    cap_reached: bool
    total_plies: int
    retained_samples: int
    # Strict-mode additions. Legacy fields/default meanings remain intact.
    native_result: str | None = None
    tablebase_witness: TablebaseWitness | None = None
    tablebase_policy_applied_plies: tuple[int, ...] = ()
    tablebase_policy_abstained_plies: tuple[int, ...] = ()


def capture_game_start(
    state: SelfplayState, slot: int, start_ordinal: int,
) -> GameStartProvenance:
    """Snapshot at slot creation, before any result-dependent game identity."""
    if state.starting_boards is None:
        raise ValueError("game provenance requires the original opening board")
    board = state.starting_boards[slot]
    return GameStartProvenance(
        start_ordinal=int(start_ordinal), slot=int(slot),
        root_fen=board.root().fen(),
        opening_moves=tuple(move.uci() for move in board.move_stack),
        start_fen=board.fen(),
        start_ply=board.ply(),
        opening_source=str(state.opening_source_arr[slot]),
        net_color=bool(state.net_color_arr[slot]),
        is_selfplay=bool(state.selfplay_arr[slot]),
    )
