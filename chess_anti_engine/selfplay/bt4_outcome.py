"""Opt-in outcome decisions for future BT4 selfplay generation.

This is a root-boundary policy, not a search or shard-writer hook. Call
``preflight_six_man_tablebases`` once in each worker before its first game,
then call ``decide_bt4_outcome`` before evaluating each root. ``None`` means
play on. A decision with ``result is None`` means discard the whole unfinished
game: replay rows require a known game outcome, and ``"*"`` must not become a
draw label.

Natural results follow the SF-rooted generator's claim-aware python-chess
rule. Syzygy results follow the existing *training-label* convention in
``tablebase.tb_adjudicate_result``: cursed wins and blessed losses are decisive
theoretical outcomes. Syzygy WDL assumes the fifty-move clock was just reset;
this adjudication is not a claim-aware verdict for the current clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chess

from chess_anti_engine import tablebase
from chess_anti_engine.utils.syzygy import SEPARATOR, require_tablebases

SIX_MAN_MAX_PIECES = 6

BT4Termination = Literal[
    "natural", "syzygy", "tablebase_unavailable", "max_plies_unresolved"
]


@dataclass(frozen=True)
class BT4OutcomeDecision:
    """Final game result, or an explicit reason to discard all its rows."""

    result: str | None
    termination: BT4Termination
    detail: str


def preflight_six_man_tablebases(syzygy_path: str) -> tuple[str, str]:
    """Refuse a missing half of the production pair before worker generation.

    This checks directory presence, opens each component, and verifies the
    combined handle has both WDL and DTZ files. It cannot prove every material
    file exists; a missing per-position probe produces a discard decision that
    the caller must count.
    """
    parts = tuple(syzygy_path.split(SEPARATOR))
    if len(parts) != 2 or any(not part or part != part.strip() for part in parts):
        raise ValueError("BT4 six-man adjudication needs both Syzygy paths")
    if Path(parts[0]).resolve() == Path(parts[1]).resolve():
        raise ValueError("BT4 six-man adjudication needs two distinct Syzygy paths")
    require_tablebases(syzygy_path, what="BT4 six-man Syzygy pair")
    for part in parts:
        component = tablebase.get_tablebase(part)
        if component is None or (not component.wdl and not component.dtz):
            raise ValueError(f"BT4 six-man Syzygy component {part!r} did not open")
    tb = tablebase.get_tablebase(syzygy_path)
    if tb is None or not tb.wdl or not tb.dtz:
        raise ValueError("BT4 six-man Syzygy pair needs WDL and DTZ files")
    return parts[0], parts[1]


def decide_bt4_outcome(
    board: chess.Board,
    *,
    plies: int,
    max_plies: int,
    syzygy_path: str,
) -> BT4OutcomeDecision | None:
    """Decide at a BT4 root, before its model inference or search.

    Natural/claimable outcomes take precedence. At six pieces or fewer, a
    failed probe (including castling rights or missing material coverage)
    discards the game immediately instead of playing on or labelling it draw.
    A covered Syzygy position is adjudicated even if the ply cap was reached
    on the same root. Above six pieces the cap discards an unresolved game.
    """
    if plies < 0 or max_plies <= 0:
        raise ValueError("BT4 plies must be nonnegative and max_plies positive")
    if not syzygy_path:
        raise ValueError("BT4 six-man adjudication needs a Syzygy path")

    natural = board.outcome(claim_draw=True)
    if natural is not None:
        return BT4OutcomeDecision(
            natural.result(), "natural", natural.termination.name.lower()
        )

    if chess.popcount(board.occupied) <= SIX_MAN_MAX_PIECES:
        if not tablebase.is_tb_eligible(board):
            return BT4OutcomeDecision(
                None, "tablebase_unavailable", "castling_rights"
            )
        result = tablebase.tb_adjudicate_result(board, syzygy_path)
        if result is None:
            return BT4OutcomeDecision(
                None, "tablebase_unavailable", "missing_material"
            )
        return BT4OutcomeDecision(result, "syzygy", "theoretical_wdl")

    if plies >= max_plies:
        return BT4OutcomeDecision(None, "max_plies_unresolved", "above_six_man")
    return None
