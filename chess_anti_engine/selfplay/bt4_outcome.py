"""Opt-in outcome decisions for future BT4 selfplay generation.

This is a root-boundary policy, not a search or shard-writer hook. Historical
``theoretical_wdl`` callers use ``preflight_six_man_tablebases`` once per
worker. ``rule50_match_v1`` callers instead own a handle returned by
``tablebase.open_strict_match_tablebase`` and pass it to each decision. Call
``decide_bt4_outcome`` before evaluating each root. ``None`` means play on.
A decision with ``result is None`` means discard the whole unfinished game:
replay rows require a known game outcome, and ``"*"`` must not become a draw.

The caller must choose an outcome mode. ``theoretical_wdl`` retains the
historical training-label convention. ``rule50_match_v1`` uses the separately
opened strict WDL/DTZ match tablebase, refuses missing eligible probes, and
discards an unresolved positive-clock decisive position. Native teacher
outputs are independent of either game-result convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import chess
import chess.syzygy

from chess_anti_engine import tablebase
from chess_anti_engine.utils.syzygy import SEPARATOR, require_tablebases

SIX_MAN_MAX_PIECES = 6
BT4OutcomeMode = Literal["theoretical_wdl", "rule50_match_v1"]

BT4Termination = Literal[
    "natural", "syzygy", "tablebase_unavailable", "rule50_unresolved",
    "max_plies_unresolved",
]


@dataclass(frozen=True)
class BT4OutcomeDecision:
    """Final game result, or an explicit reason to discard all its rows."""

    result: str | None
    termination: BT4Termination
    detail: str
    outcome_mode: BT4OutcomeMode = "theoretical_wdl"


def preflight_six_man_tablebases(syzygy_path: str) -> tuple[str, str]:
    """Preflight the historical training-label pair before worker generation.

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
    outcome_mode: BT4OutcomeMode,
    match_tablebase: chess.syzygy.Tablebase | None = None,
) -> BT4OutcomeDecision | None:
    """Decide at a BT4 root, before its model inference or search.

    Natural/claimable outcomes take precedence. At six pieces or fewer, a
    historical-mode failed probe discards the game. Match mode requires a
    separately opened strict tablebase; missing covered material raises and
    must fail the run, while an undecidable positive-clock position discards
    the game with its own counter. A covered result precedes the ply cap.
    """
    if plies < 0 or max_plies <= 0:
        raise ValueError("BT4 plies must be nonnegative and max_plies positive")
    if not syzygy_path:
        raise ValueError("BT4 six-man adjudication needs a Syzygy path")
    if cast(str, outcome_mode) not in ("theoretical_wdl", "rule50_match_v1"):
        raise ValueError("BT4 outcome mode is unsupported")
    if outcome_mode == "rule50_match_v1" and match_tablebase is None:
        raise ValueError("BT4 rule50 match mode requires a strict tablebase handle")
    if outcome_mode == "theoretical_wdl" and match_tablebase is not None:
        raise ValueError("BT4 theoretical mode cannot silently ignore a match tablebase")

    natural = board.outcome(claim_draw=True)
    if natural is not None:
        return BT4OutcomeDecision(
            natural.result(), "natural", natural.termination.name.lower(), outcome_mode,
        )

    if chess.popcount(board.occupied) <= SIX_MAN_MAX_PIECES:
        if not tablebase.is_tb_eligible(board):
            return BT4OutcomeDecision(
                None, "tablebase_unavailable", "castling_rights", outcome_mode,
            )
        if outcome_mode == "rule50_match_v1":
            assert match_tablebase is not None
            result = tablebase.rule50_match_result(
                board, match_tablebase, max_pieces=SIX_MAN_MAX_PIECES,
            )
            if result is None:
                return BT4OutcomeDecision(
                    None, "rule50_unresolved", "positive_clock_decisive_wdl",
                    outcome_mode,
                )
            return BT4OutcomeDecision(
                result, "syzygy", "rule50_match_wdl_dtz", outcome_mode,
            )
        result = tablebase.tb_adjudicate_result(board, syzygy_path)
        if result is None:
            return BT4OutcomeDecision(
                None, "tablebase_unavailable", "missing_material", outcome_mode,
            )
        return BT4OutcomeDecision(result, "syzygy", "theoretical_wdl", outcome_mode)

    if plies >= max_plies:
        return BT4OutcomeDecision(
            None, "max_plies_unresolved", "above_six_man", outcome_mode,
        )
    return None
