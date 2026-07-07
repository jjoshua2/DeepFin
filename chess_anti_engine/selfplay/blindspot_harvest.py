"""Inline blind-spot harvesting from finished selfplay games.

A value-blind position is one the net's OWN search values as fine
(``net_q = search_wdl[W]-[L] > net_ok``) while the in-loop Stockfish eval says
lost (``sf_q = sf_wdl[W]-[L] < sf_lost``). Those are the positions worth
re-seeding: the net faced the decision and got the value wrong. This module
turns a finished game's per-ply records into seed lines carrying real LC0
history (``<start_fen> | <uci moves>``, the selfplay/opening.py format), tagged
``severe`` for the high-confidence band that can be auto-fed vs the broader band
that is only collected for later threshold tuning.

Detection is a pure function (``harvest_from_records``); ``run_harvest`` is the
fail-safe finalize-side wrapper that reconstructs the pre-move boards and
appends records to a file. It never raises into finalize — a harvest bug must
not cost a game's training data.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import chess
import numpy as np

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HarvestConfig:
    net_ok: float = 0.2       # capture band: net thinks fine above this
    sf_lost: float = -0.5     # capture band: SF says lost below this
    severe_net_ok: float = 0.5    # auto-feed band (net thinks it is winning)
    severe_sf_lost: float = -0.5  # auto-feed band (SF says clearly lost)
    history_plies: int = 8    # preceding moves stored (LC0 uses 8 history steps)


@dataclass(frozen=True)
class HarvestedSeed:
    line: str        # '<start_fen> | <uci ...>' or a bare FEN (seed_board_from_line grammar)
    net_q: float
    sf_q: float
    severe: bool


def _q(wdl: np.ndarray) -> float:
    return float(wdl[0] - wdl[2])


def pre_move_boards(
    starting_board: chess.Board,
    final_move_stack: list[chess.Move],
    record_ply_indices: list[int],
    *,
    opening_len: int,
) -> list[chess.Board | None]:
    """Reconstruct each record's PRE-move board (position the net faced), with
    real history in its move_stack.

    Mirrors the syzygy-rescore walk in finalize.py: ``record_ply_index`` equals
    ``len(board_before.move_stack)`` (network_turn.py), so walking the final
    move stack and matching that count aligns each record to the position just
    before its move. ``opening_len`` skips the opening plies already present in
    ``starting_board`` on the Python play path (0 on the C-ply path)."""
    at_ply: dict[int, int] = {}
    for t, ply in enumerate(record_ply_indices):
        at_ply.setdefault(int(ply), t)  # first record at a ply wins (forced-ply dups)
    boards: list[chess.Board | None] = [None] * len(record_ply_indices)
    rb = starting_board.copy()
    for mv in final_move_stack[opening_len:]:
        t = at_ply.get(len(rb.move_stack))
        if t is not None:
            boards[t] = rb.copy()
        rb.push(mv)
    return boards


def seed_line_from_board(board: chess.Board, history_plies: int) -> str:
    """'<start_fen> | <last N uci>' so the terminal is this position with real
    history; a bare FEN when no history is available (near the game start)."""
    stack = list(board.move_stack)
    n = min(int(history_plies), len(stack))
    if n <= 0:
        return board.fen()
    moves = " ".join(m.uci() for m in stack[-n:])
    tmp = board.copy()
    for _ in range(n):
        tmp.pop()
    return f"{tmp.fen()} | {moves}"


def harvest_from_records(
    evals: Sequence[tuple[bool, np.ndarray | None, np.ndarray | None]],
    boards: Sequence[chess.Board | None],
    *,
    cfg: HarvestConfig,
) -> list[HarvestedSeed]:
    """Pure detection: per record ``(has_policy, search_wdl, sf_wdl)`` aligned
    with its pre-move ``board``, emit a HarvestedSeed for each value-blind full
    ply. Value-only / label-less / unreconstructed records are skipped."""
    out: list[HarvestedSeed] = []
    for (has_policy, search_wdl, sf_wdl), board in zip(evals, boards, strict=True):
        if not has_policy or board is None or search_wdl is None or sf_wdl is None:
            continue
        net_q, sf_q = _q(search_wdl), _q(sf_wdl)
        if not (net_q > cfg.net_ok and sf_q < cfg.sf_lost):
            continue
        severe = net_q >= cfg.severe_net_ok and sf_q <= cfg.severe_sf_lost
        out.append(HarvestedSeed(
            line=seed_line_from_board(board, cfg.history_plies),
            net_q=net_q, sf_q=sf_q, severe=severe,
        ))
    return out


def format_record(seed: HarvestedSeed, *, game_id: str) -> str:
    """One seed-file line: the seed grammar + an inline provenance comment
    (stripped by the loader). ``sev=1`` marks the auto-feed band."""
    return (f"{seed.line}  # nq={seed.net_q:.2f} sq={seed.sf_q:.2f} "
            f"sev={int(seed.severe)} game={game_id}")


def run_harvest(
    starting_board: chess.Board | None,
    final_board: chess.Board,
    records: list,
    *,
    has_c_ply: bool,
    game_id: str,
    out_path: str,
    cfg: HarvestConfig,
) -> int:
    """Finalize-side wrapper: reconstruct boards, detect blind-spots, append
    records to ``out_path``. Returns the count written. NEVER raises into
    finalize — logs and returns 0 on any error."""
    try:
        if not out_path or starting_board is None:
            return 0
        opening_len = 0 if has_c_ply else len(starting_board.move_stack)
        ply_indices = [int(rec.ply_index) for rec in records]
        boards = pre_move_boards(
            starting_board, list(final_board.move_stack), ply_indices, opening_len=opening_len,
        )
        evals = [
            (bool(rec.has_policy), getattr(rec, "search_wdl_est", None), getattr(rec, "sf_wdl", None))
            for rec in records
        ]
        seeds = harvest_from_records(evals, boards, cfg=cfg)
        if not seeds:
            return 0
        with open(out_path, "a", encoding="utf-8") as fh:
            for s in seeds:
                fh.write(format_record(s, game_id=game_id) + "\n")
        return len(seeds)
    except Exception:
        # Harvesting is a side output — it must never break finalize or cost a
        # game's training data; log and swallow.
        _log.exception("blind-spot harvest failed for game %s", game_id)
        return 0
