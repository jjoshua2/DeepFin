#!/usr/bin/env python3
"""Mine blind-spot seeds (with real history) from match-loss PGNs.

Grows the blind-spot seed set (data/blindspot_fens_v1.txt and successors) from
DeepFin's on-board losses instead of a frozen hand-curated list. For each lost
game it finds the FIRST decisive collapse — the first DeepFin move that takes
the deep-Stockfish eval (DeepFin POV) from not-yet-lost across ``--collapse-cp``
(default -150) — exactly the panel definition (scripts/blindspot_panel.py) — and
emits a seed record carrying the ~8 preceding moves so the position trains with
real LC0 history rather than repeat-filled planes (seed_board_from_line, PR
feat/seed-history).

Output line format (one per mined loss, up to two when --moves-csv is given —
see the mismatch criterion below), consumed by selfplay/opening.py:
    <start_fen> | <uci ...>   # src=<pgn> round=<r> ply=<j>
where the terminal position (after replaying the moves) is the blind-spot the
net faces, and start_fen is ``--history-plies`` plies earlier.

Curation matches the existing asset: forced / claim-draw-terminal seeds are
dropped (via _fen_reject_reason), time-forfeit games are skipped, positions in a
``--holdout`` panel are excluded (keeps panel v1 a pure generalization yardstick),
and positions already in ``--existing`` seed files or mined earlier this run are
deduplicated by position identity (EPD).

Deep-SF annotation is mandatory (match PGNs carry no eval). Example:
    PYTHONPATH=. python3 scripts/mine_blindspot_seeds.py \
        --pgn 'runs/matches/*.pgn' --deepfin-name-contains deepfin \
        --sf-path /usr/local/bin/stockfish --sf-nodes 300000 \
        --holdout data/blindspot_panel_v1.jsonl \
        --existing data/blindspot_fens_v1.txt \
        --out data/blindspot_fens_v2_mined.txt

Optional SECOND criterion — value-head miscalibration, not just move quality:
``--moves-csv`` points at the match harness's own per-move log (game, ply,
engine, score_cp_white, ... — scripts/match_vs_uci.py's --move-log-out),
which already carries DeepFin's own self-reported eval at every move it
made, for free (no re-run). Given that, each DeepFin move ply also gets a
second candidate: the one where DeepFin's own eval diverges most from
deep-SF's eval of the SAME pre-move position, both converted to a common
expected-score scale (``--mismatch-score-gap`` min gap) —
"SF says decisively lost, we still think it's fine" is the sharpest case,
but it's symmetric in either direction. Mine_game emits both the first
collapse AND the worst mismatch when they're more than ``--min-ply-gap``
plies apart (distinct teaching moments); when they're close, only the
collapse is kept (redundant).

Free first SF-refute ply (default ON via ``--append-refute-ply``): after the
historical blunder, deep-SF's best reply is baked into the seed line so the
terminal is already one punish ply in (net STM). Dedup still keys the
pre-blunder blind-spot. Pairs with the live SF-refute channel (remaining
opponent plies at selfplay time). Re-mine from PGN to upgrade old lists —
existing bare-blindspot lines do not carry the blunder UCI, so they cannot
be upgraded in place.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from chess_anti_engine.selfplay.opening import _fen_reject_reason, seed_board_from_line
from chess_anti_engine.stockfish.wdl import cp_to_wdl

# Provenance marker when blunder+refute are baked into the body (two plies past
# the blind-spot). Used by seed_blindspot_key to recover the pre-blunder EPD for
# --existing / --holdout dedup against post-refute seed lists.
_REFUTE_IN_COMMENT_RE = re.compile(r"\brefute=([a-h][1-8][a-h][1-8][qrbn]?)\b", re.I)

# Deep-SF's cp and DeepFin's own reported cp are on INCOMPATIBLE scales: SF's
# is roughly-linear classical cp; DeepFin's own UCI "score cp" comes from
# uci/score.py's q_to_cp, a Leela-style `295*tan(1.5637*(2Q-1))` map that
# saturates toward +-inf near a certain win/loss. Subtracting them directly
# produces enormous same-sign "gaps" whenever DeepFin is confident in ANY
# direction (verified empirically: every raw-cp mismatch hit in an early
# version of this script was same-sign — a scale artifact, not a real
# disagreement). Both get converted to a common expected-score scale
# ([-1, 1], DeepFin POV, +1 = certain win) before comparing.
_SF_WDL_SLOPE = 0.0060           # matches configs/pbt2_small.yaml sf_wdl_cp_slope
_SF_WDL_DRAW_WIDTH_CP = 120.0    # matches configs/pbt2_small.yaml sf_wdl_cp_draw_width
_OWN_CP_A = 295.0                # matches chess_anti_engine/uci/score.py _CP_A
_OWN_CP_K = 1.5637541897         # matches chess_anti_engine/uci/score.py _CP_K


def _sf_expected_score(effective_cp: int) -> float:
    """Deep-SF's effective cp (plain cp, or mate_to_effective_cp's output for a
    forced mate) -> expected score in [-1, 1], via the project's own WDL
    calibration — the same scale used elsewhere to judge the value head."""
    w, _d, l = cp_to_wdl(float(effective_cp), None,
                          slope=_SF_WDL_SLOPE, draw_width_cp=_SF_WDL_DRAW_WIDTH_CP)
    return float(w - l)


def _own_expected_score(own_cp: int) -> float:
    """Invert q_to_cp back to expected score in [-1, 1] — the exact inverse of
    uci/score.py's `cp = 295 * tan(1.5637 * (2Q - 1))`, so `2Q - 1` (expected
    score) = `atan(cp / 295) / 1.5637`."""
    return math.atan(float(own_cp) / _OWN_CP_A) / _OWN_CP_K

# A deep-SF eval function: (fen, nodes) -> centipawns from the side-to-move POV
# (None if unavailable). Injected so the collapse logic is testable without SF.
SfEval = Callable[[str, int], "int | None"]

# Optional best-move lookup: (fen, nodes) -> UCI of SF's best move for the
# side-to-move, or None if unavailable. Used only when --append-refute-ply is
# on: after the historical blunder, SF's first refute is baked into the seed
# so the terminal position is already one punish ply in (net STM again).
# Production wires this to the same Stockfish search as SfEval (last-result
# cache) so the post-blunder eval already paid for the bestmove.
SfBestMove = Callable[[str, int], "str | None"]


@dataclass(frozen=True)
class Collapse:
    ply: int          # index j of the blunder move (0-based half-move)
    sf_before: int    # DeepFin-POV cp before the move (>= threshold)
    sf_after: int     # DeepFin-POV cp after the move (< threshold)


def find_first_collapse(
    evals: Iterable[tuple[int, int, int]], *, collapse_cp: int, min_drop: int,
) -> Collapse | None:
    """First (ply, sf_before, sf_after) that is a decisive collapse: the DeepFin
    move leaves the position clearly lost (sf_after < collapse_cp) AND worsened
    it by at least ``min_drop`` cp. Both scores are DeepFin POV; ``evals`` are in
    play order over DeepFin's moves only.

    NOT gated on being not-yet-lost before the move: the frozen panels include
    already-losing rows that get decisively worse (e.g. -1380 -> -2993), so an
    already-lost precondition would silently drop ~half of real collapse seeds
    (Codex review)."""
    for ply, sf_before, sf_after in evals:
        if sf_after < collapse_cp and (sf_before - sf_after) >= min_drop:
            return Collapse(ply=ply, sf_before=sf_before, sf_after=sf_after)
    return None


@dataclass(frozen=True)
class Mismatch:
    ply: int            # index j of the move (0-based half-move)
    sf_before: int       # raw deep-SF effective cp, DeepFin POV (provenance only)
    own_before: int      # raw DeepFin self-reported cp, DeepFin POV (provenance only)
    sf_score: float       # deep-SF expected score, DeepFin POV, [-1, 1]
    own_score: float      # DeepFin's own expected score, [-1, 1]
    gap: float             # abs(sf_score - own_score), in [0, 2]


def find_worst_mismatch(
    evals: Iterable[tuple[int, int, int]],
    own_evals: dict[int, int],
    *, mismatch_score_gap: float,
) -> Mismatch | None:
    """Largest expected-score disagreement between deep-SF and DeepFin's own
    eval at the same decision point, among DeepFin's moves. Value-head
    miscalibration, not move quality: SF and DeepFin can agree the position
    is bad and still disagree on a blunder-free line (no seed there); this
    flags where they read the SAME position as fundamentally different games
    — "SF says decisively lost, we still think it's fine" (or the reverse).

    Compared on expected score ([-1, 1], see _sf_expected_score /
    _own_expected_score), NOT raw cp — the two engines' cp scales are
    incompatible (deep-SF ~linear classical cp; DeepFin's own UCI score is a
    saturating tan() map), so a raw cp gap is dominated by scale, not
    disagreement (every raw-cp "mismatch" found in an early version of this
    function was same-sign — confidently-agreeing positions that only looked
    like disagreements because of the scale mismatch).

    ``evals`` reuses find_first_collapse's (ply, sf_before, sf_after) stream —
    only sf_before is used here, sf_after is irrelevant to this criterion.
    ``own_evals`` maps ply -> DeepFin's own reported cp (already DeepFin POV).
    """
    best: Mismatch | None = None
    for ply, sf_before, _sf_after in evals:
        own = own_evals.get(ply)
        if own is None:
            continue
        sf_score = _sf_expected_score(sf_before)
        own_score = _own_expected_score(own)
        gap = abs(sf_score - own_score)
        if gap >= mismatch_score_gap and (best is None or gap > best.gap):
            best = Mismatch(ply=ply, sf_before=sf_before, own_before=own,
                             sf_score=sf_score, own_score=own_score, gap=gap)
    return best


def load_own_evals(csv_paths: list[str]) -> dict[tuple[int, int], int]:
    """Parse match_vs_uci.py --move-log-out CSV(s) into {(round, ply): cp}.

    score_cp_white is each engine's own self-reported eval right after IT
    moved, always normalized to White's POV in the log — callers convert to
    DeepFin's POV themselves (every row's score gets stored regardless of
    which engine played it; DeepFin-ply filtering happens at lookup time in
    mine_game, which only ever queries plies it already knows are DeepFin's).

    Keyed by (round, ply) only, NOT per-source-file: one mining run is
    expected to cover a single match (one PGN + its one move log), where
    round numbers are already unique. Multiple unrelated matches sharing
    --moves-csv would collide on round number — not handled, not needed yet.
    """
    out: dict[tuple[int, int], int] = {}
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                cp = row.get("score_cp_white", "")
                if not cp:
                    continue
                try:
                    out[(int(row["game"]), int(row["ply"]))] = int(cp)
                except (KeyError, ValueError):
                    continue
    return out


def build_seed_record(
    fens: list[str],
    ucis: list[str],
    j: int,
    *,
    history_plies: int,
    provenance: str,
    refute_uci: str | None = None,
) -> str:
    """Seed line for a blunder at half-move ``j``.

    ``fens[i]`` is the FEN before move ``i`` (fens[j] = the blind-spot the net
    faces); ``ucis[i]`` is move i. Bare records replay up to ``history_plies``
    moves from ``fens[j-h]`` up to the blind-spot (terminal = fens[j]).

    When ``refute_uci`` is set, the historical blunder (``ucis[j]``) and deep-SF's
    best reply are also appended, so the terminal is AFTER one punish ply (net
    STM again). Preceding history is clamped to ``history_plies - 2`` so the
    total replay length stays ≤ ``history_plies`` (LC0 keeps 8 history steps;
    without the clamp, default 8 + blunder + refute would drop two approach
    plies from the planes). Dedup still keys on fens[j] — see ``mine_game`` /
    ``seed_blindspot_key``. Free offline substitute for the first live
    SF-refute opponent ply (live channel then covers the rest).
    """
    if refute_uci is not None:
        if j >= len(ucis):
            raise ValueError(f"blunder ply j={j} out of range for ucis (len={len(ucis)})")
        # Reserve 2 of history_plies for blunder+refute so LC0's window still
        # covers the intended depth into the approach.
        h = min(max(0, history_plies - 2), j)
        start_fen = fens[j - h]
        moves = list(ucis[j - h:j])
        moves.extend([ucis[j], refute_uci])
    else:
        h = min(history_plies, j)
        start_fen = fens[j - h]
        moves = list(ucis[j - h:j])
    body = start_fen if not moves else f"{start_fen} | {' '.join(moves)}"
    return f"{body}  # {provenance}"


def position_key(fen: str) -> str:
    """Position identity (placement/turn/castling/ep), ignoring move counters —
    for holdout and dedup by 'same blind-spot'."""
    return chess.Board(fen).epd()


def seed_blindspot_key(raw_line: str) -> str | None:
    """Dedup key for one seed line: the pre-blunder blind-spot EPD.

    Mine ``position_key`` is always the blind-spot (fens[ply]), even when the
    seed terminal is post-refute. ``load_existing_keys`` / seed-file holdouts
    must use the same key or incremental re-mines against post-refute lists
    re-emit the same holes.

    Recovery order:
    1. Comment carries ``refute=<uci>`` → terminal is two plies past the
       blind-spot; pop those two moves and key the resulting board.
    2. Otherwise terminal is the blind-spot (legacy bare seeds / panel lines).
    Returns None if the line is unparseable.
    """
    body, _, comment = raw_line.partition("#")
    body = body.strip()
    if not body or body.startswith("{"):
        return None
    try:
        board = seed_board_from_line(body)
    except ValueError:
        return None
    if _REFUTE_IN_COMMENT_RE.search(comment) and len(board.move_stack) >= 2:
        board.pop()
        board.pop()
    return position_key(board.fen())


def deepfin_color(game: chess.pgn.Game, name_needle: str) -> chess.Color | None:
    """Which side is DeepFin, by a case-insensitive substring of the player tag;
    None if neither or both sides match (ambiguous)."""
    needle = name_needle.lower()
    white = needle in game.headers.get("White", "").lower()
    black = needle in game.headers.get("Black", "").lower()
    if white == black:  # neither or both -> ambiguous
        return None
    return chess.WHITE if white else chess.BLACK


def _is_time_forfeit(game: chess.pgn.Game) -> bool:
    term = game.headers.get("Termination", "").lower()
    return "time" in term or "forfeit" in term


def _deepfin_lost(game: chess.pgn.Game, color: chess.Color) -> bool:
    result = game.headers.get("Result", "*")
    return result == ("0-1" if color == chess.WHITE else "1-0")


def _resolve_refute_uci(
    fens: list[str],
    ucis: list[str],
    ply: int,
    *,
    sf_bestmove: SfBestMove,
    sf_nodes: int,
) -> str | None:
    """Deep-SF best reply after the historical blunder at ``ply``, or None.

    Post-blunder FEN = position after ``ucis[ply]`` from ``fens[ply]``. Returns
    None when the blunder ends the game, SF has no bestmove, or the move is
    illegal in that position (mate / stalemate / bad UCI).
    """
    if ply < 0 or ply >= len(ucis) or ply >= len(fens):
        return None
    try:
        board = chess.Board(fens[ply])
        blunder = chess.Move.from_uci(ucis[ply])
    except ValueError:
        return None
    if blunder not in board.legal_moves:
        return None
    board.push(blunder)
    if board.is_game_over(claim_draw=True):
        return None
    best = sf_bestmove(board.fen(), sf_nodes)
    if not best or best == "0000":
        return None
    try:
        refute = chess.Move.from_uci(best)
    except ValueError:
        return None
    if refute not in board.legal_moves:
        return None
    return best


def mine_game(
    game: chess.pgn.Game, *, src: str = "?", name_needle: str, sf_eval: SfEval,
    sf_nodes: int, collapse_cp: int, history_plies: int, min_drop: int = 150,
    own_evals: dict[tuple[int, int], int] | None = None,
    mismatch_score_gap: float = 0.5, min_ply_gap: int = 8,
    append_refute_ply: bool = False,
    sf_bestmove: SfBestMove | None = None,
) -> list[tuple[str, str]]:
    """Mine one game -> up to 2 (seed_record, position_key) pairs: the first
    decisive collapse (move-quality) and, when ``own_evals`` is given, the
    worst DeepFin-vs-deep-SF value mismatch (calibration) — emitted only when
    it's more than ``min_ply_gap`` plies from the collapse (otherwise it's the
    same teaching moment, keep just the collapse). Empty list when the game
    is unusable (not a DeepFin loss / ambiguous side / time forfeit) or
    neither criterion finds anything worth keeping.

    When ``append_refute_ply`` is True and ``sf_bestmove`` is provided, each
    kept seed appends the historical blunder + deep-SF's first best reply so
    the terminal is net STM after one punish ply. Dedup ``position_key`` is
    still the pre-blunder blind-spot (fens[ply]). If the refute cannot be
    resolved or the post-refute terminal is rejected by the loader, falls
    back to the pre-blunder seed (still useful for the live SF-refute channel).
    """
    color = deepfin_color(game, name_needle)
    if color is None or _is_time_forfeit(game) or not _deepfin_lost(game, color):
        return []
    try:
        round_num = int(game.headers.get("Round", "0"))
    except ValueError:
        round_num = 0

    board = game.board()
    fens: list[str] = []
    ucis: list[str] = []
    df_evals: list[tuple[int, int, int]] = []
    own_by_ply: dict[int, int] = {}
    for j, move in enumerate(game.mainline_moves()):
        fen_before = board.fen()
        fens.append(fen_before)
        ucis.append(move.uci())
        if board.turn == color:
            before = sf_eval(fen_before, sf_nodes)
            board.push(move)
            after_stm = sf_eval(board.fen(), sf_nodes)
            if before is not None and after_stm is not None:
                # after_stm is opponent-POV (opponent to move) -> negate.
                df_evals.append((j, before, -after_stm))
            if own_evals is not None:
                own_white = own_evals.get((round_num, j + 1))
                if own_white is not None:
                    # score_cp_white is always White-POV; negate for Black.
                    own_by_ply[j] = own_white if color == chess.WHITE else -own_white
        else:
            board.push(move)

    collapse = find_first_collapse(df_evals, collapse_cp=collapse_cp, min_drop=min_drop)
    mismatch = (
        find_worst_mismatch(df_evals, own_by_ply, mismatch_score_gap=mismatch_score_gap)
        if own_evals is not None else None
    )
    if mismatch is not None and collapse is not None and abs(mismatch.ply - collapse.ply) < min_ply_gap:
        mismatch = None  # same teaching moment as the collapse — don't double-mine it

    candidates: list[tuple[int, str]] = []
    if collapse is not None:
        candidates.append((collapse.ply, f"src={Path(src).name} round={round_num} "
                            f"ply={collapse.ply} sf={collapse.sf_before}->{collapse.sf_after}"))
    if mismatch is not None:
        candidates.append((mismatch.ply, f"src={Path(src).name} round={round_num} "
                            f"ply={mismatch.ply} mismatch sf_score={mismatch.sf_score:.2f} "
                            f"own_score={mismatch.own_score:.2f} gap={mismatch.gap:.2f} "
                            f"(raw sf={mismatch.sf_before} own={mismatch.own_before})"))

    want_refute = bool(append_refute_ply and sf_bestmove is not None)
    out: list[tuple[str, str]] = []
    for ply, provenance in candidates:
        record: str | None = None
        if want_refute:
            assert sf_bestmove is not None
            refute_uci = _resolve_refute_uci(
                fens, ucis, ply, sf_bestmove=sf_bestmove, sf_nodes=sf_nodes,
            )
            if refute_uci is not None:
                candidate = build_seed_record(
                    fens, ucis, ply, history_plies=history_plies,
                    provenance=f"{provenance} refute={refute_uci}",
                    refute_uci=refute_uci,
                )
                # Post-refute terminal unusable (mate/forced) → fall back below.
                if _fen_reject_reason(candidate.split("#", 1)[0].strip()) is None:
                    record = candidate
        if record is None:
            record = build_seed_record(
                fens, ucis, ply, history_plies=history_plies, provenance=provenance,
            )
            if _fen_reject_reason(record.split("#", 1)[0].strip()) is not None:
                continue
        # Dedup key is always the pre-blunder blind-spot, even when the seed
        # terminal is post-refute (same teaching hole, not a new position).
        out.append((record, position_key(fens[ply])))
    return out


def load_holdout_keys(paths: list[str]) -> set[str]:
    """Panel / seed position keys to EXCLUDE (keep panel v1 a generalization yardstick).

    Jsonl panel rows key on ``fen_before`` (the blind-spot). Seed-file lines key
    via ``seed_blindspot_key`` so post-refute terminals still exclude the
    pre-blunder hole (not the post-punish position).
    """
    keys: set[str] = set()
    for p in paths:
        for raw in Path(p).read_text(encoding="utf-8-sig").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("{"):
                fen = json.loads(stripped).get("fen_before")
                if fen:
                    keys.add(position_key(fen))
                continue
            key = seed_blindspot_key(raw)
            if key is not None:
                keys.add(key)
    return keys


def load_existing_keys(paths: list[str]) -> set[str]:
    """Blind-spot keys already seeded (dedup for incremental re-mine).

    Reads raw seed lines (comments preserved) so ``refute=`` post-refute seeds
    contribute their pre-blunder EPD — matching ``mine_game``'s ``position_key``.
    Does not use ``_load_fen_list`` (which strips comments and would leave only
    the post-refute terminal).
    """
    keys: set[str] = set()
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            continue
        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key = seed_blindspot_key(raw)
            if key is not None:
                keys.add(key)
    return keys


def _make_sf_eval(
    sf_path: str, syzygy_path: str | None,
) -> tuple[SfEval, SfBestMove, Callable[[], None]]:
    """Wire SfEval + SfBestMove to one Stockfish process with a last-result cache.

    One-slot cache: a bestmove lookup for the same (fen, nodes) as the most
    recent search is free. In ``mine_game`` the full mainline is scanned
    (every DeepFin before/after eval) *before* candidates resolve refute
    bestmoves, so the cache only hits when the kept ply's post-blunder FEN
    was the last search of the scan (e.g. late-game collapse). Otherwise
    expect a miss and one extra search per kept seed — fine at mine scale.
    """
    from chess_anti_engine.stockfish.uci import StockfishResult, StockfishUCI
    from chess_anti_engine.stockfish.wdl import mate_to_effective_cp

    sf = StockfishUCI(path=sf_path)
    # (fen, nodes, result) of the most recent search — opportunistic reuse only.
    last: list[tuple[str, int, StockfishResult] | None] = [None]

    def _search(fen: str, nodes: int) -> StockfishResult:
        hit = last[0]
        if hit is not None and hit[0] == fen and hit[1] == nodes:
            return hit[2]
        res = sf.search(fen, nodes=nodes, syzygy_path=syzygy_path)
        last[0] = (fen, nodes, res)
        return res

    def sf_eval(fen: str, nodes: int) -> int | None:
        res = _search(fen, nodes)
        if res.cp is not None:
            return res.cp
        # A forced mate is a high-value collapse, not a drop-out: map it to a
        # large signed cp so the mate side is scored, not skipped (Codex review).
        return round(mate_to_effective_cp(res.mate)) if res.mate is not None else None

    def sf_bestmove(fen: str, nodes: int) -> str | None:
        res = _search(fen, nodes)
        bm = res.bestmove_uci
        if not bm or bm == "0000":
            return None
        return bm

    return sf_eval, sf_bestmove, sf.close


def _iter_games(pgn_path: Path) -> Iterable[chess.pgn.Game]:
    with open(pgn_path, encoding="utf-8") as fh:
        while (game := chess.pgn.read_game(fh)) is not None:
            yield game


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn", nargs="+", required=True,
                    help="PGN file(s) or glob(s) of match games (runs/matches/*.pgn)")
    ap.add_argument("--deepfin-name-contains", default="deepfin",
                    help="case-insensitive substring identifying DeepFin's player tag")
    ap.add_argument("--sf-path", required=True, help="Stockfish binary")
    ap.add_argument("--sf-nodes", type=int, default=300_000,
                    help="deep-SF node budget per eval (panel used 300k)")
    ap.add_argument("--collapse-cp", type=int, default=-150,
                    help="DeepFin-POV cp the seed is below AFTER the blunder (panel: -150)")
    ap.add_argument("--min-drop", type=int, default=150,
                    help="min cp worsening the blunder must cause (filters slips; "
                         "includes already-losing positions that get decisively worse)")
    ap.add_argument("--syzygy-path", default=None,
                    help="Stockfish SyzygyPath for <=6-man endgame rows (production uses "
                         "stockfish_syzygy_path); without it, endgames get a heuristic score")
    ap.add_argument("--history-plies", type=int, default=8,
                    help="max UCI moves replayed per seed (LC0 keeps 8 history steps). "
                         "With --append-refute-ply, 2 slots are reserved for blunder+"
                         "refute so approach history is history_plies-2.")
    ap.add_argument("--moves-csv", nargs="*", default=[],
                    help="match_vs_uci.py --move-log-out CSV(s) — enables the value-"
                         "mismatch second criterion (one match's PGN+CSV per run; "
                         "round numbers must not collide across files)")
    ap.add_argument("--mismatch-score-gap", type=float, default=0.5,
                    help="min |deep-SF expected score - DeepFin's own expected score| "
                         "([-1,1] scale, NOT raw cp — see module docstring for why) at "
                         "the same decision point to count as a value-head mismatch seed")
    ap.add_argument("--min-ply-gap", type=int, default=8,
                    help="collapse and mismatch closer than this (plies) are treated "
                         "as the same teaching moment — only the collapse is kept")
    ap.add_argument("--holdout", nargs="*", default=[],
                    help="panel jsonl / seed files whose positions to EXCLUDE")
    ap.add_argument("--existing", nargs="*", default=[],
                    help="seed files to dedup against (won't re-mine known positions)")
    ap.add_argument("--out", type=Path, required=True, help="seed file to write/append")
    ap.add_argument("--append", action="store_true", help="append to --out instead of overwrite")
    ap.add_argument("--max-games", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--append-refute-ply",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bake historical blunder + deep-SF best reply into each seed so "
             "the terminal is net STM after one punish ply (default: on). "
             "Disable with --no-append-refute-ply for bare blind-spot terminals. "
             "Re-mine from PGN to upgrade old lists (cannot expand in place).",
    )
    args = ap.parse_args()

    pgns = sorted({Path(p) for pat in args.pgn for p in glob.glob(pat)})
    if not pgns:
        sys.exit(f"no PGN files matched {args.pgn}")
    exclude = load_holdout_keys(args.holdout) | load_existing_keys(args.existing)
    print(f"[mine] {len(pgns)} PGN files; {len(exclude)} holdout+existing positions excluded",
          flush=True)

    own_evals = load_own_evals(args.moves_csv) if args.moves_csv else None
    if own_evals is not None:
        print(f"[mine] {len(own_evals)} own-eval rows loaded from {len(args.moves_csv)} "
              f"move-log CSV(s); mismatch_score_gap={args.mismatch_score_gap} "
              f"min_ply_gap={args.min_ply_gap}",
              flush=True)
    if args.append_refute_ply:
        print("[mine] append-refute-ply ON — terminal = after blunder + deep-SF best reply",
              flush=True)

    sf_eval, sf_bestmove, sf_close = _make_sf_eval(args.sf_path, args.syzygy_path)
    records: list[str] = []
    seen: set[str] = set()
    n_games = 0
    try:
        for pgn in pgns:
            for game in _iter_games(pgn):
                n_games += 1
                if args.max_games and n_games > args.max_games:
                    break
                mined = mine_game(
                    game, src=str(pgn), name_needle=args.deepfin_name_contains,
                    sf_eval=sf_eval, sf_nodes=args.sf_nodes,
                    collapse_cp=args.collapse_cp, history_plies=args.history_plies,
                    min_drop=args.min_drop, own_evals=own_evals,
                    mismatch_score_gap=args.mismatch_score_gap, min_ply_gap=args.min_ply_gap,
                    append_refute_ply=bool(args.append_refute_ply),
                    sf_bestmove=sf_bestmove if args.append_refute_ply else None,
                )
                for record, key in mined:
                    if key in exclude or key in seen:
                        continue
                    seen.add(key)
                    records.append(record)
                    print(f"[mine] {record}", flush=True)
    finally:
        sf_close()

    mode = "a" if args.append else "w"
    with open(args.out, mode, encoding="utf-8") as fh:
        if mode == "w":
            header = (
                "# Mined blind-spot seeds (start_fen | moves; real LC0 history).\n"
                "# scripts/mine_blindspot_seeds.py — first decisive collapse per loss"
                + (" + worst value-head mismatch per loss" if own_evals is not None else "")
                + ("; blunder+deep-SF refute ply baked in (terminal post-punish).\n"
                   if args.append_refute_ply else ".\n")
            )
            fh.write(header)
        for r in records:
            fh.write(r + "\n")
    print(f"[mine] wrote {len(records)} new seeds ({n_games} games scanned) -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
