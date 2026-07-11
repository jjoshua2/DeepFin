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
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from chess_anti_engine.selfplay.opening import _fen_reject_reason, seed_board_from_line
from chess_anti_engine.stockfish.wdl import cp_to_wdl

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
    fens: list[str], ucis: list[str], j: int, *, history_plies: int, provenance: str,
) -> str:
    """Seed line for a blunder at half-move ``j``.

    ``fens[i]`` is the FEN before move ``i`` (fens[j] = the blind-spot the net
    faces); ``ucis[i]`` is move i. The record replays ``history_plies`` moves
    (fewer near the game start) from ``fens[j-h]`` up to the blind-spot, so its
    terminal is fens[j] carrying real history.
    """
    h = min(history_plies, j)
    start_fen = fens[j - h]
    moves = ucis[j - h:j]
    body = start_fen if not moves else f"{start_fen} | {' '.join(moves)}"
    return f"{body}  # {provenance}"


def position_key(fen: str) -> str:
    """Position identity (placement/turn/castling/ep), ignoring move counters —
    for holdout and dedup by 'same blind-spot'."""
    return chess.Board(fen).epd()


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


def mine_game(
    game: chess.pgn.Game, *, src: str = "?", name_needle: str, sf_eval: SfEval,
    sf_nodes: int, collapse_cp: int, history_plies: int, min_drop: int = 150,
    own_evals: dict[tuple[int, int], int] | None = None,
    mismatch_score_gap: float = 0.5, min_ply_gap: int = 8,
) -> list[tuple[str, str]]:
    """Mine one game -> up to 2 (seed_record, position_key) pairs: the first
    decisive collapse (move-quality) and, when ``own_evals`` is given, the
    worst DeepFin-vs-deep-SF value mismatch (calibration) — emitted only when
    it's more than ``min_ply_gap`` plies from the collapse (otherwise it's the
    same teaching moment, keep just the collapse). Empty list when the game
    is unusable (not a DeepFin loss / ambiguous side / time forfeit) or
    neither criterion finds anything worth keeping.
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

    out: list[tuple[str, str]] = []
    for ply, provenance in candidates:
        record = build_seed_record(
            fens, ucis, ply, history_plies=history_plies, provenance=provenance,
        )
        # The seed must survive the loader's own curation (forced / terminal).
        body = record.split("#", 1)[0].strip()
        if _fen_reject_reason(body) is not None:
            continue
        out.append((record, position_key(fens[ply])))
    return out


def load_holdout_keys(paths: list[str]) -> set[str]:
    """Panel position keys to EXCLUDE (keep panel v1 a generalization yardstick).
    Reads jsonl rows with a fen_before (panel) or a plain FEN/seed line."""
    keys: set[str] = set()
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8-sig").splitlines():
            line = line.split("#", 1)[0].strip() if not line.lstrip().startswith("{") else line
            if not line:
                continue
            if line.startswith("{"):
                fen = json.loads(line).get("fen_before")
                if fen:
                    keys.add(position_key(fen))
            else:
                keys.add(position_key(seed_board_from_line(line).fen()))
    return keys


def load_existing_keys(paths: list[str]) -> set[str]:
    """Terminal position keys already seeded (dedup) — reuses the seed loader."""
    from chess_anti_engine.selfplay.opening import _load_fen_list

    keys: set[str] = set()
    for p in paths:
        try:
            for line in _load_fen_list(p):
                keys.add(position_key(seed_board_from_line(line).fen()))
        except (ValueError, FileNotFoundError):
            continue
    return keys


def _make_sf_eval(sf_path: str, syzygy_path: str | None) -> tuple[SfEval, Callable[[], None]]:
    from chess_anti_engine.stockfish.uci import StockfishUCI
    from chess_anti_engine.stockfish.wdl import mate_to_effective_cp

    sf = StockfishUCI(path=sf_path)

    def sf_eval(fen: str, nodes: int) -> int | None:
        res = sf.search(fen, nodes=nodes, syzygy_path=syzygy_path)
        if res.cp is not None:
            return res.cp
        # A forced mate is a high-value collapse, not a drop-out: map it to a
        # large signed cp so the mate side is scored, not skipped (Codex review).
        return round(mate_to_effective_cp(res.mate)) if res.mate is not None else None

    return sf_eval, sf.close


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
                    help="preceding moves stored per seed (LC0 uses 8 history steps)")
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

    sf_eval, sf_close = _make_sf_eval(args.sf_path, args.syzygy_path)
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
            fh.write("# Mined blind-spot seeds (start_fen | moves; real LC0 history).\n"
                     "# scripts/mine_blindspot_seeds.py — first decisive collapse per loss"
                     + (" + worst value-head mismatch per loss.\n" if own_evals is not None else ".\n"))
        for r in records:
            fh.write(r + "\n")
    print(f"[mine] wrote {len(records)} new seeds ({n_games} games scanned) -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
