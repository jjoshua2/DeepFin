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

Output line format (one per mined loss), consumed by selfplay/opening.py:
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
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from chess_anti_engine.selfplay.opening import _fen_reject_reason, seed_board_from_line

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
) -> tuple[str, str] | None:
    """Mine one game -> (seed_record, position_key) or None.

    None when the game is unusable (not a DeepFin loss / ambiguous side / time
    forfeit / no collapse found / mate-scored / the collapse seed is
    forced-or-terminal). Only the FIRST collapse is emitted.
    """
    color = deepfin_color(game, name_needle)
    if color is None or _is_time_forfeit(game) or not _deepfin_lost(game, color):
        return None

    board = game.board()
    fens: list[str] = []
    ucis: list[str] = []
    df_evals: list[tuple[int, int, int]] = []
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
        else:
            board.push(move)

    collapse = find_first_collapse(df_evals, collapse_cp=collapse_cp, min_drop=min_drop)
    if collapse is None:
        return None

    provenance = (
        f"src={Path(src).name} round={game.headers.get('Round', '?')} "
        f"ply={collapse.ply} sf={collapse.sf_before}->{collapse.sf_after}"
    )
    record = build_seed_record(
        fens, ucis, collapse.ply, history_plies=history_plies, provenance=provenance,
    )
    # The seed must survive the loader's own curation (forced / terminal).
    body = record.split("#", 1)[0].strip()
    if _fen_reject_reason(body) is not None:
        return None
    return record, position_key(fens[collapse.ply])


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
                    min_drop=args.min_drop,
                )
                if mined is None:
                    continue
                record, key = mined
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
                     "# scripts/mine_blindspot_seeds.py — first decisive collapse per loss.\n")
        for r in records:
            fh.write(r + "\n")
    print(f"[mine] wrote {len(records)} new seeds ({n_games} games scanned) -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
