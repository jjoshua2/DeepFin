"""Blind-spot seed miner core (scripts/mine_blindspot_seeds.py)."""
from __future__ import annotations

import random

import chess
import chess.pgn
import pytest

from chess_anti_engine.selfplay.opening import seed_board_from_line
from scripts.mine_blindspot_seeds import (
    _resolve_refute_uci,
    build_seed_record,
    deepfin_color,
    find_first_collapse,
    load_existing_keys,
    mine_game,
    position_key,
    seed_blindspot_key,
)

_LINE = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5", "d7d5"]


def _fen_after(ucis: list[str], n: int) -> str:
    b = chess.Board()
    for u in ucis[:n]:
        b.push(chess.Move.from_uci(u))
    return b.fen()


def _game(ucis: list[str], *, white="deepfin_v1", black="Cheese",
          result="0-1", termination="") -> chess.pgn.Game:
    b = chess.Board()
    for u in ucis:
        b.push(chess.Move.from_uci(u))
    g = chess.pgn.Game.from_board(b)
    g.headers.update({"White": white, "Black": black, "Result": result, "Round": "1"})
    if termination:
        g.headers["Termination"] = termination
    return g


# ── pure core ────────────────────────────────────────────────────────────────

def test_find_first_collapse_picks_first_decisive_drop() -> None:
    evals = [(0, 40, 20), (2, 30, -200), (4, 10, -500)]  # ply 2 first ends <-150 w/ big drop
    c = find_first_collapse(evals, collapse_cp=-150, min_drop=150)
    assert c is not None
    assert c.ply == 2
    assert c.sf_after == -200


def test_find_first_collapse_includes_already_losing_that_worsens() -> None:
    # Already losing (-1380) that gets decisively worse (-2993) IS a valid seed
    # (matches the frozen panels), unlike a not-yet-lost precondition.
    c = find_first_collapse([(6, -1380, -2993)], collapse_cp=-150, min_drop=150)
    assert c is not None
    assert c.ply == 6


def test_find_first_collapse_skips_small_slips() -> None:
    # Below threshold but only a tiny worsening -> not a decisive collapse.
    assert find_first_collapse([(2, -160, -190)], collapse_cp=-150, min_drop=150) is None


def test_find_first_collapse_none_when_not_lost_after() -> None:
    assert find_first_collapse([(0, 50, 30), (2, 220, -80)], collapse_cp=-150, min_drop=150) is None


def test_build_seed_record_clamps_history_near_start() -> None:
    fens = [_fen_after(_LINE, i) for i in range(len(_LINE) + 1)]
    ucis = list(_LINE)
    rec = build_seed_record(fens, ucis, 6, history_plies=8, provenance="x")
    body = rec.split("#", 1)[0].strip()
    board = seed_board_from_line(body)
    assert board.fen() == fens[6]           # terminal is the blind-spot
    assert len(board.move_stack) == 6       # clamped to j=6 (< 8)


def test_build_seed_record_limits_to_history_plies() -> None:
    fens = [_fen_after(_LINE, i) for i in range(len(_LINE) + 1)]
    rec = build_seed_record(fens, list(_LINE), 6, history_plies=3, provenance="x")
    board = seed_board_from_line(rec.split("#", 1)[0].strip())
    assert board.fen() == fens[6]
    assert len(board.move_stack) == 3       # only the last 3 preceding moves


def test_build_seed_record_appends_blunder_and_refute() -> None:
    """With refute_uci, terminal is after historical blunder + SF reply (net STM)."""
    fens = [_fen_after(_LINE, i) for i in range(len(_LINE) + 1)]
    # ply 6 = white's f3g5 blunder seat; black's historical reply is d7d5 (_LINE[7]).
    refute = _LINE[7]  # d7d5 — legal after f3g5
    rec = build_seed_record(
        fens, list(_LINE), 6, history_plies=3, provenance="x", refute_uci=refute,
    )
    body = rec.split("#", 1)[0].strip()
    board = seed_board_from_line(body)
    # Terminal = after blunder (f3g5) + refute (d7d5) = fens[8]
    assert board.fen() == fens[8]
    # history_plies=3 reserves 2 for blunder+refute → 1 preceding + 2 = 3 on stack
    assert len(board.move_stack) == 3
    assert board.turn == chess.WHITE  # net (white) STM after one punish


def test_build_seed_record_refute_clamps_total_to_history_plies() -> None:
    """history_plies=8 + refute at j>=8 yields exactly 8 UCIs (clamp is load-bearing).

    At j=6, min(8, j) == min(6, j) so the clamp is a no-op; use a longer line
    where unclamped would emit 10 UCIs (8 approach + blunder + refute).
    """
    # Italian quiet: j=10 is white b1d2; legal black reply d7d6 is long_line[11].
    long_line = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6",
        "d2d3", "h7h6", "c2c3", "a7a6", "b1d2", "d7d6",
    ]
    j = 10
    refute = long_line[11]
    fens = [_fen_after(long_line, i) for i in range(len(long_line) + 1)]
    rec = build_seed_record(
        fens, list(long_line), j, history_plies=8, provenance="x", refute_uci=refute,
    )
    body = rec.split("#", 1)[0].strip()
    board = seed_board_from_line(body)
    assert board.fen() == fens[j + 2]  # after blunder + refute
    moves = body.split("|", 1)[1].split()
    assert len(moves) == 8
    assert moves[-2:] == [long_line[j], refute]
    # Clamp starts at j-(8-2)=4, not j-8=2 (which unclamped would use).
    assert moves[0] == long_line[j - 6]
    assert len(board.move_stack) == 8


def test_resolve_refute_uci_returns_legal_best() -> None:
    fens = [_fen_after(_LINE, i) for i in range(len(_LINE) + 1)]
    # After white f3g5 (ply 6), black to move — fake SF says d7d5.
    def best(fen: str, _n: int) -> str | None:
        assert fen == fens[7]
        return "d7d5"

    assert _resolve_refute_uci(fens, list(_LINE), 6, sf_bestmove=best, sf_nodes=1) == "d7d5"


def test_resolve_refute_uci_none_when_illegal_or_missing() -> None:
    fens = [_fen_after(_LINE, i) for i in range(len(_LINE) + 1)]
    assert _resolve_refute_uci(
        fens, list(_LINE), 6, sf_bestmove=lambda _f, _n: "a2a3", sf_nodes=1,
    ) is None  # a2a3 not legal for black
    assert _resolve_refute_uci(
        fens, list(_LINE), 6, sf_bestmove=lambda _f, _n: None, sf_nodes=1,
    ) is None
    assert _resolve_refute_uci(
        fens, list(_LINE), 6, sf_bestmove=lambda _f, _n: "0000", sf_nodes=1,
    ) is None


def test_position_key_ignores_move_counters() -> None:
    a = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    b = a.replace(" 3 3", " 40 41")
    assert position_key(a) == position_key(b)


def test_deepfin_color_detection() -> None:
    assert deepfin_color(_game(_LINE, white="DeepFin_g5"), "deepfin") == chess.WHITE
    assert deepfin_color(_game(_LINE, white="Cheese", black="deepfin_v2"), "deepfin") == chess.BLACK
    assert deepfin_color(_game(_LINE, white="a", black="b"), "deepfin") is None  # neither
    assert deepfin_color(_game(_LINE, white="deepfin", black="deepfin"), "deepfin") is None  # both


# ── end-to-end mine_game with a scripted fake SF ─────────────────────────────

def _fake_sf(blunder_after_fen: str):
    # side-to-move-POV cp: +30 everywhere except the opponent is winning right
    # after DeepFin's blunder move (opponent to move) -> +400 -> DeepFin POV -400.
    def sf(fen: str, _nodes: int) -> int:
        return 400 if fen == blunder_after_fen else 30
    return sf


def _mine(game: chess.pgn.Game, sf) -> list[tuple[str, str]]:
    return mine_game(game, src="t.pgn", name_needle="deepfin", sf_eval=sf,
                     sf_nodes=1, collapse_cp=-150, history_plies=8)


def test_mine_game_emits_blindspot_with_history() -> None:
    game = _game(_LINE)                       # DeepFin=white, loses
    blunder_after = _fen_after(_LINE, 7)      # after white's move[6]=f3g5, black to move
    out = mine_game(game, name_needle="deepfin", sf_eval=_fake_sf(blunder_after),
                    src="runs/matches/test.pgn", sf_nodes=1, collapse_cp=-150, history_plies=8)
    assert len(out) == 1
    record, key = out[0]
    seed_fen = _fen_after(_LINE, 6)           # white to move — the blind-spot
    assert key == position_key(seed_fen)
    board = seed_board_from_line(record.split("#", 1)[0].strip())
    assert board.fen() == seed_fen
    assert len(board.move_stack) == 6
    assert "ply=6" in record


def test_mine_game_skips_non_loss_forfeit_ambiguous() -> None:
    sf = _fake_sf(_fen_after(_LINE, 7))
    assert _mine(_game(_LINE, result="1-0"), sf) == []            # DeepFin won
    assert _mine(_game(_LINE, termination="Time forfeit"), sf) == []
    assert _mine(_game(_LINE, white="a", black="b"), sf) == []    # ambiguous side


def test_mine_game_none_when_no_collapse() -> None:
    # A fake that never returns a losing eval -> no collapse.
    assert _mine(_game(_LINE), lambda _fen, _nodes: 30) == []


def test_mine_game_appends_refute_ply_when_enabled() -> None:
    """append_refute_ply bakes blunder+SF-best; dedup key stays pre-blunder FEN."""
    game = _game(_LINE)
    blunder_after = _fen_after(_LINE, 7)  # after white f3g5
    seed_fen = _fen_after(_LINE, 6)
    post_refute = _fen_after(_LINE, 8)    # after black d7d5

    def best(fen: str, _n: int) -> str | None:
        if fen == blunder_after:
            return "d7d5"
        return None

    out = mine_game(
        game, name_needle="deepfin", sf_eval=_fake_sf(blunder_after),
        src="t.pgn", sf_nodes=1, collapse_cp=-150, history_plies=8,
        append_refute_ply=True, sf_bestmove=best,
    )
    assert len(out) == 1
    record, key = out[0]
    assert key == position_key(seed_fen)          # dedup = original blind-spot
    body = record.split("#", 1)[0].strip()
    board = seed_board_from_line(body)
    assert board.fen() == post_refute             # terminal post-punish
    assert board.turn == chess.WHITE
    assert len(board.move_stack) <= 8             # history_plies budget (clamped)
    assert "refute=d7d5" in record
    assert "ply=6" in record
    assert seed_blindspot_key(record) == position_key(seed_fen)


def test_mine_game_refute_falls_back_when_bestmove_missing() -> None:
    """If SF bestmove is unavailable, keep the bare blind-spot seed."""
    game = _game(_LINE)
    blunder_after = _fen_after(_LINE, 7)
    seed_fen = _fen_after(_LINE, 6)
    out = mine_game(
        game, name_needle="deepfin", sf_eval=_fake_sf(blunder_after),
        src="t.pgn", sf_nodes=1, collapse_cp=-150, history_plies=8,
        append_refute_ply=True, sf_bestmove=lambda _f, _n: None,
    )
    assert len(out) == 1
    record, key = out[0]
    assert key == position_key(seed_fen)
    board = seed_board_from_line(record.split("#", 1)[0].strip())
    assert board.fen() == seed_fen
    assert "refute=" not in record


def test_mine_game_refute_falls_back_when_post_refute_loader_rejects() -> None:
    """Legal SF reply that leaves a terminal/forced position → bare blind-spot."""
    # Scholar's mate line: after white Qh5 (ply 4), black Qf6? No — need a
    # position where post-blunder SF bestmates. Use K+Q vs K endgame craft:
    # White to move at blind-spot blunders into a mate-in-1 for black.
    # Simpler: monkeypatch _fen_reject_reason via a refute that is legal but
    # we force reject by choosing a real mate reply.
    #
    # Position: white king e1, black king e8, white queen d1, black queen a5 —
    # too heavy. Instead: use _LINE collapse path and a bestmove that is legal
    # after f3g5 but produce a seed we mark by patching reject for that body only.
    from unittest.mock import patch

    game = _game(_LINE)
    blunder_after = _fen_after(_LINE, 7)
    seed_fen = _fen_after(_LINE, 6)
    refute = "d7d5"

    def best(fen: str, _n: int) -> str | None:
        return refute if fen == blunder_after else None

    def reject_post_refute_only(line: str) -> str | None:
        # Reject only when the body already includes blunder+refute (longer stack).
        body = line.split("#", 1)[0].strip()
        board = seed_board_from_line(body)
        if board.fen() == _fen_after(_LINE, 8):
            return "terminal position"
        return None

    with patch("scripts.mine_blindspot_seeds._fen_reject_reason", side_effect=reject_post_refute_only):
        out = mine_game(
            game, name_needle="deepfin", sf_eval=_fake_sf(blunder_after),
            src="t.pgn", sf_nodes=1, collapse_cp=-150, history_plies=8,
            append_refute_ply=True, sf_bestmove=best,
        )
    assert len(out) == 1
    record, key = out[0]
    assert key == position_key(seed_fen)
    board = seed_board_from_line(record.split("#", 1)[0].strip())
    assert board.fen() == seed_fen
    assert "refute=" not in record


def test_seed_blindspot_key_walks_back_refute() -> None:
    fens = [_fen_after(_LINE, i) for i in range(len(_LINE) + 1)]
    rec = build_seed_record(
        fens, list(_LINE), 6, history_plies=8, provenance="ply=6 refute=d7d5",
        refute_uci="d7d5",
    )
    assert seed_blindspot_key(rec) == position_key(fens[6])
    # Bare seed keys terminal (= blind-spot).
    bare = build_seed_record(fens, list(_LINE), 6, history_plies=8, provenance="ply=6")
    assert seed_blindspot_key(bare) == position_key(fens[6])


def test_load_existing_keys_dedups_post_refute_against_blindspot(tmp_path) -> None:
    """Re-mine with --existing post-refute list must not re-emit the same hole."""
    fens = [_fen_after(_LINE, i) for i in range(len(_LINE) + 1)]
    seed_fen = fens[6]
    rec = build_seed_record(
        fens, list(_LINE), 6, history_plies=8,
        provenance="src=t.pgn ply=6 refute=d7d5", refute_uci="d7d5",
    )
    existing = tmp_path / "seeds.txt"
    existing.write_text(rec + "\n", encoding="utf-8")
    keys = load_existing_keys([str(existing)])
    assert position_key(seed_fen) in keys
    # Terminal post-refute alone is NOT the dedup key for mine_game emission.
    post = position_key(fens[8])
    assert post not in keys or post == position_key(seed_fen)

    # Incremental: mine same game with existing keys → excluded by caller pattern.
    game = _game(_LINE)
    blunder_after = _fen_after(_LINE, 7)

    def best(fen: str, _n: int) -> str | None:
        return "d7d5" if fen == blunder_after else None

    mined = mine_game(
        game, name_needle="deepfin", sf_eval=_fake_sf(blunder_after),
        src="t.pgn", sf_nodes=1, collapse_cp=-150, history_plies=8,
        append_refute_ply=True, sf_bestmove=best,
    )
    assert len(mined) == 1
    _record, key = mined[0]
    assert key in keys  # would be filtered by main()'s exclude set


def test_mine_game_bakes_deep_sf_evals_into_comment() -> None:
    """Collapse seeds carry the already-computed deep-SF evals (free reuse)."""
    game = _game(_LINE)                       # DeepFin=white, loses
    blunder_after = _fen_after(_LINE, 7)      # black to move -> fake +400 -> DeepFin -400
    out = mine_game(game, name_needle="deepfin", sf_eval=_fake_sf(blunder_after),
                    src="t.pgn", sf_nodes=1, collapse_cp=-150, history_plies=8,
                    append_refute_ply=False)
    assert len(out) == 1
    record, _key = out[0]
    assert "sfcp_before=30" in record         # DeepFin-POV eval at the blind-spot
    assert "sfcp_after=-400" in record        # DeepFin-POV eval after the blunder
    # Comment is inert to the loader; body still round-trips.
    board = seed_board_from_line(record.split("#", 1)[0].strip())
    assert board.fen() == _fen_after(_LINE, 6)
    # sfcp_* keys must not confuse the refute-marker parser.
    assert seed_blindspot_key(record) == position_key(_fen_after(_LINE, 6))


# ── history-bearing match_vs_uci PGN + move-log CSV (OpeningPlyCount) ────────
#
# match_vs_uci writes the opening history into the PGN mainline ahead of the
# engine moves (header OpeningPlyCount = N), but its --move-log-out CSV numbers
# plies from the opening position (CSV ply 1 = first ENGINE move). Both files
# are produced here by match_vs_uci's own writers so the join is tested against
# the real producer, not a hand-written imitation of it.

def _quiet_continuations(board: chess.Board) -> list[chess.Move]:
    """Legal moves that keep every position a usable seed (no repetition/draw
    claim, no game end, >= 2 replies), in a stable order."""
    out = []
    for m in sorted(board.legal_moves, key=lambda m: m.uci()):
        board.push(m)
        ok = not board.is_game_over(claim_draw=True) and board.legal_moves.count() >= 2
        board.pop()
        if ok:
            out.append(m)
    assert out
    return out
_MOVE_LOG_FIELDS = [
    "game", "opening", "ply", "color", "engine", "move", "elapsed_s", "nodes",
    "engine_time_s", "score_cp_white", "white_clock_before_s",
    "black_clock_before_s", "white_clock_after_s", "black_clock_after_s",
]


def _history_game(
    *, root_fen: str, book: list[str], deepfin: chess.Color,
    engine_plies: int = 24, seed: int = 1232, first_own: int = 2000,
):
    """One lost DeepFin game written to PGN the way match_vs_uci does.

    Engine moves are a seeded random walk over moves that keep each position
    a usable seed (the loader rejects repetition-claimable terminals).
    Returns (game, move_records, fens, book_plies, deepfin_engine_js).
    """
    import io

    from scripts import match_vs_uci as match

    start = chess.Board(root_fen)
    for u in book:
        start.push(chess.Move.from_uci(u))
    board = start.copy()
    moves: list[chess.Move] = []
    records: list = []
    rng = random.Random(seed)
    for ply in range(1, engine_plies + 1):
        mover = board.turn
        move = rng.choice(_quiet_continuations(board))
        # Own eval (White POV): DeepFin reports +30 (its POV) everywhere except
        # a "thinks it's winning" ``first_own`` on its FIRST engine move; the
        # opponent reports -2000 (DeepFin POV) so any join onto an opponent row
        # is a huge, wrongly-placed mismatch.
        if mover == deepfin:
            first = not any(r.color == ("W" if deepfin else "B") for r in records)
            df_pov = first_own if first else 30
        else:
            df_pov = -2000
        score_white = df_pov if deepfin == chess.WHITE else -df_pov
        records.append(match.MoveRecord(
            ply=ply, color="W" if mover == chess.WHITE else "B", move=move.uci(),
            elapsed_s=0.01, nodes=1, engine_time_s=0.01, score_cp=score_white,
            white_clock_before_s=None, black_clock_before_s=None,
            white_clock_after_s=None, black_clock_after_s=None,
        ))
        moves.append(move)
        board.push(move)
    record = match.GameRecord(
        result="0-1" if deepfin == chess.WHITE else "1-0", plies=engine_plies,
        start_board=start, moves=tuple(moves), move_records=tuple(records),
    )
    names = ("deepfin_v1", "Cheese") if deepfin == chess.WHITE else ("Cheese", "deepfin_v1")
    text = io.StringIO()
    match._write_pgn_game(text, record=record, event="t", white_name=names[0],
                          black_name=names[1], round_name="7")
    game = chess.pgn.read_game(io.StringIO(text.getvalue()))
    assert game is not None
    assert not game.errors
    replay = game.board()
    fens = []
    for m in game.mainline_moves():
        fens.append(replay.fen())
        replay.push(m)
    fens.append(replay.fen())
    n = len(book)
    df_engine_js = [j for j in range(n, len(fens) - 1)
                    if chess.Board(fens[j]).turn == deepfin]
    return game, records, fens, n, df_engine_js


def _write_move_log(path, attempts, *, deepfin: chess.Color, with_move: bool = True,
                    drop=()):
    """Write move-log rows for each attempt (a list of MoveRecords, all game 7)
    with match_vs_uci's own row writer, in order — a resume replay appends a
    second attempt under the same game index. ``drop`` = {(attempt, ply)}
    rows left out. ``with_move=False`` writes a CSV without the move column."""
    import csv
    import io

    from scripts import match_vs_uci as match

    rows = []
    dropped = set(drop)
    for k, records in enumerate(attempts):
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_MOVE_LOG_FIELDS)
        writer.writeheader()
        for mr in records:
            match._write_move_record(
                writer, game_index=7, a_is_white=deepfin == chess.WHITE,
                label_a="deepfin_v1", label_b="Cheese", opening_index=1, move=mr,
            )
        buf.seek(0)
        rows.extend(r for r in csv.DictReader(buf) if (k, int(r["ply"])) not in dropped)
    fields = [f for f in _MOVE_LOG_FIELDS if with_move or f != "move"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _history_match(
    tmp_path, *, root_fen: str, book: list[str], deepfin: chess.Color,
    engine_plies: int = 24,
):
    """One lost DeepFin game: PGN + its move-log CSV, both via match_vs_uci's
    writers. Returns (game, own_evals, fens, book_plies, deepfin_engine_js)."""
    from scripts.mine_blindspot_seeds import load_own_evals

    game, records, fens, n, df_js = _history_game(
        root_fen=root_fen, book=book, deepfin=deepfin, engine_plies=engine_plies)
    csv_path = _write_move_log(tmp_path / "moves.csv", [records], deepfin=deepfin)
    return game, load_own_evals([str(csv_path)]), fens, n, df_js


# (root FEN, book, DeepFin colour): odd N, even N, a black-to-move SetUp root
# with odd N, and the legacy no-history case (no header at all).
_HISTORY_CASES = {
    "odd_n5_deepfin_black": (chess.STARTING_FEN, ["g1f3", "g8f6", "f3g1", "f6g8", "e2e4"],
                             chess.BLACK),
    "even_n4_deepfin_white": (chess.STARTING_FEN, ["g1f3", "g8f6", "f3g1", "f6g8"],
                              chess.WHITE),
    "setup_black_to_move_n3_deepfin_white": (
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        ["e7e5", "g1f3", "g8f6"], chess.WHITE),
    "setup_black_to_move_n2_deepfin_black": (
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        ["e7e5", "g1f3"], chess.BLACK),
    "legacy_no_history_deepfin_white": (chess.STARTING_FEN, [], chess.WHITE),
}


@pytest.mark.parametrize("case", sorted(_HISTORY_CASES))
def test_mine_game_offsets_book_plies_for_collapse_and_own_eval_join(tmp_path, case) -> None:
    root_fen, book, deepfin = _HISTORY_CASES[case]
    game, own_evals, fens, n, df_js = _history_match(
        tmp_path, root_fen=root_fen, book=book, deepfin=deepfin)
    if n:
        assert game.headers["OpeningPlyCount"] == str(n)
    else:
        assert "OpeningPlyCount" not in game.headers
    # Trap every book move made by DeepFin's colour: SF calls it a collapse.
    # A book move was not chosen by DeepFin, so it must never be mined.
    sf_vals = {fens[j + 1]: 400 for j in range(n) if chess.Board(fens[j]).turn == deepfin}
    blunder_j = df_js[-1]                    # DeepFin's LAST engine move collapses
    sf_vals[fens[blunder_j + 1]] = 400
    out = mine_game(game, src="m.pgn", name_needle="deepfin",
                    sf_eval=lambda fen, _n: sf_vals.get(fen, 30), sf_nodes=1,
                    collapse_cp=-150, history_plies=200, own_evals=own_evals,
                    append_refute_ply=False)
    records = [r for r, _k in out]
    assert len(records) == 2, records
    collapse, mismatch = records
    assert f" ply={blunder_j} sfcp_before=30 sfcp_after=-400" in collapse
    # The worst mismatch is DeepFin's FIRST engine move, own=+2000 (its POV) —
    # the CSV row that move actually wrote, not a book-shifted neighbour.
    assert f" ply={df_js[0]} mismatch" in mismatch
    assert "own=2000)" in mismatch
    # The emitted seed still carries the full history, book plies included.
    board = seed_board_from_line(collapse.split("#", 1)[0].strip())
    assert board.fen() == fens[blunder_j]
    assert len(board.move_stack) == blunder_j


@pytest.mark.parametrize("result", ["0-1", "1-0", "1/2-1/2"])
@pytest.mark.parametrize("bad", ["-1", "abc", "1.5", "", "99"])
def test_mine_game_refuses_malformed_opening_ply_count(bad, result) -> None:
    # Refused even for a game the miner would skip (DeepFin=white won / drew):
    # a corrupt header means the file is suspect, not just this game.
    game = _game(_LINE, result=result)
    game.headers["OpeningPlyCount"] = bad
    with pytest.raises(ValueError, match=r"^t\.pgn: Round 1: OpeningPlyCount"):
        _mine(game, lambda _fen, _nodes: 30)


# ── move-verified own-eval join (resume replays, foreign CSVs, gaps) ─────────

_ODD = _HISTORY_CASES["odd_n5_deepfin_black"]


def _mine_own(game, own_evals):
    return mine_game(game, src="m.pgn", name_needle="deepfin",
                     sf_eval=lambda _fen, _n: 30, sf_nodes=1, collapse_cp=-150,
                     history_plies=200, own_evals=own_evals, append_refute_ply=False)


def _mismatch_own(out) -> str | None:
    hits = [r for r, _k in out if " mismatch " in r]
    assert len(hits) <= 1
    return hits[0].split("own=", 1)[1].rstrip(")") if hits else None


def test_own_eval_join_refuses_csv_whose_move_differs(tmp_path) -> None:
    from scripts.mine_blindspot_seeds import load_own_evals

    root, book, deepfin = _ODD
    game, records, _fens, _n, _js = _history_game(root_fen=root, book=book, deepfin=deepfin)
    _g2, other, _f2, _n2, _j2 = _history_game(root_fen=root, book=book, deepfin=deepfin, seed=7)
    path = _write_move_log(tmp_path / "m.csv", [other], deepfin=deepfin)
    bad_ply = next(i + 1 for i, (a, b) in enumerate(zip(records, other)) if a.move != b.move)
    with pytest.raises(ValueError, match=rf"^m\.pgn: Round 7: move-log CSV ply {bad_ply} has move "
                                         rf"'{other[bad_ply - 1].move}' but the PGN game has "
                                         rf"'{records[bad_ply - 1].move}'"):
        _mine_own(game, load_own_evals([str(path)]))


def test_own_eval_join_pairs_replayed_attempts_by_move_not_last_row(tmp_path) -> None:
    """--resume replays a game whose commit was lost: two PGN copies, same Round,
    duplicate CSV (game, ply) keys with differing moves. Each copy must get ITS
    evals; last-row-wins gave the first copy the replay's."""
    from scripts.mine_blindspot_seeds import load_own_evals

    root, book, deepfin = _ODD
    game_a, rec_a, *_ = _history_game(root_fen=root, book=book, deepfin=deepfin,
                                      first_own=2000)
    game_b, rec_b, *_ = _history_game(root_fen=root, book=book, deepfin=deepfin,
                                      seed=7, first_own=1500)
    assert [r.move for r in rec_a] != [r.move for r in rec_b]
    own = load_own_evals([str(_write_move_log(tmp_path / "m.csv", [rec_a, rec_b],
                                              deepfin=deepfin))])
    assert own.replayed_rounds == 1
    # Mine the replay FIRST: the first copy may never be mined (a skipped
    # non-loss), so pairing must come from the moves, not from file order.
    assert _mismatch_own(_mine_own(game_b, own)) == "1500"
    assert _mismatch_own(_mine_own(game_a, own)) == "2000"
    assert own.missing_rows == 0


def test_own_eval_join_pairs_identical_replays_in_file_order(tmp_path) -> None:
    from scripts.mine_blindspot_seeds import load_own_evals

    root, book, deepfin = _ODD
    game_a, rec_a, *_ = _history_game(root_fen=root, book=book, deepfin=deepfin,
                                      first_own=2000)
    game_b, rec_b, *_ = _history_game(root_fen=root, book=book, deepfin=deepfin,
                                      first_own=1500)
    own = load_own_evals([str(_write_move_log(tmp_path / "m.csv", [rec_a, rec_b],
                                              deepfin=deepfin))])
    assert _mismatch_own(_mine_own(game_a, own)) == "2000"
    assert _mismatch_own(_mine_own(game_b, own)) == "1500"
    with pytest.raises(ValueError, match=r"^m\.pgn: Round 7: more PGN copies"):
        _mine_own(game_a, own)  # a third copy has no run left to claim


def test_own_eval_join_counts_missing_rows_and_scores(tmp_path) -> None:
    from scripts.mine_blindspot_seeds import load_own_evals

    root, book, deepfin = _ODD
    game, records, _fens, n, df_js = _history_game(root_fen=root, book=book, deepfin=deepfin)
    first_df_ply = df_js[0] + 1 - n           # CSV ply of DeepFin's first engine move
    path = _write_move_log(tmp_path / "m.csv", [records], deepfin=deepfin,
                           drop={(0, first_df_ply)})
    own = load_own_evals([str(path)])
    out = _mine_own(game, own)
    assert own.missing_rows == 1               # counted, not silently skipped
    assert own.missing_scores == 0
    assert _mismatch_own(out) is None          # the only +2000 row is the missing one
    # A round with no CSV rows at all is counted per mined game.
    empty = load_own_evals([str(_write_move_log(tmp_path / "e.csv", [], deepfin=deepfin))])
    _mine_own(game, empty)
    assert empty.games_without_rows == 1
    assert empty.missing_rows == len(df_js)


def test_own_eval_join_without_move_column_behaves_as_before(tmp_path) -> None:
    """A CSV with no move column cannot be verified: the old join applies —
    (round, ply) only, last row wins, nothing refused — and it says so."""
    from scripts.mine_blindspot_seeds import load_own_evals

    root, book, deepfin = _ODD
    game_a, rec_a, *_ = _history_game(root_fen=root, book=book, deepfin=deepfin,
                                      first_own=2000)
    _gb, rec_b, *_ = _history_game(root_fen=root, book=book, deepfin=deepfin,
                                   seed=7, first_own=1500)
    path = _write_move_log(tmp_path / "m.csv", [rec_a, rec_b], deepfin=deepfin,
                           with_move=False)
    own = load_own_evals([str(path)])
    assert own.verified is False
    # rec_b's first DeepFin move is at the same CSV ply (same colour, same N),
    # so the last-written attempt's +1500 wins exactly as on the old loader.
    assert _mismatch_own(_mine_own(game_a, own)) == "1500"
    # Mixing a verifiable CSV with an unverifiable one is refused.
    with_move = _write_move_log(tmp_path / "v.csv", [rec_a], deepfin=deepfin)
    with pytest.raises(ValueError, match="move' column"):
        load_own_evals([str(with_move), str(path)])


def test_own_eval_join_accepts_terminal_null_move_row(tmp_path) -> None:
    """An illegal/null-move loss: match_vs_uci logs the offending move as one
    extra CSV row (ply n+1, e.g. "0000") that the PGN never plays. It must
    join, not be refused as a different game."""
    from scripts import match_vs_uci as match
    from scripts.mine_blindspot_seeds import load_own_evals

    root, book, deepfin = _ODD
    game, records, *_ = _history_game(root_fen=root, book=book, deepfin=deepfin)
    nxt = len(records) + 1
    null_row = match.MoveRecord(
        ply=nxt, color="B" if records[-1].color == "W" else "W", move="0000",
        elapsed_s=0.01, nodes=None, engine_time_s=None, score_cp=None,
        white_clock_before_s=None, black_clock_before_s=None,
        white_clock_after_s=None, black_clock_after_s=None,
    )
    own = load_own_evals([str(_write_move_log(tmp_path / "m.csv", [[*records, null_row]],
                                              deepfin=deepfin))])
    assert _mismatch_own(_mine_own(game, own)) == "2000"
    assert own.missing_rows == 0


def test_own_eval_join_prefers_complete_replay_over_crash_partial(tmp_path) -> None:
    """Crash mid-game, then --resume replays it: the CSV holds a partial run (a
    prefix of the same moves) and then the complete replay. The complete run's
    evals are the PGN game's, even though the partial run comes first."""
    from scripts.mine_blindspot_seeds import load_own_evals

    root, book, deepfin = _ODD
    _gp, partial, *_ = _history_game(root_fen=root, book=book, deepfin=deepfin,
                                     first_own=999)
    game, full, *_ = _history_game(root_fen=root, book=book, deepfin=deepfin,
                                   first_own=2000)
    assert [r.move for r in partial] == [r.move for r in full]
    own = load_own_evals([str(_write_move_log(tmp_path / "m.csv", [partial[:10], full],
                                              deepfin=deepfin))])
    assert own.replayed_rounds == 1
    assert _mismatch_own(_mine_own(game, own)) == "2000"
    assert own.missing_rows == 0
