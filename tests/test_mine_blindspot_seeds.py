"""Blind-spot seed miner core (scripts/mine_blindspot_seeds.py)."""
from __future__ import annotations

import chess
import chess.pgn

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
    """Default history_plies=8 + refute still yields ≤8 UCIs (LC0 window)."""
    fens = [_fen_after(_LINE, i) for i in range(len(_LINE) + 1)]
    rec = build_seed_record(
        fens, list(_LINE), 6, history_plies=8, provenance="x", refute_uci=_LINE[7],
    )
    body = rec.split("#", 1)[0].strip()
    board = seed_board_from_line(body)
    assert board.fen() == fens[8]
    # 6 approach available, but clamped to 8-2=6 preceding + blunder + refute would be 8;
    # j=6 so preceding = min(6, 6) = 6, total 8.
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
