"""Blind-spot seed miner core (scripts/mine_blindspot_seeds.py)."""
from __future__ import annotations

import chess
import chess.pgn

from chess_anti_engine.selfplay.opening import seed_board_from_line
from scripts.mine_blindspot_seeds import (
    build_seed_record,
    deepfin_color,
    find_first_collapse,
    mine_game,
    position_key,
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

def test_find_first_collapse_picks_first_crossing() -> None:
    evals = [(0, 40, 20), (2, 30, -200), (4, 10, -500)]  # ply 2 first crosses -150
    c = find_first_collapse(evals, collapse_cp=-150)
    assert c is not None
    assert c.ply == 2
    assert c.sf_after == -200


def test_find_first_collapse_ignores_already_lost() -> None:
    # Already lost before the move (sf_before < -150) is not a fresh collapse.
    evals = [(0, -300, -400), (2, -160, -900)]
    assert find_first_collapse(evals, collapse_cp=-150) is None


def test_find_first_collapse_none_when_no_crossing() -> None:
    assert find_first_collapse([(0, 50, 30), (2, 20, -80)], collapse_cp=-150) is None


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


def _mine(game: chess.pgn.Game, sf) -> tuple[str, str] | None:
    return mine_game(game, src="t.pgn", name_needle="deepfin", sf_eval=sf,
                     sf_nodes=1, collapse_cp=-150, history_plies=8)


def test_mine_game_emits_blindspot_with_history() -> None:
    game = _game(_LINE)                       # DeepFin=white, loses
    blunder_after = _fen_after(_LINE, 7)      # after white's move[6]=f3g5, black to move
    out = mine_game(game, name_needle="deepfin", sf_eval=_fake_sf(blunder_after),
                    src="runs/matches/test.pgn", sf_nodes=1, collapse_cp=-150, history_plies=8)
    assert out is not None
    record, key = out
    seed_fen = _fen_after(_LINE, 6)           # white to move — the blind-spot
    assert key == position_key(seed_fen)
    board = seed_board_from_line(record.split("#", 1)[0].strip())
    assert board.fen() == seed_fen
    assert len(board.move_stack) == 6
    assert "ply=6" in record


def test_mine_game_skips_non_loss_forfeit_ambiguous() -> None:
    sf = _fake_sf(_fen_after(_LINE, 7))
    assert _mine(_game(_LINE, result="1-0"), sf) is None            # DeepFin won
    assert _mine(_game(_LINE, termination="Time forfeit"), sf) is None
    assert _mine(_game(_LINE, white="a", black="b"), sf) is None    # ambiguous side


def test_mine_game_none_when_no_collapse() -> None:
    # A fake that never returns a losing eval -> no collapse.
    assert _mine(_game(_LINE), lambda _fen, _nodes: 30) is None
