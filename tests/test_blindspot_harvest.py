"""Inline blind-spot harvesting (selfplay/blindspot_harvest.py)."""
from __future__ import annotations

import chess
import numpy as np

from chess_anti_engine.selfplay.blindspot_harvest import (
    HarvestConfig,
    HarvestedSeed,
    format_record,
    harvest_from_records,
    pre_move_boards,
    seed_line_from_board,
)
from chess_anti_engine.selfplay.opening import seed_board_from_line

_LINE = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5", "d7d5", "e4d5", "c6a5"]


def _wdl(q: float) -> np.ndarray:
    # a WDL vector whose W-L equals q (draw takes the slack).
    w = max(q, 0.0)
    loss = max(-q, 0.0)
    return np.array([w, 1.0 - w - loss, loss], dtype=np.float32)


def _board_after(n: int) -> chess.Board:
    b = chess.Board()
    for u in _LINE[:n]:
        b.push(chess.Move.from_uci(u))
    return b


# ── board reconstruction ─────────────────────────────────────────────────────

def test_pre_move_boards_aligns_and_carries_history() -> None:
    final = _board_after(len(_LINE))
    # records at plies 4 and 6 (net = white, moves at even plies here).
    boards = pre_move_boards(chess.Board(), list(final.move_stack), [4, 6], opening_len=0)
    assert boards[0] is not None
    assert boards[1] is not None
    assert boards[0].fen() == _board_after(4).fen()
    assert boards[1].fen() == _board_after(6).fen()
    assert len(boards[1].move_stack) == 6  # real history preserved


def test_pre_move_boards_seeded_opening_len() -> None:
    # Python play path: starting_board already holds the opening moves, so
    # opening_len skips them in the final stack (which repeats them).
    start = _board_after(2)  # "seed" 2 plies in
    final = _board_after(len(_LINE))
    boards = pre_move_boards(start, list(final.move_stack), [4], opening_len=2)
    assert boards[0] is not None
    assert boards[0].fen() == _board_after(4).fen()


# ── seed-line emission ───────────────────────────────────────────────────────

def test_seed_line_carries_history_and_round_trips() -> None:
    board = _board_after(9)  # white to move at ply 9
    line = seed_line_from_board(board, history_plies=8)
    assert " | " in line
    rebuilt = seed_board_from_line(line)          # loads via the production loader
    assert rebuilt.fen() == board.fen()           # terminal == the blind-spot
    assert len(rebuilt.move_stack) == 8           # clamped to 8 history plies


def test_seed_line_bare_fen_at_game_start() -> None:
    assert seed_line_from_board(chess.Board(), history_plies=8) == chess.Board().fen()


# ── detection ────────────────────────────────────────────────────────────────

def _cfg() -> HarvestConfig:
    return HarvestConfig(net_ok=0.2, sf_lost=-0.5, severe_net_ok=0.5, severe_sf_lost=-0.5)


def test_harvest_flags_only_value_blind_full_plies() -> None:
    boards = [_board_after(4), _board_after(6), _board_after(8), _board_after(9)]
    evals = [
        (True, _wdl(0.6), _wdl(-0.6)),   # value-blind + severe
        (True, _wdl(0.3), _wdl(-0.55)),  # value-blind, not severe (net_q<0.5)
        (True, _wdl(0.6), _wdl(0.1)),    # net fine, SF fine -> not blind
        (False, _wdl(0.6), _wdl(-0.6)),  # value-only row (has_policy False) -> skip
    ]
    out = harvest_from_records(evals, boards, cfg=_cfg())
    assert len(out) == 2
    assert out[0].severe is True
    assert out[1].severe is False


def test_harvest_skips_missing_labels_and_boards() -> None:
    boards = [None, _board_after(6)]
    evals = [(True, _wdl(0.6), _wdl(-0.6)), (True, _wdl(0.6), None)]
    assert harvest_from_records(evals, boards, cfg=_cfg()) == []


def test_format_record_is_loadable_with_provenance() -> None:
    seed = HarvestedSeed(line=seed_line_from_board(_board_after(9), 8),
                         net_q=0.61, sf_q=-0.62, severe=True)
    rec = format_record(seed, game_id="g123")
    assert "sev=1" in rec
    assert "nq=0.61" in rec
    assert "game=g123" in rec
    # The loader strips the inline '# ...' comment and still parses the seed.
    body = rec.split("#", 1)[0].strip()
    assert seed_board_from_line(body).fen() == _board_after(9).fen()


# ── run_harvest end-to-end (reconstruct -> detect -> write file) ──────────────

class _Rec:
    def __init__(self, ply_index, has_policy, search_wdl_est, sf_wdl):
        self.ply_index = ply_index
        self.has_policy = has_policy
        self.search_wdl_est = search_wdl_est
        self.sf_wdl = sf_wdl


def test_run_harvest_writes_loadable_severe_seed(tmp_path) -> None:
    from chess_anti_engine.selfplay.blindspot_harvest import run_harvest
    from chess_anti_engine.selfplay.opening import _load_fen_list

    final = _board_after(len(_LINE))
    records = [
        _Rec(4, True, _wdl(0.6), _wdl(-0.6)),   # value-blind + severe @ ply 4
        _Rec(6, True, _wdl(0.1), _wdl(0.2)),    # fine -> not harvested @ ply 6
    ]
    out = tmp_path / "harvest.txt"
    n = run_harvest(chess.Board(), final, records, has_c_ply=True,
                    game_id="g1", out_path=str(out), cfg=_cfg())
    assert n == 1
    lines = _load_fen_list(str(out))            # loads + validates via production loader
    assert len(lines) == 1
    assert seed_board_from_line(lines[0]).fen() == _board_after(4).fen()
    assert "sev=1" in out.read_text()


def test_run_harvest_off_and_fail_safe(tmp_path) -> None:
    from chess_anti_engine.selfplay.blindspot_harvest import run_harvest

    final = _board_after(4)
    # empty out_path -> disabled, writes nothing.
    assert run_harvest(chess.Board(), final, [], has_c_ply=True, game_id="g",
                       out_path="", cfg=_cfg()) == 0
    # a broken record must be swallowed, not raised (fail-safe).
    bad = [_Rec("not-an-int", True, _wdl(0.6), _wdl(-0.6))]
    assert run_harvest(chess.Board(), final, bad, has_c_ply=True, game_id="g",
                       out_path=str(tmp_path / "h.txt"), cfg=_cfg()) == 0
