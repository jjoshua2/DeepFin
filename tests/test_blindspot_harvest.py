"""Inline blind-spot harvesting (selfplay/blindspot_harvest.py)."""
from __future__ import annotations

import chess
import numpy as np

from chess_anti_engine.moves.encode import (
    move_to_index_for_encoding,
    policy_size_for_encoding,
)
from chess_anti_engine.selfplay.blindspot_harvest import (
    HarvestConfig,
    HarvestedSeed,
    format_record,
    harvest_from_records,
    pre_move_boards,
    seed_line_from_board,
)
from chess_anti_engine.selfplay.opening import seed_board_from_line

_ENC = "lc0_1858"
_LINE = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5", "d7d5", "e4d5", "c6a5"]


def _wdl(q: float) -> np.ndarray:
    w = max(q, 0.0)
    loss = max(-q, 0.0)
    return np.array([w, 1.0 - w - loss, loss], dtype=np.float32)


def _board_after(n: int) -> chess.Board:
    b = chess.Board()
    for u in _LINE[:n]:
        b.push(chess.Move.from_uci(u))
    return b


def _probs_favoring(board: chess.Board, uci: str, p: float = 0.9) -> np.ndarray:
    """policy_probs with weight p on the given legal move (the rest uniform)."""
    v = np.full((policy_size_for_encoding(_ENC),), (1.0 - p) / policy_size_for_encoding(_ENC),
                dtype=np.float32)
    idx = int(move_to_index_for_encoding(chess.Move.from_uci(uci), board, policy_encoding=_ENC))
    v[idx] = p
    return v


def _cfg() -> HarvestConfig:
    return HarvestConfig(net_ok=0.2, sf_lost=-0.5, severe_net_ok=0.5, severe_sf_lost=-0.5)


# ── board + played-move reconstruction ───────────────────────────────────────

def test_pre_move_boards_aligns_board_and_played_move() -> None:
    final = _board_after(len(_LINE))
    boards, played = pre_move_boards(chess.Board(), list(final.move_stack), [4, 6], opening_len=0)
    assert boards[0] is not None
    assert boards[1] is not None
    assert boards[0].fen() == _board_after(4).fen()
    assert boards[1].fen() == _board_after(6).fen()
    assert played[0] == chess.Move.from_uci(_LINE[4])   # the move played from ply 4
    assert played[1] == chess.Move.from_uci(_LINE[6])
    assert len(boards[1].move_stack) == 6


# ── seed-line emission ───────────────────────────────────────────────────────

def test_seed_line_round_trips_with_history() -> None:
    board = _board_after(9)
    line = seed_line_from_board(board, history_plies=8)
    rebuilt = seed_board_from_line(line)
    assert rebuilt.fen() == board.fen()
    assert len(rebuilt.move_stack) == 8


def test_seed_line_bare_fen_at_game_start() -> None:
    assert seed_line_from_board(chess.Board(), history_plies=8) == chess.Board().fen()


# ── detection (value-blind + played move the search favored) ─────────────────

def _harvest(boards, played, evals, cfg=None):
    return harvest_from_records(evals, boards, played, cfg=cfg or _cfg(), policy_encoding=_ENC)


def test_harvest_flags_value_blind_favored_moves() -> None:
    b4, b6 = _board_after(4), _board_after(6)
    boards = [b4, b6]
    played = [chess.Move.from_uci(_LINE[4]), chess.Move.from_uci(_LINE[6])]
    evals = [
        (True, _wdl(0.6), _wdl(-0.6), _probs_favoring(b4, _LINE[4])),  # blind + severe, favored
        (True, _wdl(0.3), _wdl(-0.55), _probs_favoring(b6, _LINE[6])),  # blind, not severe, favored
    ]
    out = _harvest(boards, played, evals)
    assert len(out) == 2
    assert out[0].severe is True
    assert out[1].severe is False


def test_harvest_skips_temperature_exploration() -> None:
    # Value-blind, but the played move carried ~no improved-policy weight (the
    # search favored a DIFFERENT move) -> an exploration blunder, not blindness.
    b4 = _board_after(4)
    probs = _probs_favoring(b4, "e1e2", p=0.9)          # search favored Ke2, NOT the played move
    evals = [(True, _wdl(0.6), _wdl(-0.6), probs)]
    out = _harvest([b4], [chess.Move.from_uci(_LINE[4])], evals)
    assert out == []


def test_harvest_skips_value_only_and_missing() -> None:
    b4 = _board_after(4)
    good = _probs_favoring(b4, _LINE[4])
    played = [chess.Move.from_uci(_LINE[4])] * 3
    evals = [
        (False, _wdl(0.6), _wdl(-0.6), good),   # value-only row
        (True, _wdl(0.6), None, good),          # missing sf label
        (True, _wdl(0.6), _wdl(0.1), good),     # SF says fine -> not blind
    ]
    assert _harvest([b4, b4, b4], played, evals) == []


def test_format_record_loadable_with_provenance() -> None:
    seed = HarvestedSeed(line=seed_line_from_board(_board_after(9), 8),
                         net_q=0.61, sf_q=-0.62, severe=True)
    rec = format_record(seed, game_id="g123")
    assert "sev=1" in rec
    assert "game=g123" in rec
    assert seed_board_from_line(rec.split("#", 1)[0].strip()).fen() == _board_after(9).fen()


# ── run_harvest end-to-end (reconstruct -> detect -> split files) ────────────

class _Rec:
    def __init__(self, ply_index, has_policy, search_wdl_est, sf_wdl, policy_probs):
        self.ply_index = ply_index
        self.has_policy = has_policy
        self.search_wdl_est = search_wdl_est
        self.sf_wdl = sf_wdl
        self.policy_probs = policy_probs


def test_run_harvest_splits_severe_into_sibling_file(tmp_path) -> None:
    from chess_anti_engine.selfplay.blindspot_harvest import run_harvest, severe_path_for
    from chess_anti_engine.selfplay.opening import _load_fen_list

    final = _board_after(len(_LINE))
    records = [
        _Rec(4, True, _wdl(0.6), _wdl(-0.6), _probs_favoring(_board_after(4), _LINE[4])),   # severe
        _Rec(6, True, _wdl(0.3), _wdl(-0.55), _probs_favoring(_board_after(6), _LINE[6])),  # broad
    ]
    out = tmp_path / "harvest.txt"
    n = run_harvest(chess.Board(), final, records, has_c_ply=True, game_id="g1",
                    out_path=str(out), cfg=_cfg(), policy_encoding=_ENC)
    assert n == 2
    assert len(_load_fen_list(str(out))) == 2
    severe_lines = _load_fen_list(severe_path_for(str(out)))
    assert len(severe_lines) == 1
    assert seed_board_from_line(severe_lines[0]).fen() == _board_after(4).fen()
    assert severe_path_for("a/b/harvest.txt").endswith("harvest.severe.txt")


def test_run_harvest_off_and_fail_safe(tmp_path) -> None:
    from chess_anti_engine.selfplay.blindspot_harvest import run_harvest

    final = _board_after(4)
    assert run_harvest(chess.Board(), final, [], has_c_ply=True, game_id="g",
                       out_path="", cfg=_cfg(), policy_encoding=_ENC) == 0
    bad = [_Rec("not-an-int", True, _wdl(0.6), _wdl(-0.6), None)]
    assert run_harvest(chess.Board(), final, bad, has_c_ply=True, game_id="g",
                       out_path=str(tmp_path / "h.txt"), cfg=_cfg(), policy_encoding=_ENC) == 0
