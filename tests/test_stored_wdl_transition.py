import chess
import numpy as np
import pytest
from chess_anti_engine.encoding.lc0 import (
    encode_lc0_full_root,
    apply_lc0_root_legacy_meta_planes,
)
from scripts.stored_wdl_transition import validate_pair


def encoded(b):
    a = np.zeros((175, 8, 8), np.float32)
    a[:112] = encode_lc0_full_root(b)
    apply_lc0_root_legacy_meta_planes(a, b)
    return a.astype(np.float16).astype(np.float32)


@pytest.mark.parametrize(
    ("fen", "move"),
    [
        (chess.STARTING_FEN, "e2e4"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "e7e5"),
        ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
    ],
)
def test_legal_cases(fen, move):
    b = chess.Board(fen)
    p = encoded(b)
    b.push_uci(move)
    assert validate_pair(p, encoded(b)) == {
        "status": "qualified_narrow",
        "move_uci": move,
    }


@pytest.mark.parametrize("plane", [104, 108, 109, 110, 13])
def test_corruption_unresolved(plane):
    b = chess.Board()
    p = encoded(b)
    b.push_uci("e2e4")
    c = encoded(b)
    c[plane, 0, 0] = 0.5
    assert validate_pair(p, c)["status"] == "unresolved"


def test_saturated_clock_unknown():
    b = chess.Board()
    b.halfmove_clock = 100
    p = encoded(b)
    b.push_uci("g1f3")
    assert "saturated" in validate_pair(p, encoded(b))["reason"]


def test_wrong_legal_child():
    b = chess.Board()
    p = encoded(b)
    b.push_uci("e2e4")
    c = encoded(b)
    c[109] = np.float32(np.float16(0.02))
    assert validate_pair(p, c)["status"] == "unresolved"


def test_repetition_overlap_not_recomputed():
    b = chess.Board()
    p = encoded(b)
    b.push_uci("e2e4")
    c = encoded(b)
    p[12] = 1
    c[25] = 1
    assert validate_pair(p, c)["status"] == "qualified_narrow"
    c[25] = 0
    assert "overlap" in validate_pair(p, c)["reason"]
