import chess
import numpy as np

from chess_anti_engine.encoding import encode_position


def test_start_position_piece_counts_and_castling():
    b = chess.Board()
    x = encode_position(b, add_features=False)

    assert x.shape[1:] == (8, 8)
    # full LC0 = 112 planes
    assert x.shape[0] == 112

    # current position piece planes live at the front (first history step)
    # us pawns (plane 0) should have 8 pawns
    assert int(x[0].sum()) == 8
    # them pawns (plane 6) should have 8 pawns
    assert int(x[6].sum()) == 8

    # castling rights planes are at 96..99
    for i in range(96, 100):
        assert np.all(x[i] == 1.0)


def test_reduced_encoder_still_available():
    b = chess.Board()
    x = encode_position(b, add_features=False, use_full_lc0=False)
    assert x.shape[0] == 20


def test_en_passant_file_plane():
    # After 1. e4, black to move, ep square is e3
    b = chess.Board()
    b.push_san("e4")
    x = encode_position(b, add_features=False)

    # EP plane is at index 100
    ep_plane = x[100]
    assert np.all(ep_plane[:, 4] == 1.0)
    assert float(ep_plane.sum()) == 8.0


def test_orientation_side_to_move_flips_ranks_for_black():
    # Put a single white pawn on a2, black to move -> from black POV it should appear on a7
    b = chess.Board(fen="8/8/8/8/8/8/P7/4k2K b - - 0 1")
    x = encode_position(b, add_features=False)

    # In this position, side-to-move is black => us=black, them=white.
    # them pawn plane index (first history) = 6
    pawn_plane = x[6]
    assert pawn_plane[6, 0] == 1.0  # a7 (rank index 6)
    assert float(pawn_plane.sum()) == 1.0


def test_pin_planes_non_empty():
    # Simple pin: black king e8, black rook e7 pinned by white rook e1
    b = chess.Board(fen="4k3/4r3/8/8/8/8/8/4R2K b - - 0 1")
    x = encode_position(b, add_features=True, feature_dropout_p=0.0)

    # Full LC0 base = 112. Extra planes begin at 112.
    # In features.py: king safety (10), then pin/xray (6).
    # pin planes start at offset 112+10 = 122, and for 'us' we add pinned, ray, discovered.
    pinned_us = x[122]
    assert pinned_us.sum() >= 1.0
