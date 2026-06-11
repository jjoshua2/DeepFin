import chess
import numpy as np

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import (
    LC0_HISTORY_ROOT_LEGACY_META,
    LC0_HISTORY_MODE_LEGACY,
    LC0_HISTORY_MODE_ROOT,
    LC0_HISTORY_MODE_ROOT_LEGACY_META,
    c_input_history_mode,
    normalize_lc0_history_encoding,
)


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


def test_lc0_root_history_uses_classical_112_layout():
    b = chess.Board()
    b.push_san("e4")
    x = encode_position(b, add_features=False, input_history_encoding="lc0_root")

    assert x.shape == (112, 8, 8)
    # 8 LC0 history slots: each 12 piece planes + one repetition plane.
    assert int(x[0].sum()) == 8
    assert int(x[6].sum()) == 8
    assert int(x[13].sum()) == 8
    assert int(x[19].sum()) == 8

    # Classical LC0 aux starts at 104. Startpos after 1.e4 has all castling.
    for i in range(104, 108):
        assert np.all(x[i] == 1.0)
    # Plane 108 is the black/flipped flag, not an en-passant file plane.
    assert np.all(x[108] == 1.0)
    assert np.all(x[109] == 0.0)  # raw rule50 after a pawn move
    assert np.all(x[110] == 0.0)
    assert np.all(x[111] == 1.0)


def test_lc0_root_history_keeps_previous_slots_in_root_pov():
    b = chess.Board()
    b.push_san("e4")  # black to move; previous startpos should still be black POV.
    root = encode_position(b, add_features=False, input_history_encoding="lc0_root")

    prev = chess.Board()
    prev.turn = chess.BLACK
    ref = encode_position(prev, add_features=False)

    np.testing.assert_array_equal(root[13:25], ref[:12])


def test_lc0_root_legacy_meta_patches_rule50_and_ep():
    b = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 37 1")
    root = encode_position(b, add_features=False, input_history_encoding="lc0_root")
    meta = encode_position(b, add_features=False, input_history_encoding="lc0_root_legacy_meta")

    assert normalize_lc0_history_encoding("root_meta") == LC0_HISTORY_ROOT_LEGACY_META
    np.testing.assert_array_equal(meta[:109], root[:109])
    assert np.all(meta[109] == 0.37)
    assert np.all(meta[110, :, 4] == 1.0)
    assert float(meta[110].sum()) == 8.0
    assert np.all(meta[111] == 1.0)


def test_c_input_history_mode_ids_are_stable():
    assert c_input_history_mode(None) == LC0_HISTORY_MODE_LEGACY
    assert c_input_history_mode("legacy") == LC0_HISTORY_MODE_LEGACY
    assert c_input_history_mode("lc0_root") == LC0_HISTORY_MODE_ROOT
    assert c_input_history_mode("root_meta") == LC0_HISTORY_MODE_ROOT_LEGACY_META


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


def test_batch_encoder_selection_matches_python_per_mode():
    """Pin the name -> C-encoder selection against the Python encoders.

    The mode mapping has one home (lc0.c_input_history_mode) consumed by both
    gumbel_c._batch_encoders and the C state machine; this test catches a
    silent drift where a mode id starts selecting the wrong plane layout —
    the failure would otherwise be plausible-looking play on wrong inputs.
    """
    import random

    from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
    from chess_anti_engine.mcts.gumbel_c import _batch_encoders

    rng = random.Random(5)
    boards = []
    for _ in range(4):
        b = chess.Board()
        for _ in range(rng.randrange(0, 30)):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over():
                break
            b.push(rng.choice(moves))
        boards.append(b)

    for name in ("legacy", "lc0_root", "lc0_root_legacy_meta"):
        enc_f32, _enc_bf16 = _batch_encoders(name)
        cbs = [CBoard.from_board(b) for b in boards]
        out = np.zeros((len(cbs), 146, 8, 8), dtype=np.float32)
        enc_f32(cbs, out)
        for i, cb in enumerate(cbs):
            expect = encode_cboard(
                cb, input_history_encoding=name, input_extra_features="v1",
            )
            np.testing.assert_array_equal(
                out[i], expect,
                err_msg=f"encoder for {name!r} diverged from encode_cboard",
            )
