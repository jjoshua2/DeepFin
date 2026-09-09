"""Stored features preserve the default Board encoder's history, not just FEN."""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.ceres_tpg import (
    IS_EN_PASSANT,
    encode_ceres_tpg_bytes,
    stored_x_to_ceres_tpg_bytes,
)
from chess_anti_engine.encoding.encode import encode_position

ENCODING = 'lc0_root_legacy_meta'


def stored(board: chess.Board) -> np.ndarray:
    return encode_position(board, input_history_encoding=ENCODING,
                           input_extra_features='v2_threats').astype(np.float16)


def convert(x: np.ndarray) -> np.ndarray:
    return stored_x_to_ceres_tpg_bytes(x, input_history_encoding=ENCODING,
                                     history_rep_fix=True)


@pytest.mark.parametrize('moves', [
    '', 'e2e4', 'e2e4 e7e5', 'e2e4 a7a6 e4e5 d7d5',
    'a2a3 e7e5 a3a4 e5e4 d2d4',
    'g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8 e2e4',
    'g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8 g1f3 g8f6',
])
def test_exact_project_encoder_history_parity(moves: str) -> None:
    board = chess.Board()
    for uci in moves.split():
        board.push_uci(uci)
    x = stored(board)
    before = x.copy()
    actual = convert(x)
    assert actual.dtype == np.uint8
    assert actual.shape == (64, 137)
    np.testing.assert_array_equal(actual, encode_ceres_tpg_bytes(board))
    np.testing.assert_array_equal(x, before)
    # Distinct true frames must not collapse to a current-position-only encoding.
    if board.move_stack:
        assert not np.array_equal(actual, encode_ceres_tpg_bytes(board.copy(stack=False)))


@pytest.mark.parametrize('turn', ['w', 'b'])
@pytest.mark.parametrize('rights', ['KQkq', 'Kq', 'Qk', '-'])
def test_orthodox_castling_and_color_frame(turn: str, rights: str) -> None:
    b = chess.Board(f'r3k2r/8/8/8/8/8/8/R3K2R {turn} {rights} - 0 1')
    assert b.is_valid()
    np.testing.assert_array_equal(convert(stored(b)), encode_ceres_tpg_bytes(b))


def test_counter_grid_and_clipped_writer_counters() -> None:
    for clock in [*range(101), 101, 150]:
        b = chess.Board()
        b.halfmove_clock = clock
        np.testing.assert_array_equal(convert(stored(b)), encode_ceres_tpg_bytes(b))


def test_single_real_position_drops_ep_but_real_history_keeps_it() -> None:
    b = chess.Board()
    for uci in ['e2e4', 'a7a6', 'e4e5', 'd7d5']:
        b.push_uci(uci)
    assert convert(stored(b))[:, IS_EN_PASSANT].sum() == 100
    rootless = b.copy(stack=False)
    assert rootless.ep_square is not None
    np.testing.assert_array_equal(convert(stored(rootless)), encode_ceres_tpg_bytes(rootless))
    assert not convert(stored(rootless))[:, IS_EN_PASSANT].any()


def test_batch_preserves_order_short_fill_and_noncontiguous_view() -> None:
    boards = [chess.Board()]
    for move in ['e2e4', 'g8f6', 'e4e5']:
        board = boards[-1].copy()
        board.push_uci(move)
        boards.append(board)
    x = np.stack([stored(b) for b in boards])[::-1]
    assert not x.flags.c_contiguous
    expected = np.stack([encode_ceres_tpg_bytes(b) for b in boards[::-1]])
    np.testing.assert_array_equal(convert(x), expected)


@pytest.mark.parametrize('case', [
    'nan', 'nonbinary', 'overlap', 'no_king', 'gap', 'repetition_spatial',
    'missing_repetition', 'castling_spatial', 'color', 'bias', 'rule50_offgrid',
    'rule50_range', 'ep_spatial', 'ep_multiple', 'ep_no_pawn', 'false_castling',
])
def test_reject_malformed_writer_domain(case: str) -> None:
    b = chess.Board()
    b.push_uci('e2e4')
    x = stored(b)
    if case == 'nan':
        x[174, 0, 0] = np.nan
    elif case == 'nonbinary':
        x[0, 0, 0] = .5
    elif case == 'overlap':
        x[0, 0, 0] = x[1, 0, 0] = 1
    elif case == 'no_king':
        x[5] = 0
    elif case == 'gap':
        x[26:39] = x[:13]
        x[13:26] = 0
    elif case == 'repetition_spatial':
        x[12, 0, 0] = 1
    elif case == 'missing_repetition':
        x[38] = 1
    elif case == 'castling_spatial':
        x[104, 0, 0] = 0
    elif case == 'color':
        x[108] = .5
    elif case == 'bias':
        x[111] = 0
    elif case == 'rule50_offgrid':
        x[109] = .123
    elif case == 'rule50_range':
        x[109] = 1.1
    elif case == 'ep_spatial':
        x[110, 0, 0] = 1
    elif case == 'ep_multiple':
        x[110, :, 0] = 1
    elif case == 'ep_no_pawn':
        x[110] = 0
        x[110, :, 0] = 1
    elif case == 'false_castling':
        x[3, 0, 0] = 0
    expected = {
        'nan': 'nonfinite', 'nonbinary': 'binary', 'overlap': 'overlapping',
        'no_king': 'kings', 'gap': 'noncontiguous', 'repetition_spatial': 'repetition',
        'missing_repetition': 'repetition', 'castling_spatial': 'nonuniform',
        'color': 'color', 'bias': 'bias', 'rule50_offgrid': 'rule50',
        'rule50_range': 'rule50', 'ep_spatial': 'EP file', 'ep_multiple': 'multiple EP',
        'ep_no_pawn': 'capturable enemy pawn', 'false_castling': 'castling rights',
    }[case]
    with pytest.raises(ValueError, match=expected):
        convert(x)


@pytest.mark.parametrize('encoding', ['legacy', 'lc0_root', 'ceres_tpg', ''])
def test_refuse_other_encodings(encoding: str) -> None:
    with pytest.raises(ValueError, match='requires lc0_root_legacy_meta'):
        stored_x_to_ceres_tpg_bytes(stored(chess.Board()),
                                    input_history_encoding=encoding, history_rep_fix=True)


@pytest.mark.parametrize('fixed', [False, 1, None])
def test_require_explicit_corrected_history(fixed: bool) -> None:
    with pytest.raises(ValueError, match='history_rep_fix=True'):
        stored_x_to_ceres_tpg_bytes(stored(chess.Board()),
                                    input_history_encoding=ENCODING, history_rep_fix=fixed)


@pytest.mark.parametrize(('shape', 'dtype'), [
    ((175, 8, 8), np.float32), ((112, 8, 8), np.float16),
    ((0, 175, 8, 8), np.float16), ((175, 64), np.float16),
])
def test_refuse_wrong_storage_shape_or_dtype(shape: tuple[int, ...], dtype: type) -> None:
    with pytest.raises(ValueError, match=r'expected stored float16|empty or nonfinite'):
        convert(np.zeros(shape, dtype=dtype))
