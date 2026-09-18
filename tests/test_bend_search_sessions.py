"""Cheap session wire contracts; no compilation, model or search in pytest."""
from __future__ import annotations

import pytest

from native.bend_engine.session_probe.run_probe import (
    bits,
    evaluation,
    f32,
    move_key,
    numbers,
    position,
    unbits,
)
from native.bend_engine.legal_probe.run_probe import START, fen_position


def test_session_words_and_float_bits() -> None:
    assert numbers('result 1 2 3', 'result', 3) == [1, 2, 3]
    assert unbits(bits(-0.75)) == -0.75
    assert f32(0.1) != 0.1


@pytest.mark.parametrize('text', ['wrong 1', 'result -1', 'result 1.0',
                                 'result nan', 'result 4294967296', 'result 1 2', 'result'])
def test_session_report_rejects_malformed_words(text: str) -> None:
    with pytest.raises(ValueError, match='malformed|nondecimal|exceeds'):
        numbers(text, 'result', 1)


def test_position_roundtrip_preserves_high_bits() -> None:
    board = fen_position(START)
    words = [w for value in board[:8] for w in (value >> 32, value & 0xffffffff)] + list(board[8:])
    assert position(words) == board


@pytest.mark.parametrize('words', [[], [0] * 18, [1 << 32] * 19, [0] * 16 + [2, 0, 64]])
def test_position_shape_rejected(words: list[int]) -> None:
    with pytest.raises(ValueError, match='position'):
        position(words)


def test_move_identity_includes_underpromotion_and_special_flag() -> None:
    assert len({move_key((48, 56, promotion, 0)) for promotion in range(1, 5)}) == 4
    assert move_key((36, 43, 0, 0)) != move_key((36, 43, 0, 1))


def test_fixture_evaluator_is_not_a_constant_value_or_policy() -> None:
    board = fen_position('4k3/8/8/8/8/8/P7/4K3 w - - 0 1')
    wdl, policy = evaluation(board, [1, 2, 3, 4])
    black = (*board[:8], 0, *board[9:])
    opposite, _ = evaluation(black, [1, 2, 3, 4])
    assert wdl[0] - wdl[2] == -(opposite[0] - opposite[2])
    assert sum(wdl) == 1 and len(set(policy)) > 1
