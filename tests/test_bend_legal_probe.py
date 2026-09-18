"""Cheap wire/FEN/report contracts; no Bend compiler or perft run in pytest."""
from __future__ import annotations

import pytest

from native.bend_engine.legal_probe.run_probe import (
    START,
    fen_position,
    parse_moves,
    parse_perft,
    request,
)


def test_fen_roundtrip_wire_and_draw_counter_independence() -> None:
    position = fen_position(START)
    assert position == (
        0x00FF00000000FF00, 0x4200000000000042, 0x2400000000000024,
        0x8100000000000081, 0x0800000000000008, 0x1000000000000010,
        0xFFFF, 0xFFFF000000000000, 1, 15, 64,
    )
    assert position == fen_position(START.replace("0 1", "150 100"))
    assert tuple(int(x, 16) for x in request(position, 3).split()) == (3, 0, 1, *position)


@pytest.mark.parametrize(("fen", "reason"), [
    ("", "six fields"),
    ("8/8/8/8/8/8/8 w - - 0 1", "eight ranks"),
    ("9/8/8/8/8/8/8/8 w - - 0 1", "piece/rank"),
    ("7/8/8/8/8/8/8/8 w - - 0 1", "incomplete FEN rank"),
    (START.replace(" w ", " x "), "turn/counters"),
    (START.replace("KQkq", "KK"), "castling rights"),
    (START.replace("KQkq", "HAha"), "castling rights"),
    (START.replace(" - ", " i6 "), "en-passant square"),
    (START.replace("0 1", "-1 1"), "turn/counters"),
    (START.replace("0 1", "0 0"), "turn/counters"),
])
def test_invalid_fen_rejected(fen: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        fen_position(fen)


def test_perft_reports_preserve_u64_and_require_divide_sum() -> None:
    total, rows = parse_perft("divide 8 16 0 0 1 4\nnodes 1 4\n", 5)
    assert total == 2**32 + 4
    assert rows == {(8, 16, 0, 0): 2**32 + 4}
    assert parse_perft("nodes 0 1\n", 0) == (1, {})
    assert parse_perft("nodes 0 0\n", 1) == (0, {})


@pytest.mark.parametrize(("text", "reason"), [
    ("", "missing perft total"),
    ("nodes 0 5\n", "divide sum"),
    ("divide 8 16 0 0 0 1\nnodes 0 2\n", "divide sum"),
    ("divide 8 16 0 0 0 1\ndivide 8 16 0 1 0 1\nnodes 0 2\n", "duplicate divide move"),
    ("divide 64 16 0 0 0 1\nnodes 0 1\n", "invalid move output"),
    ("nodes -1 0\n", "nondecimal output word"),
    ("nodes 4294967296 0\n", "output word exceeds U32"),
])
def test_bad_perft_report_rejected(text: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_perft(text, 1)


def test_perft_zero_contract() -> None:
    with pytest.raises(ValueError, match="perft\\(0\\)"):
        parse_perft("nodes 0 0\n", 0)
    with pytest.raises(ValueError, match="perft\\(0\\)"):
        parse_perft("divide 8 16 0 0 0 1\nnodes 0 1\n", 0)


def test_legal_report_preserves_child_bitboards() -> None:
    # A serialized position here is parser data, not a claim of chess legality.
    state = "0 1 0 2 0 4 0 8 0 16 0 32 0 63 2147483648 0 0 15 64"
    line = f"move 8 16 0 0 {state}\n"
    rows, end = parse_moves(line + "end\n")
    assert isinstance(rows, dict)
    assert rows[(8, 16, 0, 0)] == (1, 2, 4, 8, 16, 32, 63, 2**63, 0, 15, 64)
    assert end == "end"
    with pytest.raises(ValueError, match="duplicate"):
        parse_moves(line + line + "end\n")
    with pytest.raises(ValueError, match="terminator"):
        parse_moves(line)
    with pytest.raises(ValueError, match="position"):
        parse_moves(line.replace("0 15 64", "0 15 65") + "end\n")


@pytest.mark.parametrize("ending", ["checkmate", "stalemate", "ply_limit"])
def test_trace_terminal_record(ending: str) -> None:
    assert parse_moves(ending + "\n", trace=True) == ([], ending)
    with pytest.raises(ValueError, match="malformed"):
        parse_moves("checkmate\nply_limit\n", trace=True)
