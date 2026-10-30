"""Seed-shift controls must score the same lost terminals as resolution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.selfplay.opening import seed_board_from_line
from scripts import blindspot_resolution as resolution
from tests.script_loading import load_script_module

seed_shift = load_script_module("../scratchpad/seed_shift_control.py", "seed_shift_control_test")
BACKED = chess.STARTING_FEN + " | e2e4 # dropped=e7e5,g1f3 provenance=fixture"
BARE = chess.STARTING_FEN


def test_loader_preserves_reconstructed_terminal_and_history(tmp_path: Path) -> None:
    path = tmp_path / "seeds.txt"
    path.write_text(BACKED + "\n", encoding="utf-8")
    actual = seed_shift.load_seed_lines(str(path), 1)
    assert actual == resolution.load_seed_lines(str(path)) == [BACKED]
    board = seed_board_from_line(resolution.reconstruct_lost_line(actual[0]))
    assert [move.uci() for move in board.move_stack] == ["e2e4", "e7e5", "g1f3"]
    expected = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        expected.push_uci(uci)
    assert board.fen() == expected.fen()


@pytest.mark.parametrize("limit", [1, 2, 10])
def test_limit_counts_usable_seeds_after_canonical_validation(tmp_path: Path, capsys: pytest.CaptureFixture[str], limit: int) -> None:
    path = tmp_path / "seeds.txt"
    invalid_dropped = chess.STARTING_FEN + " | e2e4 # dropped=e2e5"
    path.write_text("# comment only\n\nnot-a-fen\n" + invalid_dropped + "\n" + BACKED + "\n" + BARE + "\n", encoding="utf-8")
    actual = seed_shift.load_seed_lines(str(path), limit)
    assert actual == [BACKED, BARE][:limit]
    err = capsys.readouterr().err
    assert "skipping" in err
    assert "bad dropped= history" in err


@pytest.mark.parametrize("contents", ["", "# comment only\n\n", "not-a-fen\n"])
def test_no_usable_seeds_fail_before_scoring(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="no usable seeds"):
        seed_shift.load_seed_lines(str(path), 2)


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_seed_limit_is_rejected(tmp_path: Path, limit: int) -> None:
    with pytest.raises(ValueError, match="seed limit must be positive"):
        seed_shift.load_seed_lines(str(tmp_path / "not-read.txt"), limit)


def test_cli_preserves_paired_seed_identity_and_separate_control_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seeds = tmp_path / "seeds.txt"
    seeds.write_text("not-a-fen\n" + BACKED + "\n" + BARE + "\n", encoding="utf-8")
    board = chess.Board()
    board.push_uci("d2d4")
    control_fens = [board.fen(), chess.STARTING_FEN]
    dump = tmp_path / "control.jsonl"
    dump.write_text("\n".join(json.dumps({"fen": fen}) for fen in control_fens) + "\n", encoding="utf-8")
    calls: list[tuple[str, list[str]]] = []

    def fake_score(checkpoint: str, seed_lines: list[str], **kwargs: Any) -> np.ndarray:
        assert kwargs["device"] == "cpu"
        calls.append((checkpoint, list(seed_lines)))
        return np.arange(len(seed_lines), dtype=np.float64) / 10 + (0.1 if checkpoint == "post" else 0.0)

    monkeypatch.setattr(resolution, "score_seeds", fake_score)
    monkeypatch.setattr("sys.argv", [
        "seed_shift_control.py", "--pre", "pre", "--post", "post", "--seeds", str(seeds),
        "--control-dump", str(dump), "--n", "2", "--device", "cpu",
    ])
    seed_shift.main()
    assert calls == [
        ("pre", [BACKED, BARE]), ("post", [BACKED, BARE]),
        ("pre", control_fens), ("post", control_fens),
    ]
    assert "seeds=2  control=2" in capsys.readouterr().out


def test_bare_seed_limit_compatibility(tmp_path: Path) -> None:
    path = tmp_path / "bare.txt"
    board = chess.Board()
    board.push_uci("d2d4")
    path.write_text(BARE + "\n" + board.fen() + "\n", encoding="utf-8")
    assert seed_shift.load_seed_lines(str(path), 1) == [BARE]
    assert seed_shift.load_seed_lines(str(path), 2) == [BARE, board.fen()]


def test_control_fens_keep_their_separate_limit_contract(tmp_path: Path) -> None:
    path = tmp_path / "control.jsonl"
    board = chess.Board()
    board.push_uci("d2d4")
    path.write_text("\n".join(json.dumps(row) for row in [{}, {"fen": BARE}, {"fen": board.fen()}]) + "\n", encoding="utf-8")
    assert seed_shift.load_control_fens(str(path), 1) == [BARE]
    assert seed_shift.load_control_fens(str(path), 2) == [BARE, board.fen()]
