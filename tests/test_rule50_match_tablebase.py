"""Fake-tablebase checks for opt-in claim-aware match decisions and search leaves."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, cast

import chess
import chess.syzygy
import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel_c import _tb_override
from chess_anti_engine import tablebase as tbmod


def board_at(clock: int = 0, *, turn: bool = chess.WHITE) -> chess.Board:
    color = "w" if turn else "b"
    return chess.Board(f"7k/8/8/8/8/8/8/KQ6 {color} - - {clock} 1")


class FakeProbe:
    def __init__(self, wdl: int, dtz: int) -> None:
        self.wdl_value = wdl
        self.dtz_value = dtz
        self.wdl = {"KQvK": object()}
        self.dtz = {"KQvK": object()}
        self.calls: list[str] = []

    def probe_wdl(self, _board: chess.Board) -> int:
        self.calls.append("wdl")
        return self.wdl_value

    def probe_dtz(self, _board: chess.Board) -> int:
        self.calls.append("dtz")
        return self.dtz_value


def typed_probe(probe: FakeProbe) -> chess.syzygy.Tablebase:
    """Give a duck-typed fake the production handle type for static checking."""
    return cast(chess.syzygy.Tablebase, cast(object, probe))


@pytest.mark.parametrize(
    ("wdl", "dtz", "status"),
    [(2, 1, 1), (1, 101, 2), (0, 0, 2), (-1, -101, 2), (-2, -1, -1)],
)
@pytest.mark.parametrize("turn", [chess.WHITE, chess.BLACK])
def test_fresh_clock_wdl_and_colors(wdl: int, dtz: int, status: int, turn: bool) -> None:
    board = board_at(turn=turn)
    probe = FakeProbe(wdl, dtz)
    assert tbmod.rule50_match_status(board, typed_probe(probe)) == status
    expected = (
        "1/2-1/2" if status == 2 else
        "1-0" if (status == 1) == (turn == chess.WHITE) else "0-1"
    )
    assert tbmod.rule50_match_result(board, typed_probe(probe)) == expected
    assert probe.calls == ["wdl", "dtz", "wdl", "dtz"]


@pytest.mark.parametrize(("wdl", "dtz"), [(2, 3), (-2, -3)])
def test_positive_clock_is_unresolved_even_with_short_dtz(wdl: int, dtz: int) -> None:
    assert tbmod.rule50_match_status(board_at(0), typed_probe(FakeProbe(wdl, dtz))) == (1 if wdl > 0 else -1)
    assert tbmod.rule50_match_status(board_at(1), typed_probe(FakeProbe(wdl, dtz))) is None
    assert tbmod.rule50_match_status(board_at(95), typed_probe(FakeProbe(wdl, dtz))) is None
    assert tbmod.rule50_match_status(board_at(96), typed_probe(FakeProbe(wdl, dtz))) is None
    assert tbmod.rule50_match_status(board_at(98), typed_probe(FakeProbe(wdl, dtz))) is None


@pytest.mark.parametrize(("wdl", "dtz"), [(2, 101), (-2, -101), (1, 100), (-1, -100)])
def test_wdl_dtz_magnitude_class_must_agree(wdl: int, dtz: int) -> None:
    with pytest.raises(tbmod.MatchTablebaseError, match="magnitude mismatch"):
        tbmod.rule50_match_status(board_at(), typed_probe(FakeProbe(wdl, dtz)))


def test_natural_claimable_terminal_precedes_tablebase() -> None:
    mate = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    probe = FakeProbe(-2, -1)
    assert tbmod.rule50_match_result(mate, typed_probe(probe)) == "1-0"
    assert tbmod.rule50_match_result(board_at(100), typed_probe(probe)) == "1/2-1/2"
    repeated = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"):
        repeated.push_san(san)
    assert repeated.can_claim_threefold_repetition()
    assert tbmod.rule50_match_result(repeated, typed_probe(probe)) == "1/2-1/2"
    assert probe.calls == []


def test_ineligible_is_unknown_but_missing_eligible_probes_raise() -> None:
    probe = FakeProbe(2, 1)
    assert tbmod.rule50_match_status(chess.Board(), typed_probe(probe)) is None
    assert probe.calls == []
    class MissingWdl(FakeProbe):
        def probe_wdl(self, _board: chess.Board) -> int:
            raise chess.syzygy.MissingTableError("missing WDL")
    class MissingDtz(FakeProbe):
        def probe_dtz(self, _board: chess.Board) -> int:
            raise chess.syzygy.MissingTableError("missing DTZ")
    with pytest.raises(tbmod.MatchTablebaseError, match="missing eligible"):
        tbmod.rule50_match_status(board_at(), typed_probe(MissingWdl(2, 1)))
    with pytest.raises(tbmod.MatchTablebaseError, match="missing eligible"):
        tbmod.rule50_match_status(board_at(), typed_probe(MissingDtz(2, 1)))


def test_theoretical_label_mapping_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = FakeProbe(1, 101)
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: probe)
    assert tbmod.tb_adjudicate_result(board_at(), "fake") == "1-0"
    assert tbmod.rule50_match_result(board_at(), typed_probe(probe)) == "1/2-1/2"


class FakeOpenTablebase:
    instances: ClassVar[list[FakeOpenTablebase]] = []

    def __init__(self) -> None:
        self.wdl: dict[str, object] = {}
        self.dtz: dict[str, object] = {}
        self.closed = False
        self.instances.append(self)

    def add_directory(self, path: str) -> None:
        name = Path(path).name
        if name == "broken":
            raise OSError("broken")
        if name in ("wdl", "all"):
            self.wdl["KQRvKBN"] = object()
        if name in ("dtz", "all"):
            self.dtz["KQRvKBN"] = object()

    def close(self) -> None:
        self.closed = True


def test_strict_open_capacity_components_and_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeOpenTablebase.instances.clear()
    monkeypatch.setattr(tbmod.chess.syzygy, "Tablebase", FakeOpenTablebase)
    wdl, dtz, empty, broken = (tmp_path / name for name in ("wdl", "dtz", "empty", "broken"))
    for directory in (wdl, dtz, empty, broken):
        directory.mkdir()
    opened = tbmod.open_strict_match_tablebase(f"{wdl}{os.pathsep}{dtz}")
    assert isinstance(opened, FakeOpenTablebase)
    assert opened.wdl
    assert opened.dtz
    assert not opened.closed
    opened.close()
    for parts, problem in (
        (f"{wdl}{os.pathsep}{empty}", "no new tables"),
        (f"{wdl}{os.pathsep}{broken}", "could not open"),
        (str(wdl), "capacity below"),
        (f"{wdl}{os.pathsep}{wdl}", "duplicate"),
        (f"{wdl}{os.pathsep}{dtz}{os.pathsep}", "empty or padded"),
    ):
        before = len(FakeOpenTablebase.instances)
        with pytest.raises(tbmod.MatchTablebaseError, match=problem):
            tbmod.open_strict_match_tablebase(parts)
        for handle in FakeOpenTablebase.instances[before:]:
            assert handle.closed
    alias = tmp_path / "alias"
    alias.symlink_to(wdl, target_is_directory=True)
    with pytest.raises(tbmod.MatchTablebaseError, match="duplicate"):
        tbmod.open_strict_match_tablebase(f"{wdl}{os.pathsep}{alias}")


class FakeCBoard:
    def __init__(self, board: chess.Board) -> None:
        self.board = board
        self.occ_white = board.occupied_co[chess.WHITE]
        self.occ_black = board.occupied_co[chess.BLACK]
        self.castling = 0

    def fen(self) -> str:
        return self.board.fen()


class FakeTree:
    def __init__(self, cboard: FakeCBoard) -> None:
        self.cboard = cboard
        self.marked: tuple[np.ndarray, np.ndarray] | None = None

    def get_pending_tb_leaves(self, _max_pieces: int) -> tuple[np.ndarray, list[FakeCBoard]]:
        return np.array([0], dtype=np.int32), [self.cboard]

    def mark_tb_solved(self, indices: np.ndarray, statuses: np.ndarray) -> None:
        self.marked = indices.copy(), statuses.copy()


def test_seven_to_six_leaf_decisive_then_boundary_unknown() -> None:
    root = chess.Board("6nk/1b5p/8/8/8/8/8/KQR5 w - - 95 1")
    assert chess.popcount(root.occupied) == 7
    child = root.copy()
    child.push(chess.Move.from_uci("b1b7"))
    assert chess.popcount(child.occupied) == 6
    assert child.halfmove_clock == 0  # capture zeroes the clock
    boundary = chess.Board("6nk/7p/8/8/8/8/8/KQR5 b - - 95 1")
    assert chess.popcount(boundary.occupied) == 6
    for leaf, expected_status in ((child, 1), (boundary, 0)):
        fake_tb = FakeProbe(2, 3)
        fake_tb.wdl = {"KQRvKBN": object()}
        fake_tb.dtz = {"KQRvKBN": object()}
        probe = tbmod.SyzygyProbe(
            "fake", max_pieces=6, rule50_aware=True, tablebase=typed_probe(fake_tb),
        )
        tree = FakeTree(FakeCBoard(leaf))
        logits = np.zeros((1, 3), dtype=np.float32)
        _tb_override(cast(Any, tree), probe, logits)
        assert (tree.marked is not None) == bool(expected_status)
        if expected_status:
            assert tree.marked is not None
            assert tree.marked[1].tolist() == [expected_status]
            assert logits[0, 0] > logits[0, 1]
        else:
            assert logits.tolist() == [[0.0, 0.0, 0.0]]


def test_rule50_probe_requires_explicit_tablebase() -> None:
    with pytest.raises(tbmod.MatchTablebaseError, match="explicitly opened"):
        tbmod.SyzygyProbe("fake", max_pieces=6, rule50_aware=True)
    with pytest.raises(ValueError, match="cursed_as_draw"):
        tbmod.SyzygyProbe("fake", max_pieces=6, rule50_aware=True, cursed_as_draw=False)
    with pytest.raises(TypeError, match="rule50_aware"):
        tbmod.SyzygyProbe("fake", max_pieces=6, rule50_aware=cast(Any, 1))


def test_rule50_search_probe_raises_on_missing_eligible_dtz() -> None:
    class MissingDtz(FakeProbe):
        def probe_dtz(self, _board: chess.Board) -> int:
            raise chess.syzygy.MissingTableError("missing DTZ")

    fake_tb = MissingDtz(2, 1)
    fake_tb.wdl = {"KQRvKBN": object()}
    fake_tb.dtz = {"KQRvKBN": object()}
    probe = tbmod.SyzygyProbe("fake", max_pieces=6, rule50_aware=True, tablebase=typed_probe(fake_tb))
    leaf = FakeCBoard(chess.Board("6nk/7p/8/8/8/8/8/KQR5 b - - 0 1"))
    logits = np.zeros((1, 3), dtype=np.float32)
    solved = np.zeros(1, dtype=np.int8)
    with pytest.raises(tbmod.MatchTablebaseError, match="missing eligible"):
        probe.apply(cast(Any, [leaf]), logits, solved_out=solved)
    assert solved.tolist() == [0]
    assert logits.tolist() == [[0.0, 0.0, 0.0]]
