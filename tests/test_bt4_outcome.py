"""Small CPU-only cases for the opt-in BT4 root outcome policy."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, cast

import chess
import chess.syzygy
import pytest

from chess_anti_engine import tablebase
from chess_anti_engine.selfplay.bt4_outcome import (
    BT4OutcomeDecision,
    decide_bt4_outcome,
    preflight_six_man_tablebases,
)
from chess_anti_engine.utils.syzygy import SEPARATOR

_SIX_MAN_FEN = "7k/8/8/8/8/8/8/KQBNR3 {turn} - - {clock} 1"


class FakeTablebase:
    def __init__(self, raw_wdl: int | None) -> None:
        self.raw_wdl = raw_wdl
        self.probes = 0

    def probe_wdl(self, board: chess.Board) -> int:
        self.probes += 1
        if self.raw_wdl is None:
            raise chess.syzygy.MissingTableError(board.fen())
        return self.raw_wdl


class FakeMatchTablebase:
    def __init__(self, raw_wdl: int | None, dtz: int | None) -> None:
        self.raw_wdl = raw_wdl
        self.raw_dtz = dtz
        self.wdl = {"KQBNRvK": object()}
        self.dtz = {"KQBNRvK": object()}
        self.wdl_probes = 0
        self.dtz_probes = 0

    def probe_wdl(self, board: chess.Board) -> int:
        self.wdl_probes += 1
        if self.raw_wdl is None:
            raise chess.syzygy.MissingTableError(board.fen())
        return self.raw_wdl

    def probe_dtz(self, board: chess.Board) -> int:
        self.dtz_probes += 1
        if self.raw_dtz is None:
            raise chess.syzygy.MissingTableError(board.fen())
        return self.raw_dtz


def _decision(
    board: chess.Board, *, plies: int = 0, max_plies: int = 8
) -> BT4OutcomeDecision | None:
    return decide_bt4_outcome(
        board, plies=plies, max_plies=max_plies, syzygy_path="fake:pair",
        outcome_mode="theoretical_wdl",
    )


@pytest.mark.parametrize("turn", ["w", "b"])
@pytest.mark.parametrize("raw_wdl", [-2, -1, 0, 1, 2])
def test_six_man_training_verdict_uses_theoretical_wdl_for_both_colors(
    monkeypatch: pytest.MonkeyPatch, turn: str, raw_wdl: int
) -> None:
    fake = FakeTablebase(raw_wdl)
    monkeypatch.setattr(tablebase, "get_tablebase", lambda _path: fake)
    board = chess.Board(_SIX_MAN_FEN.format(turn=turn, clock=0))
    assert board.is_valid()

    decision = _decision(board, plies=8, max_plies=8)
    assert decision is not None
    assert decision.termination == "syzygy", "Syzygy precedes the ply cap"
    assert decision.detail == "theoretical_wdl"
    if raw_wdl == 0:
        expected = "1/2-1/2"
    elif (raw_wdl > 0) == (turn == "w"):
        expected = "1-0"
    else:
        expected = "0-1"
    assert decision.result == expected
    assert fake.probes == 1


@pytest.mark.parametrize("clock", [99, 100])
def test_claimable_fifty_moves_precedes_tablebase_at_six_men(
    monkeypatch: pytest.MonkeyPatch, clock: int
) -> None:
    fake = FakeTablebase(2)
    monkeypatch.setattr(tablebase, "get_tablebase", lambda _path: fake)
    board = chess.Board(_SIX_MAN_FEN.format(turn="w", clock=clock))
    decision = _decision(board)
    assert decision == BT4OutcomeDecision("1/2-1/2", "natural", "fifty_moves")
    assert fake.probes == 0


def test_impending_threefold_precedes_tablebase_and_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = chess.Board(_SIX_MAN_FEN.format(turn="w", clock=0))
    for uci in (
        "e1e2", "h8g8", "e2e1", "g8h8",
        "e1e2", "h8g8", "e2e1",
    ):
        board.push_uci(uci)
    fake = FakeTablebase(2)
    monkeypatch.setattr(tablebase, "get_tablebase", lambda _path: fake)

    decision = _decision(board, plies=7, max_plies=7)
    assert decision == BT4OutcomeDecision(
        "1/2-1/2", "natural", "threefold_repetition"
    )
    assert fake.probes == 0


def test_checkmate_precedes_fifty_move_claim() -> None:
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(uci)
    board.halfmove_clock = 100
    decision = _decision(board, plies=4, max_plies=4)
    assert decision == BT4OutcomeDecision("0-1", "natural", "checkmate")


def test_missing_material_is_discarded_without_a_draw_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTablebase(None)
    monkeypatch.setattr(tablebase, "get_tablebase", lambda _path: fake)
    board = chess.Board(_SIX_MAN_FEN.format(turn="w", clock=0))
    decision = _decision(board, plies=8, max_plies=8)
    assert decision == BT4OutcomeDecision(
        None, "tablebase_unavailable", "missing_material"
    )
    assert fake.probes == 1


def test_six_man_castling_rights_are_unavailable_without_a_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTablebase(2)
    monkeypatch.setattr(tablebase, "get_tablebase", lambda _path: fake)
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert board.is_valid()
    decision = _decision(board)
    assert decision == BT4OutcomeDecision(
        None, "tablebase_unavailable", "castling_rights"
    )
    assert fake.probes == 0


def test_above_six_man_cap_is_unresolved_not_draw() -> None:
    decision = _decision(chess.Board(), plies=8, max_plies=8)
    assert decision == BT4OutcomeDecision(
        None, "max_plies_unresolved", "above_six_man"
    )
    assert _decision(chess.Board(), plies=7, max_plies=8) is None


def test_pair_preflight_requires_both_components_and_wdl_dtz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wdl = tmp_path / "wdl"
    dtz = tmp_path / "dtz"
    wdl.mkdir()
    dtz.mkdir()
    (wdl / "KQvK.rtbw").touch()
    path = SEPARATOR.join((str(wdl), str(dtz)))
    with pytest.raises(ValueError, match="tablebase-BLIND"):
        preflight_six_man_tablebases(path)
    (dtz / "KQvK.rtbz").touch()

    class FakeOpened:
        wdl: ClassVar[dict[str, object]] = {"KQvK": object()}
        dtz: ClassVar[dict[str, object]] = {"KQvK": object()}

    monkeypatch.setattr(tablebase, "get_tablebase", lambda _path: FakeOpened())
    assert preflight_six_man_tablebases(path) == (str(wdl), str(dtz))
    with pytest.raises(ValueError, match="both Syzygy paths"):
        preflight_six_man_tablebases(str(wdl))
    with pytest.raises(ValueError, match="two distinct Syzygy paths"):
        preflight_six_man_tablebases(SEPARATOR.join((str(wdl), str(wdl))))
    alias = tmp_path / "wdl_alias"
    alias.symlink_to(wdl, target_is_directory=True)
    with pytest.raises(ValueError, match="two distinct Syzygy paths"):
        preflight_six_man_tablebases(SEPARATOR.join((str(wdl), str(alias))))
    with pytest.raises(ValueError, match="both Syzygy paths"):
        preflight_six_man_tablebases(SEPARATOR.join((f" {wdl}", str(dtz))))

    FakeOpened.dtz = {}
    with pytest.raises(ValueError, match="WDL and DTZ"):
        preflight_six_man_tablebases(path)
    FakeOpened.dtz = {"KQvK": object()}
    monkeypatch.setattr(
        tablebase, "get_tablebase",
        lambda candidate: None if candidate == str(dtz) else FakeOpened(),
    )
    with pytest.raises(ValueError, match="did not open"):
        preflight_six_man_tablebases(path)


@pytest.mark.parametrize(
    ("wdl", "dtz", "clock", "expected_result", "expected_termination"),
    [
        (2, 1, 0, "1-0", "syzygy"),
        (-2, -1, 0, "0-1", "syzygy"),
        (1, 101, 0, "1/2-1/2", "syzygy"),
        (-1, -101, 0, "1/2-1/2", "syzygy"),
        (0, 0, 0, "1/2-1/2", "syzygy"),
        (2, 1, 1, None, "rule50_unresolved"),
        (-2, -1, 1, None, "rule50_unresolved"),
    ],
)
def test_explicit_rule50_mode_uses_wdl_dtz_and_discards_ambiguous_clock(
    wdl: int, dtz: int, clock: int,
    expected_result: str | None, expected_termination: str,
) -> None:
    board = chess.Board(_SIX_MAN_FEN.format(turn="w", clock=clock))
    fake = FakeMatchTablebase(wdl, dtz)
    decision = decide_bt4_outcome(
        board, plies=8, max_plies=8, syzygy_path="fake:pair",
        outcome_mode="rule50_match_v1",
        match_tablebase=cast(chess.syzygy.Tablebase, cast(object, fake)),
    )
    assert decision is not None
    assert (decision.result, decision.termination, decision.outcome_mode) == (
        expected_result, expected_termination, "rule50_match_v1",
    )
    assert (fake.wdl_probes, fake.dtz_probes) == (1, 1)


@pytest.mark.parametrize(("wdl", "dtz"), [(None, 1), (2, None)])
def test_rule50_missing_eligible_component_fails_instead_of_discarding(
    wdl: int | None, dtz: int | None,
) -> None:
    board = chess.Board(_SIX_MAN_FEN.format(turn="w", clock=0))
    fake = FakeMatchTablebase(wdl, dtz)
    with pytest.raises(tablebase.MatchTablebaseError, match="missing eligible"):
        decide_bt4_outcome(
            board, plies=0, max_plies=8, syzygy_path="fake:pair",
            outcome_mode="rule50_match_v1",
            match_tablebase=cast(chess.syzygy.Tablebase, cast(object, fake)),
        )


def test_rule50_natural_claim_precedes_wdl_and_dtz_probe() -> None:
    board = chess.Board(_SIX_MAN_FEN.format(turn="w", clock=99))
    fake = FakeMatchTablebase(None, None)
    decision = decide_bt4_outcome(
        board, plies=0, max_plies=8, syzygy_path="fake:pair",
        outcome_mode="rule50_match_v1",
        match_tablebase=cast(chess.syzygy.Tablebase, cast(object, fake)),
    )
    assert decision == BT4OutcomeDecision(
        "1/2-1/2", "natural", "fifty_moves", "rule50_match_v1",
    )
    assert (fake.wdl_probes, fake.dtz_probes) == (0, 0)
