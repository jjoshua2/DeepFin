"""UCI ``searchmoves`` support in ``StockfishUCI.search``.

Protocol-level: every test asserts on the exact command strings the driver
writes to the engine, using the ``object.__new__`` + stubbed ``_send`` /
``_readline_with_deadline`` double already established by
``tests/test_cp_to_wdl.py``. No real Stockfish is spawned.

⚑ The double ANSWERS the ``go`` line it was actually handed — the PV lines it
emits are generated from that line's ``searchmoves`` tokens. A fake that
replied with a fixed script would pass whether or not ``search`` emitted the
restriction at all, which is the one thing under test here.
"""
from __future__ import annotations

import threading
from typing import Any, cast

import chess
import pytest

from chess_anti_engine.stockfish.uci import StockfishResult, StockfishUCI

_START_FEN = chess.STARTING_FEN


class _FakeEngine:
    """Records commands and replies to ``go`` from that line's own tokens."""

    def __init__(
        self,
        *,
        multipv: int = 1,
        all_moves: tuple[str, ...] = ("e2e4", "d2d4", "g1f3", "c2c4"),
    ) -> None:
        self.commands: list[str] = []
        self.multipv = multipv
        self.all_moves = list(all_moves)
        self._pending: list[str] = []

    # -- driver-facing surface -------------------------------------------
    def send(self, cmd: str) -> None:
        self.commands.append(cmd)
        if cmd.startswith("go "):
            self._pending.extend(self._reply_to(cmd))

    def readline(self, _deadline: float) -> str:
        return self._pending.pop(0)

    # -- engine behaviour -------------------------------------------------
    def _reply_to(self, go_cmd: str) -> list[str]:
        """Answer like Stockfish: PVs for the root moves this ``go`` allows."""
        toks = go_cmd.split()
        if "searchmoves" in toks:
            # Per the UCI spec searchmoves runs to the end of the line, which
            # is exactly why this fake reads it that way: a driver that
            # appended another parameter after it would feed junk moves in
            # here rather than being quietly forgiven.
            root = toks[toks.index("searchmoves") + 1 :]
        else:
            root = list(self.all_moves)
        root = root[: self.multipv]
        lines = [
            f"info depth 8 multipv {rank} score cp {30 - 5 * rank} "
            f"wdl 500 400 100 nodes 1234 pv {mv}\n"
            for rank, mv in enumerate(root, start=1)
        ]
        lines.append(f"bestmove {root[0] if root else '0000'}\n")
        return lines

    # -- assertions helpers ------------------------------------------------
    @property
    def go_line(self) -> str:
        gos = [c for c in self.commands if c.startswith("go ")]
        assert len(gos) == 1, f"expected exactly one go command, got {gos}"
        return gos[0]


def _driver(engine: _FakeEngine, *, nodes: int = 2000) -> Any:
    sf = cast(Any, object.__new__(StockfishUCI))
    sf.nodes = nodes
    sf.multipv = engine.multipv
    sf.read_timeout_s = 1.0
    sf._lock = threading.Lock()
    sf._send = engine.send
    sf._readline_with_deadline = engine.readline
    return sf


def _search(engine: _FakeEngine, **kwargs: Any) -> StockfishResult:
    fen = kwargs.pop("fen", _START_FEN)
    return StockfishUCI.search(_driver(engine), fen, **kwargs)


def test_default_search_emits_the_pre_searchmoves_go_line() -> None:
    """Byte-identical default path.

    Mutation caught: emitting the ``searchmoves`` keyword unconditionally
    (e.g. a bare ``go nodes N searchmoves`` when the list is absent), which
    would change the command every production label query sends.
    """
    engine = _FakeEngine()
    _search(engine)

    assert engine.commands == [
        f"position fen {_START_FEN}",
        "go nodes 2000",
    ]
    assert "searchmoves" not in engine.go_line


@pytest.mark.parametrize("empty", [[], (), None])
def test_empty_or_absent_searchmoves_is_byte_identical(empty: Any) -> None:
    """``[]`` and ``None`` are both "no restriction", to the byte.

    Mutation caught: treating an empty sequence as "restrict to nothing" and
    emitting a trailing bare ``searchmoves`` keyword.
    """
    engine = _FakeEngine()
    _search(engine, searchmoves=empty)

    assert engine.go_line == "go nodes 2000"


def test_restricted_search_puts_searchmoves_last_on_the_go_line() -> None:
    """The restriction is emitted, and it is the FINAL token group.

    Mutation caught: emitting ``go searchmoves e2e4 d2d4 nodes 2000`` — legal
    to build but ruinous, since UCI's ``searchmoves`` consumes the rest of the
    line and would swallow ``nodes 2000`` as two junk moves, leaving the search
    unbounded.
    """
    engine = _FakeEngine()
    _search(engine, searchmoves=["e2e4", "d2d4"], nodes=777)

    assert engine.go_line == "go nodes 777 searchmoves e2e4 d2d4"

    toks = engine.go_line.split()
    assert toks.index("nodes") < toks.index("searchmoves")
    # Everything after the keyword is moves and nothing else.
    assert toks[toks.index("searchmoves") + 1 :] == ["e2e4", "d2d4"]


def test_promotion_suffix_is_accepted() -> None:
    """A promotion move is well-formed UCI and must survive validation.

    Mutation caught: a from/to-only regex (``[a-h][1-8][a-h][1-8]$``), which
    would reject every promotion a caller wants to compare.
    """
    engine = _FakeEngine()
    fen = "8/4P3/8/8/8/8/8/K6k w - - 0 1"
    _search(engine, fen=fen, searchmoves=["e7e8q", "e7e8n"])

    assert engine.go_line == "go nodes 2000 searchmoves e7e8q e7e8n"


@pytest.mark.parametrize(
    "bad",
    [
        "e9e4",        # off-board rank
        "z2z4",        # off-board file
        "e2e4k",       # king is not a promotion piece
        "E2E4",        # uppercase is not the UCI wire form
        "0000",        # the null move: "restrict to nothing" is not a thing
        "e2e4 d2d4",   # two moves in one token
        "e2e4\nstop",  # newline injection into the go line
        "",            # empty token
    ],
)
def test_malformed_move_raises_naming_the_token(bad: str) -> None:
    """Malformed input is rejected loudly, never dropped.

    Mutation caught: filtering bad tokens out of the list instead of raising.
    A dropped move silently widens the search back toward the full move list —
    with every token dropped, the caller gets a full-width search and believes
    it got a restricted one.

    Also pins that no command reaches the engine and the engine is NOT poisoned
    by a caller-side typo: validation runs before the protocol section, so a
    bad argument costs a ``ValueError``, not a Stockfish restart.
    """
    engine = _FakeEngine()
    sf = _driver(engine)

    with pytest.raises(ValueError, match="malformed UCI move") as exc:
        StockfishUCI.search(sf, _START_FEN, searchmoves=["e2e4", bad])

    assert repr(bad) in str(exc.value)
    assert engine.commands == []
    assert sf.desynced is False


def test_illegal_move_for_the_position_raises() -> None:
    """Well-formed but not legal here: still rejected.

    Mutation caught: dropping the legality check. Stockfish silently ignores a
    root move that is illegal in the position, so a list of only-illegal moves
    would produce a full-width search with no error anywhere.
    """
    engine = _FakeEngine()

    with pytest.raises(ValueError, match="not legal in this position") as exc:
        _search(engine, searchmoves=["e2e4", "e2e5"])

    assert "'e2e5'" in str(exc.value)
    assert engine.commands == []


def test_legality_is_checked_against_the_given_fen_not_the_start_position() -> None:
    """The check uses the FEN under search.

    Mutation caught: validating against a hardcoded/starting position, which
    would pass ``e2e4`` everywhere and reject legal moves elsewhere.
    """
    engine = _FakeEngine()
    # Black to move: e2e4 is not black's move, e7e5 is.
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"

    _search(engine, fen=fen, searchmoves=["e7e5"])
    assert engine.go_line == "go nodes 2000 searchmoves e7e5"

    with pytest.raises(ValueError, match="not legal in this position"):
        _search(_FakeEngine(), fen=fen, searchmoves=["e2e4"])


def test_unreadable_fen_skips_legality_but_keeps_syntax() -> None:
    """The engine is the authority on the FEN; the driver is not.

    The two parsers genuinely disagree — python-chess raises on the
    non-numeric half-move clock below, Stockfish reads it as 0 and searches —
    so a position we cannot parse must not become unsearchable the moment a
    caller restricts the root. The syntax check still has to hold, because that
    one guards the ``go`` line rather than the chess.

    Mutation caught: letting the ``chess.Board`` ValueError escape (which would
    turn every such search into a hard failure), and separately, returning
    early past the syntax check when the FEN is unreadable.
    """
    weird_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - x 1"

    engine = _FakeEngine()
    _search(engine, fen=weird_fen, searchmoves=["e2e4", "a1a8"])
    assert engine.go_line == "go nodes 2000 searchmoves e2e4 a1a8"

    with pytest.raises(ValueError, match="malformed UCI move"):
        _search(_FakeEngine(), fen=weird_fen, searchmoves=["nonsense"])


def test_multipv3_restricted_to_two_moves_parses_without_error() -> None:
    """Fewer PV lines than MultiPV is normal under a root restriction.

    With ``searchmoves`` limiting the root to k moves, Stockfish returns at
    most k PV lines regardless of the configured MultiPV. The accumulator keys
    PVs by the rank the engine reports and never assumes MultiPV of them.

    Mutations caught: (1) ``_SearchInfoAccumulator.result`` iterating a fixed
    ``range(1, multipv + 1)`` instead of the ranks actually seen — a KeyError
    on every restricted search; (2) ``search`` dropping the restriction from
    the ``go`` line, which makes this fake engine answer with all 3 PVs.
    """
    engine = _FakeEngine(multipv=3)
    res = _search(engine, searchmoves=["d2d4", "c2c4"])

    assert [pv.move_uci for pv in res.pvs] == ["d2d4", "c2c4"]
    assert res.bestmove_uci == "d2d4"
    assert res.cp == 25
    assert res.nodes == 1234
    assert res.depth == 8
    assert res.wdl is not None


def test_unrestricted_multipv3_still_returns_three_pvs() -> None:
    """Control for the test above: the shortfall comes from the restriction.

    Mutation caught: a fake (or a driver) that always returns 2 PVs, which
    would make the MultiPV-3 test pass for the wrong reason.
    """
    engine = _FakeEngine(multipv=3)
    res = _search(engine)

    assert [pv.move_uci for pv in res.pvs] == ["e2e4", "d2d4", "g1f3"]
