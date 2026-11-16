"""The UCI search's leaf / root-eval Syzygy probe runs on the C prober (``uci/tb_leaf.py``).

Two tiers, like ``tests/test_uci_tb_root_c.py``:

* table-free tests run everywhere (CI included). They stand in for the C
  module (or use the real one with no tables) and check the wiring: the
  engine installs ``CLeafProbe`` on the C table set and never opens
  python-chess, ``CLeafProbe.apply`` writes exactly what ``SyzygyProbe.apply``
  writes for the same answers (eligibility, value, solved status, counters),
  and a probe whose table set was replaced answers nothing and is dropped;
* table tests need real Syzygy files and SKIP with the reason printed where
  there are none (CI): a leaf differential of ``CLeafProbe.apply`` against
  ``SyzygyProbe.apply`` (python-chess, the test-only reference) over several
  thousand positions, search-level identity of a fixed-seed search with each
  probe, a threaded generation-guard stress, and a real UCI subprocess whose
  python-chess Syzygy entry points are all instrumented.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import textwrap
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import chess
import chess.syzygy
import numpy as np
import pytest

import chess_anti_engine.tablebase as tbmod
import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.tb import _tb_ext
from chess_anti_engine.uci import tb_leaf, tb_root
from chess_anti_engine.uci.protocol import CmdSetOption, parse_command
from chess_anti_engine.uci.tb_leaf import CLeafProbe
from scripts.tb_prober_equivalence import (
    gen_castling,
    gen_edge_king,
    gen_ep,
    gen_random,
    material_classes,
    terminal_children,
)
from chess_anti_engine.utils.syzygy import missing_tablebase_dirs
from tests.test_uci_compliance_p2 import _go, _Out
from tests.test_uci_tb_root_c import (
    _KRK_FEN,
    _TABLES,
    REPO_ROOT,
    _c_rows,
    _engine,
    _FakeCExt,
    _read_until,
    _RecordingPyTable,
    needs_tables,
)


@pytest.fixture
def out(monkeypatch) -> _Out:
    capture = _Out()
    monkeypatch.setattr(engine_module, "sys", SimpleNamespace(stdout=capture, stderr=sys.stderr))
    return capture


@pytest.fixture(autouse=True)
def _no_c_tables_left_behind() -> Iterator[None]:
    """The C table set is process-global: start and end every test without one."""
    tb_root.unload()
    yield
    tb_root.unload()


@pytest.fixture
def fake_c(monkeypatch) -> _FakeCExt:
    fake = _FakeCExt(_c_rows)
    monkeypatch.setattr(tb_root, "_tb_ext", fake)
    monkeypatch.setattr(tb_leaf, "_tb_ext", fake)
    return fake


@pytest.fixture
def py_table(monkeypatch) -> _RecordingPyTable:
    table = _RecordingPyTable()
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: table)
    return table


@pytest.fixture(scope="module")
def tables() -> str:
    assert _TABLES is not None
    missing = missing_tablebase_dirs(_TABLES)
    if missing:
        pytest.fail(f"CAE_SYZYGY_PATH={_TABLES!r} is set but {missing} hold no tables")
    return _TABLES


_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "syzygy_3man"


def _loaded(largest_wdl: int, *, c_generation: int = 7) -> tb_root.LoadedTables:
    return tb_root.LoadedTables(
        configured="stub", resolved="/stub", largest=largest_wdl, n_wdl=2, n_dtz=2,
        largest_dtz=largest_wdl, generation=1, c_generation=c_generation,
        largest_wdl=largest_wdl,
    )


# ------------------------------------------------------- table-free stand-ins

_VALUES = (-2, -1, 0, 1, 2)


def _scripted_value(fen: str) -> int | None:
    """Deterministic 'table' shared by both stand-ins: None for ~1 in 6 FENs
    (a missing table), else a WDL in -2..2 (cursed/blessed included)."""
    h = int.from_bytes(hashlib.sha256(fen.encode()).digest()[:8], "little")
    if h % 6 == 0:
        return None
    return _VALUES[h % 5]


class _ScriptedPyTable:
    """python-chess stand-in (``get_tablebase``) answering ``_scripted_value``,
    raising exactly where python-chess raises."""

    wdl: ClassVar[dict[str, None]] = {"KQRvKR": None, "KRvK": None}
    dtz: ClassVar[dict[str, None]] = {"KQRvKR": None, "KRvK": None}

    def __init__(self) -> None:
        self.calls: list[str] = []

    def probe_wdl(self, board: chess.Board) -> int:
        self.calls.append(board.fen())
        if board.castling_rights:
            raise KeyError("syzygy tables do not contain positions with castling rights")
        value = _scripted_value(board.fen())
        if value is None:
            raise chess.syzygy.MissingTableError("did not find wdl table")
        return value


class _ScriptedC:
    """``_tb_ext`` stand-in for ``tb_leaf``: the same answers as C codes."""

    def __init__(self, generation: int = 7) -> None:
        self._generation = generation
        self.batches: list[list[str]] = []

    def generation(self) -> int:
        return self._generation

    def probe_wdl_batch(self, boards, generation: int) -> bytes | None:
        if generation != self._generation:
            return None
        fens = [cb.fen() for cb in boards]
        self.batches.append(fens)
        codes = bytearray()
        for fen in fens:
            board = chess.Board(fen)
            value = None if board.castling_rights else _scripted_value(fen)
            codes.append(5 if value is None else value + 2)
        return bytes(codes)


def _stub_leaves(seed: int, n: int) -> list[CBoard]:
    """CBoards of 4-, 5- and 6-man boards (max_pieces 5 below), some with
    castling rights, at assorted halfmove clocks."""
    rng = random.Random(seed)
    leaves: list[CBoard] = []
    while len(leaves) < n:
        kind = rng.random()
        if kind < 0.15:
            board = gen_castling(rng, "KRRvKQ")
        else:
            board = gen_random(rng, rng.choice(("KRvKP", "KQRvKR", "KQRvKRN")))
        if board is None:
            continue
        board.halfmove_clock = rng.choice((0, 50, 99, 100, 150))
        leaves.append(CBoard.from_board(board))
    return leaves


def _apply_both(
    py: tbmod.SyzygyProbe, c: CLeafProbe, leaves: list[CBoard], *, with_indices: bool,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray, int]]:
    rng = np.random.default_rng(seed)
    net = rng.normal(size=(len(leaves) + 3, 3)).astype(np.float32)
    results = []
    for probe in (py, c):
        wdl = net.copy()
        solved = np.zeros(len(leaves), dtype=np.int8)
        if with_indices:
            # Rows are deliberately NOT 0..n-1: indices[j] is the row of leaf j.
            idx = (np.arange(len(leaves), dtype=np.int32) + 3)[::-1].copy()
            hits = probe.apply(leaves, wdl, indices=idx, solved_out=solved)
        else:
            hits = probe.apply(leaves, wdl, solved_out=solved)
        results.append((wdl, solved, hits))
    return results[0], results[1]


@pytest.mark.parametrize("cursed_as_draw", [True, False])
@pytest.mark.parametrize("with_indices", [False, True])
def test_c_leaf_probe_writes_what_syzygy_probe_writes(
    monkeypatch, cursed_as_draw: bool, with_indices: bool,
) -> None:
    """Same answers in, same leaf rows, solved statuses, hits and probes out.

    The table is scripted (CI has no Syzygy files): every WDL -2..2, missing
    tables, castling rights, too many men, both ``Syzygy50MoveRule`` settings,
    with and without pre-filtered ``indices``.
    """
    table = _ScriptedPyTable()
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: table)
    stub = _ScriptedC()
    monkeypatch.setattr(tb_leaf, "_tb_ext", stub)
    py = tbmod.SyzygyProbe("stub", cursed_as_draw=cursed_as_draw)
    c = CLeafProbe(_loaded(5), cursed_as_draw=cursed_as_draw)
    # Both derive 5 (python-chess from its "KQRvKR" key, C from the largest
    # WDL table name), so the 6-man leaves exercise the filter.
    assert py.max_pieces == c.max_pieces == 5
    leaves = _stub_leaves(20261115, 400)
    if with_indices:
        # What the C tree hands over: no castling, <= max_pieces men.
        leaves = [cb for cb in leaves if cb.castling == 0
                  and (cb.occ_white | cb.occ_black).bit_count() <= 5]
    (pw, ps, ph), (cw, cs, ch) = _apply_both(py, c, leaves, with_indices=with_indices, seed=3)
    assert np.array_equal(pw, cw)
    assert np.array_equal(ps, cs)
    assert ph == ch
    assert (py.hits, py.probes) == (c.hits, c.probes)
    # The comparison is not vacuous: every outcome class occurred.
    assert ph > 50
    assert {1, -1, 2} <= set(np.unique(cs).tolist())
    assert c.probes > c.hits  # some probed leaves got no answer
    if not with_indices:
        assert c.probes < len(leaves)  # the eligibility filter dropped some
        probed = [f for batch in stub.batches for f in batch]
        assert not any(chess.Board(f).castling_rights for f in probed)
        assert all(chess.popcount(chess.Board(f).occupied) <= 5 for f in probed)


def test_value_mapping_and_counters(monkeypatch) -> None:
    """Every code, both rule settings, by hand."""
    codes = bytes([0, 1, 2, 3, 4, 5, 6])
    leaves = [CBoard.from_board(chess.Board(_KRK_FEN)) for _ in codes]

    class _C:
        def generation(self) -> int:
            return 7

        def probe_wdl_batch(self, boards, generation):
            assert generation == 7
            assert len(boards) == len(codes)
            return codes

    monkeypatch.setattr(tb_leaf, "_tb_ext", _C())
    W, D, L = tbmod._WIN_LOGITS, tbmod._DRAW_LOGITS, tbmod._LOSS_LOGITS
    for cursed_as_draw, want, solved in (
        (True, [L, D, D, D, W, None, None], [-1, 2, 2, 2, 1, 0, 0]),
        (False, [L, L, D, W, W, None, None], [-1, -1, 2, 1, 1, 0, 0]),
    ):
        probe = CLeafProbe(_loaded(3), cursed_as_draw=cursed_as_draw)
        net = np.full((len(codes), 3), 0.5, dtype=np.float32)
        out_solved = np.zeros(len(codes), dtype=np.int8)
        assert probe.apply(leaves, net, indices=np.arange(len(codes)), solved_out=out_solved) == 5
        for row, exp in zip(net, want, strict=True):
            assert np.array_equal(row, np.full(3, 0.5, np.float32) if exp is None else exp)
        assert out_solved.tolist() == solved
        assert (probe.hits, probe.probes) == (5, 7)
        probe.reset_counts()
        assert (probe.hits, probe.probes) == (0, 0)


def test_a_stale_probe_answers_nothing_and_warns_once(monkeypatch, caplog) -> None:
    stub = _ScriptedC(generation=8)  # the set was re-initialised after gen 7
    monkeypatch.setattr(tb_leaf, "_tb_ext", stub)
    probe = CLeafProbe(_loaded(6, c_generation=7), cursed_as_draw=True)
    assert not probe.is_current()
    leaves = _stub_leaves(1, 20)
    net = np.ones((len(leaves), 3), dtype=np.float32)
    with caplog.at_level(logging.WARNING, logger=tb_leaf.__name__):
        for _ in range(3):
            assert probe.apply(leaves, net) == 0
    assert np.array_equal(net, np.ones_like(net))
    assert probe.hits == 0
    assert probe.stale_batches == 3
    assert stub.batches == []
    assert sum("re-initialised" in r.message for r in caplog.records) == 1


def test_real_extension_generation_and_batch_contract(tmp_path: Path) -> None:
    """The real ``_tb_ext`` (CI: no tablebases beyond the committed 3-man
    fixture): the generation moves on every init/free -- including an
    ``init(p1)`` -> ``init(p2)`` with no ``free()`` in between, which is how
    ``tb_root.load`` switches paths when two engines use different ones -- a
    stale generation probes nothing, codes for no-table / illegal / castling
    positions, argument checking."""
    tb_root.unload()
    g0 = _tb_ext.generation()
    _tb_ext.free()
    g1 = _tb_ext.generation()
    assert g1 != g0
    with pytest.raises(ValueError, match="not a readable directory"):
        _tb_ext.init("/nonexistent-syzygy-dir-for-test")  # refused before Fathom
    assert _tb_ext.generation() == g1  # the old (empty) set stays, same identity
    krk = CBoard.from_board(chess.Board(_KRK_FEN))
    only_kq = tmp_path / "kq"
    only_kq.mkdir()
    for ext in (".rtbw", ".rtbz"):
        (only_kq / f"KQvK{ext}").symlink_to(_FIXTURE / f"KQvK{ext}")
    # init(p1): the fixture answers KRvK (White to move wins: code 4).
    _tb_ext.init(str(_FIXTURE))
    g_p1 = _tb_ext.generation()
    assert g_p1 != g1
    assert _tb_ext.probe_wdl_batch([krk], g_p1) == bytes([4])
    # init(p2) with NO free(): a new set, so the p1 generation is dead.
    _tb_ext.init(str(only_kq))
    g_p2 = _tb_ext.generation()
    assert g_p2 != g_p1
    assert _tb_ext.probe_wdl_batch([krk], g_p1) is None
    assert _tb_ext.probe_wdl_batch([krk], g_p2) == bytes([5])  # no KRvK table in p2
    # Table-free: an init that reaches Fathom and loads nothing also moves it.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no Syzygy tables"):
        _tb_ext.init(str(empty))
    assert _tb_ext.generation() not in (g_p1, g_p2)
    assert _tb_ext.probe_wdl_batch([krk], g_p2) is None
    _tb_ext.free()
    g1 = _tb_ext.generation()
    legal = CBoard.from_board(chess.Board(_KRK_FEN))
    # White to move while Black is in check: illegal (Fathom's is_valid).
    illegal = CBoard.from_board(chess.Board("7k/7R/8/8/8/8/8/K7 w - - 0 1"))
    castling = CBoard.from_board(chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"))
    assert castling.castling != 0
    assert _tb_ext.probe_wdl_batch([legal, illegal, castling], g1) == bytes([5, 6, 5])
    assert _tb_ext.probe_wdl_batch([legal], g1 + 1) is None
    assert _tb_ext.probe_wdl_batch([], g1) == b""
    with pytest.raises(TypeError):
        _tb_ext.probe_wdl_batch([legal, "not a board"], g1)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        _tb_ext.probe_wdl_batch(5, g1)  # pyright: ignore[reportArgumentType]


def _no_python_chess_tablebase(monkeypatch) -> list[str]:
    """Make every python-chess tablebase entry point record and fail."""
    calls: list[str] = []

    def refuse(name: str) -> Callable[..., object]:
        def _call(*_a, **_k):
            calls.append(name)
            raise AssertionError(f"python-chess {name} called on the UCI path")
        return _call

    monkeypatch.setattr(tbmod, "get_tablebase", refuse("get_tablebase"))
    monkeypatch.setattr(chess.syzygy, "open_tablebase", refuse("open_tablebase"))
    monkeypatch.setattr(chess.syzygy.Tablebase, "__init__", refuse("Tablebase()"))
    return calls


def test_engine_installs_the_c_leaf_probe_and_never_opens_python_chess(
    out, fake_c, monkeypatch,
) -> None:
    """SyzygyPath -> the worker's leaf probe IS the C one, on the C set; a
    ``searchmoves`` search (no root shortcut) probes the root eval and leaves
    through it, and the GUI's ``tbhits`` counts its hits."""
    calls = _no_python_chess_tablebase(monkeypatch)
    fake_c.leaf_code = lambda fen: 4 if chess.Board(fen).turn == chess.WHITE else 0
    engine, worker = _engine()
    try:
        engine.dispatch(CmdSetOption(name="SyzygyPath", value="stub-tables"))
        probe = worker._tb_probe
        root = worker._tb_root
        assert isinstance(probe, CLeafProbe)
        assert root is not None
        assert probe.tables is root.tables
        assert probe.max_pieces == 3  # "KQvK"/"KRvK" names in the stand-in
        assert "leaf prober: C Fathom" in out.text()
        engine.dispatch(parse_command(f"position fen {_KRK_FEN}"))
        _go(engine, "go nodes 64 searchmoves b1b7 a1a2")
        assert fake_c.root_calls == []  # searchmoves: no root shortcut
        assert fake_c.leaf_calls, "no leaf probe reached the C prober"
        assert fake_c.leaf_calls[0] == [_KRK_FEN]  # the root eval comes first
        last = [ln for ln in out.lines if ln.startswith("info depth")][-1]
        tbhits = int(last.split(" tbhits ")[1].split()[0])
        assert tbhits == probe.hits > 1, last
    finally:
        engine.close()
    assert calls == []


@pytest.mark.usefixtures("out", "py_table")
def test_syzygy50moverule_reaches_the_c_leaf_probe(fake_c) -> None:
    engine, worker = _engine()
    try:
        engine.dispatch(CmdSetOption(name="SyzygyPath", value="stub-tables"))
        first = worker._tb_probe
        assert isinstance(first, CLeafProbe)
        assert first.cursed_as_draw is True
        engine.dispatch(CmdSetOption(name="Syzygy50MoveRule", value="false"))
        second = worker._tb_probe
        assert isinstance(second, CLeafProbe)
        assert second is not first
        assert second.cursed_as_draw is False
        assert second.tables is first.tables  # same C set, no re-init
        assert fake_c.inits == ["stub-tables"]
    finally:
        engine.close()


def test_a_stale_leaf_probe_is_dropped_before_the_next_search(out, fake_c, py_table) -> None:
    """Another engine re-initialises the one C set: the first engine's next
    search drops its leaf probe and never probes the new set with it. The
    notice repeats on EVERY search while leaf probing stays off, and stops
    once ``SyzygyPath`` is set again (which re-installs a live probe)."""
    first, first_worker = _engine()
    second, _ = _engine()
    try:
        first.dispatch(CmdSetOption(name="SyzygyPath", value="stub-a"))
        stale = first_worker._tb_probe
        assert isinstance(stale, CLeafProbe)
        second.dispatch(CmdSetOption(name="SyzygyPath", value="stub-b"))  # re-init
        assert not stale.is_current()
        fake_c.leaf_calls.clear()
        first.dispatch(parse_command(f"position fen {_KRK_FEN}"))
        _go(first, "go nodes 64 searchmoves b1b7 a1a2")
        notes = [ln for ln in out.lines if "leaf tablebase probing disabled" in ln]
        assert len(notes) == 1, out.lines
        assert first_worker._tb_probe is None
        assert fake_c.leaf_calls == []
        assert out.bestmoves(), out.lines
        _go(first, "go nodes 64 searchmoves b1b7 a1a2")
        assert len([ln for ln in out.lines if "leaf tablebase probing disabled" in ln]) == 2
        assert py_table.calls == []
        first.dispatch(CmdSetOption(name="SyzygyPath", value="stub-a"))  # restores it
        assert isinstance(first_worker._tb_probe, CLeafProbe)
        assert first_worker._tb_probe.is_current()
        _go(first, "go nodes 64 searchmoves b1b7 a1a2")
        assert len([ln for ln in out.lines if "leaf tablebase probing disabled" in ln]) == 2
        assert fake_c.leaf_calls, "the restored probe reaches the C prober"
    finally:
        first.close()
        second.close()


def test_mid_search_reinit_is_refused_atomically(monkeypatch) -> None:
    """A batch that starts after the re-init never reaches the new set, even
    though the probe was current when the search began."""
    stub = _ScriptedC(generation=7)
    monkeypatch.setattr(tb_leaf, "_tb_ext", stub)
    probe = CLeafProbe(_loaded(6), cursed_as_draw=True)
    leaves = _stub_leaves(2, 30)
    net = np.zeros((len(leaves), 3), dtype=np.float32)
    assert probe.apply(leaves, net) > 0
    stub._generation = 8
    net2 = np.zeros_like(net)
    assert probe.apply(leaves, net2) == 0
    assert not net2.any()


def test_max_pieces_is_the_largest_wdl_table_not_fathoms_largest() -> None:
    """``TB_LARGEST`` also counts DTM tables; ``SyzygyProbe`` derived its
    filter from WDL names only, so the leaf filter must too."""
    tables = tb_root.LoadedTables(
        configured="stub", resolved="/stub", largest=7, n_wdl=1, n_dtz=1, largest_dtz=6,
        generation=1, c_generation=1, largest_wdl=6,
    )
    assert CLeafProbe(tables, cursed_as_draw=True).max_pieces == 6


def test_load_reinitialises_when_the_c_set_changed_behind_its_back(fake_c) -> None:
    """Something called ``_tb_ext.init`` directly (same path): the cached
    ``LoadedTables`` would carry a dead generation and every leaf batch would
    be refused. ``load`` must notice and re-initialise."""
    first = tb_root.load("stub-tables")
    assert first.c_generation == fake_c.generation()
    fake_c.init("stub-tables")  # behind tb_root's back
    second = tb_root.load("stub-tables")
    assert second.c_generation == fake_c.generation() != first.c_generation
    assert fake_c.inits == ["stub-tables", "stub-tables", "stub-tables"]


def test_startup_gate_requires_the_leaf_batch_symbols(monkeypatch) -> None:
    from chess_anti_engine import ext_check
    from chess_anti_engine.uci import selfcheck as sc

    real = ext_check.missing_symbols
    for symbol in ("probe_wdl_batch", "generation"):
        monkeypatch.setattr(
            ext_check, "missing_symbols",
            lambda name, symbols, _s=symbol: (
                [_s] if name == sc.TB_EXT and _s in symbols else real(name, symbols)),
        )
        problems = sc.extension_problems()
        assert any(p.startswith(f"{sc.TB_EXT} lacks {symbol}") for p in problems), problems


# ------------------------------------------------------------- table tier

def _seven_man_classes(six: list[str], rng: random.Random, n: int) -> list[str]:
    picked = rng.sample(six, min(n, len(six)))
    return [f"{a}v{b}P" for a, b in (c.split("v") for c in picked)]


def _children(cb: CBoard, limit: int, rng: random.Random) -> list[CBoard]:
    """Real tree leaves: CBoards one C ``push_index`` below ``cb``."""
    moves = list(cb.legal_move_indices())
    rng.shuffle(moves)
    out = []
    for idx in moves[:limit]:
        child = cb.copy()
        child.push_index(int(idx))
        out.append(child)
    return out


def _leaf_corpus(path: str, largest: int, seed: int) -> tuple[list[CBoard], Counter[str]]:
    rng = random.Random(seed)
    classes = [c for c in material_classes(path) if len(c) - 1 <= largest]
    small = [c for c in classes if len(c) - 1 <= 4]
    big = [c for c in classes if len(c) - 1 > 4]
    picked = small + big[:: max(1, len(big) // 60)]
    sevens = _seven_man_classes([c for c in classes if len(c) - 1 == largest], rng, 25)
    roots: list[chess.Board] = []
    counts: Counter[str] = Counter()
    for cls in picked + sevens:
        got = 0
        for _ in range(400):
            if got >= 10:
                break
            b = gen_random(rng, cls)
            if b is not None:
                b.halfmove_clock = rng.choice((0, 0, 50, 99, 100, 130))
                roots.append(b)
                got += 1
        for gen, key in ((gen_ep, "ep_root"), (gen_castling, "castling_root")):
            for _ in range(30):
                b = gen(rng, cls)
                if b is not None:
                    roots.append(b)
                    counts[key] += 1
                    break
        for _ in range(200):
            parent = gen_edge_king(rng, cls)
            kids = terminal_children(parent) if parent is not None else []
            if kids:
                roots.append(kids[0])
                counts["mate_or_stalemate"] += 1
                break
    # Cursed wins / blessed losses: rare under uniform placement.
    for cls in ("KNNvKP", "KBBvKN", "KRBvKR", "KQvKBB", "KBNvKN", "KRNvKR", "KRPvKR"):
        if cls not in classes:
            continue
        found = 0
        for _ in range(4000):
            b = gen_random(rng, cls)
            if b is not None and _tb_ext.probe_wdl_fen(b.fen()) in (-1, 1):
                roots.append(b)
                found += 1
                if found >= 4:
                    break
    leaves: list[CBoard] = []
    for b in roots:
        cb = CBoard.from_board(b)
        leaves.append(cb)
        if not b.is_game_over(claim_draw=False):
            leaves += _children(cb, 3, rng)
    for cb in leaves:
        men = (cb.occ_white | cb.occ_black).bit_count()
        counts["leaves"] += 1
        counts[f"{min(men, 7)}-man"] += 1
        if cb.castling:
            counts["castling"] += 1
        if cb.ep_square is not None:
            counts["ep_square_set"] += 1
        clock = cb.halfmove_clock
        counts["clock0" if clock == 0 else "clock50-99" if 50 <= clock <= 99
               else "clock100+" if clock >= 100 else "clock1-49"] += 1
    return leaves, counts


def _chunks(seq: list[CBoard], n: int) -> list[list[CBoard]]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


@needs_tables
def test_c_leaf_differential_against_python_chess(tables: str, capsys) -> None:
    """``CLeafProbe.apply`` == ``SyzygyProbe.apply`` (python-chess, the
    reference) on several thousand leaves: rows written, solved statuses,
    hits, probes; both ``Syzygy50MoveRule`` settings; the root-eval call
    (no indices, the probe filters) and the pre-filtered leaf call."""
    loaded = tb_root.load(tables)
    py0 = tbmod.SyzygyProbe(tables)
    assert loaded.largest_wdl == py0.max_pieces, (loaded.largest_wdl, py0.max_pieces)
    leaves, counts = _leaf_corpus(tables, loaded.largest_wdl, seed=20261115)
    mismatches: list[str] = []
    seven_diffs = 0
    for rule50 in (True, False):
        py = tbmod.SyzygyProbe(tables, cursed_as_draw=rule50)
        c = CLeafProbe(loaded, cursed_as_draw=rule50)
        prefiltered = [cb for cb in leaves if cb.castling == 0
                       and (cb.occ_white | cb.occ_black).bit_count() <= c.max_pieces]
        for mode, corpus, with_idx in (("root-eval", leaves, False),
                                       ("leaf", prefiltered, True)):
            for k, batch in enumerate(_chunks(corpus, 64)):
                (pw, ps, ph), (cw, cs, ch) = _apply_both(py, c, batch, with_indices=with_idx,
                                                         seed=k)
                if not (np.array_equal(pw, cw) and np.array_equal(ps, cs) and ph == ch):
                    bad = [batch[j].fen() for j in range(len(batch)) if ps[j] != cs[j]]
                    mismatches.append(f"{mode} rule50={rule50} batch {k}: {bad[:3]}")
                if rule50 and mode == "root-eval":
                    counts["answered"] += int(np.count_nonzero(cs))
                    vals = [_tb_ext.probe_wdl(cb) for cb in batch if cb.castling == 0
                            and (cb.occ_white | cb.occ_black).bit_count() <= c.max_pieces]
                    counts["cursed_or_blessed"] += sum(v in (-1, 1) for v in vals)
        assert (py.hits, py.probes) == (c.hits, c.probes), rule50
        # Leaves above the tables, handed over as if pre-filtered (no caller
        # does this): python-chess sometimes answers through a capture, C never.
        over = [cb for cb in leaves if cb.castling == 0
                and (cb.occ_white | cb.occ_black).bit_count() > c.max_pieces]
        for batch in _chunks(over, 64):
            (pw, ps, _), (cw, cs, _) = _apply_both(py, c, batch, with_indices=True, seed=1)
            assert not cs.any()  # C: no answer above largest()
            seven_diffs += int((ps != cs).sum())
    with capsys.disabled():
        print(f"\nC-vs-python-chess UCI leaf differential on {tables} (WDL up to "
              f"{loaded.largest_wdl}-man): {dict(sorted(counts.items()))}; "
              f">largest leaves python-chess answered via a capture (no caller passes them): "
              f"{seven_diffs}")
    assert not mismatches, f"{len(mismatches)} mismatching batches, first: {mismatches[:5]}"
    assert counts["leaves"] >= 2000
    for name, floor in (("answered", 1000), ("castling", 20), ("ep_square_set", 20),
                        ("clock0", 200), ("clock50-99", 200), ("clock100+", 100),
                        ("mate_or_stalemate", 20), ("7-man", 100)):
        assert counts[name] >= floor, (name, dict(counts))
    if loaded.largest_wdl >= 5:
        assert counts["cursed_or_blessed"] >= 5, dict(counts)


def _subset(directory: Path, path: str, stems: tuple[str, ...]) -> str:
    directory.mkdir()
    for part in path.split(os.pathsep):
        for stem in stems:
            for ext in (".rtbw", ".rtbz"):
                src = Path(part) / f"{stem}{ext}"
                if src.exists() and not (directory / src.name).exists():
                    (directory / src.name).symlink_to(src)
    return str(directory)


@needs_tables
def test_missing_tables_inside_the_piece_limit_agree(tables: str, tmp_path: Path) -> None:
    """An incomplete set: KQvKR and KQvK present, KQvKN and KRvK missing.
    Leaves whose own table is missing but a capture reaches a present one
    (python-chess's / Fathom's capture cutoff), and leaves with no route at
    all, must agree -- including which ones get no answer."""
    path = _subset(tmp_path / "part", tables, ("KQvKR", "KQvK", "KNvK"))
    loaded = tb_root.load(path)
    rng = random.Random(5)
    boards = []
    for cls in ("KQvKN", "KRvK", "KQvKR", "KQvK", "KNvK"):
        n = 0
        while n < 300:
            b = gen_random(rng, cls)
            if b is not None:
                boards.append(CBoard.from_board(b))
                n += 1
    for rule50 in (True, False):
        py = tbmod.SyzygyProbe(path, cursed_as_draw=rule50)
        c = CLeafProbe(loaded, cursed_as_draw=rule50)
        assert py.max_pieces == c.max_pieces == 4
        answered = 0
        for k, batch in enumerate(_chunks(boards, 64)):
            (pw, ps, ph), (cw, cs, ch) = _apply_both(py, c, batch, with_indices=False, seed=k)
            assert np.array_equal(ps, cs), [batch[j].fen() for j in np.flatnonzero(ps != cs)][:3]
            assert np.array_equal(pw, cw)
            assert ph == ch
            answered += ch
        assert (py.hits, py.probes) == (c.hits, c.probes)
        assert answered == c.hits
        assert 0 < c.hits < c.probes, (c.hits, c.probes)  # some missing, some answered


@needs_tables
def test_leaf_probes_under_concurrent_reinit_never_answer_from_another_set(
    tables: str, tmp_path: Path,
) -> None:
    """Four threads hammer one probe (as walker / pool threads do) while the
    main thread re-initialises and frees the C set. Every answer given while
    the probe was current equals python-chess's; after the first re-init the
    probe never answers again; nothing crashes."""
    kq_only = _subset(tmp_path / "kq", tables, ("KQvK",))
    loaded = tb_root.load(tables)
    probe = CLeafProbe(loaded, cursed_as_draw=True)
    py = tbmod.SyzygyProbe(tables, cursed_as_draw=True)
    rng = random.Random(11)
    leaves = []
    while len(leaves) < 256:
        b = gen_random(rng, rng.choice(("KRvK", "KQvKR", "KRvKB", "KPvK")))
        if b is not None:
            leaves.append(CBoard.from_board(b))
    want = np.zeros((len(leaves), 3), dtype=np.float32)
    want_solved = np.zeros(len(leaves), dtype=np.int8)
    py.apply(leaves, want, solved_out=want_solved)
    stop = threading.Event()
    reinit_done = threading.Event()
    errors: list[str] = []
    late_hits = [0]

    def hammer() -> None:
        while not stop.is_set():
            after = reinit_done.is_set()
            wdl = np.zeros_like(want)
            solved = np.zeros_like(want_solved)
            hits = probe.apply(leaves, wdl, solved_out=solved)
            if hits:
                if after:
                    late_hits[0] += hits
                if not (np.array_equal(solved, want_solved) and np.array_equal(wdl, want)):
                    errors.append("an answered batch differs from python-chess")

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for _ in range(20):
            tb_root.load(kq_only)
            reinit_done.set()
            tb_root.unload()
            tb_root.load(tables)  # same path, NEW generation: the probe stays stale
    finally:
        stop.set()
        for t in threads:
            t.join(30)
    assert not any(t.is_alive() for t in threads)
    assert not errors, errors[:3]
    assert late_hits[0] == 0
    assert probe.stale_batches > 0
    assert not probe.is_current()


_IDENTITY_CASES: list[tuple[str, tuple[str, ...]]] = [
    # 7 men at the root: no shortcut, captures lead into the tables.
    ("8/8/8/2br4/3P2k1/4B3/2p5/4K3 w - - 95 40", ()),
    ("8/8/8/2br4/3P2k1/4B3/2p5/4K3 w - - 0 40", ()),
    ("8/5pk1/6p1/8/3R4/6P1/5PK1/3r4 w - - 0 40", ()),
    ("2r5/5pk1/8/8/3R4/6P1/5PK1/8 b - - 3 50", ()),
    ("8/3k4/8/1p1P4/1P1K4/8/7r/7R w - - 0 60", ()),
    # <= 6 men with searchmoves: the shortcut is skipped, leaves are probed.
    ("8/8/8/4k3/8/8/4P3/R3K3 w - - 0 1", ("a1a7", "e1d2", "e2e4")),
    ("8/3k4/8/8/8/1q6/8/K1R5 w - - 30 70", ("c1c3", "c1b1")),
    ("8/8/8/3k4/8/3KP3/8/8 w - - 0 1", ("e3e4", "d3e2")),
]

#: (position, go) pairs the UCI subprocess test plays.
_SUBPROCESS_GOES: list[tuple[str, str]] = [
    ("8/8/8/2br4/3P2k1/4B3/2p5/4K3 w - - 0 40", "go nodes 400"),
    ("8/8/8/4k3/8/8/4P3/R3K3 w - - 0 1", "go nodes 300 searchmoves a1a7 e2e4"),
    ("8/8/8/4k3/8/8/8/KR6 w - - 0 1", "go nodes 64"),
]


def _all_case_fens() -> list[tuple[str, tuple[str, ...]]]:
    out = list(_IDENTITY_CASES)
    for fen, go in _SUBPROCESS_GOES:
        words = go.split()
        moves = tuple(words[words.index("searchmoves") + 1:]) if "searchmoves" in words else ()
        out.append((fen, moves))
    return out


def test_every_tb_case_position_is_legal() -> None:
    """A case list with an illegal FEN (e.g. the side NOT to move in check)
    silently searches nothing: guard every one, and every searchmove."""
    for fen, moves in _all_case_fens():
        board = chess.Board(fen)
        assert board.is_valid(), (fen, board.status())
        assert not board.is_game_over(claim_draw=False), fen
        for uci in moves:
            assert chess.Move.from_uci(uci) in board.legal_moves, (fen, uci)


# Search-level identity: the same fixed-seed search with each leaf probe.
_IDENTITY = r'''
import json, sys, threading
import chess, torch
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.tablebase import SyzygyProbe
from chess_anti_engine.uci import tb_root
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.tb_leaf import CLeafProbe
from chess_anti_engine.uci.time_manager import Deadline

torch.set_num_threads(1)
path, mode, probe_kind, rule50 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"

def ev():
    torch.manual_seed(1234)
    m = build_model(ModelConfig(input_extra_features="v1", embed_dim=32, num_layers=2,
                                num_heads=2, ffn_mult=2.0))
    m.eval()
    return ThreadSafeGPUDispatcher(DirectGPUEvaluator(
        m, device="cpu", max_batch=512, use_amp=False, n_slots=2))

tables = tb_root.load(path)
w = SearchWorker(ev(), device="cpu",
                 gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=256, add_noise=False),
                 chunk_sims=128, n_walkers=1)
if mode == "pool1":
    w.install_multi_gpu_pucv([ev()], gather=24, as_factories=False)
probe = (SyzygyProbe(path, cursed_as_draw=rule50) if probe_kind == "py"
         else CLeafProbe(tables, cursed_as_draw=rule50))
w.set_tb_probe(probe)
w.set_tb_root(tb_root.RootTablebase(tables, rule50=rule50))
CASES = json.loads(sys.argv[5])
out = []
for fen, moves in CASES:
    w.reset_tree()
    r = w.run(chess.Board(fen), stop_event=threading.Event(), deadline=Deadline(None),
              max_nodes=600, root_moves=tuple(moves))
    try:
        mv, vis = w._tree.get_children_visits(w._root_id)
        vis = [int(v) for v in vis]
        mv = [int(x) for x in mv]
    except Exception as e:
        mv, vis = [], [str(e)]
    out.append(dict(fen=fen, best=r.bestmove_uci, nodes=r.nodes, pv=list(r.pv),
                    score=r.score_cp, mate=r.score_mate, tbhits=r.tbhits, visits=vis, moves=mv))
w.close()
tb_root.unload()
print(json.dumps(out, sort_keys=True))
'''


@needs_tables
@pytest.mark.parametrize("mode", ["classic", "pool1"])
@pytest.mark.parametrize("rule50", [True, False])
def test_search_identity_python_chess_leaf_vs_c_leaf(tables: str, tmp_path: Path, mode: str,
                                                     rule50: bool) -> None:
    """Fixed-seed CPU search from TB-rich roots: bestmove, score, PV, nodes,
    tbhits and the root visit vector are identical with the python-chess leaf
    probe (main) and the C leaf probe (this PR)."""
    script = tmp_path / "identity.py"
    script.write_text(_IDENTITY)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    results = {}
    for kind in ("py", "c"):
        proc = subprocess.run([sys.executable, str(script), tables, mode, kind,
                               "1" if rule50 else "0", json.dumps(_IDENTITY_CASES)],
                              capture_output=True, text=True, env=env,
                              cwd=str(tmp_path), timeout=900, check=False)
        assert proc.returncode == 0, proc.stderr[-3000:]
        results[kind] = json.loads(proc.stdout.strip().splitlines()[-1])
    assert results["py"] == results["c"]
    assert all(r["best"] != "0000" for r in results["c"]), results["c"]
    # Leaves were probed on every case (not only the root shortcut's 1).
    assert all(r["tbhits"] > 1 for r in results["c"]), results["c"]


_LAUNCHER = textwrap.dedent("""
    import os
    import runpy
    import sys

    import chess.syzygy

    def _mark(name):
        sys.stderr.write(f"PYCHESS_SYZYGY {name}\\n")
        sys.stderr.flush()

    def _wrap(owner, name):
        original = getattr(owner, name)

        def call(*args, **kwargs):
            _mark(name)
            return original(*args, **kwargs)

        setattr(owner, name, call)

    for _name in ("probe_wdl", "probe_dtz", "probe_wdl_table", "probe_dtz_table",
                  "get_wdl", "get_dtz", "add_directory", "add_file", "__init__"):
        _wrap(chess.syzygy.Tablebase, _name)
    _wrap(chess.syzygy, "open_tablebase")

    import chess_anti_engine.tb._tb_ext as _tb_ext

    _batches = [0, 0]
    _orig_batch = _tb_ext.probe_wdl_batch

    def _counting(boards, generation):
        out = _orig_batch(boards, generation)
        _batches[0] += 1
        _batches[1] += len(boards)
        return out

    _tb_ext.probe_wdl_batch = _counting

    def _report():
        sys.stderr.write(f"LEAFSTATS batches={_batches[0]} leaves={_batches[1]}\\n")
        sys.stderr.flush()

    import atexit
    atexit.register(_report)
    _real_exit = os._exit

    def _exit(code):  # the engine's quit path ends with os._exit (no atexit)
        _report()
        _real_exit(code)

    os._exit = _exit
    sys.argv[0] = "deepfin"
    runpy.run_module("chess_anti_engine.uci", run_name="__main__", alter_sys=True)
""")


@needs_tables
def test_uci_subprocess_plays_with_zero_python_chess_syzygy_calls(tables: str,
                                                                  tmp_path: Path) -> None:
    """Take-effect proof on the real entry point: every python-chess Syzygy
    entry point (opening a Tablebase, adding directories/files, every probe)
    writes a marker. Searches from TB-rich positions -- a 7-man root, a
    ``go searchmoves`` root, a root the shortcut answers -- run with zero
    markers and a non-zero count of C leaf batches."""
    from chess_anti_engine.uci.selfcheck import _write_tiny_checkpoint

    ckpt = _write_tiny_checkpoint(tmp_path)
    launcher = tmp_path / "launch_engine.py"
    launcher.write_text(_LAUNCHER, encoding="utf-8")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO_ROOT))
    cmd = [sys.executable, "-u", str(launcher), "--checkpoint", str(ckpt), "--no-compile",
           "--device", "cpu", "--compile-cache-dir", str(tmp_path / "cache")]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    lines: list[str] = []
    try:
        assert proc.stdin is not None
        proc.stdin.write("uci\n")
        proc.stdin.flush()
        _read_until(proc, lines, "uciok", 180)
        proc.stdin.write(f"setoption name SyzygyPath value {tables}\nisready\n")
        proc.stdin.flush()
        _read_until(proc, lines, "readyok", 180)
        for fen, go in _SUBPROCESS_GOES:
            proc.stdin.write(f"position fen {fen}\n{go}\n")
            proc.stdin.flush()
            _read_until(proc, lines, "bestmove", 180)
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait(timeout=60)
        assert proc.stdout is not None
        lines += proc.stdout.read().splitlines()  # what followed the last bestmove
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    assert [ln for ln in lines if "leaf prober: C Fathom" in ln], lines[-30:]
    assert not [ln for ln in lines if "PYCHESS_SYZYGY" in ln], \
        [ln for ln in lines if "PYCHESS_SYZYGY" in ln][:5]
    stats = [ln for ln in lines if ln.startswith("LEAFSTATS ")]
    assert stats, lines[-30:]
    fields = dict(kv.split("=") for kv in stats[-1].split()[1:])
    assert int(fields["batches"]) > 0, stats[-1]
    assert int(fields["leaves"]) > 0, stats[-1]
    tbhits = [int(ln.split(" tbhits ")[1].split()[0]) for ln in lines
              if ln.startswith("info depth") and " tbhits " in ln]
    assert max(tbhits) > 1, tbhits
    print(f"\n{stats[-1]}")
