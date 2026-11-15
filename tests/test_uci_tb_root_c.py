"""The UCI root Syzygy shortcut runs on the C prober (``uci/tb_root.py``).

Two tiers, like ``tests/test_tb_prober.py``:

* table-free tests run everywhere (CI included). They replace the C module
  ``tb_root`` talks to with a scripted stand-in, or use the real one on paths
  that load nothing, and check the ENGINE wiring: the root shortcut asks the C
  prober and never python-chess, ``searchmoves`` bypasses it, a ``SyzygyPath``
  the C prober refuses is loud and disables tablebases, a ``SyzygyPath`` change
  re-initialises it, ``close`` frees it;
* table tests need real Syzygy files and SKIP with the reason printed where
  there are none (CI). Where tables exist they run: a differential test of the
  C root against the python-chess root (``tablebase.try_tb_root_move``, kept as
  the test-only reference), engine-level ``go`` on real tables, and a UCI
  subprocess whose python-chess Syzygy probes are instrumented, proving the
  production path probes the root through C.
"""
from __future__ import annotations

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
import pytest

import chess_anti_engine.tablebase as tbmod
import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.uci import tb_root
from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import CmdSetOption, parse_command
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.utils.syzygy import default_syzygy_path, missing_tablebase_dirs, syzygy_roots
from scripts.tb_prober_equivalence import (
    gen_edge_king,
    gen_ep,
    gen_random,
    material_classes,
    terminal_children,
)
from tests.test_uci_compliance_p2 import _go, _Out, _tiny_worker

REPO_ROOT = Path(__file__).resolve().parents[1]
_TB_WIN_LINE = "score cp 41890 "


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


# ------------------------------------------------------- table-free stand-ins

_RowsFn = Callable[[chess.Board], "list[tuple[str, int, int, int]] | None"]


class _FakeCExt:
    """Stands in for ``chess_anti_engine.tb._tb_ext`` inside ``tb_root``."""

    FATHOM_COMMIT = "f" * 40

    def __init__(self, rows: _RowsFn, *, largest: int = 3) -> None:
        self.rows = rows
        self.largest = largest
        self.inits: list[str] = []
        self.root_calls: list[str] = []
        self._path: str | None = None

    def init(self, path: str) -> int:
        self.inits.append(path)
        if not path:
            self._path = None
            return 0
        if "missing" in path:
            raise ValueError(f"Syzygy path component is not a readable directory: {path}")
        self._path = "/fake/" + path
        return self.largest

    def free(self) -> None:
        self._path = None

    def path(self) -> str | None:
        return self._path

    def table_counts(self) -> tuple[int, int, int]:
        return (2, 2, self.largest) if self._path else (0, 0, 0)

    def probe_root_fen(self, fen: str, has_repeated: bool = False, use_rule50: bool = True):
        # tb_root reads only the per-move DTZ, which neither flag changes; it
        # passes Fathom's defaults and applies the repo's own claim rules.
        assert has_repeated is False
        assert use_rule50 is True
        self.root_calls.append(fen)
        return self.rows(chess.Board(fen))


class _RecordingPyTable:
    """python-chess stand-in for ``get_tablebase``: records every probe.

    The leaf probe (``SyzygyProbe``) still opens python-chess; the ROOT must
    never probe it. Its answers make a python-chess root pick ``a1a2`` -- a
    different move from the C stand-in's -- so a root routed back to
    python-chess also changes the bestmove.
    """

    wdl: ClassVar[dict[str, None]] = {"KRvK": None}
    dtz: ClassVar[dict[str, None]] = {"KRvK": None}

    def __init__(self) -> None:
        self.calls: list[str] = []

    def probe_wdl(self, board: chess.Board) -> int:
        self.calls.append("wdl " + board.fen())
        return 2 if board.turn == chess.WHITE else -2

    def probe_dtz(self, board: chess.Board) -> int:
        self.calls.append("dtz " + board.fen())
        last = board.peek().uci() if board.move_stack else ""
        return -3 if last == "a1a2" else -40


_KRK_FEN = "7k/8/8/8/8/8/8/KR6 w - - 0 1"
_C_BEST = "b1b7"


def _c_rows(board: chess.Board) -> list[tuple[str, int, int, int]]:
    """Every move wins in 30 plies except ``b1b7`` (wins in 5)."""
    return [(m.uci(), 5 if m.uci() == _C_BEST else 30, 0, 0) for m in board.legal_moves]


@pytest.fixture
def fake_c(monkeypatch) -> _FakeCExt:
    fake = _FakeCExt(_c_rows)
    monkeypatch.setattr(tb_root, "_tb_ext", fake)
    return fake


@pytest.fixture
def py_table(monkeypatch) -> _RecordingPyTable:
    table = _RecordingPyTable()
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: table)
    monkeypatch.setattr(engine_module, "get_tablebase", lambda _path: table)
    return table


def _engine() -> tuple[Engine, SearchWorker]:
    worker = _tiny_worker()
    return Engine(worker), worker


def _info_lines(out: _Out) -> list[str]:
    return [ln for ln in out.lines if ln.startswith("info depth")]


def test_engine_root_shortcut_asks_the_c_prober_not_python_chess(out, fake_c, py_table) -> None:
    engine, worker = _engine()
    try:
        engine.dispatch(CmdSetOption(name="SyzygyPath", value="stub-tables"))
        assert "info string Syzygy root prober: C Fathom" in out.text()
        assert worker._tb_root is not None
        engine.dispatch(parse_command(f"position fen {_KRK_FEN}"))
        _go(engine, "go nodes 64")
        assert out.bestmoves()[-1] == _C_BEST
        assert len(fake_c.root_calls) == 1
        assert py_table.calls == []
        last = _info_lines(out)[-1]
        assert _TB_WIN_LINE in last, last
        assert " tbhits 1" in last, last
    finally:
        engine.close()


@pytest.mark.usefixtures("py_table")
def test_searchmoves_bypasses_the_c_root_probe(out, fake_c) -> None:
    engine, _ = _engine()
    try:
        engine.dispatch(CmdSetOption(name="SyzygyPath", value="stub-tables"))
        engine.dispatch(parse_command(f"position fen {_KRK_FEN}"))
        _go(engine, "go nodes 64 searchmoves a1a2")
        assert out.bestmoves()[-1] == "a1a2"
        assert fake_c.root_calls == []
    finally:
        engine.close()


def test_a_syzygy_path_the_c_prober_refuses_is_loud_and_disables_tablebases(
    out, tmp_path: Path,
) -> None:
    """Real ``_tb_ext``: an empty directory loads no table, a missing one is
    not a directory. Either way the engine says so and probes nothing."""
    engine, worker = _engine()
    try:
        for bad in (str(tmp_path), str(tmp_path / "missing")):
            out.lines.clear()
            engine.dispatch(CmdSetOption(name="SyzygyPath", value=bad))
            text = out.text()
            assert f"info string error: SyzygyPath {bad!r} REFUSED by the C tablebase prober" in text
            assert "tablebase probing disabled" in text
            assert worker._tb_probe is None
            assert worker._tb_root is None
            assert tb_root.loaded() is None
    finally:
        engine.close()


@pytest.mark.usefixtures("py_table")
def test_refusal_after_a_good_path_drops_the_old_tables(out, fake_c) -> None:
    engine, worker = _engine()
    try:
        engine.dispatch(CmdSetOption(name="SyzygyPath", value="stub-tables"))
        assert worker._tb_root is not None
        engine.dispatch(CmdSetOption(name="SyzygyPath", value="missing-dir"))
        assert "REFUSED by the C tablebase prober" in out.text()
        assert worker._tb_root is None
        assert worker._tb_probe is None
        assert fake_c.path() is None
        engine.dispatch(parse_command(f"position fen {_KRK_FEN}"))
        _go(engine, "go nodes 64")
        assert fake_c.root_calls == []
    finally:
        engine.close()


@pytest.mark.usefixtures("out", "py_table")
def test_syzygy_path_change_mid_session_reinitialises_the_c_prober(fake_c) -> None:
    engine, worker = _engine()
    try:
        engine.dispatch(CmdSetOption(name="SyzygyPath", value="stub-a"))
        first = worker._tb_root
        assert first is not None
        assert first.tables.configured == "stub-a"
        engine.dispatch(CmdSetOption(name="SyzygyPath", value="stub-b"))
        second = worker._tb_root
        assert second is not None
        assert second.tables.configured == "stub-b"
        assert fake_c.inits == ["stub-a", "stub-b"]
        with pytest.raises(tb_root.RootTablebaseError, match="re-initialised"):
            first.root_move(chess.Board(_KRK_FEN))
        # Syzygy50MoveRule rebuilds the handle on the SAME tables (no re-init).
        engine.dispatch(CmdSetOption(name="Syzygy50MoveRule", value="false"))
        third = worker._tb_root
        assert third is not None
        assert third is not second
        assert third.rule50 is False
        assert fake_c.inits == ["stub-a", "stub-b"]
        assert third.root_move(chess.Board(_KRK_FEN)) == (chess.Move.from_uci(_C_BEST), 2)
        engine.dispatch(CmdSetOption(name="SyzygyPath", value="<empty>"))
        assert worker._tb_root is None
        assert tb_root.loaded() is None
        assert fake_c.path() is None
    finally:
        engine.close()


@pytest.mark.usefixtures("out", "py_table")
def test_engine_close_frees_the_c_tables(fake_c) -> None:
    engine, _ = _engine()
    engine.dispatch(CmdSetOption(name="SyzygyPath", value="stub-tables"))
    assert fake_c.path() is not None
    engine.close()
    assert fake_c.path() is None
    assert tb_root.loaded() is None


def test_root_move_keeps_the_repo_claim_and_ranking_rules(fake_c) -> None:
    """At halfmove 99 every quiet move is a 50-move claim (distance 0) with the
    rule on, whatever C's table distance says; the pawn push then wins."""
    board = chess.Board("7k/8/8/8/8/8/P7/KR6 w - - 99 150")
    fake_c.rows = lambda b: [
        (m.uci(), 101 if b.piece_type_at(m.from_square) == chess.PAWN else 30, 0, 0)
        for m in b.legal_moves
    ]
    fake_c.largest = 5
    tables = tb_root.load("stub")
    on = tb_root.RootTablebase(tables, rule50=True).root_move(board)
    off = tb_root.RootTablebase(tables, rule50=False).root_move(board)
    assert on is not None
    assert off is not None
    assert board.piece_type_at(on[0].from_square) == chess.PAWN
    assert on[1] == 1
    assert board.piece_type_at(off[0].from_square) != chess.PAWN
    assert off[1] == 2


def test_c_refused_input_is_no_tb_answer(fake_c) -> None:
    def refuse(_board: chess.Board):
        raise ValueError("not a legal chess position")

    fake_c.rows = refuse
    handle = tb_root.RootTablebase(tb_root.load("stub"), rule50=True)
    assert handle.root_move(chess.Board(_KRK_FEN)) is None


@pytest.mark.usefixtures("fake_c")
def test_load_normalises_like_the_python_chess_opener() -> None:
    sep = os.pathsep
    tables = tb_root.load(f" stub-a {sep}{sep} stub-b ")
    assert tables.configured == f"stub-a{sep}stub-b"
    with pytest.raises(tb_root.RootTablebaseError):
        tb_root.load(f" {sep} ")


# ------------------------------------------------------------- table tier


def _find_tables() -> str | None:
    if os.environ.get("CAE_SYZYGY_PATH", "").strip():
        return os.environ["CAE_SYZYGY_PATH"].strip()
    path = default_syzygy_path()
    if not missing_tablebase_dirs(path):
        return path
    for root in syzygy_roots():
        smoke = root / "data" / "syzygy_3-4man"
        if not missing_tablebase_dirs(str(smoke)):
            return str(smoke)
    return None


_TABLES = _find_tables()
needs_tables = pytest.mark.skipif(
    _TABLES is None,
    reason=(
        "no Syzygy tables here (CI ships none): the C-vs-python-chess UCI root "
        "differential and the real-table engine tests are NOT run; set CAE_SYZYGY_PATH"
    ),
)


@pytest.fixture(scope="module")
def tables() -> str:
    assert _TABLES is not None
    missing = missing_tablebase_dirs(_TABLES)
    if missing:
        pytest.fail(f"CAE_SYZYGY_PATH={_TABLES!r} is set but {missing} hold no tables")
    return _TABLES


def _py_move_dtz(board: chess.Board, path: str, *, rule50: bool) -> list[tuple[chess.Move, int]] | None:
    """The python-chess reference: ``tablebase._dtz_of_played_move`` per legal move."""
    tb = tbmod.get_tablebase(path)
    assert tb is not None
    if not tbmod.is_tb_eligible(board) or board.is_game_over(claim_draw=False):
        return None
    out: list[tuple[chess.Move, int]] = []
    for move in list(board.legal_moves):
        board.push(move)
        try:
            out.append((move, tbmod._dtz_of_played_move(board, tb, rule50=rule50)))
        except tbmod._TB_PROBE_ERRORS:
            return None
        finally:
            board.pop()
    return out


def _reversible_cycle(board: chess.Board, rng: random.Random) -> list[chess.Move] | None:
    """Four quiet king/piece moves that bring ``board`` back to itself."""
    def quiet(b: chess.Board) -> list[chess.Move]:
        return [m for m in b.legal_moves
                if not b.is_capture(m) and b.piece_type_at(m.from_square) != chess.PAWN]

    for _ in range(20):
        work = board.copy()
        first = quiet(work)
        if not first:
            return None
        m1 = rng.choice(first)
        work.push(m1)
        second = quiet(work)
        if not second:
            continue
        m2 = rng.choice(second)
        work.push(m2)
        m3 = chess.Move(m1.to_square, m1.from_square)
        if m3 not in work.legal_moves:
            continue
        work.push(m3)
        m4 = chess.Move(m2.to_square, m2.from_square)
        if m4 not in work.legal_moves:
            continue
        return [m1, m2, m3, m4]
    return None


def _repetition_boards(path: str, rng: random.Random, n: int) -> list[chess.Board]:
    """Roots whose move stack makes some child a threefold repetition."""
    out: list[chess.Board] = []
    classes = [c for c in ("KRvK", "KQvK", "KQvKR", "KRvKB", "KBNvK") if c in material_classes(path)]
    tries = 0
    while len(out) < n and tries < n * 200 and classes:
        tries += 1
        board = gen_random(rng, rng.choice(classes))
        if board is None or board.is_game_over(claim_draw=False):
            continue
        board.halfmove_clock = rng.choice([0, 10, 40])
        cycle = _reversible_cycle(board, rng)
        if cycle is None:
            continue
        for _ in range(2):
            for move in cycle:
                board.push(move)
        if board.is_game_over(claim_draw=False):
            continue
        out.append(board)
    return out


def _differential_boards(path: str, largest: int, seed: int, per_class: int) -> list[chess.Board]:
    classes = [c for c in material_classes(path) if len(c) - 1 <= largest]
    small = [c for c in classes if len(c) - 1 <= 4]
    big = [c for c in classes if len(c) - 1 > 4]
    picked = small + big[:: max(1, len(big) // 45)]
    out: list[chess.Board] = []
    for cls in picked:
        rng = random.Random(f"{seed}:{cls}")
        got = tries = 0
        while got < per_class and tries < per_class * 100:
            tries += 1
            b = gen_random(rng, cls)
            if b is None:
                continue
            out.append(b)
            got += 1
            if got % 3 == 0:  # the same board just before the 50-move boundary
                near = b.copy(stack=False)
                near.halfmove_clock = rng.choice([90, 95, 97, 98, 99])
                out.append(near)
        for _ in range(8):  # only classes with a pawn on each side yield one
            for _ in range(20):
                b = gen_ep(rng, cls)
                if b is not None:
                    out.append(b)
                    break
        found = 0
        for _ in range(400):  # edge-king roots: mates (and stalemates) in one
            parent = gen_edge_king(rng, cls)
            if parent is None:
                continue
            kids = terminal_children(parent)
            if kids:
                out.append(parent)
                out.append(kids[0])
                found += 1
                if found >= 2:
                    break
    out += _repetition_boards(path, random.Random(f"{seed}:rep"), 24)
    out += _cursed_boards(path, largest, random.Random(f"{seed}:cursed"), 16)
    return out


_CURSED_CLASSES = ("KNNvKP", "KBBvKN", "KRBvKR", "KQvKBB", "KBNvKN", "KRNvKR", "KRPvKR")


def _cursed_boards(path: str, largest: int, rng: random.Random, n: int) -> list[chess.Board]:
    """Roots whose table result is a cursed win / blessed loss (WDL +-1).

    Rare among uniform placements, so they are searched for in the classes
    where they are common; the C WDL probe only SELECTS them (the comparison
    is against python-chess like every other board).
    """
    from chess_anti_engine.tb import _tb_ext

    classes = [c for c in _CURSED_CLASSES if c in material_classes(path) and len(c) - 1 <= largest]
    out: list[chess.Board] = []
    for _ in range(40000):
        if len(out) >= n or not classes:
            break
        board = gen_random(rng, rng.choice(classes))
        if board is None:
            continue
        board.halfmove_clock = rng.choice([0, 0, 5, 30])
        wdl = _tb_ext.probe_wdl_fen(board.fen())
        if wdl in (-1, 1):
            out.append(board)
    return out


def _stats(board: chess.Board, ref: list[tuple[chess.Move, int]] | None, rule50: bool,
           counts: Counter[str]) -> None:
    counts["white" if board.turn == chess.WHITE else "black"] += 1
    if board.ep_square is not None and any(board.generate_legal_ep()):
        counts["en_passant"] += 1
    if board.halfmove_clock >= 90:
        counts["halfmove>=90"] += 1
    if ref is None:
        counts["no_answer"] += 1
        return
    clock = board.halfmove_clock
    key = (lambda d: tbmod._rule50_dtz_key(d, clock)) if rule50 else tbmod._plain_dtz_key
    best = max(ref, key=lambda md: key(md[1]))
    # WDL-greedy: the first legal move with the best result class.
    greedy = next(m for m, d in ref if key(d)[0] == key(best[1])[0])
    if greedy != best[0]:
        counts["dtz_best!=wdl_greedy"] += 1
    if any(abs(d) > 100 for _, d in ref):
        counts["cursed_or_blessed_move"] += 1
    for move, d in ref:
        board.push(move)
        try:
            if board.is_checkmate():
                counts["mate_in_1_root"] += 1
                break
            if board.is_stalemate():
                counts["stalemate_move_root"] += 1
                break
            if board.is_repetition(3):
                counts["threefold_claim_move"] += 1
                break
            if rule50 and board.is_fifty_moves() and d == 0:
                counts["fifty_claim_move"] += 1
                break
        finally:
            board.pop()


@needs_tables
def test_c_root_matches_the_python_chess_root(tables: str, capsys) -> None:
    """Per-move DTZ, chosen move and reported WDL: C root == python-chess root.

    python-chess (``tablebase.try_tb_root_move``) is the reference this switch
    must reproduce exactly, for both ``Syzygy50MoveRule`` settings.
    """
    loaded = tb_root.load(tables)
    largest = min(loaded.largest_dtz, 6)
    boards = _differential_boards(tables, largest, seed=20261114, per_class=12)
    counts: Counter[str] = Counter()
    mismatches: list[str] = []
    for board in boards:
        for rule50 in (True, False):
            handle = tb_root.RootTablebase(loaded, rule50=rule50)
            ref = _py_move_dtz(board, tables, rule50=rule50)
            got = handle.move_dtz(board)
            if got != ref:
                mismatches.append(f"move_dtz rule50={rule50} {board.fen()}: c={got} py={ref}")
                continue
            want = tbmod.try_tb_root_move(board, tables, rule50=rule50)
            pick = handle.root_move(board)
            if pick != want:
                mismatches.append(f"root_move rule50={rule50} {board.fen()}: c={pick} py={want}")
            if rule50:
                counts["positions"] += 1
                _stats(board, ref, rule50, counts)
    with capsys.disabled():
        print(f"\nC-vs-python-chess UCI root differential on {tables} "
              f"(DTZ up to {loaded.largest_dtz}-man): {dict(sorted(counts.items()))}")
    assert not mismatches, f"{len(mismatches)} mismatches, first: {mismatches[:5]}"
    assert counts["positions"] >= 300
    for name, floor in (("white", 50), ("black", 50), ("en_passant", 5), ("halfmove>=90", 50),
                        ("dtz_best!=wdl_greedy", 30), ("mate_in_1_root", 5),
                        ("threefold_claim_move", 3), ("fifty_claim_move", 5)):
        assert counts[name] >= floor, (name, dict(counts))
    if loaded.largest_dtz >= 5:
        assert counts["cursed_or_blessed_move"] >= 1, dict(counts)


def _ref_move(fen: str, path: str) -> str:
    want = tbmod.try_tb_root_move(chess.Board(fen), path)
    assert want is not None
    return want[0].uci()


_REAL_FEN = "8/8/8/4k3/8/8/8/KR6 w - - 0 1"


@needs_tables
def test_engine_go_on_real_tables_plays_the_tb_optimal_move(out, tables: str) -> None:
    engine, worker = _engine()
    try:
        engine.dispatch(CmdSetOption(name="SyzygyPath", value=tables))
        assert "info string Syzygy root prober: C Fathom" in out.text()
        root = worker._tb_root
        assert root is not None
        assert root.tables.n_wdl > 0
        assert root.tables.n_dtz > 0
        engine.dispatch(parse_command(f"position fen {_REAL_FEN}"))
        _go(engine, "go nodes 64")
        assert out.bestmoves()[-1] == _ref_move(_REAL_FEN, tables)
        assert root.root_probes == 1
        last = _info_lines(out)[-1]
        assert _TB_WIN_LINE in last, last
        assert " tbhits 1" in last, last
    finally:
        engine.close()


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
def test_syzygy_path_change_on_real_tables_takes_effect(out, tables: str, tmp_path: Path) -> None:
    """A KRvK root: no shortcut on a KQvK-only path, the TB move after switching
    SyzygyPath to a KRvK-only path, in one session."""
    only_kq = _subset(tmp_path / "kq", tables, ("KQvK",))
    only_kr = _subset(tmp_path / "kr", tables, ("KRvK",))
    engine, worker = _engine()
    try:
        engine.dispatch(CmdSetOption(name="SyzygyPath", value=only_kq))
        engine.dispatch(parse_command(f"position fen {_REAL_FEN}"))
        _go(engine, "go nodes 64")
        assert not any(_TB_WIN_LINE in ln for ln in _info_lines(out))
        engine.dispatch(CmdSetOption(name="SyzygyPath", value=only_kr))
        root = worker._tb_root
        assert root is not None
        assert root.tables.configured == only_kr
        _go(engine, "go nodes 64")
        assert out.bestmoves()[-1] == _ref_move(_REAL_FEN, tables)
        assert _TB_WIN_LINE in _info_lines(out)[-1]
    finally:
        engine.close()


_LAUNCHER = textwrap.dedent("""
    import runpy
    import sys

    import chess.syzygy

    def _instrument(name):
        original = getattr(chess.syzygy.Tablebase, name)

        def probe(self, board, *args, **kwargs):
            sys.stderr.write(f"PYCHESS_SYZYGY_PROBE {name} {board.fen()}\\n")
            sys.stderr.flush()
            return original(self, board, *args, **kwargs)

        setattr(chess.syzygy.Tablebase, name, probe)

    for _name in ("probe_wdl", "probe_dtz", "probe_wdl_table", "probe_dtz_table"):
        _instrument(_name)
    sys.argv[0] = "deepfin"
    runpy.run_module("chess_anti_engine.uci", run_name="__main__", alter_sys=True)
""")


def _read_until(proc: subprocess.Popen[str], lines: list[str], token: str, timeout_s: float) -> None:
    done = threading.Event()

    def pump() -> None:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            lines.append(line.rstrip("\n"))
            if line.split(maxsplit=1)[:1] == [token]:
                done.set()
                return
        done.set()

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    if not done.wait(timeout_s):
        proc.kill()
        pytest.fail(f"no {token!r} within {timeout_s}s; transcript: {lines[-20:]}")
    thread.join(5)


@needs_tables
def test_uci_subprocess_probes_the_root_through_c(tables: str, tmp_path: Path) -> None:
    """Take-effect proof on the real entry point (``python -m chess_anti_engine.uci``).

    Every python-chess Syzygy probe in the engine process writes a marker to
    stderr. The engine must announce the C prober with real table counts and
    answer a KRvK root with the TB move -- without one python-chess probe.
    """
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
        proc.stdin.write(f"position fen {_REAL_FEN}\ngo nodes 64\n")
        proc.stdin.flush()
        _read_until(proc, lines, "bestmove", 180)
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    prober = [ln for ln in lines if ln.startswith("info string Syzygy root prober: C Fathom")]
    assert prober, lines[-30:]
    assert " 0 WDL" not in prober[0], prober[0]
    assert " 0 DTZ" not in prober[0], prober[0]
    assert not [ln for ln in lines if "PYCHESS_SYZYGY_PROBE" in ln], lines[-30:]
    best = [ln for ln in lines if ln.startswith("bestmove ")]
    assert best, lines[-30:]
    assert best[-1].split()[1] == _ref_move(_REAL_FEN, tables), lines[-30:]
    assert any(" tbhits 1" in ln and _TB_WIN_LINE in ln for ln in lines if ln.startswith("info depth"))


@needs_tables
def test_wdl_tables_without_dtz_are_warned_about(out, tables: str, tmp_path: Path) -> None:
    """A path whose KQvK has WDL but no DTZ (half of a split set): loaded, with
    a warning that the root shortcut cannot answer it."""
    wdl_only = tmp_path / "wdl_only"
    wdl_only.mkdir()
    for part in tables.split(os.pathsep):
        src = Path(part) / "KQvK.rtbw"
        if src.exists():
            (wdl_only / src.name).symlink_to(src)
            break
    engine, worker = _engine()
    try:
        engine.dispatch(CmdSetOption(name="SyzygyPath", value=str(wdl_only)))
        root = worker._tb_root
        assert root is not None
        assert (root.tables.n_wdl, root.tables.n_dtz, root.tables.largest_dtz) == (1, 0, 0)
        assert "info string warning: 1 of 1 WDL tables have no DTZ table in SyzygyPath" in out.text()
    finally:
        engine.close()


@needs_tables
def test_selfcheck_syzygy_probes_through_the_c_prober(tables: str, monkeypatch) -> None:
    from chess_anti_engine.uci import selfcheck as sc

    detail = sc.check_syzygy(tables)
    assert "root prober C Fathom" in detail
    monkeypatch.setattr(sc, "_probe_wdl_c", lambda _path, _fen: 99)
    with pytest.raises(sc.CheckFailed, match="C prober 99"):
        sc.check_syzygy(tables)


def test_startup_gate_requires_the_c_tablebase_prober(monkeypatch) -> None:
    from chess_anti_engine import ext_check
    from chess_anti_engine.uci import selfcheck as sc

    real = ext_check.missing_symbols
    monkeypatch.setattr(
        ext_check, "missing_symbols",
        lambda name, symbols: ["table_counts"] if name == sc.TB_EXT else real(name, symbols),
    )
    problems = sc.extension_problems()
    assert any(p.startswith(f"{sc.TB_EXT} lacks table_counts") for p in problems), problems
    assert sc.TB_EXT in sc.UCI_EXTENSIONS
