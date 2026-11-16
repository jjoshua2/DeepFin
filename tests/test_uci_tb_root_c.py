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
from chess_anti_engine.uci import tb_leaf, tb_root
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
        self.validated: list[int] = []
        self.corrupt: dict[int, str] = {}
        self._path: str | None = None
        self._generation = 1
        # Leaf probes (``uci/tb_leaf.py``): every batch's FENs, and the code
        # each leaf gets (default 5: "no table", so leaves keep the net's WDL).
        self.leaf_calls: list[list[str]] = []
        self.leaf_code: Callable[[str], int] = lambda _fen: 5

    def init(self, path: str) -> int:
        self.inits.append(path)
        if not path:
            self._path = None
            return 0
        if "missing" in path:
            raise ValueError(f"Syzygy path component is not a readable directory: {path}")
        self._generation += 1
        self._path = "/fake/" + path
        return self.largest

    def free(self) -> None:
        self._generation += 1
        self._path = None

    def generation(self) -> int:
        return self._generation

    def probe_wdl_batch(self, boards, generation: int) -> bytes | None:
        if generation != self._generation:
            return None
        fens = [cb.fen() for cb in boards]
        self.leaf_calls.append(fens)
        return bytes(self.leaf_code(f) for f in fens)

    def path(self) -> str | None:
        return self._path

    def table_counts(self) -> tuple[int, int, int]:
        return (2, 2, self.largest) if self._path else (0, 0, 0)

    def table_names(self) -> list[tuple[str, bool]]:
        return [("KQvK", True), ("KRvK", True)] if self._path else []

    def validate_table(self, i: int) -> str | None:
        self.validated.append(i)
        return self.corrupt.get(i)

    def probe_root_fen(self, fen: str, has_repeated: bool = False, use_rule50: bool = True):
        # tb_root reads only the per-move DTZ, which neither flag changes; it
        # passes Fathom's defaults and applies the repo's own claim rules.
        assert has_repeated is False
        assert use_rule50 is True
        self.root_calls.append(fen)
        return self.rows(chess.Board(fen))


class _RecordingPyTable:
    """python-chess stand-in for ``get_tablebase``: records every probe.

    The UCI engine must never probe it (root shortcut, root eval and leaves
    all run on the C prober). Its answers make a python-chess root pick
    ``a1a2`` -- a different move from the C stand-in's -- so a root routed
    back to python-chess also changes the bestmove.
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
    monkeypatch.setattr(tb_leaf, "_tb_ext", fake)
    return fake


@pytest.fixture
def py_table(monkeypatch) -> _RecordingPyTable:
    table = _RecordingPyTable()
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: table)
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


# ------------------------------------------- review follow-ups (table-free)


def test_load_validates_every_table_and_refuses_a_corrupt_one(fake_c) -> None:
    tables = tb_root.load("stub-good")
    assert fake_c.validated == [0, 1]
    assert tables.n_wdl == 2
    fake_c.corrupt = {1: "KRvK.rtbz"}
    with pytest.raises(tb_root.RootTablebaseError, match=r"corrupt .*KRvK\.rtbz"):
        tb_root.load("stub-bad")
    assert fake_c.path() is None
    assert tb_root.loaded() is None


def test_reported_wdl_at_halfmove_100_after_a_winning_zeroing_move(fake_c) -> None:
    """Counter 100: every quiet move is a 50-move claim (0); the pawn push
    zeroes the counter and wins in 1, so the reported WDL is a full win (2),
    judged at counter 0, not at 100."""
    board = chess.Board("7k/8/8/8/8/8/P7/KR6 w - - 100 150")
    fake_c.rows = lambda b: [
        (m.uci(), 1 if b.piece_type_at(m.from_square) == chess.PAWN else 30, 0, 0)
        for m in b.legal_moves
    ]
    fake_c.largest = 5
    picked = tb_root.RootTablebase(tb_root.load("stub"), rule50=True).root_move(board)
    assert picked is not None
    assert board.piece_type_at(picked[0].from_square) == chess.PAWN
    assert picked[1] == 2


def test_equal_moves_break_ties_by_python_chess_move_order(fake_c) -> None:
    board = chess.Board(_KRK_FEN)
    fake_c.rows = lambda b: [(m.uci(), 30, 0, 0) for m in b.legal_moves]
    picked = tb_root.RootTablebase(tb_root.load("stub"), rule50=True).root_move(board)
    assert picked is not None
    assert picked[0] == next(iter(board.legal_moves))
    assert picked[0] != list(board.legal_moves)[-1]


def test_a_threefold_child_is_a_draw_whatever_the_table_says(fake_c) -> None:
    """After two a1a2/h8g8/a2a1/g8h8 cycles, a1a2 repeats a position for the
    third time: distance 0, so a longer real win beats the table's win in 3."""
    board = chess.Board(_KRK_FEN)
    for uci in ("a1a2", "h8g8", "a2a1", "g8h8") * 2:
        board.push_uci(uci)
    fake_c.rows = lambda b: [(m.uci(), 3 if m.uci() == "a1a2" else 30, 0, 0) for m in b.legal_moves]
    handle = tb_root.RootTablebase(tb_root.load("stub"), rule50=True)
    dtz = {m.uci(): d for m, d in handle.move_dtz(board) or []}
    assert dtz["a1a2"] == 0
    picked = handle.root_move(board)
    assert picked is not None
    assert picked[0].uci() != "a1a2"
    assert picked[1] == 2


def test_stale_handle_falls_through_to_search_and_close_keeps_the_other_engines_tables(
    out, fake_c, py_table,
) -> None:
    """Two engines in one process share Fathom's single table set."""
    first, first_worker = _engine()
    second, second_worker = _engine()
    try:
        first.dispatch(CmdSetOption(name="SyzygyPath", value="stub-a"))
        second.dispatch(CmdSetOption(name="SyzygyPath", value="stub-b"))  # re-init
        first.dispatch(parse_command(f"position fen {_KRK_FEN}"))
        _go(first, "go nodes 64")
        notes = [ln for ln in out.lines if "root tablebase shortcut disabled" in ln]
        assert len(notes) == 1, out.lines
        assert first_worker._tb_root is None
        assert out.bestmoves(), out.lines
        assert fake_c.root_calls == []  # the stale handle never reached C
        assert "bestmove_fallback_used" not in out.text()
        assert py_table.calls == []
        first.close()
        loaded = tb_root.loaded()
        assert loaded is not None
        assert loaded.configured == "stub-b"
        assert fake_c.path() is not None
        assert second_worker._tb_root is not None
        second.dispatch(parse_command(f"position fen {_KRK_FEN}"))
        _go(second, "go nodes 64")
        assert out.bestmoves()[-1] == _C_BEST
    finally:
        first.close()
        second.close()
    assert fake_c.path() is None


def test_a_refused_path_is_repeated_on_every_isready(out, tmp_path: Path) -> None:
    engine, _ = _engine()
    try:
        engine.dispatch(CmdSetOption(name="SyzygyPath", value=str(tmp_path)))
        for _ in range(2):
            out.lines.clear()
            engine.dispatch(parse_command("isready"))
            lines = list(out.lines)
            assert lines[-1] == "readyok", lines
            assert any("REFUSED by the C tablebase prober" in ln for ln in lines[:-1]), lines
        engine.dispatch(CmdSetOption(name="SyzygyPath", value="<empty>"))
        out.lines.clear()
        engine.dispatch(parse_command("isready"))
        assert not any("REFUSED" in ln for ln in out.lines), out.lines
    finally:
        engine.close()


def test_selfcheck_names_the_table_a_crashing_validation_child_was_on(tmp_path: Path) -> None:
    from chess_anti_engine.uci import selfcheck as sc

    crash = (
        "import os, signal, sys\n"
        f"print({tb_root.VALIDATE_START!r} + 'KQvK', flush=True)\n"
        f"print({tb_root.VALIDATE_START!r} + 'KQvKR', flush=True)\n"
        "os.kill(os.getpid(), signal.SIGSEGV)\n"
    )
    with pytest.raises(sc.CheckFailed, match=r"crashed \(exit -11\) while initialising table KQvKR"):
        sc.validate_tables_subprocess(str(tmp_path), command=[sys.executable, "-c", crash])


# --------------------------------------- review follow-ups (smoke tables)


def _smoke_set() -> Path | None:
    for root in syzygy_roots():
        smoke = root / "data" / "syzygy_3-4man"
        if not missing_tablebase_dirs(str(smoke)):
            return smoke
    return None


_SMOKE = _smoke_set()
needs_smoke = pytest.mark.skipif(
    _SMOKE is None,
    reason="3-4 man smoke Syzygy set (data/syzygy_3-4man) not present (CI ships none): "
           "the corrupt-table refusal tests are NOT run",
)
_CORRUPT_STEMS = ("KQvK", "KRvK", "KQvKR", "KBvK", "KNvK")
_KQKR_FEN = "8/8/2k5/8/8/3K4/1Q6/7r b - - 0 1"


def _corrupt_set(directory: Path, *, mode: str) -> str:
    """Five smoke tables copied; KQvKR corrupted (random tail) or truncated."""
    assert _SMOKE is not None
    directory.mkdir()
    for stem in _CORRUPT_STEMS:
        for ext in (".rtbw", ".rtbz"):
            (directory / f"{stem}{ext}").write_bytes((_SMOKE / f"{stem}{ext}").read_bytes())
    rng = random.Random(20261115)
    for ext in (".rtbw", ".rtbz"):
        f = directory / f"KQvKR{ext}"
        data = bytearray(f.read_bytes())
        if mode == "random":
            data[16:] = bytes(rng.randrange(256) for _ in range(len(data) - 16))
        else:
            data = data[:-7]  # size no longer 16 mod 64: Fathom drops it
        f.write_bytes(bytes(data))
    return str(directory)


_CHILD = textwrap.dedent("""
    import sys
    import chess
    from chess_anti_engine.uci import tb_root
    try:
        tables = tb_root.load(sys.argv[1])
    except tb_root.RootTablebaseError as exc:
        print("REFUSED", exc, flush=True)
        sys.exit(0)
    # Only reached if validation missed it: probe the corrupt material.
    print("LOADED", flush=True)
    print(tb_root.RootTablebase(tables, rule50=True).root_move(chess.Board(sys.argv[2])), flush=True)
""")


@needs_smoke
@pytest.mark.parametrize(("mode", "named", "why"), [
    ("random", "KQvKR.rtbw", "corrupt"),
    ("truncated", "KQvKR.rtbw", "were not loaded"),
])
def test_a_corrupt_table_is_refused_at_load_before_any_probe(
    tmp_path: Path, mode: str, named: str, why: str,
) -> None:
    path = _corrupt_set(tmp_path / "tb", mode=mode)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, path, _KQKR_FEN], capture_output=True, text=True,
        timeout=120, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO_ROOT)),
        check=False,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr[-2000:])
    verdict = [ln for ln in proc.stdout.splitlines() if not ln.startswith("info string ")]
    assert verdict, proc.stdout
    assert verdict[0].startswith("REFUSED"), proc.stdout
    assert named in verdict[0]
    assert why in verdict[0]


@needs_smoke
def test_selfcheck_fails_naming_the_corrupt_table(tmp_path: Path) -> None:
    from chess_anti_engine.uci import selfcheck as sc

    path = _corrupt_set(tmp_path / "tb", mode="random")
    with pytest.raises(sc.CheckFailed, match=r"^table validation: .*KQvKR\.rtbw"):
        sc.check_syzygy(path)


def _spawn_engine(tmp_path: Path) -> subprocess.Popen[str]:
    from chess_anti_engine.uci.selfcheck import _write_tiny_checkpoint

    ckpt = _write_tiny_checkpoint(tmp_path)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO_ROOT))
    cmd = [sys.executable, "-u", "-m", "chess_anti_engine.uci", "--checkpoint", str(ckpt),
           "--no-compile", "--device", "cpu", "--compile-cache-dir", str(tmp_path / "cache")]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)


@needs_smoke
def test_engine_refuses_a_corrupt_set_before_readyok_and_keeps_playing(tmp_path: Path) -> None:
    """The real entry point: the refusal is printed before `readyok` (twice:
    at setoption and at isready), and the corrupt material is then SEARCHED,
    never probed, so the engine answers and exits 0."""
    path = _corrupt_set(tmp_path / "tb", mode="random")
    proc = _spawn_engine(tmp_path)
    lines: list[str] = []
    try:
        assert proc.stdin is not None
        proc.stdin.write("uci\n")
        proc.stdin.flush()
        _read_until(proc, lines, "uciok", 180)
        proc.stdin.write(f"setoption name SyzygyPath value {path}\nisready\n")
        proc.stdin.flush()
        _read_until(proc, lines, "readyok", 180)
        refused = [ln for ln in lines if "REFUSED by the C tablebase prober" in ln]
        assert len(refused) == 2, lines[-20:]
        assert "KQvKR.rtbw" in refused[0]
        proc.stdin.write(f"position fen {_KQKR_FEN}\ngo nodes 32\n")
        proc.stdin.flush()
        _read_until(proc, lines, "bestmove", 180)
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        assert proc.wait(timeout=60) == 0, lines[-20:]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


# ------------------------------- follow-ups to the #1327 independent review


@pytest.mark.usefixtures("py_table")
def test_two_engines_on_one_path_share_tables_until_the_last_one_closes(out, fake_c) -> None:
    """Same SyzygyPath -> same generation. The first engine to close must not
    free the tables the second is still using; the last one must."""
    d, _ = _engine()
    e, e_worker = _engine()
    try:
        d.dispatch(CmdSetOption(name="SyzygyPath", value="stub-tables"))
        e.dispatch(CmdSetOption(name="SyzygyPath", value="stub-tables"))
        assert fake_c.inits == ["stub-tables"]  # one init, one shared generation
        assert d._tb_generation == e._tb_generation
        d.close()
        assert fake_c.path() is not None
        assert tb_root.loaded() is not None
        e.dispatch(parse_command(f"position fen {_KRK_FEN}"))
        _go(e, "go nodes 64")
        assert not [ln for ln in out.lines if "root tablebase shortcut disabled" in ln], out.lines
        assert e_worker._tb_root is not None
        assert fake_c.root_calls == [_KRK_FEN]
        assert out.bestmoves()[-1] == _C_BEST
    finally:
        d.close()
        e.close()
    assert fake_c.path() is None  # the last owner freed them
    assert tb_root.loaded() is None
    # And a fresh engine can load the same path again after the last close.
    f, _ = _engine()
    try:
        f.dispatch(CmdSetOption(name="SyzygyPath", value="stub-tables"))
        assert fake_c.inits == ["stub-tables", "stub-tables"]
        assert fake_c.path() is not None
    finally:
        f.close()
    assert fake_c.path() is None


@pytest.mark.usefixtures("out", "py_table")
def test_rule50_rebuild_and_clear_release_one_owner_once(fake_c) -> None:
    """Re-installing the same path (Syzygy50MoveRule) must not count the engine
    twice: one clear by D then still leaves E's tables, and E's clear frees."""
    d, _ = _engine()
    e, _ = _engine()
    try:
        d.dispatch(CmdSetOption(name="SyzygyPath", value="stub-tables"))
        d.dispatch(CmdSetOption(name="Syzygy50MoveRule", value="false"))
        e.dispatch(CmdSetOption(name="SyzygyPath", value="stub-tables"))
        e.dispatch(CmdSetOption(name="SyzygyPath", value="<empty>"))
        assert fake_c.path() is not None  # D still holds it
        d.dispatch(CmdSetOption(name="SyzygyPath", value="<empty>"))
        assert fake_c.path() is None
    finally:
        d.close()
        e.close()


def test_unload_without_owner_or_generation_still_forces_a_free(fake_c) -> None:
    owner = _Owner()
    other = _Owner()
    tables = tb_root.load("stub-tables", owner=owner)
    assert tb_root.unload(tables.generation, owner=other) is False  # not the owner
    assert fake_c.path() is not None
    assert tb_root.unload() is True
    assert fake_c.path() is None


def test_selfcheck_names_the_table_a_hanging_validation_child_was_on(tmp_path: Path) -> None:
    from chess_anti_engine.uci import selfcheck as sc

    hang = (
        "import time\n"
        f"print({tb_root.VALIDATE_START!r} + 'KQvK', flush=True)\n"
        f"print({tb_root.VALIDATE_START!r} + 'KRvKN', flush=True)\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(sc.CheckFailed, match=r"hung \(no result within 2 s, child killed\) "
                                             r"while initialising table KRvKN"):
        sc.validate_tables_subprocess(str(tmp_path), timeout_s=2,
                                      command=[sys.executable, "-c", hang])


def test_a_refusal_prints_exactly_twice_by_the_first_readyok_then_once_per_isready(
    out, tmp_path: Path,
) -> None:
    """Deliberate: once at `setoption`, once before `readyok` (engine.py,
    _install_tablebase). Two by the first readyok, never three."""
    engine, _ = _engine()
    try:
        engine.dispatch(CmdSetOption(name="SyzygyPath", value=str(tmp_path)))
        engine.dispatch(parse_command("isready"))
        refused = [ln for ln in out.lines if "REFUSED by the C tablebase prober" in ln]
        assert len(refused) == 2, out.lines
        assert out.lines[-1] == "readyok"
        out.lines.clear()
        engine.dispatch(parse_command("isready"))
        assert sum("REFUSED" in ln for ln in out.lines) == 1, out.lines
    finally:
        engine.close()


class _Owner:
    """A weak-referenceable stand-in for an engine owning the C tables."""


@pytest.mark.usefixtures("out", "py_table")
def test_an_owner_collected_without_close_does_not_pin_the_tables(fake_c) -> None:
    """An engine dropped without ``close()`` must not count forever: once it is
    garbage-collected, the remaining owner's release frees the set."""
    import gc

    dropped = _Owner()
    tables = tb_root.load("stub-tables", owner=dropped)
    e, _ = _engine()
    try:
        e.dispatch(CmdSetOption(name="SyzygyPath", value="stub-tables"))
        assert e._tb_generation == tables.generation
        del dropped
        gc.collect()
    finally:
        e.close()
    assert fake_c.path() is None
    assert tb_root.loaded() is None


#: Stray names written by ``_stray_dir`` beyond the three special cases: more
#: than any short cap, so a refusal that names only the first few is caught.
_PLAIN_STRAYS = ("notes.rtbw", "old1.rtbw", "old2.rtbw", "old3.rtbw", "readme.rtbz", "x.rtbz")
_STRAYS = (*_PLAIN_STRAYS, "KvKR.rtbw", "KBvK.rtbw")


def _stray_dir(d: Path, *, tables: tuple[str, ...]) -> Path:
    d.mkdir()
    for name in tables:
        (d / name).write_bytes(b"\0" * 80)
    for name in _PLAIN_STRAYS:
        (d / name).write_text("my notes")
    (d / "KvKR.rtbw").write_bytes(b"\0" * 80)
    os.symlink(d / "gone" / "KBvK.rtbw", d / "KBvK.rtbw")
    return d


@pytest.mark.usefixtures("py_table")
def test_stray_files_refuse_the_path_naming_each_and_saying_remove_or_rename(
    out, fake_c, tmp_path: Path,
) -> None:
    # A long directory name: the engine's refusal line must name every file in
    # full, so it is not cut at the 600 characters other info lines get.
    d = _stray_dir(tmp_path / ("tb-" + "x" * 150),
                   tables=("KQvK.rtbw", "KQvK.rtbz", "KRvK.rtbw", "KRvK.rtbz"))
    with pytest.raises(tb_root.RootTablebaseError) as info:
        tb_root.load(str(d))
    msg = str(info.value)
    assert (f"{len(_STRAYS)} file(s) in the table directories are not loadable Syzygy tables; "
            "remove or rename each one") in msg
    for name in _PLAIN_STRAYS:
        assert f"{d / name} (not a Syzygy table name)" in msg
    assert f"{d / 'notes.rtbw'} (not a Syzygy table name)" in msg
    assert f"{d / 'KvKR.rtbw'} (mirrored name; the table is KRvK.rtbw)" in msg
    assert f"{d / 'KBvK.rtbw'} (broken symlink -> " in msg
    assert fake_c.path() is None  # all tablebases off, no partial set
    engine, _ = _engine()
    try:
        engine.dispatch(CmdSetOption(name="SyzygyPath", value=str(d)))
        line = next(ln for ln in out.lines if "REFUSED by the C tablebase prober" in ln)
        assert len(line) > 600
        for name in _STRAYS:
            assert str(d / name) in line, line
        assert "remove or rename" in line
        assert line.endswith("; tablebase probing disabled")
    finally:
        engine.close()


@needs_smoke
def test_stray_files_beside_real_tables_refuse_the_path(tmp_path: Path) -> None:
    assert _SMOKE is not None
    d = tmp_path / "tb"
    _stray_dir(d, tables=())
    for name in ("KQvK.rtbw", "KQvK.rtbz"):
        (d / name).write_bytes((_SMOKE / name).read_bytes())
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(d), "8/8/8/8/8/2k5/8/KQ6 w - - 0 1"],
        capture_output=True, text=True, timeout=120,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO_ROOT)), check=False,
    )
    verdict = [ln for ln in proc.stdout.splitlines() if not ln.startswith("info string ")]
    assert verdict, (proc.stdout, proc.stderr[-2000:])
    assert verdict[0].startswith("REFUSED"), (proc.stdout, proc.stderr[-2000:])
    assert "remove or rename" in verdict[0]
    for name in _STRAYS:
        assert str(d / name) in verdict[0]
