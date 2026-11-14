"""C Syzygy prober (``chess_anti_engine.tb._tb_ext``, vendored Fathom) vs python-chess.

Two tiers:

* table-free tests run everywhere (CI included): vendored-file hashes, the FEN
  and CBoard readers against python-chess, empty-table and illegal-input
  behaviour;
* table tests need real Syzygy files. CI ships none, so they SKIP there with
  the reason printed. When tables ARE found (``$CAE_SYZYGY_PATH``, the
  production pair via ``utils.syzygy.default_syzygy_path``, or the
  ``data/syzygy_3-4man`` smoke set) they run and any disagreement FAILS.

The full 200k-position proof is ``scripts/tb_prober_equivalence.py``; these
tests reuse its generators on a smaller deterministic sample.
"""
from __future__ import annotations

import hashlib
import random
import re
import threading
from collections.abc import Iterator
from pathlib import Path

import chess
import chess.syzygy
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.tb import _tb_ext
from chess_anti_engine.utils.syzygy import default_syzygy_path, missing_tablebase_dirs, syzygy_roots
from scripts.tb_prober_equivalence import (
    compare_root,
    gen_castling,
    gen_edge_king,
    gen_ep,
    gen_random,
    material_classes,
    open_python_tablebase,
    py_probe,
    py_root,
    terminal_children,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_Row = tuple[int | None, int | None, list[tuple[str, int, int, int]] | None]
FATHOM_DIR = REPO_ROOT / "chess_anti_engine" / "tb" / "fathom"


# ---------------------------------------------------------------- helpers

def _python_bitboards(board: chess.Board) -> tuple[int, ...]:
    """The tuple ``_tb_ext.fen_bitboards`` returns, built from python-chess."""
    castling = 0
    for bit, sq in ((1, chess.H1), (2, chess.A1), (4, chess.H8), (8, chess.A8)):
        if board.castling_rights & chess.BB_SQUARES[sq]:
            castling |= bit
    return (
        board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
        board.kings, board.queens, board.rooks, board.bishops, board.knights, board.pawns,
        board.ep_square or 0, int(board.turn), castling, board.halfmove_clock,
    )


def _sample_boards(seed: int, n: int) -> list[chess.Board]:
    """Legal positions over a spread of material, with ep, castling and clocks."""
    rng = random.Random(seed)
    classes = ["KQvK", "KRvKB", "KPvKP", "KRPvKR", "KBNvKP", "KQRvKQ", "KPPvKPP", "KRvKR"]
    out: list[chess.Board] = []
    gens = (gen_random, gen_ep, gen_castling)
    while len(out) < n:
        b = rng.choice(gens)(rng, rng.choice(classes))
        if b is not None:
            out.append(b)
    return out


@pytest.fixture(autouse=True)
def _restore_tables():
    """_tb_ext is a process-wide singleton: put back whatever was loaded."""
    before = _tb_ext.path()
    yield
    _tb_ext.init(before or "")


# ---------------------------------------------------------- table-free tier

def test_vendored_fathom_files_match_recorded_hashes() -> None:
    """VENDORED.md records the sha256 of every upstream file; they must hold."""
    table = (FATHOM_DIR / "VENDORED.md").read_text()
    rows = dict(re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$", table, re.M))
    vendored = sorted(
        p.relative_to(FATHOM_DIR).as_posix()
        for p in FATHOM_DIR.rglob("*") if p.is_file() and p.name != "VENDORED.md"
    )
    assert sorted(rows) == vendored
    for rel, digest in rows.items():
        assert hashlib.sha256((FATHOM_DIR / rel).read_bytes()).hexdigest() == digest, rel


def test_fen_and_cboard_readers_match_python_chess() -> None:
    boards = _sample_boards(7, 400)
    assert any(b.turn == chess.BLACK for b in boards)
    assert any(b.turn == chess.WHITE for b in boards)
    assert any(b.ep_square for b in boards)
    assert any(b.castling_rights for b in boards)
    for board in boards:
        want = _python_bitboards(board)
        assert _tb_ext.fen_bitboards(board.fen(en_passant="fen")) == want, board.fen()
        assert _tb_ext.cboard_bitboards(CBoard.from_board(board)) == want, board.fen()


def test_no_tables_means_every_probe_is_unavailable(tmp_path: Path) -> None:
    assert _tb_ext.init(str(tmp_path)) == 0
    assert _tb_ext.largest() == 0
    assert _tb_ext.path() == str(tmp_path)
    fen = "8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1"
    assert _tb_ext.probe_wdl_fen(fen) is None
    assert _tb_ext.probe_dtz_fen(fen) is None
    assert _tb_ext.probe_root_fen(fen) is None
    assert _tb_ext.probe_wdl(CBoard.from_board(chess.Board(fen))) is None
    assert _tb_ext.init("") == 0
    assert _tb_ext.path() is None


@pytest.mark.parametrize("fen", [
    "8/8/8/8/8/8/8/8 w - - 0 1",                 # no kings
    "4k3/4R3/8/8/8/8/8/4K3 w - - 0 1",           # black (not to move) in check
    "P3k3/8/8/8/8/8/8/4K3 w - - 0 1",            # pawn on the back rank
])
def test_illegal_positions_raise(fen: str) -> None:
    assert not chess.Board(fen).is_valid()
    with pytest.raises(ValueError, match="not a legal chess position"):
        _tb_ext.probe_wdl_fen(fen)
    with pytest.raises(ValueError, match="not a legal chess position"):
        _tb_ext.probe_dtz_fen(fen)


@pytest.mark.parametrize("fen", [
    "", "8/8/8/8/8/8/8 w - - 0 1", "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - e4 0 1", "4k3/8/8/8/8/8/8/4K3 w Z - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - - x 1", "4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra",
])
def test_malformed_fen_raises(fen: str) -> None:
    with pytest.raises(ValueError, match="malformed FEN"):
        _tb_ext.fen_bitboards(fen)


def test_probe_rejects_non_cboard() -> None:
    with pytest.raises(TypeError):
        _tb_ext.probe_wdl(chess.Board())  # pyright: ignore[reportArgumentType]


# -------------------------------------------------------------- table tier

def _find_tables() -> str | None:
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
        "no Syzygy tables here (CI ships none): C-vs-python-chess equivalence NOT "
        "checked; set CAE_SYZYGY_PATH to run it"
    ),
)


@pytest.fixture(scope="module")
def tables() -> Iterator[tuple[str, chess.syzygy.Tablebase]]:
    assert _TABLES is not None
    tb = open_python_tablebase(_TABLES)
    yield _TABLES, tb
    tb.close()


def _covered_classes(path: str, largest: int) -> list[str]:
    classes = [c for c in material_classes(path) if len(c) - 1 <= largest]
    # A deterministic spread: every 3-4 man class, then a stride through the rest.
    small = [c for c in classes if len(c) - 1 <= 4]
    big = [c for c in classes if len(c) - 1 > 4]
    return small + big[:: max(1, len(big) // 40)]


def _class_sample(path: str, largest: int, per_class: int, seed: int) -> list[chess.Board]:
    out: list[chess.Board] = []
    for cls in _covered_classes(path, largest):
        rng = random.Random(f"{seed}:{cls}")
        got = tries = 0
        while got < per_class and tries < per_class * 100:
            tries += 1
            b = gen_random(rng, cls)
            if b is not None:
                out.append(b)
                got += 1
        for gen in (gen_ep, gen_castling):
            for _ in range(20):
                b = gen(rng, cls)
                if b is not None:
                    out.append(b)
                    break
        for _ in range(200):
            parent = gen_edge_king(rng, cls)
            kids = terminal_children(parent) if parent is not None else []
            if kids:
                out.append(kids[0])
                break
    return out


@needs_tables
def test_wdl_and_dtz_equal_python_chess(tables) -> None:
    path, tb = tables
    largest = _tb_ext.init(path)
    assert largest >= 4
    boards = _class_sample(path, largest, per_class=12, seed=11)
    seen = {"black": 0, "ep": 0, "castling": 0, "mate": 0, "clock99": 0, "probed": 0}
    bad = []
    for board in boards:
        py_wdl, py_dtz = py_probe(tb, board)
        fen = board.fen(en_passant="fen")
        cb = CBoard.from_board(board)
        got = (_tb_ext.probe_wdl_fen(fen), _tb_ext.probe_dtz_fen(fen),
               _tb_ext.probe_wdl(cb), _tb_ext.probe_dtz(cb))
        if got != (py_wdl, py_dtz, py_wdl, py_dtz):
            bad.append((fen, (py_wdl, py_dtz), got))
        seen["black"] += board.turn == chess.BLACK
        seen["ep"] += board.has_legal_en_passant()
        seen["castling"] += bool(board.castling_rights)
        seen["mate"] += board.is_checkmate()
        seen["clock99"] += board.halfmove_clock == 99
        seen["probed"] += py_wdl is not None
    assert not bad, bad[:10]
    assert all(seen.values()), seen  # every category was actually exercised


@needs_tables
def test_every_directory_of_the_path_is_searched(tables, tmp_path: Path) -> None:
    """An empty FIRST component must not hide the tables in the second."""
    path, tb = tables
    largest = _tb_ext.init(path)
    assert _tb_ext.init(f"{tmp_path}:{path}") == largest
    fen = "8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1"
    assert _tb_ext.probe_dtz_fen(fen) == tb.probe_dtz(chess.Board(fen))
    if largest >= 6:
        # Production layout: 6-man DTZ lives only in the second directory.
        six = "8/8/8/3k4/8/2R5/1KB3N1/7n w - - 0 1"
        assert _tb_ext.init(path) == 6
        assert _tb_ext.probe_dtz_fen(six) == tb.probe_dtz(chess.Board(six))
        assert _tb_ext.probe_dtz_fen(six) is not None


@needs_tables
def test_root_ranking_matches_repo_root_logic(tables) -> None:
    """Per-move DTZ and Fathom rank vs ``tablebase._dtz_of_played_move``.

    Includes halfmove clocks of 99, where Fathom's rank depends on the clock
    and the repo's root code treats a non-zeroing move as a fifty-move claim.
    """
    path, tb = tables
    largest = _tb_ext.init(path)
    rng = random.Random(5)
    boards = [b for b in _class_sample(path, largest, per_class=3, seed=17)
              if not b.castling_rights and not b.is_game_over()]
    rng.shuffle(boards)
    boards = boards[:250]
    for board in boards[::2]:
        board.halfmove_clock = 99
    clock_sensitive = 0
    for board in boards:
        c_rows = _tb_ext.probe_root(CBoard.from_board(board))
        res = compare_root(board, py_root(board, tb), c_rows)
        assert res["ok"], (board.fen(), res["why"])
        if board.halfmove_clock == 99 and c_rows:
            clock_sensitive += any(rank not in (1000, -1000, 0) for _, _, rank, _ in c_rows)
    assert clock_sensitive > 10


@needs_tables
def test_castling_rights_are_never_probed(tables) -> None:
    path, tb = tables
    _tb_ext.init(path)
    fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    board = chess.Board(fen)
    with pytest.raises(KeyError):
        tb.probe_wdl(board)
    assert _tb_ext.probe_wdl_fen(fen) is None
    assert _tb_ext.probe_dtz_fen(fen) is None
    assert _tb_ext.probe_root_fen(fen) is None
    assert _tb_ext.probe_wdl(CBoard.from_board(board)) is None
    assert _tb_ext.probe_wdl_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1") == 2


@needs_tables
def test_concurrent_probes_agree_with_serial(tables) -> None:
    """GIL-released WDL/DTZ probes and lock-serialised root probes from threads."""
    path, _ = tables
    largest = _tb_ext.init(path)
    boards = [b for b in _class_sample(path, largest, per_class=2, seed=23) if not b.castling_rights]
    cbs = [CBoard.from_board(b) for b in boards]
    serial = [(_tb_ext.probe_wdl(cb), _tb_ext.probe_dtz(cb), _tb_ext.probe_root(cb)) for cb in cbs]
    results: dict[int, list[_Row | None]] = {}
    errors: list[BaseException] = []

    def work(tid: int) -> None:
        try:
            order = list(range(len(cbs)))
            random.Random(tid).shuffle(order)
            out: list[_Row | None] = [None] * len(cbs)
            for i in order:
                out[i] = (_tb_ext.probe_wdl(cbs[i]), _tb_ext.probe_dtz(cbs[i]), _tb_ext.probe_root(cbs[i]))
            results[tid] = out
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for out in results.values():
        assert out == serial
