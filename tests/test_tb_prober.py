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
import os
import random
import re
import shutil
import subprocess
import sys
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
    ep_must_be_refused,
    expected_answers,
    gen_bogus_ep,
    gen_castling,
    gen_castling_letters,
    gen_ep_a1,
    gen_ep_origin_occupied,
    gen_edge_king,
    gen_ep,
    gen_random,
    material_classes,
    open_python_tablebase,
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
    rights = board.castling_rights  # raw, as python-chess's probe and CBoard read them
    for bit, sq in ((1, chess.H1), (2, chess.A1), (4, chess.H8), (8, chess.A8)):
        if rights & chess.BB_SQUARES[sq]:
            castling |= bit
    if rights & ~chess.BB_CORNERS:
        castling |= 16  # a right on a non-corner rook
    return (
        board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
        board.kings, board.queens, board.rooks, board.bishops, board.knights, board.pawns,
        board.ep_square or 0, int(board.turn), castling, board.halfmove_clock,
    )


def _dirty_castling(rng: random.Random, cls: str) -> chess.Board | None:
    """A legal board carrying castling flags its king/rook placement cannot back."""
    b = gen_random(rng, cls)
    if b is None:
        return None
    b.castling_rights = rng.choice([chess.BB_CORNERS, chess.BB_H1, chess.BB_A8, chess.BB_H1 | chess.BB_A8])
    return b


def _sample_boards(seed: int, n: int) -> list[chess.Board]:
    """Legal positions over a spread of material, with ep, castling and clocks."""
    rng = random.Random(seed)
    classes = ["KQvK", "KRvKB", "KPvKP", "KRPvKR", "KBNvKP", "KQRvKQ", "KPPvKPP", "KRvKR"]
    out: list[chess.Board] = []
    gens = (gen_random, gen_ep, gen_castling, _dirty_castling)
    while len(out) < n:
        b = rng.choice(gens)(rng, rng.choice(classes))
        if b is not None:
            out.append(b)
    return out


def _raw_fen(board: chess.Board) -> str:
    """``board.fen()`` with the RAW castling flags (fen() writes cleaned ones,
    which would hide a stale flag from the FEN path)."""
    parts = board.fen(en_passant="fen").split(" ")
    flags = "".join(ch for ch, sq in (("K", chess.H1), ("Q", chess.A1), ("k", chess.H8), ("q", chess.A8))
                    if board.castling_rights & chess.BB_SQUARES[sq])
    parts[2] = flags or "-"
    return " ".join(parts)


def _c_or_refused(fn, arg) -> int | str | None:
    try:
        return fn(arg)
    except ValueError:
        return "ValueError"


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
    assert any(b.castling_rights != b.clean_castling_rights() for b in boards)  # stale flags
    for board in boards:
        want = _python_bitboards(board)
        # The FEN text is judged against python-chess's parse of that text (its
        # parser may drop or re-target a letter); the CBoard against the board.
        fen = _raw_fen(board)
        assert _tb_ext.fen_bitboards(fen) == _python_bitboards(chess.Board(fen)), fen
        cb_want = (*want[:10], want[10] & 15, want[11])  # CBoard: corner bits only
        assert _tb_ext.cboard_bitboards(CBoard.from_board(board)) == cb_want, board.fen()


def test_fen_castling_letters_follow_python_chess_parser() -> None:
    """Each letter is kept, dropped or re-targeted exactly as python-chess's
    Board._set_castling_fen does, so the FEN path refuses what its probe refuses."""
    rng = random.Random(3)
    fens = [f for f in (gen_castling_letters(rng, c) for c in ["KRvK", "KRRvKR", "KQvKR", "KRvKN", "KPvK"] * 80) if f]
    fens += ["8/7K/8/8/5k2/8/8/8 b Qq - 0 1", "4k3/8/8/8/8/8/8/4KR2 w K - 0 1",
             "4k3/8/8/8/8/8/8/3K3R w KQkq - 0 1", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"]
    kinds = set()
    for fen in fens:
        want = _python_bitboards(chess.Board(fen))
        assert _tb_ext.fen_bitboards(fen) == want, fen
        kinds.add((bool(want[10] & 15), bool(want[10] & 16), "-" not in fen.split()[2]))
    assert (False, False, True) in kinds  # letters python-chess drops entirely
    assert (False, True, True) in kinds   # re-targeted to a non-corner rook
    assert (True, False, True) in kinds   # kept on a corner


@pytest.mark.parametrize("castling", ["QQQ", "KKK", "Kkkk", "qqq", "KQkqq"])
def test_fen_castling_field_capped_at_two_letters_per_case(castling: str) -> None:
    """python-chess 1.11's FEN_CASTLING_REGEX allows at most two letters per
    case; a third is malformed there and must be malformed here too."""
    fen = f"4k3/8/8/8/8/8/8/4KQ2 w {castling} - 0 1"
    with pytest.raises(ValueError, match="invalid castling"):
        chess.Board(fen)
    with pytest.raises(ValueError, match="malformed FEN"):
        _tb_ext.fen_bitboards(fen)
    with pytest.raises(ValueError, match="malformed FEN"):
        _tb_ext.probe_wdl_fen(fen)
    ok = f"4k3/8/8/8/8/8/8/4KQ2 w {castling[:2]} - 0 1"
    assert _tb_ext.fen_bitboards(ok) == _python_bitboards(chess.Board(ok))


def test_fen_needs_single_spaces_unlike_python_chess() -> None:
    """Documented difference: python-chess accepts runs of spaces."""
    fen = "8/8/8/8/8/8/8/K6k  w - - 0 1"
    assert chess.Board(fen).fen() == "8/8/8/8/8/8/8/K6k w - - 0 1"
    with pytest.raises(ValueError, match="malformed FEN"):
        _tb_ext.probe_wdl_fen(fen)


def test_ep_on_a1_means_no_ep_like_python_chess() -> None:
    fen = "8/8/8/1P6/8/1k6/4K3/8 w - a1 0 1"
    assert not chess.Board(fen).ep_square
    assert _tb_ext.fen_bitboards(fen)[8] == 0
    assert _tb_ext.probe_wdl_fen(fen) == _tb_ext.probe_wdl_fen(fen.replace(" a1 ", " - "))


def test_relative_path_survives_a_chdir(tmp_path: Path) -> None:
    """Fathom opens tables lazily; a relative component would be re-resolved
    against the cwd of the first probe. init() must hand it absolute paths."""
    if _TABLES is None:
        pytest.skip("no Syzygy tables here: the lazy-open path cannot be exercised")
    first = _TABLES.split(":")[0]
    rel = os.path.relpath(first, tmp_path)
    code = (
        "import os, sys\n"
        "from chess_anti_engine.tb import _tb_ext\n"
        f"os.chdir({str(tmp_path)!r})\n"
        f"n = _tb_ext.init({rel!r})\n"
        "assert os.path.isabs(_tb_ext.path()), _tb_ext.path()\n"
        "os.chdir('/')\n"
        "print(n, _tb_ext.probe_dtz_fen('8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1'))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=120, check=False, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    assert proc.returncode == 0, proc.stderr
    largest, dtz = proc.stdout.split()
    assert int(largest) >= 4
    assert dtz == "-53"


def test_init_refuses_a_path_that_loads_nothing(tmp_path: Path) -> None:
    """A missing component, or a path holding no tables, is an error -- never
    an accepted value that silently probes nothing."""
    with pytest.raises(ValueError, match="no Syzygy tables found"):
        _tb_ext.init(str(tmp_path))
    assert _tb_ext.path() is None
    assert _tb_ext.largest() == 0
    with pytest.raises(ValueError, match="not a readable directory"):
        _tb_ext.init(str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="not a readable directory"):
        _tb_ext.init(f"{tmp_path}:{tmp_path / 'missing'}")
    with pytest.raises(ValueError, match="empty or oversized component"):
        _tb_ext.init(f"{tmp_path}::{tmp_path}")
    assert _tb_ext.init("") == 0  # empty path: a valid "no tablebases"
    assert _tb_ext.path() is None


def test_no_tables_means_every_probe_is_unavailable() -> None:
    assert _tb_ext.init("") == 0
    assert _tb_ext.largest() == 0
    fen = "8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1"
    assert _tb_ext.probe_wdl_fen(fen) is None
    assert _tb_ext.probe_dtz_fen(fen) is None
    assert _tb_ext.probe_root_fen(fen) is None
    assert _tb_ext.probe_wdl(CBoard.from_board(chess.Board(fen))) is None


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
    "4k3/8/8/8/8/8/8/4K3 w - e9 0 1", "4k3/8/8/8/8/8/8/4K3 w Z - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - - x 1", "4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra",
])
def test_malformed_fen_raises(fen: str) -> None:
    with pytest.raises(ValueError, match="malformed FEN"):
        _tb_ext.fen_bitboards(fen)


# Round-2 review cases: each must equal python-chess.
A1_FEN = "8/8/8/1P6/8/1k6/4K3/8 w - a1 0 1"          # ep a1 = none
ORIGIN_FEN = "8/8/8/8/3pP3/1K6/4N3/4k3 b - e3 0 1"   # real pawn on e4, knight on e2
LETTERS_FEN = "8/7K/8/8/5k2/8/8/8 b Qq - 0 1"        # letters python-chess's parser drops
NONCORNER_FEN = "4k3/8/8/8/8/8/8/4KR2 w K - 0 1"     # right on f1: python-chess KeyError
# Round-3: 4-man origin-occupied ep (a king on ep+/-8), one pair per side to
# move, so the 3-4 man smoke set covers the pass-through on both branches.
ORIGIN_4MAN = (
    ("8/8/8/8/3pP3/8/4K3/7k b - e3 0 1", (0, 0)),   # black to move, white king on e2
    ("8/8/8/8/3pP3/8/4k3/K7 b - e3 0 1", (2, 1)),   # black to move, black king on e2
    ("7k/3K4/8/3pP3/8/8/8/8 w - d6 0 1", (2, 1)),   # white to move, white king on d7
    ("K7/3k4/8/3pP3/8/8/8/8 w - d6 0 1", (0, 0)),   # white to move, black king on d7
)

# The review's crash: Fathom's ep capture deletes whatever stands on ep-8, here
# the black king, and the probe segfaulted (exit 139) on every entry point.
CRASH_FEN = "8/8/8/5Pk1/8/8/3K4/1R6 w - g6 0 1"
# Same cause without a crash: an impossible ep square nobody can capture onto
# changed Fathom's DTZ (C said 3, python-chess 1). It must now be ignored.
WRONG_DTZ_FEN = "8/8/8/1P6/8/1k6/4K3/8 w - b6 52 1"


def test_impossible_ep_with_a_capturer_is_refused_on_every_path() -> None:
    """Table-free: the gate runs before any table is consulted."""
    board = chess.Board(CRASH_FEN)
    assert ep_must_be_refused(board)
    cb = CBoard.from_board(board)
    for fen_fn in (_tb_ext.probe_wdl_fen, _tb_ext.probe_dtz_fen, _tb_ext.probe_root_fen):
        with pytest.raises(ValueError, match="not a legal chess position"):
            fen_fn(CRASH_FEN)
    for cb_fn in (_tb_ext.probe_wdl, _tb_ext.probe_dtz, _tb_ext.probe_root):
        with pytest.raises(ValueError, match="not a legal chess position"):
            cb_fn(cb)


def test_illegal_in_check_position_raises_before_any_init() -> None:
    """Fathom's legality check reads attack tables built by its first tb_init;
    the module builds them at import, so a fresh process already refuses."""
    code = (
        "from chess_anti_engine.tb import _tb_ext\n"
        "try:\n"
        "    _tb_ext.probe_wdl_fen('4k3/4R3/8/8/8/8/8/4K3 w - - 0 1')\n"
        "except ValueError:\n"
        "    print('refused')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=120, check=False,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "refused"


def test_crash_fuzz_in_a_subprocess(tmp_path: Path) -> None:
    """20k deterministic random/mutated inputs through every entry point.

    In a subprocess so a segfault fails this test with the offending FEN
    instead of killing pytest. Uses tables when present (the crash lived in
    the probe itself, after the gate); without them it still exercises the
    gate and both readers.
    """
    out = tmp_path / "fuzz.out"
    cmd = [sys.executable, "-X", "faulthandler", str(REPO_ROOT / "scripts" / "tb_prober_fuzz.py"),
           "--n", "20000", "--seed", "1114"]
    if _TABLES is not None:
        cmd += ["--syzygy-path", _TABLES]
    with out.open("w") as fh:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.PIPE, text=True,
                              timeout=1800, check=False,
                              env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "CUDA_VISIBLE_DEVICES": ""})
    lines = out.read_text().splitlines()
    last_fen = next((ln[4:] for ln in reversed(lines) if ln.startswith("FEN ")), None)
    assert proc.returncode == 0, f"prober died (rc={proc.returncode}) on {last_fen!r}\n{proc.stderr[-3000:]}"
    assert lines[-1].startswith("DONE"), lines[-1:]
    assert "'inputs': 20000" in lines[-1]


def test_only_the_module_init_symbol_is_exported() -> None:
    nm = shutil.which("nm")
    if nm is None:
        pytest.skip("nm not installed: cannot list dynamic symbols")
    so = Path(_tb_ext.__file__)
    proc = subprocess.run([nm, "-D", "--defined-only", str(so)], capture_output=True, text=True,
                          check=True)
    names = {line.split()[-1] for line in proc.stdout.splitlines() if line.strip()}
    assert names == {"PyInit__tb_ext"}, sorted(names)


def test_fathom_notice_ships_in_the_package(tmp_path: Path) -> None:
    """MIT requires the notice with every copy: it must be package data."""
    import setuptools

    major = int(setuptools.__version__.split(".")[0])
    if major < 61:
        pytest.skip(f"setuptools {setuptools.__version__} < 61 cannot read [tool.setuptools.package-data] "
                    "from pyproject.toml (build-system requires >= 70)")
    proc = subprocess.run([sys.executable, "setup.py", "-q", "egg_info", "--egg-base", str(tmp_path)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=600, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    sources = next(tmp_path.glob("*.egg-info")).joinpath("SOURCES.txt").read_text().splitlines()
    for rel in ("fathom/LICENSE", "fathom/VENDORED.md"):
        assert f"chess_anti_engine/tb/{rel}" in sources


def test_probe_rejects_non_cboard() -> None:
    with pytest.raises(TypeError):
        _tb_ext.probe_wdl(chess.Board())  # pyright: ignore[reportArgumentType]


# -------------------------------------------------------------- table tier

def _find_tables() -> str | None:
    if os.environ.get("CAE_SYZYGY_PATH", "").strip():
        # Explicitly requested: use exactly that or fail (see `tables`), never
        # fall back to some other table set.
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
        "no Syzygy tables here (CI ships none): C-vs-python-chess equivalence NOT "
        "checked; set CAE_SYZYGY_PATH to run it"
    ),
)


@pytest.fixture(scope="module")
def tables() -> Iterator[tuple[str, chess.syzygy.Tablebase]]:
    assert _TABLES is not None
    missing = missing_tablebase_dirs(_TABLES)
    if missing:
        pytest.fail(f"CAE_SYZYGY_PATH={_TABLES!r} is set but {missing} hold no tables")
    tb = open_python_tablebase(_TABLES)
    yield _TABLES, tb
    tb.close()


def _covered_classes(path: str, largest: int) -> list[str]:
    classes = [c for c in material_classes(path) if len(c) - 1 <= largest]
    # A deterministic spread: every 3-4 man class, then a stride through the rest.
    small = [c for c in classes if len(c) - 1 <= 4]
    big = [c for c in classes if len(c) - 1 > 4]
    return small + big[:: max(1, len(big) // 40)]


def _class_sample(path: str, largest: int, per_class: int, seed: int,
                  *, odd_inputs: bool = False) -> list[chess.Board]:
    """``odd_inputs`` adds a bogus-ep and a stale-castling board per class."""
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
        gens = (gen_ep, gen_castling, gen_bogus_ep, _dirty_castling) if odd_inputs else (gen_ep, gen_castling)
        for gen in gens:
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
    boards = _class_sample(path, largest, per_class=12, seed=11, odd_inputs=True)
    boards += [chess.Board(WRONG_DTZ_FEN), chess.Board(CRASH_FEN)]
    # Every item is (FEN text, python-chess's parse of it): a board whose raw
    # castling mask was set directly may not survive the text round trip.
    items = [(_raw_fen(b), chess.Board(_raw_fen(b))) for b in boards]
    rng = random.Random(29)
    for cls in _covered_classes(path, largest)[:60]:
        for gen in (gen_ep_a1, gen_castling_letters):
            fen = gen(rng, cls)
            if fen:
                items.append((fen, chess.Board(fen)))
        origin = gen_ep_origin_occupied(rng, cls)
        if origin is not None:
            items.append((_raw_fen(origin), chess.Board(_raw_fen(origin))))
    items += [(f, chess.Board(f)) for f in (A1_FEN, ORIGIN_FEN, LETTERS_FEN, NONCORNER_FEN)
              if chess.popcount(chess.Board(f).occupied) <= largest]
    seen = {"black": 0, "ep": 0, "castling": 0, "stale_castling": 0, "bogus_ep_ignored": 0,
            "bogus_ep_refused": 0, "origin_occupied": 0, "noncorner": 0, "mate": 0, "clock99": 0,
            "probed": 0}
    bad = []
    for fen, board in items:
        refuse = ep_must_be_refused(board)
        want, cb_want = expected_answers(tb, board)
        cb = CBoard.from_board(board)
        got = (_c_or_refused(_tb_ext.probe_wdl_fen, fen), _c_or_refused(_tb_ext.probe_dtz_fen, fen),
               _c_or_refused(_tb_ext.probe_wdl, cb), _c_or_refused(_tb_ext.probe_dtz, cb))
        if got != (*want, *cb_want):
            bad.append((fen, want, cb_want, got))
        seen["origin_occupied"] += (not refuse and bool(board.ep_square)
                                    and bool(board.occupied & chess.BB_SQUARES[board.ep_square + (8 if board.turn else -8)])
                                    if board.ep_square and 16 <= board.ep_square < 48 else False)
        seen["noncorner"] += bool(board.castling_rights & ~chess.BB_CORNERS)
        invalid_ep = bool(board.status() & chess.STATUS_INVALID_EP_SQUARE)
        seen["black"] += board.turn == chess.BLACK
        seen["ep"] += not invalid_ep and board.has_legal_en_passant()
        seen["castling"] += bool(board.castling_rights)
        seen["stale_castling"] += board.castling_rights != board.clean_castling_rights()
        seen["bogus_ep_ignored"] += invalid_ep and not refuse
        seen["bogus_ep_refused"] += refuse
        seen["mate"] += not invalid_ep and board.is_checkmate()
        seen["clock99"] += board.halfmove_clock == 99
        seen["probed"] += isinstance(want[0], int)
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
def test_impossible_ep_without_a_capturer_is_ignored_like_python_chess(tables) -> None:
    path, tb = tables
    _tb_ext.init(path)
    board = chess.Board(WRONG_DTZ_FEN)
    assert not ep_must_be_refused(board)
    want = (tb.probe_wdl(board), tb.probe_dtz(board))
    assert want == (2, 1)
    cb = CBoard.from_board(board)
    assert (_tb_ext.probe_wdl_fen(WRONG_DTZ_FEN), _tb_ext.probe_dtz_fen(WRONG_DTZ_FEN)) == want
    assert (_tb_ext.probe_wdl(cb), _tb_ext.probe_dtz(cb)) == want
    plain = WRONG_DTZ_FEN.replace(" b6 ", " - ")
    assert _tb_ext.probe_root_fen(WRONG_DTZ_FEN) == _tb_ext.probe_root_fen(plain)
    assert _tb_ext.probe_root(cb) == _tb_ext.probe_root_fen(plain)


ROOT_KBN = "8/8/8/4k3/8/8/8/KBN5 w - - {clock} 1"


@needs_tables
def test_root_score_follows_use_rule50_and_rank_follows_the_clock(tables) -> None:
    """``use_rule50`` changes only Fathom's score; the rank is always clock-aware;
    ``has_repeated`` drops a win out of the 1000 bucket."""
    path, _ = tables
    _tb_ext.init(path)
    mate_score = 31744  # TB_VALUE_MATE - TB_MAX_MATE_PLY - 1 in Fathom's scale

    def first(clock: int, **kw) -> tuple[str, int, int, int]:
        rows = _tb_ext.probe_root_fen(ROOT_KBN.format(clock=clock), **kw)
        assert rows is not None
        return next(r for r in rows if r[0] == "a1a2")

    assert first(90) == ("a1a2", 60, 850, 450)                       # cursed band, scored as such
    assert first(90, use_rule50=False) == ("a1a2", 60, 850, mate_score)  # same rank, win score
    assert first(0) == ("a1a2", 60, 1000, mate_score)
    assert first(0, use_rule50=False) == ("a1a2", 60, 1000, mate_score)
    assert first(0, has_repeated=True)[:3] == ("a1a2", 60, 940)       # 1000 - (60 + 0)
    cb = CBoard.from_board(chess.Board(ROOT_KBN.format(clock=90)))
    assert next(r for r in _tb_ext.probe_root(cb, use_rule50=False) or [] if r[0] == "a1a2")[3] == mate_score


@needs_tables
def test_round_two_review_positions(tables) -> None:
    path, tb = tables
    largest = _tb_ext.init(path)
    for fen, want in ((A1_FEN, (2, 1)), (ORIGIN_FEN, (2, 1)), (LETTERS_FEN, (0, 0)), *ORIGIN_4MAN):
        board = chess.Board(fen)
        if chess.popcount(board.occupied) > largest:
            continue  # above the loaded tables (the 3-4 man smoke set): not a table answer
        assert (tb.get_wdl(board), tb.get_dtz(board)) == want, fen
        assert (_tb_ext.probe_wdl_fen(fen), _tb_ext.probe_dtz_fen(fen)) == want, fen
        cb = CBoard.from_board(board)
        assert (_tb_ext.probe_wdl(cb), _tb_ext.probe_dtz(cb)) == want, fen
    # The one castling difference, by design: CBoard cannot see an f1 right.
    board = chess.Board(NONCORNER_FEN)
    assert tb.get_wdl(board) is None
    assert _tb_ext.probe_wdl_fen(NONCORNER_FEN) is None
    assert _tb_ext.probe_wdl(CBoard.from_board(board)) == 2


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
    # Stale flags (king off e1): python-chess's probe reads the RAW rights and
    # refuses (KeyError), and CBoard copies them raw -- the FEN path agrees.
    stale = "4k3/8/8/8/8/8/8/3K3R w KQkq - 0 1"
    assert chess.Board(stale).clean_castling_rights() == 0
    assert chess.Board(stale).castling_rights != 0
    with pytest.raises(KeyError):
        tb.probe_wdl(chess.Board(stale))
    assert _tb_ext.probe_wdl_fen(stale) is None
    assert _tb_ext.probe_wdl(CBoard.from_board(chess.Board(stale))) is None


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
