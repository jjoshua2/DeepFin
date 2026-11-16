"""Real-table tests that run in CI: the committed 3-man Syzygy fixture.

CI ships no tablebases, so every other real-table test of the C prober SKIPs
there. ``tests/fixtures/syzygy_3man/`` holds ten standard table files
(the complete 3-man set; 25,824 bytes; README beside them), enough for CI to see a
wrong VALUE from the C batch path, not just wiring:

* known W/D/L codes from ``probe_wdl_batch`` (both colours, mate, stalemate,
  a draw by capture into KvK);
* ``CLeafProbe.apply`` == ``SyzygyProbe.apply`` (python-chess) over sampled
  KQvK / KRvK / KPvK leaves and their children, both ``Syzygy50MoveRule``
  settings;
* the C root shortcut == ``tablebase.try_tb_root_move`` on the same roots;
* a real ``SyzygyPath`` engine subprocess on the fixture.

Nothing here reads a path outside the repository.
"""
from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import chess
import numpy as np
import pytest

import chess_anti_engine.tablebase as tbmod
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.tb import _tb_ext
from chess_anti_engine.uci import tb_root
from chess_anti_engine.uci.tb_leaf import CLeafProbe
from tests.test_uci_tb_leaf_c import _LAUNCHER, _apply_both
from tests.test_uci_tb_root_c import _read_until

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "syzygy_3man"
STEMS = ("KQvK", "KRvK", "KPvK", "KBvK", "KNvK")
SAMPLED = ("KQvK", "KRvK", "KPvK")
MANIFEST = REPO_ROOT / "chess_anti_engine" / "tb" / "syzygy_standard_3-4-5-6.md5"


@pytest.fixture(autouse=True)
def _no_c_tables_left_behind() -> Iterator[None]:
    tb_root.unload()
    yield
    tb_root.unload()


def test_fixture_is_the_standard_tables_and_small() -> None:
    files = sorted(p.name for p in FIXTURE.iterdir() if p.suffix in (".rtbw", ".rtbz"))
    assert files == sorted(f"{s}{e}" for s in STEMS for e in (".rtbw", ".rtbz"))
    manifest = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split()[:2]
            manifest[name.lstrip("*").rsplit("/", 1)[-1]] = digest
    total = 0
    for name in files:
        data = (FIXTURE / name).read_bytes()
        total += len(data)
        assert hashlib.md5(data).hexdigest() == manifest[name], name
    assert total < 64_000, total


# (fen, WDL from the side to move) -- checked against python-chess too.
KNOWN = [
    ("8/8/8/4k3/8/8/8/KQ6 w - - 0 1", 2),       # KQvK, White to move wins
    ("8/8/8/4k3/8/8/8/KQ6 b - - 0 1", -2),      # same board, Black to move loses
    ("kq6/8/8/8/4K3/8/8/8 b - - 0 1", 2),       # colours swapped
    ("kq6/8/8/8/4K3/8/8/8 w - - 0 1", -2),
    ("8/8/8/4k3/8/8/8/KR6 w - - 0 1", 2),       # KRvK
    ("7k/8/8/8/8/8/8/KR6 b - - 0 1", -2),
    ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", 2),     # KPvK win
    ("k7/8/8/8/8/8/P7/K7 w - - 0 1", 0),        # rook-pawn draw, both sides to move
    ("k7/8/8/8/8/8/P7/K7 b - - 0 1", 0),
    ("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1", 0),      # stalemate
    ("k7/1Q6/1K6/8/8/8/8/8 b - - 0 1", -2),     # checkmated side to move
    ("8/8/8/8/8/8/1q6/K6k w - - 0 1", 0),       # Kxb2 into KvK: draw
]


def test_probe_wdl_batch_known_values_on_the_fixture() -> None:
    import chess.syzygy

    _tb_ext.init(str(FIXTURE))
    try:
        gen = _tb_ext.generation()
        boards = [chess.Board(fen) for fen, _ in KNOWN]
        for board in boards:
            assert board.is_valid(), board.fen()
        codes = _tb_ext.probe_wdl_batch([CBoard.from_board(b) for b in boards], gen)
        assert codes is not None
        got = [c - 2 for c in codes]
        assert got == [w for _, w in KNOWN], list(zip([f for f, _ in KNOWN], got, strict=True))
        with chess.syzygy.open_tablebase(str(FIXTURE)) as tb:
            assert [tb.probe_wdl(b) for b in boards] == got
    finally:
        _tb_ext.free()


def _sample_boards(seed: int, per_class: int) -> list[chess.Board]:
    from scripts.tb_prober_equivalence import gen_random

    rng = random.Random(seed)
    out: list[chess.Board] = []
    for cls in SAMPLED:
        got = 0
        while got < per_class:
            b = gen_random(rng, cls)
            if b is None:
                continue
            b.halfmove_clock = rng.choice((0, 0, 50, 95, 99, 100))
            out.append(b)
            got += 1
    return out


def test_leaf_differential_c_vs_python_chess_on_the_fixture() -> None:
    """Every sampled KQvK / KRvK / KPvK leaf and its C-pushed children
    (captures into KvK, promotions into KQvK..KNvK): identical
    rows, solved statuses, hits and probes."""
    tables = tb_root.load(str(FIXTURE))
    leaves: list[CBoard] = []
    rng = random.Random(7)
    for b in _sample_boards(20261115, 400):
        cb = CBoard.from_board(b)
        leaves.append(cb)
        moves = list(cb.legal_move_indices())
        rng.shuffle(moves)
        for idx in moves[:2]:
            child = cb.copy()
            child.push_index(int(idx))
            leaves.append(child)
    for rule50 in (True, False):
        py = tbmod.SyzygyProbe(str(FIXTURE), cursed_as_draw=rule50)
        c = CLeafProbe(tables, cursed_as_draw=rule50)
        assert py.max_pieces == c.max_pieces == 3
        solved_kinds: set[int] = set()
        for k in range(0, len(leaves), 64):
            batch = leaves[k:k + 64]
            for with_idx in (False, True):
                if with_idx:
                    batch = [cb for cb in batch
                             if (cb.occ_white | cb.occ_black).bit_count() <= 3]
                (pw, ps, ph), (cw, cs, ch) = _apply_both(py, c, batch, with_indices=with_idx,
                                                         seed=k)
                assert np.array_equal(ps, cs), [batch[j].fen() for j in np.flatnonzero(ps != cs)]
                assert np.array_equal(pw, cw)
                assert ph == ch
                solved_kinds |= set(np.unique(cs).tolist())
        assert (py.hits, py.probes) == (c.hits, c.probes)
        assert {1, -1, 2} <= solved_kinds, solved_kinds  # wins, losses and draws seen
        assert c.hits > 1000  # complete 3-man set: every eligible leaf is answered


def test_root_shortcut_differential_on_the_fixture() -> None:
    tables = tb_root.load(str(FIXTURE))
    roots = _sample_boards(99, 120)
    answered = 0
    for rule50 in (True, False):
        handle = tb_root.RootTablebase(tables, rule50=rule50)
        for board in roots:
            want = tbmod.try_tb_root_move(board, str(FIXTURE), rule50=rule50)
            got = handle.root_move(board)
            assert got == want, (board.fen(), rule50, got, want)
            answered += got is not None
    assert answered > 500


def test_engine_subprocess_on_the_fixture(tmp_path: Path) -> None:
    """The real entry point with ``SyzygyPath`` = the fixture: the load line
    counts its 5 tables, a ``searchmoves`` search reports leaf tbhits, the
    root shortcut answers KRvK, and no python-chess Syzygy entry point runs."""
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
        proc.stdin.write(f"setoption name SyzygyPath value {FIXTURE}\nisready\n")
        proc.stdin.flush()
        _read_until(proc, lines, "readyok", 180)
        for fen, go in (("8/8/8/4k3/8/8/8/KQ6 w - - 0 1", "go nodes 200 searchmoves b1b5 b1e1"),
                        ("8/8/8/4k3/8/8/8/KR6 w - - 0 1", "go nodes 64")):
            proc.stdin.write(f"position fen {fen}\n{go}\n")
            proc.stdin.flush()
            _read_until(proc, lines, "bestmove", 180)
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait(timeout=60)
        assert proc.stdout is not None
        lines += proc.stdout.read().splitlines()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    load = [ln for ln in lines if ln.startswith("info string SyzygyPath loaded from")]
    assert load, lines[-30:]
    assert "5 WDL + 5 DTZ tables, up to 3-piece positions; leaf prober: C Fathom" in load[0]
    assert not [ln for ln in lines if "PYCHESS_SYZYGY" in ln], lines[-30:]
    stats = [ln for ln in lines if ln.startswith("LEAFSTATS ")]
    assert stats, lines[-30:]
    assert int(dict(kv.split("=") for kv in stats[-1].split()[1:])["batches"]) > 0
    infos = [ln for ln in lines if ln.startswith("info depth") and " tbhits " in ln]
    tbhits = [int(ln.split(" tbhits ")[1].split()[0]) for ln in infos]
    assert max(tbhits) > 1, tbhits
    best = [ln.split()[1] for ln in lines if ln.startswith("bestmove ")]
    assert best[0] in ("b1b5", "b1e1"), best
    want = tbmod.try_tb_root_move(chess.Board("8/8/8/4k3/8/8/8/KR6 w - - 0 1"), str(FIXTURE))
    assert want is not None
    assert best[1] == want[0].uci(), best
