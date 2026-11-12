"""Every Syzygy opener builds handles with python-chess LRU eviction OFF.

python-chess 1.11's eviction can deadlock a handle probed from several threads
(lock-order inversion between ``Tablebase.lru_lock`` and a table's
``write_lock``/``read_count``; see :func:`chess_anti_engine.tablebase.new_tablebase`).
Selfplay probes one cached handle from every selfplay thread, so the openers must
pass ``max_fds=None``.

The first two tests run everywhere (empty, correctly named table files: python-chess
only parses the file NAME until a probe). The last two need the 3-4 man smoke set
and skip without it; CI has no tablebases.
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import threading
from pathlib import Path

import chess
import chess.syzygy
import pytest

import chess_anti_engine.tablebase as tbmod
from chess_anti_engine.utils.syzygy import syzygy_roots
from tests.script_loading import load_script_module


def _smoke_syzygy() -> Path | None:
    """The 3-4 man smoke set, in this checkout or the main one (``data/`` is untracked)."""
    for root in syzygy_roots():
        candidate = root / "data" / "syzygy_3-4man"
        if candidate.is_dir() and any(candidate.glob("*.rtbw")):
            return candidate
    return None


def _fake_table_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "tb"
    directory.mkdir()
    for name in ("KQvK.rtbw", "KQvK.rtbz", "KRvK.rtbw", "KRvK.rtbz"):
        (directory / name).write_bytes(b"")
    return directory


def test_every_opener_builds_a_real_handle_without_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _fake_table_dir(tmp_path)
    monkeypatch.setattr(tbmod, "_tablebases", {})

    cached = tbmod.get_tablebase(str(directory))
    assert type(cached) is chess.syzygy.Tablebase
    assert cached.max_fds is None

    strict = tbmod.open_strict_match_tablebase(str(directory), max_pieces=3)
    try:
        assert type(strict) is chess.syzygy.Tablebase
        assert strict.max_fds is None
    finally:
        strict.close()

    match_vs_uci = load_script_module("match_vs_uci.py", "match_vs_uci_no_eviction_test")
    adjudicator = match_vs_uci._open_syzygy_tablebase(str(directory))
    try:
        assert type(adjudicator) is chess.syzygy.Tablebase
        assert adjudicator.max_fds is None
    finally:
        adjudicator.close()
    cached.close()


class _RecordingTable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_no_eviction_handle_never_closes_a_table_under_lru_lock() -> None:
    """Structural: the only ``close()`` python-chess runs under ``lru_lock`` is
    ``_bump_lru``'s eviction, and ``max_fds=None`` makes it return first. The
    ``max_fds=2`` arm is the positive control that this probe can see eviction."""
    off = tbmod.new_tablebase()
    tables = [_RecordingTable() for _ in range(300)]
    for table in tables:
        off._bump_lru(table)  # pyright: ignore[reportArgumentType]
    assert not any(t.closed for t in tables)
    assert len(off.lru) == 0

    evicting = chess.syzygy.Tablebase(max_fds=2)
    control = [_RecordingTable() for _ in range(5)]
    for table in control:
        evicting._bump_lru(table)  # pyright: ignore[reportArgumentType]
    assert sum(t.closed for t in control) == 3


def _random_board(rng: random.Random, material: str) -> chess.Board:
    white, black = material.split("v")
    pieces = [(c, chess.WHITE) for c in white] + [(c, chess.BLACK) for c in black]
    while True:
        board = chess.Board(None)
        ok = True
        for (symbol, color), square in zip(pieces, rng.sample(range(64), len(pieces))):
            piece_type = chess.Piece.from_symbol(symbol).piece_type
            if piece_type == chess.PAWN and chess.square_rank(square) in (0, 7):
                ok = False
                break
            board.set_piece_at(square, chess.Piece(piece_type, color))
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        if ok and board.is_valid():
            return board


def test_smoke_tables_touched_from_threads_are_never_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = _smoke_syzygy()
    if smoke is None:
        pytest.skip("3-4 man smoke Syzygy set (data/syzygy_3-4man) not present")
    monkeypatch.setattr(tbmod, "_tablebases", {})
    closes: list[str] = []
    real_close = chess.syzygy.Table.close

    def recording_close(self: chess.syzygy.Table) -> None:
        closes.append(self.path)
        real_close(self)

    monkeypatch.setattr(chess.syzygy.Table, "close", recording_close)
    tb = tbmod.get_tablebase(str(smoke))
    assert tb is not None
    assert tb.max_fds is None
    materials = sorted({p.stem for p in smoke.glob("*.rtbw")})
    rng = random.Random(7)
    boards = [_random_board(rng, m) for m in materials for _ in range(3)]
    errors: list[BaseException] = []

    def worker(offset: int) -> None:
        try:
            for i in range(len(boards)):
                board = boards[(i + offset) % len(boards)].copy(stack=False)
                tb.probe_wdl(board)
                tb.probe_dtz(board)
        except BaseException as exc:  # surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k * 11,), daemon=True) for k in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads), "probe threads wedged"
    assert not errors, errors[:3]
    # Every WDL table and the DTZ tables a decisive probe reaches are mapped and
    # still open (drawn materials answer DTZ from WDL and never open their DTZ).
    wdl_mapped = {id(t) for t in tb.wdl.values() if t.data is not None}
    dtz_mapped = {id(t) for t in tb.dtz.values() if t.data is not None}
    assert len(wdl_mapped) == len(materials)
    assert len(dtz_mapped) >= 20
    assert closes == []
    assert len(tb.lru) == 0
    tb.close()


# Runs in a child process so a deadlock (the defect) is killed with the child
# instead of leaving wedged threads in the test process.
_STRESS_CHILD = r'''
import faulthandler, os, random, sys, threading, time
import chess
import chess_anti_engine.tablebase as tbmod

directory, n_threads, seconds, stall_s = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
sys.setswitchinterval(1e-6)  # maximal interleaving: the inversion needs a preemption inside probe
tb = tbmod.get_tablebase(directory)
print(f"max_fds={tb.max_fds}", flush=True)
materials = sorted({f[:-5] for f in os.listdir(directory) if f.endswith(".rtbw")})
rng = random.Random(1)
boards = []
while len(boards) < 2000:
    material = rng.choice(materials)
    white, black = material.split("v")
    pieces = [(c, chess.WHITE) for c in white] + [(c, chess.BLACK) for c in black]
    board = chess.Board(None)
    ok = True
    for (symbol, color), square in zip(pieces, rng.sample(range(64), len(pieces))):
        pt = chess.Piece.from_symbol(symbol).piece_type
        if pt == chess.PAWN and chess.square_rank(square) in (0, 7):
            ok = False
            break
        board.set_piece_at(square, chess.Piece(pt, color))
    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    if ok and board.is_valid():
        boards.append(board)
progress = [0] * n_threads
stop = threading.Event()

def run(tid):
    r = random.Random(1000 + tid)
    while not stop.is_set():
        board = boards[r.randrange(len(boards))].copy(stack=False)  # probes mutate the board
        tb.probe_wdl(board)
        progress[tid] += 1

for tid in range(n_threads):
    threading.Thread(target=run, args=(tid,), daemon=True).start()
start = last_change = time.monotonic()
last_total = -1
while time.monotonic() - start < seconds:
    time.sleep(0.25)
    total = sum(progress)
    now = time.monotonic()
    if total != last_total:
        last_total, last_change = total, now
    elif now - last_change > stall_s:
        print(f"HANG after {now - start:.1f}s, probes={total}", flush=True)
        faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
        sys.stdout.flush()
        os._exit(3)
stop.set()
print(f"OK probes={sum(progress)}", flush=True)
os._exit(0)
'''


def test_shared_cached_handle_survives_multithreaded_stress(tmp_path: Path) -> None:
    """32 threads (production ``--selfplay-threads``) on main's cached handle.

    A liveness smoke, not a detector: the smoke set has too few tables to
    evict at the default ``max_fds``, and even a forced ``max_fds=2`` wedges
    only stochastically (seen at 5-13 s, never inside this 6 s window). The
    deterministic guards are the opener and ``_bump_lru`` tests above; this
    one checks the realized ``max_fds`` and that 32 threads keep progressing.
    The child's no-progress watchdog reports a wedge and the outer timeout is
    the backstop."""
    smoke = _smoke_syzygy()
    if smoke is None:
        pytest.skip("3-4 man smoke Syzygy set (data/syzygy_3-4man) not present")
    script = tmp_path / "stress_child.py"
    script.write_text(_STRESS_CHILD, encoding="utf-8")
    repo_root = Path(tbmod.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo_root), "CUDA_VISIBLE_DEVICES": ""}
    try:
        result = subprocess.run(
            [sys.executable, str(script), str(smoke), "32", "6", "3"],
            capture_output=True, text=True, env=env, timeout=120, check=False,
        )
    except subprocess.TimeoutExpired as exc:  # backstop: child watchdog itself wedged
        pytest.fail(f"stress child hung past its own watchdog: {exc.stdout!r}")
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert "max_fds=None" in result.stdout
    assert "OK probes=" in result.stdout
