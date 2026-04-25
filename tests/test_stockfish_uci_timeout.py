"""Regression for F004: Stockfish reads must time out instead of hanging.

Pre-fix, ``StockfishUCI._wait_for`` and ``search`` called
``proc.stdout.readline()`` in unbounded loops. A stalled Stockfish (e.g.
protocol-deadlocked, hung at startup) would block selfplay/arena
indefinitely while holding ``self._lock``. Post-fix, both go through
``_readline_with_deadline`` which uses ``select`` with a configurable
``read_timeout_s`` and raises ``StockfishTimeoutError``.
"""
from __future__ import annotations

import stat
import sys

import pytest

from chess_anti_engine.stockfish.uci import StockfishTimeoutError, StockfishUCI


_SILENT_ENGINE = """\
import sys
for _line in sys.stdin:
    pass
"""

_HANDSHAKE_THEN_HANG = """\
import sys
for line in sys.stdin:
    line = line.strip()
    if line == 'uci':
        print('uciok', flush=True)
    elif line == 'isready':
        print('readyok', flush=True)
    # 'go' and everything else: read but never respond.
"""


def _make_engine_wrapper(tmp_path, body: str, name: str) -> str:
    py = tmp_path / f"{name}.py"
    py.write_text(body, encoding="utf-8")
    sh = tmp_path / f"{name}.sh"
    sh.write_text(
        f"#!/usr/bin/env bash\nexec {sys.executable} {py}\n",
        encoding="utf-8",
    )
    sh.chmod(sh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(sh)


def test_handshake_times_out_on_silent_engine(tmp_path) -> None:
    engine = _make_engine_wrapper(tmp_path, _SILENT_ENGINE, "silent")
    with pytest.raises(StockfishTimeoutError):
        StockfishUCI(engine, nodes=1, read_timeout_s=0.3)


def test_search_times_out_when_engine_hangs_after_go(tmp_path) -> None:
    engine = _make_engine_wrapper(tmp_path, _HANDSHAKE_THEN_HANG, "lazy")
    sf = StockfishUCI(engine, nodes=1, read_timeout_s=0.3)
    try:
        with pytest.raises(StockfishTimeoutError):
            sf.search("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    finally:
        sf.close()
