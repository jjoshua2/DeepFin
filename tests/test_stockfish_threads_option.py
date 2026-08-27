"""``StockfishUCI(threads=...)`` reaches the engine's stdin, and 1 is byte-stable.

⚑⚑ THE KNOB IS PROVED AT THE CONSUMER, NOT AT THE CONSTRUCTOR. This repository's
signature defect is a value that is accepted, stored, announced and then never
read; `self.threads` being an int is not evidence of anything. UCI has no option
readback -- an engine cannot be asked what `Threads` it ended up with -- so the
strongest available observation is the exact line written to the engine, which
is what this file captures.

⚑ AND THE DEFAULT MUST BE BYTE-IDENTICAL TO THE LITERAL IT REPLACED. Every
production caller (selfplay labels, the arena, the deep-SF tools) constructs
`StockfishUCI` without `threads`, and the class previously sent
`setoption name Threads value 1` unconditionally. If the default emitted
anything else -- a different spelling or a different value -- this parameter
would have silently re-tuned every engine in the project. (Handshake ORDER is
deliberately not asserted: `setoption` lines before `isready` are order-free
in UCI, and the filter below discards ordering by construction.)
"""
from __future__ import annotations

import stat
import sys
from pathlib import Path

from chess_anti_engine.stockfish.uci import StockfishUCI


def _fake_engine(tmp_path: Path, log_path: Path) -> Path:
    engine_py = tmp_path / "engine.py"
    engine_py.write_text(
        f"""
import sys
from pathlib import Path
log = Path({str(log_path)!r})
for line in sys.stdin:
    cmd = line.strip()
    with log.open('a', encoding='utf-8') as f:
        f.write(cmd + '\\n')
    if cmd == 'uci':
        print('uciok', flush=True)
    elif cmd == 'isready':
        print('readyok', flush=True)
    elif cmd.startswith('go '):
        print('info depth 1 multipv 1 score cp 0 pv e2e4', flush=True)
        print('bestmove e2e4', flush=True)
""",
        encoding="utf-8",
    )
    engine_sh = tmp_path / "engine.sh"
    engine_sh.write_text(
        f"#!/usr/bin/env bash\nexec {sys.executable} {engine_py}\n", encoding="utf-8",
    )
    engine_sh.chmod(
        engine_sh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
    )
    return engine_sh


def _threads_lines(tmp_path: Path, threads: int | None = None) -> list[str]:
    """Every ``setoption name Threads`` line one handshake wrote to the engine.

    ``threads=None`` constructs the engine with NO ``threads`` argument, which
    is what every production caller does; that path must stay byte-identical to
    the literal the class used to send.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    log_path = tmp_path / "commands.log"
    engine = _fake_engine(tmp_path, log_path)
    sf = (
        StockfishUCI(str(engine), nodes=1, read_timeout_s=5.0) if threads is None
        else StockfishUCI(str(engine), nodes=1, read_timeout_s=5.0, threads=threads)
    )
    sf.close()
    return [
        line for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("setoption name Threads")
    ]


def test_the_default_sends_exactly_the_line_it_always_sent(tmp_path: Path) -> None:
    assert _threads_lines(tmp_path) == ["setoption name Threads value 1"]


def test_a_requested_thread_count_reaches_the_engine(tmp_path: Path) -> None:
    assert _threads_lines(tmp_path / "four", 4) == [
        "setoption name Threads value 4",
    ]


def test_a_nonsensical_thread_count_is_clamped_up_to_one(tmp_path: Path) -> None:
    """0 threads is not a UCI value any engine accepts; clamping to 1 keeps the
    handshake valid rather than sending a line the engine will reject and then
    continuing as if it had not."""
    assert _threads_lines(tmp_path / "zero", 0) == [
        "setoption name Threads value 1",
    ]
