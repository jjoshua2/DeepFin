from __future__ import annotations

import stat
import sys
from pathlib import Path
from types import SimpleNamespace

from chess_anti_engine.selfplay.config import GameConfig
from chess_anti_engine.selfplay.stockfish_turn import _sf_syzygy_path_for_slot
from chess_anti_engine.stockfish.uci import StockfishUCI


def _state_for_path(
    *, pieces: int, adjudicate: bool = True, adj_roll: int = 0,
    normal_path: str | None = "/ssd/tb",
    full_path: str | None = "/ssd/tb:/mnt/e/dtz",
):
    return SimpleNamespace(
        game=GameConfig(
            syzygy_path=full_path,
            stockfish_syzygy_path=normal_path,
            syzygy_adjudicate=adjudicate,
        ),
        cboards=[
            SimpleNamespace(
                occ_white=(1 << min(pieces, 63)) - 1,
                occ_black=0,
                castling=0,
            )
        ],
        tb_adj_roll_arr=[adj_roll],
    )


def test_stockfish_syzygy_path_switches_only_for_playthrough_tb_roots() -> None:
    assert _sf_syzygy_path_for_slot(_state_for_path(pieces=7, adj_roll=0), 0) == "/ssd/tb"
    assert _sf_syzygy_path_for_slot(_state_for_path(pieces=6, adj_roll=1), 0) == "/ssd/tb"
    assert _sf_syzygy_path_for_slot(_state_for_path(pieces=6, adj_roll=0), 0) == "/ssd/tb:/mnt/e/dtz"
    assert (
        _sf_syzygy_path_for_slot(_state_for_path(pieces=6, adjudicate=False), 0)
        == "/ssd/tb:/mnt/e/dtz"
    )


def test_stockfish_uci_caches_per_search_syzygy_path_switch(tmp_path: Path) -> None:
    log_path = tmp_path / "commands.log"
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
        print('info depth 1 multipv 1 score cp 0 wdl 1 998 1 pv e2e4', flush=True)
        print('bestmove e2e4', flush=True)
""",
        encoding="utf-8",
    )
    engine_sh = tmp_path / "engine.sh"
    engine_sh.write_text(f"#!/usr/bin/env bash\nexec {sys.executable} {engine_py}\n", encoding="utf-8")
    engine_sh.chmod(engine_sh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    sf = StockfishUCI(str(engine_sh), nodes=1, syzygy_path="/ssd/tb", read_timeout_s=1.0)
    try:
        fen = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
        sf.search(fen, syzygy_path="/ssd/tb")
        sf.search(fen, syzygy_path="/ssd/tb:/mnt/e/dtz")
        sf.search(fen, syzygy_path="/ssd/tb:/mnt/e/dtz")
        sf.search(fen, syzygy_path="/ssd/tb")
    finally:
        sf.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    syzygy_sets = [line for line in lines if line.startswith("setoption name SyzygyPath value ")]
    assert syzygy_sets == [
        "setoption name SyzygyPath value /ssd/tb",
        "setoption name SyzygyPath value /ssd/tb:/mnt/e/dtz",
        "setoption name SyzygyPath value /ssd/tb",
    ]
