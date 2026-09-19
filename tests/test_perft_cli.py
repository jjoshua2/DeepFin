"""Exercise the installed perft CLI, including its failure exit statuses."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from chess_anti_engine.encoding.perft import main


def test_perft_cli_module_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "chess_anti_engine.encoding.perft", "2", "--expect", "400"],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Nodes: 400" in result.stdout


def test_perft_cli_divide_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["2", "--divide", "--json", "--expect", "400"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["nodes"] == 400
    assert len(output["divide"]) == 20
    assert set(output["divide"].values()) == {20}
    assert output["matches_expected"] is True
    assert output["slider_backend"] in {"magic", "pext", "rays"}


def test_perft_cli_mismatch(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["1", "--expect", "21"]) == 1
    output = capsys.readouterr()
    assert "Nodes: 20" in output.out
    assert "expected 21, got 20" in output.err


def test_perft_cli_root_moves(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["1", "--moves", "e2e4", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["nodes"] == 20
    assert output["fen"].split()[1:4] == ["b", "KQkq", "e3"]


@pytest.mark.parametrize(
    "args",
    [
        ["-1"], ["65"], ["0", "--divide"], ["1", "--expect", "-1"],
        ["1", "--fen", "not a FEN"],
        ["1", "--fen", "8/8/8/8/8/8/8/8 w - - 0 1"],
        ["1", "--moves", "e2e5"],
    ],
)
def test_perft_cli_invalid_input(args: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX interval timers")
def test_native_perft_is_interruptible() -> None:
    # Isolate signals and bound the whole test if a signal-polling regression
    # accidentally makes this enormous traversal uninterruptible.
    code = """
import signal
from chess import Board
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.perft import perft

def interrupt(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGALRM, interrupt)
signal.setitimer(signal.ITIMER_REAL, 0.05)
try:
    perft(CBoard.from_board(Board()), 10)
except KeyboardInterrupt:
    print('interrupted')
else:
    raise AssertionError('timer did not interrupt perft')
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        check=False, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "interrupted"
