"""LineReader must attribute a response to its UCI command word.

``info string bestmove_fallback_used=...`` contains the letters of the
command the bench and smoke client wait for. A substring match returns
before the real ``bestmove`` line and leaves that line for the next read.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from chess_anti_engine.uci.subprocess_client import LineReader, send_line


def _fake_engine(tmp_path: Path) -> subprocess.Popen[str]:
    script = tmp_path / "fake_uci.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.write('id name Fake\\n')\n"
        "sys.stdout.write('uciok\\n')\n"
        "sys.stdout.write('readyok\\n')\n"
        "sys.stdout.write("
        "'info string bestmove_fallback_used=1 source=root_policy "
        "move=a7a6 exception=RuntimeError\\n')\n"
        "sys.stdout.write('bestmove a7a6\\n')\n"
        "sys.stdout.write('info depth 1 nodes 1 score cp 3 pv e2e4\\n')\n"
        "sys.stdout.write('bestmove e2e4\\n')\n"
        "sys.stdout.flush()\n",
        encoding="utf-8",
    )
    return subprocess.Popen(
        [sys.executable, "-u", str(script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def test_bestmove_wait_skips_info_substring_and_keeps_the_next_reply(
    tmp_path: Path,
) -> None:
    proc = _fake_engine(tmp_path)
    reader = LineReader(proc)
    try:
        send_line(proc, "uci")
        handshake = reader.read_until("uciok", timeout_s=2.0)
        assert handshake[-1] == "uciok"
        ready = reader.read_until("readyok", timeout_s=2.0)
        assert ready == ["readyok"]

        first = reader.read_until("bestmove", timeout_s=2.0)
        assert first[0].startswith("info string bestmove_fallback_used=")
        assert first[-1] == "bestmove a7a6"

        second = reader.read_until("bestmove", timeout_s=2.0)
        assert second[-1] == "bestmove e2e4"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


@pytest.mark.parametrize(
    ("reply", "needle"),
    [
        ("bestmove\te2e4", "bestmove"),
        ("bestmove \t e2e4", "bestmove"),
        (" \tbestmove\te2e4", "bestmove"),
        (" \treadyok \t", "readyok"),
        ("readyok", "readyok"),
        ("bestmove e2e4 ponder e7e5", "bestmove e2e4 ponder e7e5"),
    ],
    ids=["tab", "mixed", "leading", "padded-token", "exact-token", "whole-line"],
)
def test_whitespace_responses_preserve_consecutive_attribution(reply: str, needle: str) -> None:
    diagnostic = "info string " + needle + " is not a response"
    lines = [diagnostic, reply, "info string bestmove_fallback_used=1", "bestmove d2d4"]
    script = "import sys; sys.stdout.write(" + repr("\n".join(lines) + "\n") + "); sys.stdout.flush()"
    with subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE,
        text=True,
    ) as proc:
        reader = LineReader(proc)
        assert reader.read_until(needle, timeout_s=2.0) == [diagnostic, reply]
        assert reader.read_until("bestmove", timeout_s=2.0) == lines[2:]
        proc.wait(timeout=5)


@pytest.mark.parametrize("line", [" \t", "bestmove\te2e4"])
def test_whole_line_needle_requires_exact_text(line: str) -> None:
    script = "print(" + repr(line) + ", flush=True)"
    with subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE,
        text=True,
    ) as proc:
        reader = LineReader(proc)
        with pytest.raises(RuntimeError, match="engine exited before"):
            reader.read_until("bestmove e2e4", timeout_s=2.0)
        proc.wait(timeout=5)
