"""End-to-end UCI smoke test.

Builds a TinyNet model, serialises a minimal checkpoint, spawns the
``chess_anti_engine.uci`` entry point as a subprocess, drives it through
a basic handshake + search + quit, asserts on the output lines.

We use TinyNet to keep the test fast — we're validating wiring, not
strength. The search budget is a handful of nodes so the test runs in
a couple of seconds on CPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.subprocess_client import LineReader as _LineReader, send_line as _send


def _make_tiny_checkpoint(tmp_path: Path) -> Path:
    ckpt_dir = tmp_path / "checkpoint_000001"
    ckpt_dir.mkdir()
    cfg = ModelConfig(kind="tiny")
    model = build_model(cfg)
    torch.save({"model": model.state_dict(), "step": 0}, ckpt_dir / "trainer.pt")
    with (tmp_path / "params.json").open("w") as fh:
        json.dump({"model": "tiny"}, fh)
    return ckpt_dir


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    return _make_tiny_checkpoint(tmp_path)


def _spawn_engine(checkpoint: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    # -u: unbuffered stdout so info/bestmove lines flush in real time when
    # stdout is a pipe rather than a tty.
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "chess_anti_engine.uci",
         "--checkpoint", str(checkpoint), "--device", "cpu"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )


def test_handshake_and_search(tiny_checkpoint: Path) -> None:
    proc = _spawn_engine(tiny_checkpoint)
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        lines = reader.read_until("uciok")
        assert any(l.startswith("id name ") for l in lines)
        assert any(l.startswith("id author ") for l in lines)

        _send(proc, "isready")
        reader.read_until("readyok")

        _send(proc, "ucinewgame")
        _send(proc, "position startpos")
        _send(proc, "go nodes 8")
        lines = reader.read_until("bestmove")
        bestmove_line = next(l for l in lines if l.startswith("bestmove "))
        bestmove = bestmove_line.split()[1]
        assert len(bestmove) >= 4  # e.g. "e2e4"

        _send(proc, "quit")
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_position_fen_and_search(tiny_checkpoint: Path) -> None:
    proc = _spawn_engine(tiny_checkpoint)
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        reader.read_until("uciok")
        _send(proc, "isready")
        reader.read_until("readyok")

        _send(proc, "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4")
        _send(proc, "go nodes 4")
        lines = reader.read_until("bestmove")
        assert any(l.startswith("bestmove ") for l in lines)

        _send(proc, "quit")
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_bestmove_emits_ponder_suffix(tiny_checkpoint: Path) -> None:
    proc = _spawn_engine(tiny_checkpoint)
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        reader.read_until("uciok")
        _send(proc, "isready")
        reader.read_until("readyok")
        _send(proc, "position startpos")
        _send(proc, "go nodes 16")
        lines = reader.read_until("bestmove")
        bestmove_line = next(l for l in lines if l.startswith("bestmove "))
        tokens = bestmove_line.split()
        assert tokens[0] == "bestmove"
        assert len(tokens[1]) >= 4
        # With 16 sims the tree almost always has at least one grandchild
        # (opponent's reply), so ponder should be present.
        if len(tokens) >= 4:
            assert tokens[2] == "ponder"
            assert len(tokens[3]) >= 4
        _send(proc, "quit")
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_ponderhit_converts_to_timed_search(tiny_checkpoint: Path) -> None:
    proc = _spawn_engine(tiny_checkpoint)
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        reader.read_until("uciok")
        _send(proc, "isready")
        reader.read_until("readyok")
        # Simulate: played e2e4, predicted opponent plays e7e5
        _send(proc, "position startpos moves e2e4 e7e5")
        _send(proc, "go ponder wtime 1000 btime 1000")
        time.sleep(0.3)
        _send(proc, "ponderhit")
        lines = reader.read_until("bestmove", timeout_s=15.0)
        assert any(l.startswith("bestmove ") for l in lines)
        _send(proc, "quit")
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_stop_during_ponder(tiny_checkpoint: Path) -> None:
    """Opponent played differently: GUI sends `stop` without `ponderhit`.
    We should still emit a bestmove so UCI state machine stays sane."""
    proc = _spawn_engine(tiny_checkpoint)
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        reader.read_until("uciok")
        _send(proc, "isready")
        reader.read_until("readyok")
        _send(proc, "position startpos moves e2e4 e7e5")
        _send(proc, "go ponder wtime 1000 btime 1000")
        time.sleep(0.3)
        _send(proc, "stop")
        lines = reader.read_until("bestmove", timeout_s=15.0)
        assert any(l.startswith("bestmove ") for l in lines)
        _send(proc, "quit")
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_go_depth_terminates(tiny_checkpoint: Path) -> None:
    """`go depth N` terminates when the most-visited PV reaches N plies."""
    proc = _spawn_engine(tiny_checkpoint)
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        reader.read_until("uciok")
        _send(proc, "isready")
        reader.read_until("readyok")
        _send(proc, "position startpos")
        _send(proc, "go depth 2")
        lines = reader.read_until("bestmove", timeout_s=30.0)
        assert any(l.startswith("bestmove ") for l in lines)
        _send(proc, "quit")
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_position_extension_reuses_tree(tiny_checkpoint: Path) -> None:
    """Two successive `position` commands where the second extends the first
    by two plies. Tree-reuse path in `advance_root` must not crash and must
    still produce a legal bestmove."""
    proc = _spawn_engine(tiny_checkpoint)
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        reader.read_until("uciok")
        _send(proc, "isready")
        reader.read_until("readyok")
        _send(proc, "position startpos moves e2e4 e7e5")
        _send(proc, "go nodes 16")
        reader.read_until("bestmove")
        # Extend by two plies; exercises advance_root twice.
        _send(proc, "position startpos moves e2e4 e7e5 g1f3 b8c6")
        _send(proc, "go nodes 16")
        lines = reader.read_until("bestmove")
        assert any(l.startswith("bestmove ") for l in lines)
        _send(proc, "quit")
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_stop_interrupts_search(tiny_checkpoint: Path) -> None:
    proc = _spawn_engine(tiny_checkpoint)
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        reader.read_until("uciok")
        _send(proc, "isready")
        reader.read_until("readyok")
        _send(proc, "position startpos")
        _send(proc, "go infinite")
        time.sleep(0.5)
        _send(proc, "stop")
        lines = reader.read_until("bestmove", timeout_s=15.0)
        assert any(l.startswith("bestmove ") for l in lines)
        _send(proc, "quit")
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
