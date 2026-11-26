"""End-to-end UCI smoke test.

Builds a TinyNet model, serialises a minimal checkpoint, spawns the
``chess_anti_engine.uci`` entry point as a subprocess, drives it through
a basic handshake + search + quit, asserts on the output lines.

TinyNet and small search budgets keep protocol checks cheap. One handshake
case also exercises the default compiled CPU path; the rest run eagerly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import chess
import pytest
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.subprocess_client import LineReader as _LineReader
from chess_anti_engine.uci.subprocess_client import send_line as _send
from tests._child_env import gpu_free_env


def _make_tiny_checkpoint(tmp_path: Path) -> Path:
    ckpt_dir = tmp_path / "checkpoint_000001"
    ckpt_dir.mkdir()
    cfg = ModelConfig(kind="tiny")
    model = build_model(cfg)
    torch.save({"model": model.state_dict(), "step": 0}, ckpt_dir / "trainer.pt")
    with (tmp_path / "params.json").open("w") as fh:
        json.dump({"model": "tiny", "input_extra_features": "v2_threats"}, fh)
    return ckpt_dir


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    return _make_tiny_checkpoint(tmp_path)


def _spawn_engine(
    checkpoint: Path, *, compiled: bool = False, devices: str = "cpu",
    extra: tuple[str, ...] = (),
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    if compiled:
        # This case qualifies compilation even when a developer's shell has
        # disabled it or allowed Torch to hide compiler errors with eager fallback.
        for key in ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE"):
            env[key] = "0"
        # Torch interprets any nonempty SUPPRESS_ERRORS value, including "0", as true.
        env.pop("TORCHDYNAMO_SUPPRESS_ERRORS", None)
    # -u: unbuffered stdout so info/bestmove lines flush in real time when
    # stdout is a pipe rather than a tty.
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "chess_anti_engine.uci",
         "--checkpoint", str(checkpoint),
         *(("--device", devices) if "," not in devices else ("--devices", devices)),
         "--compile" if compiled else "--no-compile", *extra],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=gpu_free_env(env),
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_handshake_and_search(tiny_checkpoint: Path, compiled: bool) -> None:
    proc = _spawn_engine(tiny_checkpoint, compiled=compiled)
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        lines = reader.read_until("uciok")
        assert any(l.startswith("id name ") for l in lines)
        assert any(l.startswith("id author ") for l in lines)

        transcript = list(lines)
        _send(proc, "isready")
        transcript.extend(reader.read_until("readyok", timeout_s=120.0 if compiled else 60.0))

        _send(proc, "ucinewgame")
        _send(proc, "position startpos")
        _send(proc, "go nodes 8")
        lines = reader.read_until("bestmove")
        transcript.extend(lines)
        assert not any(
            marker in line
            for line in transcript
            for marker in ("search error:", "bestmove_fallback_used=", "engine load failed:")
        ), "\n".join(transcript)
        bestmove_line = next(l for l in lines if l.startswith("bestmove "))
        bestmove = bestmove_line.split()[1]
        assert chess.Move.from_uci(bestmove) in chess.Board().legal_moves

        _send(proc, "quit")
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


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
        proc.wait(timeout=5)


def test_bestmove_omits_ponder_suffix_by_default(tiny_checkpoint: Path) -> None:
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
        assert "ponder" not in tokens
        _send(proc, "quit")
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


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
        proc.wait(timeout=5)


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
        proc.wait(timeout=5)


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
        proc.wait(timeout=5)


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
        proc.wait(timeout=5)


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
        proc.wait(timeout=5)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_multi_device_routing_chain_starts(tiny_checkpoint: Path, compiled: bool) -> None:
    """``--devices cpu,cpu --no-multi-gpu-pucv --max-batch 256`` (the
    MultiGPUDispatcher routing chain) must reach ``readyok`` and search. It
    used to exit 1 at startup two ways: compiled, ``'MultiGPUDispatcher' object
    has no attribute 'evaluate_encoded_async'``; eager or compiled at a MaxBatch
    below the Gumbel leaf buffer, ``batch 384 > max 256``."""
    proc = _spawn_engine(
        tiny_checkpoint, compiled=compiled, devices="cpu,cpu",
        extra=("--no-multi-gpu-pucv", "--max-batch", "256"),
    )
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        reader.read_until("uciok")
        _send(proc, "isready")
        transcript = reader.read_until("readyok", timeout_s=240.0)
        assert not any("startup failed" in line for line in transcript), transcript
        _send(proc, "position startpos")
        _send(proc, "go nodes 64")
        lines = reader.read_until("bestmove", timeout_s=60.0)
        best = lines[-1].split()[1]
        assert chess.Move.from_uci(best) in chess.Board().legal_moves
        _send(proc, "quit")
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def test_pool_vlgather_change_mid_game_does_not_crash(tiny_checkpoint: Path) -> None:
    """Regression: on main before the full bucket warm-up, this sequence
    segfaulted (rc -11, 3 of 3) -- after `setoption VLGather 96` the rebuilt
    pool's two worker threads met a cold batch shape and compiled it
    concurrently on the next `go`. With every reachable shape warmed per
    device before `readyok`, no compile happens during the search."""
    proc = _spawn_engine(
        tiny_checkpoint, compiled=True, devices="cpu,cpu",
        extra=("--max-batch", "256", "--vl-gather", "200"),
    )
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        reader.read_until("uciok")
        _send(proc, "isready")
        reader.read_until("readyok", timeout_s=300.0)
        _send(proc, "position startpos moves e2e4")
        _send(proc, "go nodes 300")
        reader.read_until("bestmove", timeout_s=120.0)
        _send(proc, "setoption name VLGather value 96")
        _send(proc, "isready")
        reader.read_until("readyok", timeout_s=300.0)
        _send(proc, "position startpos moves e2e4 e7e5")
        _send(proc, "go nodes 300")
        lines = reader.read_until("bestmove", timeout_s=120.0)
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        assert chess.Move.from_uci(lines[-1].split()[1]) in board.legal_moves
        assert not any("first-seen batch shape" in line for line in lines), lines
        _send(proc, "quit")
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
