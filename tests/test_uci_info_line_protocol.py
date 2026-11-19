"""UCI output protocol fixes found by the 735-game CPU soak (2026-11-18).

- F6: nothing may follow ``pv`` on an ``info`` line (the multi-GPU pool used to
  append ``string pucv leaves=...``, which a GUI reads as moves).
- F7: per-search diagnostics (pool batch statistics, root-parallel Gumbel
  phase lines) print only under UCI ``debug on``.
- F10: an explicit multi-GPU mode on one device refuses to start, and
  ``searchconfig`` reports the real device count.
- F12: an unsearched forced move does not print a search score: an exact mate
  on the board is reported as one, anything else is labelled as the network's
  evaluation.
- F11: a mate in one is played at a root that is already a draw by rule.

The end-to-end tests drive the real entry point (``python -m
chess_anti_engine.uci``) on a tiny random-weight checkpoint with CPU devices.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves import POLICY_SIZE, move_to_index
from chess_anti_engine.uci import protocol
from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.search import SearchWorker, _board_line_mate
from chess_anti_engine.uci.time_manager import Deadline

FORCED_INTO_MATE = "7k/8/6KQ/8/8/8/8/8 b - - 0 1"  # Kg8 only, then Qg7#
FORCED = "7k/8/8/8/8/8/6q1/R6K w - - 0 1"  # Kxg2 only: K+R v K, nothing settled
FORCED_INTO_BARE_KINGS = "k7/8/8/8/8/8/1r6/K7 w - - 0 1"  # Kxb2 only: K v K
FORCED_INTO_STALEMATE = "5k2/3Q4/7N/8/8/8/1q6/1K6 w - - 0 1"  # Kxb2 only: Kf8 stalemated


@pytest.fixture(autouse=True)
def _debug_off() -> Iterator[None]:
    protocol.set_uci_debug(False)
    yield
    protocol.set_uci_debug(False)


class _SpreadEvaluator:
    """Deterministic pseudo-random policy / WDL per position."""

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        b = int(x.shape[0])
        rows = np.ascontiguousarray(x).reshape(b, -1)
        pol = np.zeros((b, POLICY_SIZE), dtype=np.float32)
        wdl = np.zeros((b, 3), dtype=np.float32)
        for i in range(b):
            digest = hashlib.blake2b(rows[i].tobytes(), digest_size=8).digest()
            r = np.random.default_rng(int.from_bytes(digest, "little"))
            pol[i, :] = r.standard_normal(POLICY_SIZE, dtype=np.float32)
            wdl[i, :] = r.standard_normal(3, dtype=np.float32)
        return pol, wdl


def _worker(walkers: int = 1) -> SearchWorker:
    return SearchWorker(
        _SpreadEvaluator(), device="cpu", chunk_sims=64, n_walkers=walkers,
        gumbel_cfg=GumbelConfig(simulations=64, topk=16, temperature=0.0, add_noise=False),
    )


def _dispatch(engine: Engine, *lines: str) -> None:
    for line in lines:
        engine.dispatch(parse_command(line))
    engine._wait_for_search()


def _assert_pv_lines_legal(lines: list[str], board: chess.Board) -> int:
    """Every token after ``pv`` on every ``info`` line is a legal move sequence
    from ``board``. Returns how many PV lines were checked."""
    checked = 0
    for line in lines:
        tokens = line.split()
        if not tokens or tokens[0] != "info" or "pv" not in tokens:
            continue
        if "string" in tokens and tokens.index("string") < tokens.index("pv"):
            continue  # `string` takes the rest of the line: free text, not a pv
        b = board.copy()
        for tok in tokens[tokens.index("pv") + 1:]:
            try:
                move = chess.Move.from_uci(tok)
            except ValueError:
                raise AssertionError(f"non-move {tok!r} after pv: {line}") from None
            assert move in b.legal_moves, f"illegal {tok!r} after pv: {line}"
            b.push(move)
        checked += 1
    return checked


# --- in-process ----------------------------------------------------------------


def test_debug_command_toggles_the_diagnostic_gate(capsys) -> None:
    engine = Engine(_worker())
    try:
        _dispatch(engine, "debug on")
        assert protocol.uci_debug_enabled() is True
        _dispatch(engine, "debug off")
        assert protocol.uci_debug_enabled() is False
        capsys.readouterr()
        _dispatch(engine, "debug loud")
        assert protocol.uci_debug_enabled() is False
        assert "info string debug: 'loud' is not 'on' or 'off'; ignored" in capsys.readouterr().out
    finally:
        engine.close()


@pytest.mark.parametrize(("devices", "expected"), [
    (None, "devices=1"),
    (("cpu",), "devices=1"),
    (("cpu", "cpu"), "devices=2 (routing dispatcher: one device call at a time)"),
])
def test_searchconfig_reports_the_real_device_count(capsys, devices, expected) -> None:
    """``--devices cpu,cpu --no-multi-gpu-pucv`` runs the classic path over the
    routing dispatcher on two devices; searchconfig used to say "1 device"."""
    engine = Engine(_worker(), search_devices=devices)
    try:
        capsys.readouterr()
        _dispatch(engine, "searchconfig")
        first = capsys.readouterr().out.splitlines()[0]
    finally:
        engine.close()
    assert first.startswith("info string searchconfig path=gumbel ")
    assert first.endswith(f" {expected}")
    assert "1 device" not in first


def _run_forced(fen: str, walkers: int = 1) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    _worker(walkers).run(
        chess.Board(fen), stop_event=threading.Event(), deadline=Deadline(20_000),
        max_nodes=None, optimum_ms=2_500, info_cb=lambda **kw: lines.append(kw),
    )
    return lines


@pytest.mark.parametrize("walkers", [1, 2])
def test_forced_move_into_mate_reports_mate_minus_one(walkers: int) -> None:
    """Soak F12: the classic path banks a forced move without visiting it and
    printed the root's network eval (``score cp 17``) where the truth is mated
    in one. The walker path searches it and already said so."""
    lines = _run_forced(FORCED_INTO_MATE, walkers)
    assert len(lines) == 1
    assert tuple(lines[0]["pv"])[:1] == ("h8g8",)
    assert lines[0]["score_mate"] == -1


@pytest.mark.parametrize(("halfmove", "expected"), [(0, -1), (98, -1), (99, None)])
def test_board_mate_is_not_claimed_past_a_claimable_draw(halfmove: int, expected) -> None:
    """Kg8 at halfmove 99 makes the 50-move draw claimable: the mover may claim
    it instead of facing Qg7#, so "mated in one" is no longer exact."""
    board = chess.Board(FORCED_INTO_MATE.replace(" 0 1", f" {halfmove} 1"))
    action = int(move_to_index(chess.Move.from_uci("h8g8"), board))
    assert _board_line_mate(board, action) == expected


def test_board_mate_is_not_claimed_when_the_move_completes_a_threefold() -> None:
    """Kh8 is forced and repeats its position a third time: the mover may claim
    the draw, so the opponent's Qg7# / Qb8# does not make it "mated in one"."""
    board = chess.Board("6k1/Q7/6K1/8/8/B7/8/R7 b - - 0 1")
    for mv in ("g8h8", "a1a2", "h8g8", "a2a1", "g8h8", "a1a2", "h8g8", "a2a1"):
        board.push_uci(mv)
    move = chess.Move.from_uci("g8h8")
    assert list(board.legal_moves) == [move]
    action = int(move_to_index(move, board))
    assert _board_line_mate(board, action) is None
    # The same forced move without the repetition history is mated in one.
    fresh = chess.Board(board.fen())
    assert _board_line_mate(fresh, int(move_to_index(move, fresh))) == -1


@pytest.mark.parametrize("walkers", [1, 2])
@pytest.mark.parametrize("fen", [FORCED_INTO_BARE_KINGS, FORCED_INTO_STALEMATE])
def test_forced_move_into_a_board_draw_scores_zero(fen: str, walkers: int) -> None:
    """Review item 6: a forced move that leaves insufficient material or
    stalemate is an exact draw; the classic path printed the root's eval."""
    lines = _run_forced(fen, walkers)
    assert len(lines) == 1
    assert tuple(lines[0]["pv"])[:1] == ("b1b2",) if fen == FORCED_INTO_STALEMATE else ("a1b2",)
    assert lines[0]["score_mate"] is None
    assert lines[0]["score_cp"] == 0
    assert "unsearched" not in (lines[0]["string"] or "")


def test_tb_root_shortcut_reports_the_board_mate() -> None:
    """Review item 5: the tablebase root shortcut printed a saturated cp for a
    root that is mated next move (Syzygy has no distances; the board does)."""
    from chess_anti_engine.uci.search import _try_tb_root_bestmove

    class _LossTable:
        def root_move(self, board: chess.Board) -> tuple[chess.Move, int]:
            del board
            return chess.Move.from_uci("h8g8"), -2

    result = _try_tb_root_bestmove(
        chess.Board(FORCED_INTO_MATE), _LossTable(), include_ponder=False,  # pyright: ignore[reportArgumentType]
    )
    assert result is not None
    assert result.bestmove_uci == "h8g8"
    assert result.score_mate == -1


def test_go_mate_on_a_forced_root_answers_at_once() -> None:
    """Review item 7: `go mate 1` on a forced root idled to the 60 s cap at
    nodes 0 (the classic path never simulates a forced root, so no mate can be
    proven and the move is decided anyway)."""
    w = _worker()
    t0 = time.monotonic()
    result = w.run(
        chess.Board(FORCED_INTO_MATE), stop_event=threading.Event(), deadline=Deadline(60_000),
        max_nodes=None, mate_moves=1,
    )
    assert time.monotonic() - t0 < 5.0
    assert result.bestmove_uci == "h8g8"
    assert w.last_stop_reason == "mate_forced_root"


def test_unsearched_forced_move_score_is_labelled() -> None:
    lines = _run_forced(FORCED)
    assert len(lines) == 1
    assert lines[0]["nodes"] == 0
    assert lines[0]["score_mate"] is None
    assert "score is the network's evaluation of this position" in (lines[0]["string"] or "")


def test_searched_move_carries_no_unsearched_label() -> None:
    lines: list[dict[str, Any]] = []
    _worker().run(
        chess.Board(), stop_event=threading.Event(), deadline=Deadline(20_000),
        max_nodes=128, info_cb=lambda **kw: lines.append(kw),
    )
    assert lines
    assert not any("unsearched" in (kw.get("string") or "") for kw in lines)


def test_unsearched_label_goes_on_its_own_line(capsys) -> None:
    engine = Engine(_worker())
    try:
        capsys.readouterr()
        _dispatch(engine, f"position fen {FORCED}", "go wtime 60000 btime 60000")
        out = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("info")]
    finally:
        engine.close()
    assert _assert_pv_lines_legal(out, chess.Board(FORCED)) == 1
    assert any(ln.startswith("info string move unsearched") for ln in out), out


def _threefold_mate_root() -> chess.Board:
    board = chess.Board("7k/Q7/6K1/8/8/8/8/Rn6 w - - 0 1")
    for mv in ("a1a2", "b1c3", "a2a1", "c3b1", "a1a2", "b1c3", "a2a1", "c3b1"):
        board.push_uci(mv)
    assert board.is_repetition(3)
    return board


_MATES_IN_ONE = {"a7a8", "a7b8", "a7g7"}


@pytest.mark.parametrize("walkers", [1, 2])
@pytest.mark.parametrize("root", ["halfmove100", "threefold"])
def test_mate_in_one_is_played_at_a_root_drawn_by_rule(walkers: int, root: str) -> None:
    """Soak F11: checkmate takes precedence over a claimable draw. At such a
    root the C Gumbel path declines to search and answers from the prior; the
    root mate check runs first and must still find the mate."""
    board = (chess.Board("7k/Q7/6K1/8/8/8/8/8 w - - 100 80") if root == "halfmove100"
             else _threefold_mate_root())
    assert not board.is_game_over()
    result = _worker(walkers).run(
        board, stop_event=threading.Event(), deadline=Deadline(20_000), max_nodes=64,
    )
    assert result.bestmove_uci in _MATES_IN_ONE, result
    assert result.score_mate == 1


# --- end to end: the real entry point --------------------------------------------


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("info_line_ckpt")
    ckpt_dir = root / "checkpoint_000001"
    ckpt_dir.mkdir()
    torch.manual_seed(0)
    model = build_model(ModelConfig(kind="tiny"))
    torch.save({"model": model.state_dict(), "step": 0}, ckpt_dir / "trainer.pt")
    (root / "params.json").write_text(
        json.dumps({"model": "tiny", "input_extra_features": "v2_threats"}), encoding="utf-8",
    )
    return ckpt_dir


def _engine_cmd(ckpt: Path, args: list[str]) -> tuple[list[str], dict[str, str]]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    repo = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-u", "-m", "chess_anti_engine.uci", "--checkpoint", str(ckpt),
           "--device", "cpu", "--no-compile", *args]
    return cmd, env


class _Session:
    def __init__(self, ckpt: Path, args: list[str]) -> None:
        cmd, env = _engine_cmd(ckpt, args)
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, bufsize=1,
        )
        self._q: queue.Queue[str | None] = queue.Queue()
        self.lines: list[str] = []

        def pump() -> None:
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self._q.put(line.rstrip("\n"))
            self._q.put(None)

        threading.Thread(target=pump, daemon=True).start()

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, pred: Callable[[str], bool], timeout: float = 120.0) -> list[str]:
        got: list[str] = []
        end = time.monotonic() + timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                raise AssertionError("timed out; transcript:\n" + "\n".join(self.lines[-60:]))
            try:
                line = self._q.get(timeout=min(left, 0.5))
            except queue.Empty:
                continue
            if line is None:
                raise AssertionError("engine exited; transcript:\n" + "\n".join(self.lines[-60:]))
            self.lines.append(line)
            got.append(line)
            if pred(line):
                return got

    def go(self, moves: tuple[str, ...], go: str) -> tuple[chess.Board, list[str]]:
        board = chess.Board()
        for mv in moves:
            board.push_uci(mv)
        position = "position startpos" + (f" moves {' '.join(moves)}" if moves else "")
        self.send(position)
        self.send(go)
        return board, self.read_until(lambda ln: ln.startswith("bestmove"))

    def close(self) -> None:
        if self.proc.poll() is None:
            with contextlib.suppress(Exception):
                self.send("quit")
                self.proc.wait(timeout=30)
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.communicate()


# (flags, the debug-only diagnostic this mode prints, or None)
_MODES: dict[str, tuple[list[str], str | None]] = {
    "classic": ([], None),
    "pool": (["--devices", "cpu,cpu"], "info string pucv leaves="),
    "rpg": (["--devices", "cpu,cpu", "--search-parallel", "gumbel"], "info string rpg phase="),
    "chain": (["--devices", "cpu,cpu", "--no-multi-gpu-pucv"], None),
    "walkers": (["--walkers", "4"], None),
}
_DIAGNOSTICS = ("pucv leaves=", "pucv pending=", "rpg phase=")


@pytest.mark.parametrize("mode", sorted(_MODES))
def test_every_info_pv_is_a_legal_move_sequence_in_every_search_mode(ckpt, mode) -> None:
    """Soak F6 + F7 end to end: parse every ``info`` line the engine prints and
    require the tokens after ``pv`` to be a legal line; diagnostics appear only
    under ``debug on``, each on its own ``info string`` line."""
    flags, diagnostic = _MODES[mode]
    sess = _Session(ckpt, flags)
    try:
        sess.send("uci")
        sess.read_until(lambda ln: ln == "uciok")
        sess.send("isready")
        sess.read_until(lambda ln: ln == "readyok")
        sess.send("searchconfig")
        path_line = sess.read_until(lambda ln: ln.startswith("info string searchconfig path="))[-1]
        assert ("devices=2" if "--devices" in flags else "devices=1") in path_line, path_line
        checked = 0
        quiet: list[str] = []
        for moves in ((), ("e2e4",)):
            board, got = sess.go(moves, "go nodes 512")
            checked += _assert_pv_lines_legal(got, board)
            quiet += got
        assert not [ln for ln in quiet if any(d in ln for d in _DIAGNOSTICS)], quiet
        sess.send("debug on")
        board, loud = sess.go(("e2e4", "e7e5"), "go nodes 512")
        checked += _assert_pv_lines_legal(loud, board)
        assert checked >= 3
        if diagnostic is not None:
            assert any(ln.startswith(diagnostic) for ln in loud), loud
        sess.send("debug off")
        board, quiet2 = sess.go(("d2d4",), "go nodes 512")
        _assert_pv_lines_legal(quiet2, board)
        assert not [ln for ln in quiet2 if any(d in ln for d in _DIAGNOSTICS)], quiet2
    finally:
        sess.close()


def test_pool_reports_exact_board_scores_for_forced_moves(ckpt) -> None:
    """Review item 5/6 on the TCEC path (multi-GPU pool): `score mate -1` for a
    forced move into mate (it printed `score cp -17 pv h8g8 h6c1`), `cp 0` for a
    forced move into a bare-kings draw."""
    sess = _Session(ckpt, ["--devices", "cpu,cpu"])
    try:
        sess.send("uci")
        sess.read_until(lambda ln: ln == "uciok")
        sess.send("isready")
        sess.read_until(lambda ln: ln == "readyok")
        for fen, score in ((FORCED_INTO_MATE, "score mate -1"), (FORCED_INTO_BARE_KINGS, "score cp 0")):
            sess.send(f"position fen {fen}")
            sess.send("go wtime 10000 btime 10000")
            got = sess.read_until(lambda ln: ln.startswith("bestmove"))
            pv_lines = [ln for ln in got if ln.startswith("info") and " pv " in ln]
            assert pv_lines, got
            assert all(f" {score} " in ln for ln in pv_lines), got
    finally:
        sess.close()


def test_multi_gpu_pucv_on_one_device_refuses_to_start(ckpt) -> None:
    """Soak F10 end to end: exit 1 with the reason on stderr, before any UCI line."""
    cmd, env = _engine_cmd(ckpt, ["--multi-gpu-pucv"])
    proc = subprocess.run(cmd, input="uci\nquit\n", capture_output=True, text=True,
                          env=env, timeout=120, check=False)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "--multi-gpu-pucv with 1 device" in proc.stderr
    assert "uciok" not in proc.stdout


def _bench(ckpt: Path, *args: str) -> subprocess.CompletedProcess[str]:
    _, env = _engine_cmd(ckpt, [])
    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "bench_uci_engine.py"), "--checkpoint", str(ckpt),
         "--device", "cpu", "--no-compile", "--repeats", "1", *args],
        capture_output=True, text=True, env=env, timeout=600, check=False,
    )


def test_bench_pool_run_still_reports_the_pucv_profile(ckpt) -> None:
    """The bench's `pucv profile:` line and cache rates are built from the pool
    statistics that now print only under `debug on`; the bench must send it.
    (Without it the bench raises instead of silently dropping the profile.)"""
    proc = _bench(ckpt, "--devices", "cpu,cpu", "--multi-gpu-pucv", "--nodes", "300",
                  "--eval-cache-entries", "1000")
    assert proc.returncode == 0, (proc.stdout[-3000:], proc.stderr[-3000:])
    profile = [ln for ln in proc.stdout.splitlines() if "pucv profile:" in ln]
    assert len(profile) == 1, proc.stdout
    assert " cache=" in profile[0], profile


def test_bench_refuses_multi_gpu_pucv_without_devices(ckpt) -> None:
    proc = _bench(ckpt, "--multi-gpu-pucv", "--nodes", "64")
    assert proc.returncode != 0
    assert "--multi-gpu-pucv needs --devices with at least two entries" in proc.stderr, proc.stderr
