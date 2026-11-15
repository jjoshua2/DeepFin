"""End-to-end: the real ``deepfin`` entry point under injected device faults.

A driver script runs ``chess_anti_engine.uci.__main__.main()`` on a tiny
random-weight checkpoint with CPU "devices" (``--devices cpu,cpu``,
``--no-compile``). It swaps ``DirectGPUEvaluator`` for a subclass that turns
one device bad, chosen by the thread that builds it (pool workers are
``pucv-gpu-<k>`` / ``rpg-group-<k>``; routing-chain devices are numbered in
build order). Faults:

- ``raise``: every device call raises (CUDA error) once armed;
- ``stall``: the call never returns -- ``STALL_KIND=spin`` spins in torch
  (the shape that used to abort interpreter finalization at ``quit``, rc 134),
  ``STALL_KIND=block`` blocks without the GIL (the closer model of a CUDA
  hang);
- ``startup``: the device's first call (its warm-up) never returns.

A mid-search fault is ARMED by the first ``go`` (so the warm-up is clean).
Every ``go`` must get exactly one legal ``bestmove`` (inside ``searchmoves``
when given), and ``quit`` must exit 0 -- never 134 / -11.
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

_DRIVER = r'''
import os, sys, threading
import torch
from chess_anti_engine.uci import __main__ as uci_main
from chess_anti_engine.uci import engine as engine_mod
from chess_anti_engine.uci import startup_progress
from chess_anti_engine.uci.protocol import CmdGo

torch.set_num_threads(1)
MODE = os.environ["FAULT_MODE"]          # raise | stall | startup | none
WHERE = os.environ.get("FAULT_WHERE", "")  # comma list: pool:1, rpg:1, chain:1, primary, all-pool
AFTER = int(os.environ.get("FAULT_AFTER", "2"))
STALL_KIND = os.environ.get("STALL_KIND", "spin")  # spin (holds the GIL in torch) | block (GIL-free)
if os.environ.get("BUILDER_GRACE"):
    startup_progress.BUILDER_TIMEOUT_GRACE_S = float(os.environ["BUILDER_GRACE"])
ARMED = threading.Event()
if MODE == "startup":
    ARMED.set()
_orig_dispatch = engine_mod.Engine.dispatch
def dispatch(self, cmd):
    if isinstance(cmd, CmdGo):
        ARMED.set()
    return _orig_dispatch(self, cmd)
engine_mod.Engine.dispatch = dispatch

_chain_count = [0]
_lock = threading.Lock()
Real = uci_main.DirectGPUEvaluator

def _role():
    name = threading.current_thread().name
    for prefix in ("pucv-gpu-", "rpg-group-"):
        if name.startswith(prefix):
            return f"{'pool' if prefix == 'pucv-gpu-' else 'rpg'}:{name[len(prefix):]}"
    return "build"

class Faulty(Real):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        role = _role()
        if role == "build":
            with _lock:
                idx = _chain_count[0]
                _chain_count[0] += 1
            role = "primary" if idx == 0 else "build"
            self._chain_role = f"chain:{idx}"
        else:
            self._chain_role = ""
        targets = set(WHERE.split(","))
        self._bad = (role in targets or self._chain_role in targets
                     or ("all-pool" in targets and role.startswith(("pool:", "rpg:"))))
        self._calls = 0
        if self._bad:
            for name in dir(Real):
                if name.startswith("evaluate"):
                    setattr(self, name, self._gate(getattr(self, name)))

    def _gate(self, fn):
        def gated(*a, **k):
            if ARMED.is_set():
                self._calls += 1
                if self._calls > (0 if MODE == "startup" else AFTER):
                    if MODE == "raise":
                        raise RuntimeError("CUDA error: device lost (injected)")
                    if STALL_KIND == "block":
                        threading.Event().wait()  # GIL-free: the closer model of a CUDA hang
                    t = torch.ones(64, 64)
                    while True:
                        t = torch.mm(t, t).clamp_(-1.0, 1.0)
            return fn(*a, **k)
        return gated

uci_main.DirectGPUEvaluator = Faulty
sys.argv = ["deepfin", *sys.argv[1:]]
sys.exit(uci_main.main())
'''


def _tiny_checkpoint(tmp_path: Path) -> Path:
    ckpt_dir = tmp_path / "checkpoint_000001"
    ckpt_dir.mkdir()
    torch.manual_seed(0)
    model = build_model(ModelConfig(kind="tiny"))
    torch.save({"model": model.state_dict(), "step": 0}, ckpt_dir / "trainer.pt")
    (tmp_path / "params.json").write_text(
        json.dumps({"model": "tiny", "input_extra_features": "v2_threats"}), encoding="utf-8",
    )
    return ckpt_dir


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory) -> Path:
    return _tiny_checkpoint(tmp_path_factory.mktemp("degrade_ckpt"))


class _Session:
    def __init__(self, ckpt: Path, tmp_path: Path, args: list[str], env: dict[str, str]) -> None:
        driver = tmp_path / "driver.py"
        driver.write_text(_DRIVER)
        full_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1",
                        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", **env)
        repo = str(Path(__file__).resolve().parents[1])
        full_env["PYTHONPATH"] = repo + os.pathsep + full_env.get("PYTHONPATH", "")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(driver), "--checkpoint", str(ckpt), "--no-compile",
             "--chunk-sims", "64", *args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=full_env, cwd=str(tmp_path), bufsize=1,
        )
        self.lines: list[str] = []
        import queue
        import threading
        self._q: queue.Queue[str | None] = queue.Queue()

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

    def read_until(self, pred, timeout: float) -> list[str]:
        import queue
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

    def ready(self, timeout: float = 120.0) -> list[str]:
        self.send("uci")
        self.read_until(lambda ln: ln == "uciok", timeout)
        self.send("isready")
        return self.read_until(lambda ln: ln == "readyok", timeout)

    def go(self, position: str, go: str, *, stop_after: float | None = None,
           timeout: float = 60.0) -> tuple[str, list[str]]:
        self.send(position)
        self.send(go)
        if stop_after is not None:
            time.sleep(stop_after)
            self.send("stop")
        got = self.read_until(lambda ln: ln.startswith("bestmove"), timeout)
        return got[-1].split()[1], got

    def quit(self, timeout: float = 60.0) -> tuple[int, str, str]:
        self.send("quit")
        rest = ""
        try:
            # ``communicate`` also reads stdout, racing the pump thread: keep
            # whatever it wins, or the quit line can vanish from the transcript.
            rest, err = self.proc.communicate(timeout=timeout)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.communicate()
        self.drain()
        self.lines.extend(ln for ln in (rest or "").splitlines() if ln)
        return int(self.proc.returncode), "\n".join(self.lines), err or ""

    def drain(self) -> None:
        """Move what the pump thread queued into ``lines`` (it owns stdout,
        so ``communicate`` never sees it). Call after the process exited."""
        import queue
        while True:
            try:
                line = self._q.get(timeout=2.0)
            except queue.Empty:
                break
            if line is None:
                break
            self.lines.append(line)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.communicate()


def _legal(position_moves: str, move: str, searchmoves: tuple[str, ...] = ()) -> None:
    board = chess.Board()
    for mv in position_moves.split():
        board.push_uci(mv)
    assert chess.Move.from_uci(move) in board.legal_moves, (position_moves, move)
    if searchmoves:
        assert move in searchmoves, (move, searchmoves)


# Each go variant: (go command, stop_after, searchmoves restriction).
_GOS = {
    "nodes": ("go nodes 600", None, ()),
    "movetime": ("go movetime 2500", None, ()),
    "lowclock": ("go wtime 2000 btime 2000", None, ()),
    "infinite": ("go infinite", 2.5, ()),
    "searchmoves": ("go movetime 2500 searchmoves a7a6 h7h6 g8f6", None, ("a7a6", "h7h6", "g8f6")),
}
# ``--vl-gather 32``: small batches, so both devices get work even on short
# searches (at the default 512 one worker can drain a whole low-clock budget).
_PATHS = {
    "pool": ["--devices", "cpu,cpu", "--vl-gather", "32"],
    "chain": ["--devices", "cpu,cpu", "--no-multi-gpu-pucv"],
    "rpg": ["--devices", "cpu,cpu", "--search-parallel", "gumbel", "--vl-gather", "32"],
}
_WHERE = {"pool": "pool:1", "chain": "chain:1", "rpg": "rpg:1"}


def _fault_env(mode: str, where: str) -> dict[str, str]:
    """``mode``: raise | stall (spinning in torch) | block (GIL-free stall)."""
    env = {"FAULT_MODE": "stall" if mode == "block" else mode, "FAULT_WHERE": where}
    if mode == "block":
        env["STALL_KIND"] = "block"
    return env


def _run_fault_session(ckpt: Path, tmp_path: Path, path: str, mode: str, go_kind: str,
                       where: str | None = None, after: int | None = None) -> None:
    env = _fault_env(mode, where or _WHERE[path])
    if after is not None:
        env["FAULT_AFTER"] = str(after)
    sess = _Session(ckpt, tmp_path, _PATHS[path], env)
    try:
        sess.ready()
        go, stop_after, restrict = _GOS[go_kind]
        move, _got = sess.go("position startpos moves e2e4", go, stop_after=stop_after)
        _legal("e2e4", move, restrict)
        # A second, clean go: searched on the survivors (never parked).
        move2, got2 = sess.go("position startpos moves e2e4 e7e5", "go nodes 300")
        _legal("e2e4 e7e5", move2)
        assert not any("parked" in ln or "search not started" in ln for ln in got2), got2
        rc, text, err = sess.quit()
    finally:
        sess.close()
    assert text.count("\nbestmove ") + text.startswith("bestmove ") == 2, text
    assert "WARNING multi-GPU DEGRADED" in text, text
    assert "terminate called" not in err, err
    assert rc == 0, (rc, text[-3000:], err[-3000:])


@pytest.mark.parametrize("mode", ["raise", "stall"])
@pytest.mark.parametrize("go_kind", sorted(_GOS))
def test_pool_device_fault_under_every_go_kind(ckpt, tmp_path, mode, go_kind) -> None:
    _run_fault_session(ckpt, tmp_path, "pool", mode, go_kind)


@pytest.mark.parametrize("mode", ["raise", "stall", "block"])
@pytest.mark.parametrize("go_kind", ["nodes", "movetime", "infinite", "searchmoves"])
@pytest.mark.parametrize("path", ["chain", "rpg"])
def test_chain_and_rpg_device_fault(ckpt, tmp_path, path, mode, go_kind) -> None:
    _run_fault_session(ckpt, tmp_path, path, mode, go_kind)


@pytest.mark.parametrize("mode", ["raise", "stall", "block"])
@pytest.mark.parametrize("where", ["primary", "rpg:0,primary"])
def test_rpg_survives_losing_the_primary_or_gpu0(ckpt, tmp_path, where, mode) -> None:
    """Root-parallel Gumbel losing the primary evaluator, or devices[0] (its
    group AND the primary that shares it): searched moves continue on the
    surviving group, announced, and quit exits 0."""
    # after=0: the first armed call fails (the primary makes about one root
    # evaluation per go; its install-time warm call runs before arming).
    _run_fault_session(ckpt, tmp_path, "rpg", mode, "movetime", where=where, after=0)


@pytest.mark.parametrize("path", ["pool", "chain", "rpg"])
def test_rebuild_after_a_wedge_leaves_the_device_out(ckpt, tmp_path, path) -> None:
    """A device wedges mid-search, then ``setoption MaxBatch`` rebuilds the
    evaluators and pool: the wedged device is left out at once (no startup-
    timeout wait before readyok), the engine keeps searching, quit exits 0."""
    sess = _Session(ckpt, tmp_path, _PATHS[path], _fault_env("block", _WHERE[path]))
    try:
        sess.ready()
        move, _ = sess.go("position startpos moves e2e4", "go movetime 2500")
        _legal("e2e4", move)
        started = time.monotonic()
        sess.send("setoption name MaxBatch value 512")
        sess.send("isready")
        sess.read_until(lambda ln: ln == "readyok", 120)
        assert time.monotonic() - started < 60
        move2, got2 = sess.go("position startpos moves e2e4 e7e5", "go nodes 300")
        _legal("e2e4 e7e5", move2)
        assert not any("parked" in ln or "search not started" in ln for ln in got2), got2
        rc, text, err = sess.quit()
    finally:
        sess.close()
    assert "WARNING multi-GPU DEGRADED" in text, text
    assert "terminate called" not in err, err
    assert rc == 0, (rc, text[-3000:], err[-3000:])


@pytest.mark.parametrize("stall_kind", ["spin", "block"])
def test_rpg_device_hung_at_startup_exits_1(ckpt, tmp_path, stall_kind) -> None:
    """Root-parallel Gumbel needs every group: a group whose build hangs fails
    the start with exit 1 (documented), never an abort or a crash (-6/-11)."""
    sess = _Session(ckpt, tmp_path, [*_PATHS["rpg"], "--device-startup-timeout", "3"],
                    {"FAULT_MODE": "startup", "FAULT_WHERE": "rpg:1", "BUILDER_GRACE": "1",
                     "STALL_KIND": stall_kind})
    try:
        sess.send("uci")
        sess.send("isready")
        rc = sess.proc.wait(timeout=90)
        _out, err = sess.proc.communicate(timeout=10)
        sess.drain()
    finally:
        sess.close()
    text = "\n".join(sess.lines)
    assert rc == 1, (rc, text[-2000:], (err or "")[-3000:])
    assert "readyok" not in text
    assert "engine startup failed" in text, text[-2000:]
    assert "terminate called" not in (err or ""), err


@pytest.mark.parametrize("stall_kind", ["spin", "block"])
@pytest.mark.parametrize("path", ["pool", "chain"])
def test_a_device_hung_at_startup_is_dropped_and_the_engine_plays(ckpt, tmp_path, path,
                                                                  stall_kind) -> None:
    sess = _Session(ckpt, tmp_path, [*_PATHS[path], "--device-startup-timeout", "3"],
                    {"FAULT_MODE": "startup", "FAULT_WHERE": _WHERE[path], "STALL_KIND": stall_kind})
    try:
        started = time.monotonic()
        before = sess.ready(timeout=90)
        assert time.monotonic() - started < 60
        assert any("DEGRADED" in ln and "at startup" in ln for ln in sess.lines), sess.lines
        assert any(ln.startswith("info string multi-GPU degraded: 1 of 2") for ln in before), before
        move, _ = sess.go("position startpos", "go nodes 300")
        _legal("", move)
        rc, text, err = sess.quit()
    finally:
        sess.close()
    assert "terminate called" not in err, err
    assert rc == 0, (rc, text[-2000:], err[-2000:])


@pytest.mark.parametrize(("args", "where"), [
    (["--devices", "cpu,cpu"], "all-pool"),     # every pool device hangs
    (["--device", "cpu"], "primary"),           # the only device hangs (builder watchdog)
])
def test_no_device_coming_up_fails_loudly_not_a_hang(ckpt, tmp_path, args, where) -> None:
    sess = _Session(ckpt, tmp_path, [*args, "--device-startup-timeout", "2"],
                    {"FAULT_MODE": "startup", "FAULT_WHERE": where, "BUILDER_GRACE": "1"})
    try:
        started = time.monotonic()
        sess.send("uci")
        sess.send("isready")
        rc = sess.proc.wait(timeout=90)
        took = time.monotonic() - started
        _out, err = sess.proc.communicate(timeout=10)
        sess.drain()
    finally:
        sess.close()
    text = "\n".join(sess.lines)
    assert rc == 1, (rc, text[-2000:], (err or "")[-2000:])
    assert took < 60
    assert "readyok" not in text
    assert "engine startup failed" in text, text[-2000:]
    assert "made no progress" in text, text[-2000:]
    assert "terminate called" not in (err or "")


def test_every_device_failing_mid_search_still_answers_each_go(ckpt, tmp_path) -> None:
    sess = _Session(ckpt, tmp_path, _PATHS["pool"],
                    {"FAULT_MODE": "raise", "FAULT_WHERE": "all-pool,primary", "FAULT_AFTER": "0"})
    try:
        sess.ready()
        for moves, go in ([("e2e4", "go movetime 800"), ("e2e4 e7e5", "go nodes 200"),
                                         ("e2e4 e7e5 g1f3", "go wtime 2000 btime 2000")]):
            move, _ = sess.go(f"position startpos moves {moves}", go)
            _legal(moves, move)
        rc, text, err = sess.quit()
    finally:
        sess.close()
    assert text.count("bestmove ") == 3
    assert "bestmove_fallback_used" in text
    assert "terminate called" not in err
    assert rc == 0, (rc, text[-2000:], err[-2000:])


@pytest.mark.parametrize("mode", ["stall", "block"])
@pytest.mark.parametrize("path", ["pool", "chain", "rpg"])
def test_quit_during_a_stall_exits_zero(ckpt, tmp_path, path, mode) -> None:
    """`quit` while a dropped device's thread is still stuck (spinning in
    torch, or blocked GIL-free): exit 0 with the reason line, never the
    finalization abort (134) or a crash (-11)."""
    sess = _Session(ckpt, tmp_path, _PATHS[path], _fault_env(mode, _WHERE[path]))
    try:
        sess.ready()
        move, _ = sess.go("position startpos moves d2d4", "go movetime 2000")
        _legal("d2d4", move)
        rc, text, err = sess.quit()
    finally:
        sess.close()
    assert "terminate called" not in err, err
    assert rc == 0, (rc, text[-2000:], err[-2000:])
    assert "info string quit:" in text, text[-2000:]
    assert "exiting with status 0" in text, text[-2000:]
