from pathlib import Path

import pytest

from chess_anti_engine.worker_pool import build_worker_command


def test_build_worker_command_appends_child_workdir():
    cmd = build_worker_command(
        worker_args=["--server-url", "http://127.0.0.1:8000", "--update"],
        worker_dir=Path("/tmp/pool/worker_00"),
        shared_cache_dir=Path("/tmp/pool/shared_cache"),
    )
    assert cmd[:3] == ["python", "-m", "chess_anti_engine.worker"] or cmd[1:3] == ["-m", "chess_anti_engine.worker"]
    assert "--server-url" in cmd
    assert "--update" in cmd
    assert cmd[-4:] == [
        "--work-dir",
        "/tmp/pool/worker_00",
        "--shared-cache-dir",
        "/tmp/pool/shared_cache",
    ]


def test_build_worker_command_rejects_user_workdir_override():
    with pytest.raises(ValueError, match="do not pass --work-dir"):
        build_worker_command(
            worker_args=["--server-url", "http://127.0.0.1:8000", "--work-dir", "/tmp/other"],
            worker_dir=Path("/tmp/pool/worker_00"),
            shared_cache_dir=Path("/tmp/pool/shared_cache"),
        )


def test_build_worker_command_rejects_user_shared_cache_override():
    with pytest.raises(ValueError, match="do not pass --shared-cache-dir"):
        build_worker_command(
            worker_args=["--server-url", "http://127.0.0.1:8000", "--shared-cache-dir", "/tmp/other_cache"],
            worker_dir=Path("/tmp/pool/worker_00"),
            shared_cache_dir=Path("/tmp/pool/shared_cache"),
        )


def test_main_reaps_already_started_children_when_later_spawn_fails(tmp_path, monkeypatch):
    import sys

    import chess_anti_engine.worker_pool as worker_pool

    class Child:
        def __init__(self):
            self.terminated = False
            self.killed = False
            self.reaped = False

        def poll(self):
            return 0 if self.terminated or self.killed else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            del timeout
            self.reaped = True
            return 0

    child = Child()
    calls = 0

    def spawn(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return child
        raise OSError("process quota reached")

    monkeypatch.setattr(worker_pool.subprocess, "Popen", spawn)
    monkeypatch.setattr(worker_pool.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(worker_pool.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(
        sys, "argv",
        ["worker_pool", "--workers", "2", "--pool-work-dir", str(tmp_path)],
    )

    import pytest

    with pytest.raises(OSError, match="process quota reached"):
        worker_pool.main()

    assert child.terminated


# ── C-core refusal (exit 78) is not respawned ───────────────────────────────


class _ScriptedChild:
    """A Popen stand-in whose poll() returns a fixed exit code (None = alive)."""

    def __init__(self, rc):
        self.rc = rc
        self.terminated = False

    def poll(self):
        return self.rc

    def terminate(self):
        self.terminated = True
        if self.rc is None:
            self.rc = -15

    def kill(self):
        self.rc = -9

    def wait(self, timeout=None):
        del timeout
        return self.rc


def _drive_pool(monkeypatch, tmp_path, rcs, *, loops):
    """Run worker_pool.main(--respawn) with children exiting with ``rcs``.

    Returns (spawn_count, pool exit, stderr). The pool is stopped through its own
    SIGTERM handler after ``loops`` supervision passes.
    """
    import sys

    import chess_anti_engine.worker_pool as worker_pool

    handlers = {}
    spawned = []
    pending = list(rcs)

    def spawn(*_args, **_kwargs):
        child = _ScriptedChild(pending.pop(0) if pending else None)
        spawned.append(child)
        return child

    passes = 0

    def sleep(seconds):
        nonlocal passes
        if seconds == 1.0:  # one supervision pass
            passes += 1
            if passes > loops:
                handlers[worker_pool.signal.SIGTERM](worker_pool.signal.SIGTERM, None)

    monkeypatch.setattr(worker_pool.subprocess, "Popen", spawn)
    monkeypatch.setattr(worker_pool.signal, "signal", handlers.__setitem__)
    monkeypatch.setattr(worker_pool.time, "sleep", sleep)
    monkeypatch.setattr(sys, "argv", [
        "worker_pool", "--workers", str(len(rcs)), "--pool-work-dir", str(tmp_path),
        "--respawn", "--stagger-seconds", "0", "--restart-delay-seconds", "0.5",
    ])
    exit_code = None
    try:
        worker_pool.main()
    except SystemExit as exc:
        exit_code = exc.code
    return spawned, exit_code


def test_pool_does_not_respawn_a_child_that_refused_on_the_c_core(tmp_path, monkeypatch, capsys):
    from chess_anti_engine.ext_check import C_CORE_REFUSAL_EXIT_CODE

    spawned, exit_code = _drive_pool(
        monkeypatch, tmp_path, [C_CORE_REFUSAL_EXIT_CODE, None], loops=5,
    )
    assert len(spawned) == 2  # never relaunched
    assert exit_code is None  # one child is still healthy: the pool keeps running
    err = capsys.readouterr().err
    assert err.count("worker_00 exited 78") == 1  # logged once, not every pass
    assert "Not respawning" in err


def test_pool_still_respawns_every_other_exit_code(tmp_path, monkeypatch):
    for rc in (0, 1, 2, -9, -15):
        spawned, exit_code = _drive_pool(monkeypatch, tmp_path, [rc, None], loops=1)
        assert len(spawned) == 3, rc  # worker_00 relaunched once
        assert exit_code is None, rc


def test_pool_exits_78_when_every_child_refused(tmp_path, monkeypatch):
    from chess_anti_engine.ext_check import C_CORE_REFUSAL_EXIT_CODE

    spawned, exit_code = _drive_pool(
        monkeypatch, tmp_path, [C_CORE_REFUSAL_EXIT_CODE, C_CORE_REFUSAL_EXIT_CODE], loops=50,
    )
    assert len(spawned) == 2
    assert exit_code == C_CORE_REFUSAL_EXIT_CODE
