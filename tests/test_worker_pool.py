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
