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
    with pytest.raises(ValueError):
        build_worker_command(
            worker_args=["--server-url", "http://127.0.0.1:8000", "--work-dir", "/tmp/other"],
            worker_dir=Path("/tmp/pool/worker_00"),
            shared_cache_dir=Path("/tmp/pool/shared_cache"),
        )


def test_build_worker_command_rejects_user_shared_cache_override():
    with pytest.raises(ValueError):
        build_worker_command(
            worker_args=["--server-url", "http://127.0.0.1:8000", "--shared-cache-dir", "/tmp/other_cache"],
            worker_dir=Path("/tmp/pool/worker_00"),
            shared_cache_dir=Path("/tmp/pool/shared_cache"),
        )
