"""Harness server paths share the caller-relative run-directory contract."""
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.tune import distributed_runtime, harness


@pytest.mark.parametrize("absolute_work_dir", [False, True])
@pytest.mark.parametrize("override_kind", ["wsl", "ordinary", "relative", "none"])
def test_server_launch_uses_caller_run_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    absolute_work_dir: bool, override_kind: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    relative = Path("runs") / "experiment"
    work_dir = tmp_path / relative if absolute_work_dir else relative
    override = {
        "wsl": "/mnt/c/chess_active/server",
        "ordinary": str(tmp_path / "explicit_server"),
        "relative": "explicit_server",
        "none": "",
    }[override_kind]
    expected = (
        tmp_path / "runs" / "experiment_server" if override_kind == "wsl"
        else tmp_path / "explicit_server" if override_kind in {"ordinary", "relative"}
        else tmp_path / relative / "server"
    )
    mkdir_paths: list[Path] = []
    monkeypatch.setattr(Path, "mkdir", lambda self, **kwargs: mkdir_paths.append(self))
    auth_roots: list[Path] = []

    def auth(*, server_root: Path, **_kwargs: Any) -> tuple[str, Path]:
        auth_roots.append(server_root)
        return "fixture-user", tmp_path / "fixture-password"

    monkeypatch.setattr(harness, "_prepare_distributed_worker_auth", auth)
    spawned: list[dict[str, Any]] = []
    process = object()

    def spawn(**kwargs: Any) -> object:
        spawned.append(kwargs)
        return process

    monkeypatch.setattr(distributed_runtime, "_spawn_with_reap", spawn)
    ready: list[dict[str, Any]] = []
    monkeypatch.setattr(harness, "_wait_for_server_ready", lambda **kw: ready.append(kw))
    config: dict[str, Any] = {
        "distributed_workers_per_trial": 1,
        "distributed_server_port": 45999,
        "distributed_server_root_override": override,
    }
    assert harness._launch_distributed_server(base_config=config, work_dir=work_dir) is process
    assert mkdir_paths == [expected]
    assert auth_roots == [expected]
    assert len(spawned) == 1
    cmd = spawned[0]["cmd"]
    assert cmd[cmd.index("--server-root") + 1] == str(expected)
    assert spawned[0]["log_path"] == expected / "server.log"
    assert spawned[0]["reap_terms"] == ["--server-root", str(expected)]
    assert ready == [{"base_url": "http://127.0.0.1:45999", "proc": process}]
    assert config["distributed_server_root"] == str(expected)
